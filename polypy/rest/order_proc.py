import datetime
import re
import warnings
from typing import Literal, NoReturn

import msgspec
import requests

from polypy.exceptions import (
    OrderPlacementDelayed,
    OrderPlacementFailure,
    OrderPlacementMarketNotReady,
    OrderPlacementUnmatched,
    OrderUpdateException,
    PolyPyException,
)
from polypy.order.common import INSERT_STATUS, SIDE, OrderProtocol, update_order
from polypy.structs import CancelOrdersResponse, PostOrderResponse


def _not_frozen(orders: OrderProtocol | list[OrderProtocol]) -> None:
    if not isinstance(orders, list):  # single order
        if hasattr(orders, "_wrapped_order"):
            raise PolyPyException(
                "Cannot handle frozen order object. Use Order Manager instead."
            )
        return

    for order in orders:
        if hasattr(order, "_wrapped_order"):
            raise PolyPyException(
                "Cannot handle frozen order object. Use Order Manager instead."
            )


# todo test
def _find_status_unmatched_delayed_from_exc_note(
    note: str, default_return: INSERT_STATUS
) -> INSERT_STATUS:
    search_note = note.lower()

    if "status" not in search_note:
        return default_return

    if re.search(r'["\']?status["\']?\s*:\s*["\']?unmatched["\']?', search_note):
        return INSERT_STATUS.UNMATCHED
    elif re.search(r'["\']?status["\']?\s*:\s*["\']?delayed["\']?', search_note):
        return INSERT_STATUS.DELAYED

    # live and matched would not have caused an exception in the first place
    #   and in any other case (e.g. market not ready) we just leave it as INSERT_STATUS.DEFINED

    return default_return


def _find_order_id_from_str(note: str) -> str:
    id_cands = re.findall(r"0x[a-fA-F0-9]{64}", note)
    return id_cands[0] if len(id_cands) == 1 else ""


def _optimistic_check_errmsg_order_id(
    order: OrderProtocol, note: str, exc_str: str
) -> None:
    idx = _find_order_id_from_str(note)

    if idx and order.id != idx:
        raise OrderPlacementFailure(
            f"order.id={order.id} does not match orderID inferred from exception note={idx}. {exc_str}"
        )
    else:
        warnings.warn(
            f"Cannot infer orderID from exception note. "
            f"Optimistic assumption: response covers order.id={order.id}. "
            f"\nTraceback: {exc_str}"
        )
        # optimistic update, that response actually covers specified order object...


# todo test
def _raise_post_order_exception(
    exc: requests.HTTPError, order: OrderProtocol
) -> NoReturn:
    # we for sure know, that e.__notes__ exists
    note = str(exc.__notes__[0])

    exc_str = (
        f"Original exception: {exc}. Original exception note: {note}. Order: {order}."
    ).replace("\n", "")

    _optimistic_check_errmsg_order_id(order, note, exc_str)
    # todo optimistic assumption: response covers specified order object

    # in case of an exception, we still try to infer the order state
    # todo: in case of DELAYED or UNMATCHED we assume, that no size was matched at all (not even partially)
    if "delayed" in note:
        order.status = INSERT_STATUS.DELAYED
        raise OrderPlacementDelayed(
            f"Order marketable, but subject to matching delay. {exc_str}"
        ) from exc
    elif re.search(r"not\s+.*?\s+ready", note, re.IGNORECASE):
        order.status = _find_status_unmatched_delayed_from_exc_note(note, order.status)
        raise OrderPlacementMarketNotReady(
            f"The market is not yet ready to process new orders. {exc_str}"
        ) from exc
    else:
        order.status = _find_status_unmatched_delayed_from_exc_note(note, order.status)
        raise OrderPlacementFailure(f"Order placement failed. {exc_str}") from exc


def _assert_post_order_errmsg(
    resp_struct: PostOrderResponse, order: OrderProtocol
) -> None:
    if resp_struct.errorMsg == "":
        # empty error message, everything went well
        return

    # non-empty error message: we know for sure, failure has happened

    # try to check order id first
    if resp_struct.orderID and resp_struct.orderID != order.id:
        raise OrderPlacementFailure(
            f"order.id={order.id} does not match response.orderID={resp_struct.orderID}."
        )
    elif not resp_struct.orderID:
        _optimistic_check_errmsg_order_id(
            order, resp_struct.errorMsg, resp_struct.errorMsg
        )

    if resp_struct.status.lower() not in {"", "live", "matched"}:
        # Following states are invalid at this stage:
        #   - "":..........not state available in response
        #   - "live":......this would be an error-free state, but we know we encountered an error though
        #   - "matched":...this would be an error-free state, but we know we encountered an error though

        # if we got here, state is not "", "live", "matched", which means we can set state from response
        order.status = INSERT_STATUS(resp_struct.status.upper())
        return

    # struct was not filled properly, so we try to infer from error message
    search_msg = resp_struct.errorMsg.lower()

    if "delayed" in search_msg:
        order.status = INSERT_STATUS.DELAYED
        raise OrderPlacementDelayed(
            f"Order marketable, but subject to matching delay. "
            f"Server response: {msgspec.structs.asdict(resp_struct)}."
            f"Order: {order}."
        )
    elif "unmatched" in search_msg:
        order.status = INSERT_STATUS.UNMATCHED
        raise OrderPlacementUnmatched(
            f"Order marketable, but failure delaying, placement not successful. "
            f"Server response: {msgspec.structs.asdict(resp_struct)}."
            f"Order: {order}."
        )
    else:
        raise OrderPlacementFailure(
            f"Client-side error when posting order. "
            f"Cannot infer state from: {msgspec.structs.asdict(resp_struct)}."
            f"Order: {order}."
        )


def _assert_post_order_server_success(
    resp_struct: PostOrderResponse, order: OrderProtocol
) -> None:
    if resp_struct.success is False:
        raise OrderPlacementFailure(
            f"Server-side error when posting order. "
            f"Original response: {msgspec.structs.asdict(resp_struct)}."
            f"Order: {order}"
        )


def _assert_post_order_insert_state(order: OrderProtocol) -> None:
    if not isinstance(order.status, INSERT_STATUS):
        raise OrderPlacementFailure(
            f"order.status is not of type INSERT_STATUS. Got: {order.status}."
        )

    if order.status in {INSERT_STATUS.LIVE, INSERT_STATUS.MATCHED}:
        return

    if order.status is INSERT_STATUS.UNMATCHED:
        raise OrderPlacementUnmatched(
            f"Order marketable, but failure delaying, placement not successful. "
            f"Order: {order}."
        )
    elif order.status is INSERT_STATUS.DELAYED:
        raise OrderPlacementDelayed(
            f"Order marketable, but subject to matching delay. Order: {order}."
        )
    else:
        raise OrderPlacementFailure(
            f"Order state invalid after posting order. Order: {order}."
        )


def _update_post_order_fill(order: OrderProtocol, resp: PostOrderResponse) -> None:
    if order.id != resp.orderID:
        raise OrderPlacementFailure(
            f"order.id={order.id} does not match response.orderID={resp.orderID}."
        )

    status = INSERT_STATUS(resp.status.upper())

    if order.side is SIDE.BUY:
        size_matched = resp.takingAmount
    elif order.side is SIDE.SELL:
        size_matched = resp.makingAmount
    else:
        raise PolyPyException(f"Unknown side: {order.side}.")

    size_matched = order.numeric_type(size_matched or "0")

    # we use update_order instead of directly setting the attributes to assure, that
    #   we don't set outdated data (update_order has some basic checks)
    update_order(order=order, status=status, size_matched=size_matched)


def _update_cancel_orders(
    orders: list[OrderProtocol], resp: CancelOrdersResponse
) -> list[OrderProtocol]:
    if resp.not_canceled:
        canceled_ids = set(resp.canceled)
        for order in orders:
            if order.id in canceled_ids:
                order.status = INSERT_STATUS.CANCELED
    else:
        # safe to assume all orders canceled
        for order in orders:
            order.status = INSERT_STATUS.CANCELED

    return orders


def _assert_cancel_orders_mode(
    mode_not_canceled: Literal["except", "warn", "ignore"], resp: CancelOrdersResponse
) -> None:
    if resp.not_canceled:
        if mode_not_canceled == "except":
            raise OrderUpdateException(
                f"Cancellation of the following orders was not possible: {resp.not_canceled}."
            )
        elif mode_not_canceled == "warn":
            warnings.warn(
                f"{datetime.datetime.now()} | "
                f"Cancellation of the following orders was not possible: {resp.not_canceled}."
            )
