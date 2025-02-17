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
    id_cands = re.findall(r"0x[a-fA-F0-9]{24,128}", note)
    return id_cands[0] if len(id_cands) == 1 else ""


def _update_order_id(
    order: OrderProtocol, resp_struc: PostOrderResponse | None, note: str
) -> OrderProtocol:
    if order.id is not None and order.id != "":
        return order

    if (
        resp_struc is not None
        and resp_struc.orderID != ""
        and resp_struc.orderID is not None
    ):
        order.id = resp_struc.orderID
        return order

    if (idx := _find_order_id_from_str(note)) != "":
        order.id = idx
        return order

    if order.id is None or order.id == "":
        # we do not raise because we do not know order status for sure at this point (might be assigned downstream)
        warnings.warn(
            f"{datetime.datetime.now()} | Could not infer order id. Order: {order}."
        )

    return order


# todo test
def _raise_post_order_exception(
    exc: requests.HTTPError, order: OrderProtocol
) -> NoReturn:
    # we for sure know, that e.__notes__ exists
    note = str(exc.__notes__[0])

    exc_str = (
        f"Original exception: {exc}. Original exception note: {note}. Order: {order}."
    ).replace("\n", "")

    order = _update_order_id(order, None, note)

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


def _assert_post_order_server_success(
    resp_struct: PostOrderResponse, order: OrderProtocol
) -> None:
    if resp_struct.success is False:
        raise OrderPlacementFailure(
            f"Server-side error when posting order. "
            f"Original response: {msgspec.structs.asdict(resp_struct)}."
            f"Order: {order}"
        )


def _assert_post_order_errmsg(
    resp_struct: PostOrderResponse, order: OrderProtocol
) -> None:
    if resp_struct.errorMsg == "":
        # empty error message, everything went well
        return

    # non-empty error message: we know for sure, failure has happened

    order = _update_order_id(order, resp_struct, resp_struct.errorMsg)

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


def _assert_post_order_valid(order: OrderProtocol) -> None:
    if not isinstance(order.status, INSERT_STATUS):
        raise OrderPlacementFailure(
            f"order.status is not of type INSERT_STATUS. Got: {order.status}."
        )

    if order.status is INSERT_STATUS.UNMATCHED:
        raise OrderPlacementUnmatched(
            f"Order marketable, but failure delaying, placement not successful. "
            f"Order: {order}."
        )
    elif order.status is INSERT_STATUS.DELAYED:
        raise OrderPlacementDelayed(
            f"Order marketable, but subject to matching delay. Order: {order}."
        )
    elif order.status not in {INSERT_STATUS.LIVE, INSERT_STATUS.MATCHED}:
        raise OrderPlacementFailure(
            f"Order state invalid after posting order. Order: {order}."
        )

    if order.id is None or order.id == "":
        raise OrderPlacementFailure(
            f"order.id is invalid, got: {order.id}. Order: {order}."
        )


def _update_post_order_fill(
    order: OrderProtocol, resp: PostOrderResponse
) -> OrderProtocol:
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
    order = update_order(order=order, status=status, size_matched=size_matched)
    return order


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
