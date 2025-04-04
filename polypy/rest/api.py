import datetime
import math
import time
import warnings
from functools import lru_cache
from typing import Any, Callable, Literal

import msgspec
import requests
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey

from polypy.constants import ENDPOINT, SIG_DIGITS_SIZE
from polypy.exceptions import OrderUpdateException, PolyPyException
from polypy.order.common import SIDE, OrderProtocol, update_order
from polypy.position import PositionFactory, PositionProtocol
from polypy.rest.order_processing import (
    _assert_cancel_orders_mode,
    _assert_post_order_errmsg,
    _assert_post_order_server_success,
    _assert_post_order_valid,
    _not_frozen,
    _raise_post_order_exception,
    _update_cancel_orders,
    _update_order_id,
    _update_post_order_fill,
)
from polypy.rounding import round_floor_tenuis_ceil
from polypy.signing import (
    SIGNATURE_TYPE,
    build_hmac_signature,
    private_key_checksum_address,
)
from polypy.structs import (
    CancelOrdersResponse,
    MarketInfo,
    MarketsResponse,
    OpenOrderInfo,
    OpenOrderResponse,
    PostOrderResponse,
)
from polypy.typing import NumericAlias

# todo msgspec create specialized Decoders


END_CURSOR = "LTE="
DEFAULT_HEADERS = {
    "Accept": "*/*",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
}


def build_auth_header(
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    method: str,
    request_path: str,
    body: Any | None,
):
    timestamp = int(datetime.datetime.now().timestamp())
    hmac_signature = build_hmac_signature(secret, timestamp, method, request_path, body)

    return {
        "POLY_ADDRESS": private_key_checksum_address(private_key),
        "POLY_SIGNATURE": hmac_signature,
        "POLY_TIMESTAMP": str(timestamp),
        "POLY_API_KEY": api_key,
        "POLY_PASSPHRASE": passphrase,
    }


def _overload_headers(headers: dict | None, method: str) -> dict | None:
    if headers is None:
        headers = DEFAULT_HEADERS
    else:
        # only overwrite, if not yet defined
        headers.setdefault("Accept", "*/*")
        headers.setdefault("Connection", "keep-alive")
        headers.setdefault("Content-Type", "application/json")

    if method == "GET":
        headers.setdefault("Accept-Encoding", "gzip")

    return headers


def _parse_to_list(x: Any | list[Any]) -> tuple[bool, list[Any]]:
    return (False, x) if isinstance(x, list) else (True, [x])


def _request(
    url: str,
    method: Literal["GET", "POST", "DELETE"],
    headers: dict | None,
    data: Any | None,
) -> requests.Response:
    headers = _overload_headers(headers, method)
    resp = requests.request(method, url, json=data, headers=headers)

    try:
        resp.raise_for_status()
    except Exception as e:
        e.add_note(resp.text)
        raise e

    return resp


def get_market(endpoint: str | ENDPOINT, market: str) -> MarketInfo:
    resp = _request(f"{endpoint}/markets/{market}", "GET", None, None)
    return msgspec.json.decode(resp.text, type=MarketInfo)


def get_markets(
    endpoint: str | ENDPOINT,
    next_cursor: str | None,
    max_pages: int | None,
    throttle_s: float | None,
) -> tuple[list[MarketInfo], str]:
    """Query markets.

    Parameters
    ----------
    endpoint: str | Endpoint,
        REST endpoint to use, most likely polypy.ENDPOINT.REST
    next_cursor: str | None
        Pagination item to retrieve the next page base64 encoded. 'LTE=' means the end and empty ('')
        means the beginning. None will be parsed to "".
    max_pages: int | None
        maximum number of pages to query. If None, no upper limit.
    throttle_s: float | None
        sleep duration (seconds) to throttle for-loop inbetween pages. None defaults to 0.

    Returns
    -------
    tuple[list[MarketInfo], str]
        - list of market info structs
        - last cursor returned to retrieve next page

    Notes
    -----
    datetime in MarketInfo (e.g., MarketInfo.end_date_iso) are timezone-aware.
    """
    if next_cursor is None:
        next_cursor = ""

    if max_pages is None:
        max_pages = math.inf

    ret = []
    i = 0
    throttle_s = 0 if throttle_s is None else throttle_s

    while next_cursor != END_CURSOR and i < max_pages:
        i += 1
        resp = _request(
            f"{endpoint}/markets?next_cursor={next_cursor}",
            "GET",
            None,
            None,
        )
        market_response = msgspec.json.decode(
            resp.text, type=MarketsResponse, strict=False
        )

        next_cursor = market_response.next_cursor
        ret.extend(market_response.data)

        time.sleep(throttle_s)

    return ret, next_cursor


def _get_book_summary(endpoint: str | ENDPOINT, asset_id: str) -> dict[str, Any]:
    resp = _request(f"{endpoint}/book?token_id={asset_id}", "GET", None, None)
    return msgspec.json.decode(resp.text)


def get_book_summaries(
    endpoint: str | ENDPOINT, asset_ids: str | list[str]
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    if isinstance(asset_ids, str):
        return _get_book_summary(endpoint, asset_ids)

    body = [{"token_id": x} for x in asset_ids]
    resp = _request(url=f"{endpoint}/books", method="POST", headers=None, data=body)
    resp = msgspec.json.decode(resp.text)

    return {x["asset_id"]: x for x in resp}


def _get_midpoint(
    endpoint: str | ENDPOINT,
    asset_id: str,
    numeric_type: NumericAlias | type | Callable[[str], Any],
) -> NumericAlias:
    resp = _request(f"{endpoint}/midpoint?token_id={asset_id}", "GET", None, None)
    return numeric_type(msgspec.json.decode(resp.text)["mid"])


def get_midpoints(
    endpoint: str | ENDPOINT,
    asset_ids: str | list[str] | set[str],
    numeric_type: NumericAlias | type | Callable[[str], Any],
) -> NumericAlias | dict[str, NumericAlias]:
    if isinstance(asset_ids, str):
        return _get_midpoint(endpoint, asset_ids, numeric_type)

    body = [{"token_id": x} for x in asset_ids]
    resp: requests.Response = _request(
        url=f"{endpoint}/midpoints",
        method="POST",
        headers=None,
        data=body,
    )
    resp: dict[str, str] = msgspec.json.decode(resp.text)

    return {key: numeric_type(val) for key, val in resp.items()}


def get_tick_size(endpoint: str | ENDPOINT, asset_id: str) -> float:
    resp = _request(f"{endpoint}/tick-size?token_id={asset_id}", "GET", None, None)
    # {"minimum_tick_size":0.01}
    return msgspec.json.decode(resp.text)["minimum_tick_size"]


@lru_cache(maxsize=64)
def get_neg_risk(endpoint: str | ENDPOINT, asset_id: str) -> bool:
    resp = _request(f"{endpoint}/neg-risk?token_id={asset_id}", "GET", None, None)
    return msgspec.json.decode(resp.text)["neg_risk"]


def _get_last_trade_price(
    endpoint: str | ENDPOINT,
    asset_id: str,
    numeric_type: type[NumericAlias] | Callable[[str], NumericAlias],
) -> tuple[NumericAlias, SIDE]:
    # {"price":"0.2","side":"BUY"}
    resp = _request(
        f"{endpoint}/last-trade-price?token_id={asset_id}", "GET", None, None
    )
    resp = msgspec.json.decode(resp.text)
    return numeric_type(resp["price"]), SIDE(resp["side"])


def get_last_trades_prices(
    endpoint: str | ENDPOINT,
    asset_ids: str | list[str],
    numeric_type: type[NumericAlias] | Callable[[str], NumericAlias],
) -> tuple[NumericAlias, SIDE] | dict[str, tuple[NumericAlias, SIDE]]:
    if isinstance(asset_ids, str):
        return _get_last_trade_price(endpoint, asset_ids, numeric_type)

    # [
    #  {"price":"0.04","side":"SELL","token_id":"108121897270065793553958888713640886117036125727878261495100162544755789776283"},
    #  {"price":"0.19","side":"SELL","token_id":"84888738552927367074645370589243532246434846262240067551074642775011167355434"}
    # ]
    body = [{"token_id": x} for x in asset_ids]
    resp = _request(f"{endpoint}/last-trades-prices", "POST", None, body)
    resp = msgspec.json.decode(resp.text)

    return {x["token_id"]: (numeric_type(x["price"]), SIDE(x["side"])) for x in resp}


def get_balance_allowances(
    endpoint: str | ENDPOINT,
    numeric_type: type[NumericAlias] | Callable[[str], NumericAlias],
    signature_type: SIGNATURE_TYPE,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    nb_decimals: int = 6,
    extra_precision_buffer: int = 8,
) -> tuple[NumericAlias, dict[str, str]]:
    resp = _request(
        url=f"{endpoint}/balance-allowance?asset_type=COLLATERAL&signature_type={signature_type}",
        method="GET",
        headers=build_auth_header(
            private_key, api_key, secret, passphrase, "GET", "/balance-allowance", None
        ),
        data=None,
    )
    resp = msgspec.json.decode(resp.text)

    balance = resp["balance"]
    balance = round_floor_tenuis_ceil(
        numeric_type(balance) / 1_000_000, nb_decimals, extra_precision_buffer
    )
    return balance, resp["allowances"]


def get_balance(
    endpoint: str | ENDPOINT,
    numeric_type: type[NumericAlias] | Callable[[str], NumericAlias],
    signature_type: SIGNATURE_TYPE,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    nb_decimals: int = 6,
    extra_precision_buffer: int = 8,
) -> NumericAlias:
    """Get cash position in USDC."""
    return get_balance_allowances(
        endpoint,
        numeric_type,
        signature_type,
        private_key,
        api_key,
        secret,
        passphrase,
        nb_decimals,
        extra_precision_buffer,
    )[0]


def get_allowances(
    endpoint: str | ENDPOINT,
    numeric_type: type[NumericAlias] | Callable[[str], NumericAlias],
    signature_type: SIGNATURE_TYPE,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
):
    return get_balance_allowances(
        endpoint,
        numeric_type,
        signature_type,
        private_key,
        api_key,
        secret,
        passphrase,
        6,
        8,
    )[1]


def post_order(
    endpoint: str | ENDPOINT,
    order: OrderProtocol,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
) -> tuple[OrderProtocol, PostOrderResponse]:
    """Post order and update the order object accordingly with response information (status and order_id).

    Parameters
    ----------
    endpoint
    order
    private_key
    api_key
    secret
    passphrase

    Returns
    -------
    order: OrderProtocol
    response: PostOrderResponse

    Raises
    ------
    OrderPlacementUnmatched: Raised if status is INSERT_STATUS.UNMATCHED.
        Order status has been changed, order id has been assigned (in case order object is used after try-except).
    OrderPlacementDelayed: Raised if status is INSERT_STATUS.DELAYED.
        Order status has been changed, order id has been assigned (in case order object is used after try-except).
    OrderPlacementMarketNotReady: Raised if market not ready to accept orders yet.
        Order status MIGHT NOT BE CHANGED! Order id might not be assigned!
    OrderPlacementFailure: Raised if any other failure to place order.
        Order status MIGHT NOT BE CHANGED! Order id might not be assigned!
        This can be caused by either a server-side or client-side error.
    msgspec.ValidationError: if any exception during JSON decoding (this indicates an internal error).

    Notes
    -----
    User has responsibility that order is in a state, that can be posted (INSERT_STATE.DEFINED).
    This is not checked separately.
    Any INSERT_STATUS other than LIVE or MATCHED (other: DELAYED, UNMATCHED) raises an exception. DELAYED and
    MATCHED raise specific errors after which the order object is updated accordingly to be used further on after
    try-except. This is NOT the case for other exceptions.
    """
    _not_frozen(order)

    body = order.to_payload(api_key)

    try:
        resp = _request(
            url=f"{endpoint}/order",
            method="POST",
            headers=build_auth_header(
                private_key, api_key, secret, passphrase, "POST", "/order", body
            ),
            data=body,
        )
    except requests.HTTPError as e:
        _raise_post_order_exception(e, order)

    resp = msgspec.json.decode(resp.text, type=PostOrderResponse)

    # update order
    order = _update_order_id(order, resp, resp.errorMsg)
    order = _update_post_order_fill(order, resp)

    _assert_post_order_server_success(resp, order)
    _assert_post_order_errmsg(resp, order)
    _assert_post_order_valid(order)

    return order, resp


# todo overload type annotation
# todo Sequence instead of list for inputs
def cancel_orders(
    endpoint: str | ENDPOINT,
    orders: OrderProtocol | list[OrderProtocol],
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
) -> tuple[OrderProtocol | list[OrderProtocol], CancelOrdersResponse]:
    """Cancel order/s and change order.status to INSERT_STATUS.CANCELED if successful.

    Parameters
    ----------
    endpoint
    orders
    private_key
    api_key
    secret
    passphrase
    mode_not_canceled

    Returns
    -------

    Raises
    ------
    OrderUpdateException: if `mode_not_canceled="except"` and any order was not canceled.
        Order.status of orders which could not be canceled remains unchanged!
        Order.status of successfully canceled order is changed to INSERT_STATUS.CANCELED.

    Notes
    -----
    User has to take responsibility to check, if order is a cancelable state. This is not checked separately.
    """
    _single, orders = _parse_to_list(orders)
    _not_frozen(orders)

    body = [order.id for order in orders]
    resp = _request(
        url=f"{endpoint}/orders",
        method="DELETE",
        headers=build_auth_header(
            private_key, api_key, secret, passphrase, "DELETE", "/orders", body
        ),
        data=body,
    )
    resp = msgspec.json.decode(resp.text, type=CancelOrdersResponse)

    orders = _update_cancel_orders(orders, resp)
    _assert_cancel_orders_mode(mode_not_canceled, resp)

    if _single:
        orders = orders[0]

    return orders, resp


def cancel_orders_by_ids_(
    endpoint: str | ENDPOINT,
    order_ids: str | list[str],
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
) -> CancelOrdersResponse:
    warnings.warn(
        "Status of order objects corresponding to `order_ids` will not be updated automatically. "
        "Use of this function is discouraged. Use `cancel_orders` instead."
    )

    _, body = _parse_to_list(order_ids)

    resp = _request(
        url=f"{endpoint}/orders",
        method="DELETE",
        headers=build_auth_header(
            private_key, api_key, secret, passphrase, "DELETE", "/orders", body
        ),
        data=body,
    )
    resp = msgspec.json.decode(resp.text, type=CancelOrdersResponse)
    _assert_cancel_orders_mode(mode_not_canceled, resp)

    return resp


def _cancel_market_asset(
    endpoint: str | ENDPOINT,
    market_id: str,
    asset_id: str,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
) -> CancelOrdersResponse:
    body = {"market": market_id, "asset_id": asset_id}
    resp = _request(
        url=f"{endpoint}/cancel-market-orders",
        method="DELETE",
        headers=build_auth_header(
            private_key,
            api_key,
            secret,
            passphrase,
            "DELETE",
            "/cancel-market-orders",
            body,
        ),
        data=body,
    )
    resp = msgspec.json.decode(resp.text, type=CancelOrdersResponse)
    _assert_cancel_orders_mode(mode_not_canceled, resp)

    return resp


def cancel_market(
    endpoint: str | ENDPOINT,
    market_id: str,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
) -> CancelOrdersResponse:
    return _cancel_market_asset(
        endpoint,
        market_id,
        "",
        private_key,
        api_key,
        secret,
        passphrase,
        mode_not_canceled,
    )


def cancel_asset(
    endpoint: str | ENDPOINT,
    asset_id: str,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
) -> CancelOrdersResponse:
    return _cancel_market_asset(
        endpoint,
        "",
        asset_id,
        private_key,
        api_key,
        secret,
        passphrase,
        mode_not_canceled,
    )


# todo overload
def get_orders(
    endpoint: str | ENDPOINT,
    orders: OrderProtocol | list[OrderProtocol],
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    mode_not_existent: Literal["warn", "except", "ignore"] = "except",
) -> tuple[
    OrderProtocol | list[OrderProtocol],
    OpenOrderInfo | list[OpenOrderInfo | None],
]:
    """Update order with info pulled from exchange.

    Only updates status, size_matched and created_at (created_at will only be modified only if it is not already set,
    else created_at will be ignored silently).
    If order not found, raises, warns or ignores depending on `mode_not_existent`. If exception is raised,
    orders, that could be retrieved successfully, are still updated accordingly.

    Parameters
    ----------
    endpoint
    orders
    private_key
    api_key
    secret
    passphrase
    mode_not_existent

    Returns
    -------

    """
    _single, orders = _parse_to_list(orders)
    _not_frozen(orders)

    ret_responses = []
    _missed = []
    for order in orders:
        resp = _request(
            url=f"{endpoint}/data/order/{order.id}",
            method="GET",
            headers=build_auth_header(
                private_key,
                api_key,
                secret,
                passphrase,
                "GET",
                f"/data/order/{order.id}",
                None,
            ),
            data=None,
        )

        if resp.text in ["null", "", "none", "None"]:
            ret_responses.append(None)
            _missed.append(order.id)
            continue

        resp = msgspec.json.decode(resp.text, type=OpenOrderInfo, strict=False)
        ret_responses.append(resp)

        update_order(
            order,
            status=resp.status,
            size_matched=order.numeric_type(resp.size_matched),
            created_at=order.numeric_type(resp.created_at),
        )

    if len(_missed):
        if mode_not_existent == "except":
            raise OrderUpdateException(f"Could not get all orders. Missed: {_missed}.")
        elif mode_not_existent == "warn":
            warnings.warn(f"Could not get all orders. Missed: {_missed}.")

    if _single:
        orders, ret_responses = orders[0], ret_responses[0]

    return orders, ret_responses


def get_orders_by_ids_(
    endpoint: str | ENDPOINT,
    order_ids: str | list[str],
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    mode_not_existent: Literal["warn", "except", "ignore"] = "except",
) -> OpenOrderInfo | None | list[OpenOrderInfo | None]:
    warnings.warn(
        "Status of order objects corresponding to `order_ids` will not be updated automatically. "
        "Use of this function is discouraged. Use `get_orders` instead."
    )

    _single, order_ids = _parse_to_list(order_ids)

    ret_responses = []
    _missed = []
    for order_id in order_ids:
        resp = _request(
            url=f"{endpoint}/data/order/{order_id}",
            method="GET",
            headers=build_auth_header(
                private_key,
                api_key,
                secret,
                passphrase,
                "GET",
                f"/data/order/{order_id}",
                None,
            ),
            data=None,
        )

        if resp.text in ["null", "", "none", "None"]:
            ret_responses.append(None)
            _missed.append(order_id)
            continue

        resp = msgspec.json.decode(resp.text, type=OpenOrderInfo, strict=False)
        ret_responses.append(resp)

    if len(_missed):
        if mode_not_existent == "except":
            raise OrderUpdateException(f"Could not get all orders. Missed: {_missed}.")
        elif mode_not_existent == "warn":
            warnings.warn(f"Could not get all orders. Missed: {_missed}.")

    if _single:
        ret_responses = ret_responses[0]

    return ret_responses


def _parse_active_orders_url(
    endpoint: str,
    order_id: str | None,
    market_id: str | None,
    asset_id: str | None,
    next_cursor: str,
) -> str:
    url = f"{endpoint}/data/orders?next_cursor={next_cursor}"

    if order_id:
        url = f"{url}&id={order_id}"
    if market_id:
        url = f"{url}&market={market_id}"
    if asset_id:
        url = f"{url}&asset_id={asset_id}"

    return url


def get_active_orders(
    endpoint: str | ENDPOINT,
    market_id: str | None,
    asset_id: str | None,
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
    next_cursor: str | None = None,
) -> list[OpenOrderInfo]:
    if next_cursor is None:
        next_cursor = ""

    auth_header = build_auth_header(
        private_key, api_key, secret, passphrase, "GET", "/data/orders", None
    )

    ret = []
    while next_cursor != END_CURSOR:
        resp = _request(
            _parse_active_orders_url(endpoint, None, market_id, asset_id, next_cursor),
            method="GET",
            headers=auth_header,
            data=None,
        )
        resp = msgspec.json.decode(resp.text, type=OpenOrderResponse, strict=False)

        next_cursor = resp.next_cursor
        ret.extend(resp.data)

    return ret


def _are_orders_scoring_by_ids_(
    endpoint: str | ENDPOINT,
    order_ids: str | list[str],
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
) -> dict[str, bool]:
    _, order_ids = _parse_to_list(order_ids)

    resp = _request(
        url=f"{endpoint}/orders-scoring",
        method="POST",
        headers=build_auth_header(
            private_key,
            api_key,
            secret,
            passphrase,
            "POST",
            "/orders-scoring",
            order_ids,
        ),
        data=order_ids,
    )
    return resp.json()


def are_orders_scoring_by_ids_(
    endpoint: str | ENDPOINT,
    order_ids: str | list[str],
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
) -> dict[str, bool]:
    warnings.warn(
        "Status of order objects corresponding to `order_ids` will not be updated automatically. "
        "Use of this function is discouraged. Use `are_orders_scoring` instead."
    )

    return _are_orders_scoring_by_ids_(
        endpoint, order_ids, private_key, api_key, secret, passphrase
    )


def are_orders_scoring(
    endpoint: str | ENDPOINT,
    orders: OrderProtocol | list[OrderProtocol],
    private_key: str | PrivateKey | PrivateKeyType,
    api_key: str,
    secret: str,
    passphrase: str,
) -> bool | list[bool]:
    _single, orders = _parse_to_list(orders)
    order_ids = [order.id for order in orders]

    ret_dict = _are_orders_scoring_by_ids_(
        endpoint, order_ids, private_key, api_key, secret, passphrase
    )

    ret = [ret_dict[order.id] for order in orders]

    if _single:
        ret = ret[0]

    return ret


def _parse_positions_url(
    endpoint: str,
    user: str,
    size_threshold: float,
    limit: int,
    offset: int,
    market: str | list[str] | None,
    redeemable: bool | None,
    mergeable: bool | None,
    title: str | None,
) -> str:
    if limit > 500:
        raise PolyPyException("`limit` must be <= 500.")

    url = f"{endpoint}/positions?=user={user}&sizeThreshold={size_threshold}&limit={limit}&offset={offset}"

    if market:
        _, market = _parse_to_list(market)
        url = f"{url}&market={','.join(market)}"
    if isinstance(redeemable, bool):
        url = f"{url}&redeemable={redeemable}"
    if isinstance(mergeable, bool):
        url = f"{url}&mergeable={mergeable}"
    if title:
        url = f"{url}&title={title}"

    return url


def get_positions(
    endpoint_data: str | ENDPOINT,
    user: str,
    position_factory: type[PositionProtocol] | PositionFactory | None,
    market: str | list[str] | None,
    redeemable: bool | None,
    mergeable: bool | None,
    title: str | None,
    size_threshold: float = 1,
    limit: int = 500,
    offset: int = 0,
    size_sig_digits: int = SIG_DIGITS_SIZE,
) -> list[PositionProtocol] | list[dict[str, Any]]:
    if isinstance(endpoint_data, ENDPOINT) and endpoint_data is not ENDPOINT.DATA:
        raise PolyPyException(
            f"Incorrect ENDPOINT={endpoint_data}. Did you mean ENDPOINT.DATA?"
        )

    position_factory = (
        position_factory.create
        if hasattr(position_factory, "create")
        else position_factory
    )

    url = _parse_positions_url(
        endpoint=endpoint_data,
        user=user,
        size_threshold=size_threshold,
        limit=limit,
        offset=offset,
        market=market,
        redeemable=redeemable,
        mergeable=mergeable,
        title=title,
    )
    resp = _request(url, "GET", None, None)
    resp = msgspec.json.decode(resp.text)

    if position_factory is None:
        return resp

    return [
        position_factory(
            asset_id=r["asset"], size=r["size"], size_sig_digits=size_sig_digits
        )
        for r in resp
    ]
