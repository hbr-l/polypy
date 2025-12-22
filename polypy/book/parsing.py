from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable

import msgspec

from polypy.exceptions import EventTypeException, OrderBookException
from polypy.order.common import SIDE
from polypy.structs import (
    BookEvent,
    BookSummary,
    MarketEvent,
    OrderSummary,
    PriceChangeEvent,
    TickSizeEvent,
)
from polypy.typing import ArrayCoercible, NumericAlias

if TYPE_CHECKING:
    from polypy.book.order_book import OrderBookProtocol


def _number_to_str(x: NumericAlias) -> str:
    # todo optimize performance

    # f"{x:.16g} breaks (i.e. truncation) if more than 16 chars:
    #  for $xxx$.¢¢ notation this is somewhere greater than around 100 tn
    #  AND is wrong for e.g. x=0.691 -> '0.6909999999999999'

    nb = str(x)
    return nb.rstrip("0").rstrip(".") if "." in nb else nb


def merge_quotes_to_order_summaries(
    prices: list[NumericAlias] | ArrayCoercible,
    sizes: list[NumericAlias] | ArrayCoercible,
    reverse: bool,
    order_summary_factory: type[dict] | type[OrderSummary] = dict,
) -> list[dict[str, str]] | list[OrderSummary]:
    # Polymarket orders price level s.t. best bid is at index -1,
    #   whilst we order in reverse (i.e. to apply np.cumsum more easily)
    # remove trailing 0s and if necessary decimal point

    # todo optimize for performance:
    #   - prices could be cached (either cache manually or use python's mem_cache)
    #   - if sizes are stored as scaled ints: insert decimal point char manually

    if reverse:
        prices = reversed(prices)
        sizes = reversed(sizes)

    return [
        order_summary_factory(
            price=_number_to_str(price),
            size=_number_to_str(size),
        )
        for price, size in zip(prices, sizes)
    ]


def split_order_summaries_to_quotes(
    x: list[OrderSummary],
    dtype_factory: Callable[[str], Any],
) -> tuple[list[NumericAlias], list[NumericAlias]]:
    if not len(x):
        return [], []
    return [dtype_factory(d.price) for d in x], [dtype_factory(d.size) for d in x]


def _quotes_from_book_event(
    msg: BookEvent | BookSummary, dtype_factory: Callable[[str], Any]
) -> tuple[
    list[NumericAlias], list[NumericAlias], list[NumericAlias], list[NumericAlias]
]:
    bid_prices, bid_sizes = split_order_summaries_to_quotes(msg.bids, dtype_factory)
    ask_prices, ask_sizes = split_order_summaries_to_quotes(msg.asks, dtype_factory)
    return bid_prices, bid_sizes, ask_prices, ask_sizes


def _quotes_from_price_change_event(
    msg: PriceChangeEvent, token_id: str, dtype_factory: Callable[[str], Any]
) -> tuple[list, list, list, list]:
    bid_prices, bid_sizes, ask_prices, ask_sizes = [], [], [], []

    for change in msg.price_changes:
        if change.asset_id != token_id:
            continue
        if change.side is SIDE.BUY:
            bid_prices.append(dtype_factory(change.price))
            bid_sizes.append(dtype_factory(change.size))
        elif change.side is SIDE.SELL:
            ask_prices.append(dtype_factory(change.price))
            ask_sizes.append(dtype_factory(change.size))
        else:
            raise ValueError(f"Unknown side: {change.side}")

    return bid_prices, bid_sizes, ask_prices, ask_sizes


def dict_to_book_struct(msg: dict[str, Any]) -> MarketEvent | BookSummary:
    msg_type = BookSummary
    if "event_type" in msg:
        msg_type = MarketEvent

    try:
        return msgspec.convert(msg, type=msg_type, strict=False)
    except msgspec.ValidationError as e:
        raise EventTypeException(f"Unknown msg: {msg}.") from e


def guess_tick_size(
    msg: BookEvent | dict | BookSummary,
    n: int = 12,
) -> float:
    if isinstance(msg, dict):
        msg: MarketEvent | BookSummary = dict_to_book_struct(msg)

    if msg.event_type == "summary":
        return float(msg.tick_size)

    min_exponent = min(
        [Decimal(i.price).as_tuple().exponent for i in msg.bids[:n]]
        + [Decimal(i.price).as_tuple().exponent for i in msg.asks[:n]],
    )
    return 10**min_exponent


def _set_book_event(
    msg: BookEvent | BookSummary,
    book: "OrderBookProtocol",
    dtype_factory: Callable[[str], Any] | None = None,
) -> "OrderBookProtocol":
    if msg.asset_id != book.token_id:
        raise OrderBookException(
            f"asset_id != token_id. Got asset_id={msg.asset_id} and token_id={book.token_id}"
        )

    bid_prices, bid_sizes, ask_prices, ask_sizes = _quotes_from_book_event(
        msg, dtype_factory
    )

    if len(bid_prices):
        book.reset_bids(bid_prices, bid_sizes)
    else:
        book.null_bids()

    if len(ask_prices):
        book.reset_asks(ask_prices, ask_sizes)
    else:
        book.null_asks()

    if msg.event_type == "summary":
        book.tick_size = float(msg.tick_size)
        book.min_order_size = int(msg.min_order_size)
        book.neg_risk = msg.neg_risk
        book.market_id = msg.market

    return book


def _set_price_change_event(
    msg: PriceChangeEvent,
    book: "OrderBookProtocol",
    dtype_factory: Callable[[str], Any] | None = None,
) -> "OrderBookProtocol":
    (
        bid_prices,
        bid_sizes,
        ask_prices,
        ask_sizes,
    ) = _quotes_from_price_change_event(msg, book.token_id, dtype_factory)

    if len(bid_prices):
        book.set_bids(bid_prices, bid_sizes)
    if len(ask_prices):
        book.set_asks(ask_prices, ask_sizes)

    if not bid_prices and not ask_prices:
        asset_ids = set(change.asset_id for change in msg.price_changes)
        raise OrderBookException(
            f"No bids and asks to parse. "
            f"Make sure asset_id == token_id. "
            f"book.token_id='{book.token_id}', asset_ids='{asset_ids}'."
        )
    return book


def _set_tick_size_event(
    msg: TickSizeEvent, book: "OrderBookProtocol"
) -> "OrderBookProtocol":
    if msg.asset_id != book.token_id:
        raise OrderBookException(
            f"asset_id != token_id. Got asset_id={msg.asset_id} and token_id={book.token_id}"
        )

    book.tick_size = msg.new_tick_size
    return book


def message_to_orderbook(
    msg: dict[str, Any] | MarketEvent | BookSummary,
    book: "OrderBookProtocol",
    dtype_factory: Callable[[str], Any] | type | None = None,
) -> "OrderBookProtocol":
    if isinstance(msg, dict):
        msg: BookSummary | MarketEvent = dict_to_book_struct(msg)

    dtype_factory = book.dtype if dtype_factory is None else dtype_factory

    if msg.event_type == "summary" or msg.event_type == "book":
        return _set_book_event(msg, book, dtype_factory)

    if msg.event_type == "price_change":
        return _set_price_change_event(msg, book, dtype_factory)

    if msg.event_type == "tick_size_change":
        return _set_tick_size_event(msg, book)

    raise EventTypeException(f"Unknown msg: {msg}.")
