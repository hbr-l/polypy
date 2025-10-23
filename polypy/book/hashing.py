import hashlib
from typing import TYPE_CHECKING, Sequence

from msgspec import json as msgspec_json

from polypy.exceptions import OrderBookException

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from polypy.book.order_book import OrderBookProtocol


def dict_to_sha1(x: dict) -> str:
    return hashlib.sha1(msgspec_json.encode(x)).hexdigest()


def str_to_sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def guess_check_orderbook_hash(
    target_hash: str,
    book: "OrderBookProtocol | dict",
    timestamps: Sequence[int | float | str],
    market_id: str | None,
    min_order_size: str | int | None,
    neg_risk: bool | None,
    strict: bool,
) -> tuple[bool, int | None, str | None]:
    if isinstance(book, dict):
        market_id = market_id if market_id is not None else book["market_id"]
        min_order_size = (
            min_order_size if min_order_size is not None else book["min_order_size"]
        )
        neg_risk = neg_risk if neg_risk is not None else book["neg_risk"]
        book["hash"] = ""
    else:
        market_id = market_id if market_id is not None else book.market_id
        min_order_size = (
            min_order_size if min_order_size is not None else book.min_order_size
        )
        neg_risk = neg_risk if neg_risk is not None else book.neg_risk
        book = book.to_dict(
            timestamp="",
            hash_str="",
            market_id=market_id,
            min_order_size=min_order_size,
            neg_risk=neg_risk,
        )

    if market_id is None or min_order_size is None or neg_risk is None:
        if strict:
            raise OrderBookException(
                "`market_id`, `min_order_size` and `neg_risk` must not be None AFTER parsing. "
                "Cannot compute hash for order book."
            )
        return False, None, None

    # todo suboptimal approach (brute-forcing through timestamps), but currently no known alternative
    for i, ts in enumerate(timestamps):
        if not isinstance(ts, str):
            ts = f"{ts:.0f}"

        book["timestamp"] = ts

        book_hash = dict_to_sha1(book)
        if book_hash == target_hash:
            return True, i, book_hash

    return False, None, None
