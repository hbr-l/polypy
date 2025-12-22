import hashlib
from typing import TYPE_CHECKING, Sequence

from msgspec import json as msgspec_json

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from polypy.book.order_book import OrderBookProtocol


def dict_to_sha1(x: dict) -> str:
    return hashlib.sha1(msgspec_json.encode(x)).hexdigest()


def str_to_sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def _book_to_dict(
    book: "OrderBookProtocol|dict",
    market_id: str | None,
    min_order_size: str | int | None,
    neg_risk: bool | None,
) -> dict:
    if not isinstance(book, dict):
        book = book.to_dict(
            timestamp="",
            hash_str="",
            market_id=book.market_id,
            min_order_size=book.min_order_size,
            neg_risk=book.neg_risk,
        )

    if market_id is not None:
        book["market"] = market_id
    if min_order_size is not None:
        book["min_order_size"] = min_order_size
    if neg_risk is not None:
        # noinspection PyTypeChecker
        book["neg_risk"] = neg_risk
    book["hash"] = ""

    return book


def check_orderbook_hash(
    target_hash: str,
    book: "OrderBookProtocol | dict",
    timestamps: Sequence[int | float | str],
    market_id: str | None,
    min_order_size: str | int | None,
    neg_risk: bool | None,
) -> tuple[bool, int | None, str | None]:
    book = _book_to_dict(book, market_id, min_order_size, neg_risk)

    assert None not in (book["market"], book["min_order_size"], book["neg_risk"]), (
        "`market_id`, `min_order_size` or `neg_risk` must not be None AFTER parsing. "
        "Cannot compute hash for order book."
    )

    # todo suboptimal approach (brute-forcing through timestamps), but currently no known alternative
    for i, ts in enumerate(timestamps):
        if not isinstance(ts, str):
            ts = f"{ts:.0f}"

        book["timestamp"] = ts

        book_hash = dict_to_sha1(book)
        if book_hash == target_hash:
            return True, i, book_hash

    return False, None, None
