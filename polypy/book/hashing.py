import hashlib
from typing import TYPE_CHECKING, Sequence

from msgspec import json as msgspec_json

from polypy.exceptions import OrderBookException

if TYPE_CHECKING:
    from polypy.book.order_book import OrderBookProtocol


def dict_to_sha1(x: dict) -> str:
    return hashlib.sha1(msgspec_json.encode(x)).hexdigest()


def guess_check_orderbook_hash(
    target_hash: str,
    book: "OrderBookProtocol | dict",
    market_id: str | None,
    timestamps: Sequence[int | float | str],
) -> tuple[bool, int | None, str | None]:
    book_dict = book if isinstance(book, dict) else book.to_dict(market_id, None, "")

    if book_dict["market"] is None:
        raise OrderBookException(
            "market_id not set, cannot compute hash for order book."
        )

    # todo suboptimal approach (brute-forcing through timestamps), but currently no known alternative
    for i, ts in enumerate(timestamps):
        if not isinstance(ts, str):
            ts = f"{ts:.0f}"

        book_dict["timestamp"] = ts

        book_hash = dict_to_sha1(book_dict)
        if book_hash == target_hash:
            return True, i, book_hash

    return False, None, None
