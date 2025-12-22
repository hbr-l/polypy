"""
dict gen: 	 0.05538800099981017 ms
str gen: 	 0.05854117900016718 ms
complete hash gen dict: 	 0.06419529700069689 ms
complete hash gen str: 		 0.06471271799993701 ms
iter over timestamp range:  0.1489447579998523 ms
"""

import json
import timeit
from typing import Sequence

from polypy.book import OrderBook, dict_to_sha1, message_to_orderbook
from polypy.book.hashing import str_to_sha1
from polypy.book.order_book import OrderBookProtocol
from polypy.typing import NumericAlias


def _stringify_quotes(
    prices: Sequence[str | NumericAlias], sizes: Sequence[str | NumericAlias]
) -> list[str]:
    return [
        f'{{"price":"{price}","size":"{str(size).rstrip("0").rstrip(".")}"}}'
        for price, size in zip(reversed(prices), reversed(sizes))
    ]


def stringify_orderbook(
    book: OrderBookProtocol,
    hash_str: str,
    timestamp: int | float | str,
    market_id: str,
    min_order_size: str | int,
    neg_risk: bool,
) -> str:
    bids = _stringify_quotes(book.bid_prices, book.bid_sizes)
    asks = _stringify_quotes(book.ask_prices, book.ask_sizes)

    return (
        f'{{"market":"{market_id}","asset_id":"{book.token_id}","timestamp":"{timestamp}",'
        f'"hash":"{hash_str}","bids":[{",".join(bids)}],"asks":[{",".join(asks)}],"min_order_size":"{min_order_size}",'
        f'"tick_size":"{book.tick_size}","neg_risk":{str(neg_risk).lower()}}}'
    )


def main():
    with open("../tests/orderbook/data/messages_hash.txt", "r") as f:
        data = json.loads(f.readlines()[5])
    market_id = data["market"]
    timestamp = int(data["timestamp"])

    book = OrderBook(data["asset_id"], 0.001)
    message_to_orderbook(data, book)

    nb_iter = 100_000

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        book.to_dict(timestamp, "", market_id, data["min_order_size"], data["neg_risk"])
    end_t = timeit.default_timer()
    print("dict gen: \t", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        stringify_orderbook(
            book, "", timestamp, market_id, data["min_order_size"], data["neg_risk"]
        )
    end_t = timeit.default_timer()
    print("str gen: \t", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        dict_to_sha1(
            book.to_dict(
                timestamp, "", market_id, data["min_order_size"], data["neg_risk"]
            )
        )
    end_t = timeit.default_timer()
    print("complete hash gen dict: \t", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        str_to_sha1(
            stringify_orderbook(
                book, "", timestamp, market_id, data["min_order_size"], data["neg_risk"]
            )
        )
    end_t = timeit.default_timer()
    print("complete hash gen str: \t\t", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        book_dict = book.to_dict(
            None, "", market_id, data["min_order_size"], data["neg_risk"]
        )
        for i in range(15):
            book_dict["timestamp"] = str(timestamp - i)
            dict_to_sha1(book_dict)
    end_t = timeit.default_timer()
    print("iter over timestamp range: ", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    # detect regression bugs
    assert (end_t - start_t) / nb_iter < 0.00055


if __name__ == "__main__":
    main()
