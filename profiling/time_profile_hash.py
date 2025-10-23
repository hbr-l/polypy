"""
dict gen: 	 0.05538800099981017 ms
str gen: 	 0.05854117900016718 ms
complete hash gen dict: 	 0.06419529700069689 ms
complete hash gen str: 		 0.06471271799993701 ms
iter over timestamp range:  0.1489447579998523 ms
"""

import json
import timeit

from polypy.book import OrderBook, dict_to_sha1, message_to_orderbook
from polypy.book.hashing import str_to_sha1
from polypy.book.parsing import stringify_orderbook


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
