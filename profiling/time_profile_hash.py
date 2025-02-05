"""
Polymarket websocket 'price_change' message contains timestamp of emission instead of book generation, s.t.
we have to guess the correct timestamp by testing, i.e. last 15 ms timestamps (iter).

dict gen:  0.180089000496082 ms
complete hash gen:  0.18984199967235327 ms
iter over timestamp range:  0.4052450001472607 ms
"""

import json
import timeit

from polypy.book import OrderBook, dict_to_sha1, message_to_orderbook


def main():
    with open("../tests/data/messages_hash.txt", "r") as f:
        data = json.loads(f.readlines()[1])
    market_id = data["market"]
    timestamp = int(data["timestamp"])

    book = OrderBook(data["asset_id"], 0.001)
    message_to_orderbook(data, book)

    nb_iter = 100

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        book.to_dict(market_id, timestamp, "")
    end_t = timeit.default_timer()
    print("dict gen: ", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        book.hash(market_id, timestamp)
    end_t = timeit.default_timer()
    print("complete hash gen: ", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        book_dict = book.to_dict(market_id, None, "")
        for i in range(15):
            book_dict["timestamp"] = str(timestamp - i)
            dict_to_sha1(book_dict)
    end_t = timeit.default_timer()
    print("iter over timestamp range: ", ((end_t - start_t) / nb_iter) * 1_000, "ms")

    # detect regression bugs
    assert (end_t - start_t) / nb_iter < 0.00055


if __name__ == "__main__":
    main()
