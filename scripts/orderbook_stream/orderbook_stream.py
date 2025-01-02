import time
import timeit

import numpy as np
from matplotlib import pyplot as plt

from polypy.orderbook import OrderBook
from polypy.stream import OrderBookStream


def main():
    book = OrderBook(
        "72936048731589292555781174533757608024096898681344338816447372274344589246891",
        0.01,
    )

    streamer = OrderBookStream(
        "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        book,
        (1, 15),
        endpoint=None,
        nb_redundant_skt=2,
    )

    streamer.start()

    start_t = timeit.default_timer()
    while (timeit.default_timer() - start_t) < 20:
        print(streamer.status_orderbook(book.token_id))
        time.sleep(0.5)

    streamer.stop(True, None)

    print(streamer.last_traded_price(book.token_id))

    plt.figure()
    plt.bar(
        book.bids[0],
        np.cumsum(book.bids[1]),
        alpha=0.5,
        color="green",
        width=min(book.allowed_tick_sizes),
    )
    plt.bar(
        book.asks[0],
        np.cumsum(book.asks[1]),
        alpha=0.5,
        color="red",
        width=min(book.allowed_tick_sizes),
    )
    plt.show()


if __name__ == "__main__":
    main()
