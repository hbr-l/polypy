import threading
import time
import timeit

import numpy as np
from matplotlib import pyplot as plt

from polypy.constants import ENDPOINT
from polypy.orderbook import OrderBook
from polypy.stream import OrderBookStream


def callback_thread_id(_, __):
    print("ID: ", threading.get_ident())


def main():
    book = OrderBook(
        "88458672007514219171605090869548159546185169218791748266793909997093690233909",
        0.01,
    )

    streamer = OrderBookStream(
        ENDPOINT.WS,
        book,
        (1, 250),
        rest_endpoint=None,
        nb_redundant_skt=5,
        callback_msg=callback_thread_id,
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
