"""
Live plot of order book using OrderBook and MarketStream.
Please enter token ID into prompt.
Exit via KeyboardInterrupt.
"""
import time

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

import polypy as plp


def main(token_id: str) -> None:
    book = plp.OrderBook(token_id, plp.get_tick_size(plp.ENDPOINT.REST, token_id))
    stream = plp.MarketStream(plp.ENDPOINT.WS, book, None, plp.ENDPOINT.REST)
    stream.start()

    time.sleep(0.5)

    fig, ax = plt.subplots()
    plt.title(f"Order Book: {token_id}")
    bid_chart = ax.bar(
        book.bids[0],
        np.cumsum(book.bids[1]),
        alpha=0.5,
        color="green",
        width=book.min_tick_size,
    )
    ask_chart = ax.bar(
        book.asks[0],
        np.cumsum(book.asks[1]),
        alpha=0.5,
        color="red",
        width=book.min_tick_size,
    )

    def animate(*_, **__):
        cumsum_bid_sizes = np.cumsum(book.bids[1])
        cumsum_ask_sizes = np.cumsum(book.asks[1])

        for i, rect in enumerate(bid_chart):
            rect.set_height(cumsum_bid_sizes[i])
        for i, rect in enumerate(ask_chart):
            rect.set_height(cumsum_ask_sizes[i])

        return bid_chart + ask_chart

    _ = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
    plt.show()

    stream.stop(True, 1)


if __name__ == "__main__":
    asset_id = input("Please enter token_id: ")
    main(asset_id)
