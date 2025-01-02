import json

import numpy as np
from matplotlib import pyplot as plt

from polypy.orderbook import OrderBook, message_to_orderbook


def main():
    with open("_messages_bitcoin.txt") as f:
        txt = f.readlines()

    txt = txt[0]
    txt = txt.replace("'", '"')
    print(txt)

    data = json.loads(txt)
    orderbook = OrderBook(data["asset_id"], 0.01)

    orderbook, _ = message_to_orderbook(data, orderbook)

    plt.figure()
    plt.bar(
        orderbook.bids[0],
        np.cumsum(orderbook.bids[1]),
        width=0.001,
        alpha=0.5,
        color="green",
    )
    plt.bar(
        orderbook.asks[0],
        np.cumsum(orderbook.asks[1]),
        width=0.001,
        alpha=0.5,
        color="red",
    )
    plt.bar(
        orderbook.bids[0],
        orderbook.bids[1],
        width=0.001,
        alpha=0.5,
        color="green",
    )
    plt.bar(
        orderbook.asks[0],
        orderbook.asks[1],
        width=0.001,
        alpha=0.5,
        color="red",
    )
    plt.show()


if __name__ == "__main__":
    main()
