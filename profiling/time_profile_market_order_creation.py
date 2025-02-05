"""
CLOB client: 9293.912499560975 us/order
polypy:     3115.4959998093545 us/order

polypy is roughly 3x faster regarding market order creation.
"""
import json
import timeit

import responses
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs
from py_clob_client.constants import POLYGON

from polypy.constants import CHAIN_ID
from polypy.order import SIDE, create_market_order
from polypy.signing import SIGNATURE_TYPE


@responses.activate
def benchmark_clob_client(token_id, tick_size, private_key, amount, nb_iter):
    host = "http://localhost:8080"

    with open("../tests/order/data/book_data_a.txt", "r") as f:
        book = json.load(f)["yes_book"]

    responses.get(
        f"{host}/tick-size?token_id={token_id}", json={"minimum_tick_size": tick_size}
    )
    responses.get(f"{host}/neg-risk?token_id={token_id}", json={"neg_risk": False})
    responses.get(f"{host}/book?token_id={token_id}", json=book)

    client = ClobClient(host, POLYGON, private_key)

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        client.create_market_order(MarketOrderArgs(token_id, amount))
    end_t = timeit.default_timer()
    print(f"CLOB client: {((end_t - start_t) / nb_iter) * 1_000_000} us/order")


def benchmark_polypy(token_id, tick_size, private_key, amount, nb_iter):
    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        create_market_order(
            amount,
            token_id,
            SIDE.BUY,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )
    end_t = timeit.default_timer()
    print(f"polypy: {((end_t - start_t) / nb_iter) * 1_000_000} us/order")


def main():
    private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    token_id = (
        "72936048731589292555781174533757608024096898681344338816447372274344589246891"
    )
    amount = 101.1
    tick_size = 0.001
    nb_iter = 200

    benchmark_clob_client(token_id, tick_size, private_key, amount, nb_iter)
    benchmark_polypy(token_id, tick_size, private_key, amount, nb_iter)


if __name__ == "__main__":
    main()
