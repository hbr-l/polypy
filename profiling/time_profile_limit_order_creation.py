"""
CLOB client: 8206.130999606103 us/order
polypy:     3016.5119998855516 us/order

polypy is more than 2.5x faster regarding limit order creation.
"""

import timeit

import responses
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.constants import POLYGON
from py_clob_client.order_builder.constants import BUY

from polypy.constants import CHAIN_ID
from polypy.order import SIDE, create_limit_order
from polypy.signing import SIGNATURE_TYPE


@responses.activate
def benchmark_clob_client(token_id, tick_size, private_key, price, size, nb_iter):
    host = "http://localhost:8080"

    responses.get(
        f"{host}/tick-size?token_id={token_id}", json={"minimum_tick_size": tick_size}
    )
    responses.get(f"{host}/neg-risk?token_id={token_id}", json={"neg_risk": False})

    client = ClobClient(host, POLYGON, private_key)

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        client.create_order(OrderArgs(token_id, price, size, BUY))
    end_t = timeit.default_timer()
    print(f"CLOB client: {((end_t - start_t) / nb_iter) * 1_000_000} us/order")


def benchmark_polypy(token_id, tick_size, private_key, price, size, nb_iter):
    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        create_limit_order(
            price,
            size,
            token_id,
            SIDE.BUY,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            maker=None,
            signature_type=SIGNATURE_TYPE.EOA,
        )
    end_t = timeit.default_timer()
    print(f"polypy: {((end_t - start_t) / nb_iter) * 1_000_000} us/order")


def main():
    private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    token_id = (
        "72936048731589292555781174533757608024096898681344338816447372274344589246891"
    )
    price = 0.5
    size = 10
    tick_size = 0.001
    nb_iter = 200

    benchmark_clob_client(token_id, tick_size, private_key, price, size, nb_iter)
    benchmark_polypy(token_id, tick_size, private_key, price, size, nb_iter)


if __name__ == "__main__":
    main()
