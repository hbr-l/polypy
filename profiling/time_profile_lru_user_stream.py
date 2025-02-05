"""
nb_iter = 1_000

No cache: 		0.0010500999633222818 ms
With cache: 	0.0012493000831454992 ms

lru_cache is only effective for nb_iter > 5_000
"""
import json
import time
import timeit

import msgspec

from polypy.stream.user import (
    TRADER_SIDE,
    TradeWSInfo,
    _maker_order_side,
    _TradeOrderInfo,
    lru_cache_non_empty,
)


def _filter_orders_trade_info_no_cache(
    msg: TradeWSInfo, api_key
) -> list[_TradeOrderInfo]:
    if msg.trader_side is TRADER_SIDE.TAKER and msg.owner == api_key:
        return [
            _TradeOrderInfo(
                msg.taker_order_id,
                msg.size,
                msg.price,
                msg.asset_id,
                msg.side,
            )
        ]
    elif msg.trader_side is TRADER_SIDE.MAKER:
        return [
            _TradeOrderInfo(
                maker_order.order_id,
                maker_order.matched_amount,
                maker_order.price,
                maker_order.asset_id,
                _maker_order_side(maker_order, msg),
            )
            for maker_order in msg.maker_orders
            if maker_order.owner == api_key
        ]
    return []


@lru_cache_non_empty(max_size=64, copy_mode=None)
def _filter_orders_trade_info_with_cache(
    msg: TradeWSInfo, api_key
) -> list[_TradeOrderInfo]:
    if msg.trader_side is TRADER_SIDE.TAKER and msg.owner == api_key:
        return [
            _TradeOrderInfo(
                msg.taker_order_id,
                msg.size,
                msg.price,
                msg.asset_id,
                msg.side,
            )
        ]
    elif msg.trader_side is TRADER_SIDE.MAKER:
        return [
            _TradeOrderInfo(
                maker_order.order_id,
                maker_order.matched_amount,
                maker_order.price,
                maker_order.asset_id,
                _maker_order_side(maker_order, msg),
            )
            for maker_order in msg.maker_orders
            if maker_order.owner == api_key
        ]
    return []


def main():
    with open("../tests/stream/data/test_trade_info_maker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )
    api_key = "000000000-0000-0000-0000-000000000000"
    nb_iter = 1_000

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        _filter_orders_trade_info_no_cache(data, api_key)
    end_t = timeit.default_timer()
    dt = ((end_t - start_t) / nb_iter) * 1_000
    print(f"No cache: \t\t{dt} ms")

    time.sleep(0.5)

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        _filter_orders_trade_info_with_cache(data, api_key)
    end_t = timeit.default_timer()
    dt = ((end_t - start_t) / nb_iter) * 1_000
    print(f"With cache: \t{dt} ms")


if __name__ == "__main__":
    main()
