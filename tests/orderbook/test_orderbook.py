import collections
import json
import math
import pathlib
from decimal import Decimal

import numpy as np
import pytest
import responses
from attrs.exceptions import FrozenAttributeError
from py_clob_client.clob_types import OrderSummary
from py_clob_client.utilities import parse_raw_orderbook_summary

from polypy.exceptions import EventTypeException, OrderBookException

# noinspection PyProtectedMember
from polypy.orderbook import (
    HASH_STATUS,
    OrderBook,
    _quotes_event_type_price_change,
    _valid_message_asset_id,
    guess_check_orderbook_hash,
    merge_quotes_to_order_summaries,
    message_to_orderbook,
    split_order_summaries_to_quotes,
)

test_pth = pathlib.Path(__file__).parent


@pytest.fixture
def unified_book_yes(json_book_to_arrays):
    return json_book_to_arrays(test_pth / "data/ws_msg_book_yes.json")


@pytest.fixture
def book_t0_ws_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return json.loads(f.readlines()[1])


@pytest.fixture
def price_change_t1_ws_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return json.loads(f.readlines()[4])


@pytest.fixture
def book_t1_rest_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return json.loads(f.readlines()[5])


@pytest.fixture
def price_change_t2_ws_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return json.loads(f.readlines()[8])


@pytest.fixture
def book_t2_rest_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return json.loads(f.readlines()[9])


# noinspection PyUnresolvedReferences,DuplicatedCode
def test_orderbook_set():
    orderbook = OrderBook("test_token_id", 0.01)

    assert np.issubdtype(orderbook.dtype, np.floating)

    bid_p = [0.10, 0.09, 0.08, 0.05]
    bid_s = [10, 9, 8, 500]

    ask_p = [0.12, 0.2]
    ask_s = [1000, 0]

    orderbook.set_bids(bid_p, bid_s)
    orderbook.set_asks(ask_p, ask_s)

    assert orderbook.bid_size(0.10) == 10
    assert orderbook.bid_size(0) == 0

    assert orderbook.ask_size(0.12) == 1000
    assert orderbook.ask_size(0.2) == 0
    assert orderbook.ask_size(0.9) == 0
    assert orderbook.ask_size(1) == 0

    assert orderbook.bid_prices.tolist() == bid_p
    assert orderbook.bid_sizes.tolist() == bid_s

    assert orderbook.ask_prices.tolist() == [0.12]
    assert orderbook.ask_sizes.tolist() == [1000]

    assert orderbook.best_bid_price == 0.1
    assert orderbook.best_bid_size == 10
    assert orderbook.best_ask_price == 0.12
    assert orderbook.best_ask_size == 1000

    assert type(orderbook.bids[1][0]) is orderbook.dtype


def test_orderbook_set_unordered():
    orderbook = OrderBook("test_token_id", 0.01)

    bid_p = [0.10, 0.05, 0.08, 0.09]
    bid_s = [10, 500, 8, 9]

    ask_p = [0.2, 0.12]
    ask_s = [0, 1000]

    orderbook.set_bids(bid_p, bid_s)
    orderbook.set_asks(ask_p, ask_s)

    assert orderbook.bid_prices.tolist() == [0.10, 0.09, 0.08, 0.05]
    assert orderbook.bid_sizes.tolist() == [10, 9, 8, 500]
    assert orderbook.ask_prices.tolist() == [0.12]
    assert orderbook.ask_sizes.tolist() == [1000]

    assert orderbook.best_bid_price == 0.1
    assert orderbook.best_bid_size == 10
    assert orderbook.best_ask_price == 0.12
    assert orderbook.best_ask_size == 1000


def test_orderbook_raise_price_gt_1():
    orderbook = OrderBook("test_token_id", 0.01)

    bid_p = [0.10, 0.09, 0.08, 0.05]
    bid_s = [10, 9, 8, 5]
    orderbook.set_bids(bid_p, bid_s)

    with pytest.raises(IndexError):
        orderbook.set_bids([1.1], [100])
    with pytest.raises(IndexError):
        orderbook.set_asks([1.1], [100])

    with pytest.raises(IndexError):
        orderbook.bid_size(1.1)
    with pytest.raises(IndexError):
        orderbook.ask_size(1.1)


def test_orderbook_raise_price_lt_0():
    orderbook = OrderBook("test_token_id", 0.01)

    bid_p = [0.10, 0.09, 0.08, 0.05]
    bid_s = [10, 9, 8, 5]
    orderbook.set_bids(bid_p, bid_s)

    with pytest.raises(OrderBookException):
        orderbook.set_bids([-0.1], [100])
    with pytest.raises(OrderBookException):
        orderbook.set_asks([-0.1], [100])

    with pytest.raises(OrderBookException):
        orderbook.bid_size(-0.1)
    with pytest.raises(OrderBookException):
        orderbook.ask_size(-0.1)


# noinspection DuplicatedCode,PyUnresolvedReferences
def test_orderbook_array_factory():
    def factory(x: int, *_):
        return np.array([0] * x, dtype=float)

    orderbook = OrderBook("test_token_id", 0.01, zeros_factory=factory)

    bid_p = [0.10, 0.09, 0.08, 0.05]
    bid_s = [10, 9, 8, 500]

    ask_p = [0.12, 0.2]
    ask_s = [1000, 0]

    orderbook.set_bids(bid_p, bid_s)
    orderbook.set_asks(ask_p, ask_s)

    assert orderbook.bid_size(0.10) == 10
    assert orderbook.bid_size(0) == 0

    assert orderbook.ask_size(0.12) == 1000
    assert orderbook.ask_size(0.2) == 0
    assert orderbook.ask_size(0.9) == 0
    assert orderbook.ask_size(1) == 0

    assert orderbook.bid_prices.tolist() == bid_p
    assert orderbook.bid_sizes.tolist() == bid_s

    assert orderbook.ask_prices.tolist() == [0.12]
    assert orderbook.ask_sizes.tolist() == [1000]

    assert orderbook.best_bid_price == 0.1
    assert orderbook.best_bid_size == 10
    assert orderbook.best_ask_price == 0.12
    assert orderbook.best_ask_size == 1000


def test_orderbook_array_factory_raise_set():
    def factory(x: int, *_):
        return np.array([0] * x, dtype=int)

    orderbook = OrderBook("test_token_id", 0.01, zeros_factory=factory)

    assert np.issubdtype(orderbook.dtype, np.integer)

    orderbook.set_bids([0.1], [10])

    with pytest.raises(TypeError):
        orderbook.set_bids([0.1], [0.5])


def test_orderbook_tick_size():
    book = OrderBook("test_token_id", 0.01)

    assert book.tick_size == 0.01

    book.tick_size = 0.001

    assert book.tick_size == 0.001


def test_orderbook_raise_tick_size_base10():
    with pytest.raises(OrderBookException):
        OrderBook("test_token_id", 0.011)


def test_orderbook_raise_tick_size_allowed():
    with pytest.raises(OrderBookException):
        OrderBook("test_token_id", 0.00001)

    book = OrderBook("test_token_id", 0.01)
    with pytest.raises(OrderBookException):
        book.tick_size = 0.00001


@responses.activate
def test_orderbook_update_tick_size():
    book = OrderBook("test_token_id", 0.01)
    endpoint = "https://test_endpoint.com"

    # mock response
    rest_data = {"minimum_tick_size": 0.001}
    responses.get(f"{endpoint}/tick-size?token_id={book.token_id}", json=rest_data)

    book.update_tick_size(endpoint)
    assert book.tick_size == 0.001


@responses.activate
def test_orderbook_update_tick_size_raise_allowed_tick_sizes():
    book = OrderBook("test_token_id", 0.01)
    endpoint = "https://test_endpoint.com"

    # mock response
    rest_data = {"minimum_tick_size": 0.000001}
    responses.get(f"{endpoint}/tick-size?token_id={book.token_id}", json=rest_data)

    with pytest.raises(OrderBookException):
        book.update_tick_size(endpoint)


@responses.activate
def test_orderbook_update_tick_size_raise_base10():
    book = OrderBook("test_token_id", 0.01)
    endpoint = "https://test_endpoint.com"

    # mock response
    rest_data = {"minimum_tick_size": 0.02}
    responses.get(f"{endpoint}/tick-size?token_id={book.token_id}", json=rest_data)

    with pytest.raises(OrderBookException):
        book.update_tick_size(endpoint)


# noinspection PyUnresolvedReferences
def test_orderbook_bids_asks(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)

    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bidsp, bidsq = orderbook.bids
    asksp, asksq = orderbook.asks

    assert bidsp.tolist() == np.round(np.linspace(0, 1, 1001), 3)[::-1].tolist()
    assert asksp.tolist() == np.round(np.linspace(0, 1, 1001), 3).tolist()

    cmp_bidsq, cmp_asksq = np.zeros(1001), np.zeros(1001)
    cmp_bidsq[(bid_p * 1000).astype(int)] = bid_q
    cmp_asksq[(ask_p * 1000).astype(int)] = ask_q

    assert bidsq.tolist() == cmp_bidsq[::-1].tolist()
    assert asksq.tolist() == cmp_asksq.tolist()


def test_orderbook_null(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    assert len(orderbook.bid_sizes) > 0
    assert len(orderbook.ask_sizes) > 0

    orderbook.null_bids()
    orderbook.null_asks()

    assert len(orderbook.bid_sizes) == 0
    assert len(orderbook.ask_sizes) == 0
    assert np.sum(orderbook.bids[1]) == 0
    assert np.sum(orderbook.asks[1]) == 0


def test_orderbook_reset(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    assert len(orderbook.bid_prices) > 1
    assert len(orderbook.ask_prices) > 1
    assert np.sum(orderbook.bids[1]) != 2
    assert np.sum(orderbook.asks[1]) != 6

    bid_p, bid_q = [0.1], [2]
    ask_p, ask_q = [0.5], [6]

    orderbook.reset_bids(bid_p, bid_q)
    orderbook.reset_asks(ask_p, ask_q)

    assert orderbook.bid_size(0.1) == 2
    assert orderbook.ask_size(0.5) == 6
    assert len(orderbook.bid_prices) == 1
    assert len(orderbook.ask_prices) == 1
    assert np.sum(orderbook.bids[1]) == 2
    assert np.sum(orderbook.asks[1]) == 6

    orderbook.reset_bids()
    orderbook.reset_asks()

    assert len(orderbook.bid_sizes) == 0
    assert len(orderbook.ask_sizes) == 0
    assert np.sum(orderbook.bids[1]) == 0
    assert np.sum(orderbook.asks[1]) == 0

    assert type(orderbook.bids[1][0]) is orderbook.dtype


def test_orderbook_set_raise_type(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bids_q, asks_q = orderbook.bids[1], orderbook.asks[1]

    with pytest.raises(TypeError):
        orderbook.set_bids([Decimal(1)], [Decimal(1)])
    with pytest.raises(TypeError):
        orderbook.set_bids([1], [Decimal(1)])
    with pytest.raises(TypeError):
        orderbook.set_asks([Decimal(1)], [Decimal(1)])
    with pytest.raises(TypeError):
        orderbook.set_asks([1], [Decimal(1)])

    assert bids_q.tolist() == orderbook.bids[1].tolist()
    assert asks_q.tolist() == orderbook.asks[1].tolist()

    orderbook.set_bids([Decimal(1)], [1])
    orderbook.set_asks([Decimal(1)], [1])
    assert orderbook.ask_size(1) == 1
    assert orderbook.bid_size(1) == 1


def test_orderbook_reset_raise_type(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bids_q, asks_q = orderbook.bids[1], orderbook.asks[1]

    with pytest.raises(TypeError):
        orderbook.reset_bids([Decimal(1)], [Decimal(1)])
    with pytest.raises(TypeError):
        orderbook.reset_bids([1], [Decimal(1)])
    with pytest.raises(TypeError):
        orderbook.reset_asks([Decimal(1)], [Decimal(1)])
    with pytest.raises(TypeError):
        orderbook.reset_asks([1], [Decimal(1)])

    assert bids_q.tolist() == orderbook.bids[1].tolist()
    assert asks_q.tolist() == orderbook.asks[1].tolist()

    orderbook.reset_bids([Decimal(1)], [1])
    orderbook.reset_asks([Decimal(1)], [1])

    assert np.sum(orderbook.bids[1]) == 1
    assert np.sum(orderbook.asks[1]) == 1


# noinspection DuplicatedCode
def test_orderbook_set_raise_empty(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bid_quants = orderbook.bids[1].copy()
    ask_quants = orderbook.asks[1].copy()

    with pytest.raises(IndexError):
        orderbook.set_bids([0.1], [])

    with pytest.raises(IndexError):
        orderbook.set_asks([0.1], [])

    with pytest.raises(IndexError):
        orderbook.set_bids([], [10])

    with pytest.raises(IndexError):
        orderbook.set_asks([], [10])

    with pytest.raises(IndexError):
        orderbook.set_bids([], [])

    with pytest.raises(IndexError):
        orderbook.set_asks([], [])

    assert bid_quants.tolist() == orderbook.bids[1].tolist()
    assert ask_quants.tolist() == orderbook.asks[1].tolist()


# noinspection DuplicatedCode
def test_orderbook_reset_raise_empty(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bid_quants = orderbook.bids[1].copy()
    ask_quants = orderbook.asks[1].copy()

    with pytest.raises(IndexError):
        orderbook.reset_bids([0.1], [])

    with pytest.raises(IndexError):
        orderbook.reset_asks([0.1], [])

    with pytest.raises(IndexError):
        orderbook.reset_bids([], [10])

    with pytest.raises(IndexError):
        orderbook.reset_asks([], [10])

    with pytest.raises(IndexError):
        orderbook.reset_bids([], [])

    with pytest.raises(IndexError):
        orderbook.reset_asks([], [])

    assert bid_quants.tolist() == orderbook.bids[1].tolist()
    assert ask_quants.tolist() == orderbook.asks[1].tolist()


def test_orderbook_frozen_attributes(unified_book_yes):
    orderbook = OrderBook("test_token", 0.001)

    with pytest.raises(FrozenAttributeError):
        orderbook.allowed_tick_sizes = {0.1, 0.00001}

    with pytest.raises(FrozenAttributeError):
        orderbook.zeros_factory = np.zeros


# noinspection DuplicatedCode
def test_orderbook_set_decimal():
    def arr_factory(x: int, *_):
        return np.array([Decimal(0)] * x, dtype=object)

    orderbook = OrderBook("test_token_id", 0.01, zeros_factory=arr_factory)

    assert orderbook.dtype is Decimal

    bid_p = [Decimal("0.10"), Decimal("0.09"), Decimal("0.08"), Decimal("0.05")]
    bid_s = [Decimal(i) for i in [10, 9, 8, 500]]

    ask_p = [Decimal("0.12"), Decimal("0.2")]
    ask_s = [Decimal(i) for i in [1000, 0]]

    orderbook.set_bids(bid_p, bid_s)
    orderbook.set_asks(ask_p, ask_s)

    assert orderbook.bid_size(0.10) == Decimal(10)
    assert orderbook.bid_size(Decimal("0.10")) == Decimal(10)
    assert orderbook.bid_size(0) == Decimal(0)

    assert orderbook.ask_size(0.12) == Decimal(1000)
    assert orderbook.ask_size(Decimal("0.12")) == Decimal(1000)
    assert orderbook.ask_size(0.2) == Decimal(0)
    assert orderbook.ask_size(Decimal("0.2")) == Decimal(0)
    assert orderbook.ask_size(0.9) == Decimal(0)
    assert orderbook.ask_size(Decimal("0.9")) == Decimal(0)
    assert orderbook.ask_size(1) == Decimal(0)

    assert [Decimal(str(i)) for i in orderbook.bid_prices.tolist()] == bid_p
    assert orderbook.bid_sizes.tolist() == bid_s

    assert [Decimal(str(i)) for i in orderbook.ask_prices.tolist()] == [Decimal("0.12")]
    assert orderbook.ask_sizes.tolist() == [1000]

    assert orderbook.best_bid_price == 0.1
    assert orderbook.best_bid_size == Decimal(10)
    assert orderbook.best_ask_price == 0.12
    assert orderbook.best_ask_size == Decimal(1000)


# noinspection DuplicatedCode
def test_orderbook_reset_decimal():
    def arr_factory(x: int, *_):
        return np.array([Decimal(0)] * x, dtype=object)

    orderbook = OrderBook("test_token_id", 0.01, zeros_factory=arr_factory)

    assert orderbook.dtype is Decimal

    orderbook.set_bids([Decimal("0.2")], [Decimal(10)])
    orderbook.set_asks([Decimal("0.5")], [Decimal(10)])
    assert orderbook.bid_size(0.2) == Decimal(10)
    assert orderbook.ask_size(0.5) == Decimal(10)
    assert np.sum(orderbook.bids[1]) == 10
    assert np.sum(orderbook.asks[1]) == 10

    bid_p = [Decimal("0.10"), Decimal("0.09"), Decimal("0.08"), Decimal("0.05")]
    bid_s = [Decimal(i) for i in [10, 9, 8, 500]]

    ask_p = [Decimal("0.12"), Decimal("0.2")]
    ask_s = [Decimal(i) for i in [1000, 0]]

    orderbook.reset_bids(bid_p, bid_s)
    orderbook.reset_asks(ask_p, ask_s)

    assert orderbook.bid_size(0.2) == Decimal(0)
    assert orderbook.ask_size(0.5) == Decimal(0)

    assert orderbook.bid_size(0.10) == Decimal(10)
    assert orderbook.bid_size(Decimal("0.10")) == Decimal(10)
    assert orderbook.bid_size(0) == Decimal(0)

    assert orderbook.ask_size(0.12) == Decimal(1000)
    assert orderbook.ask_size(Decimal("0.12")) == Decimal(1000)
    assert orderbook.ask_size(0.2) == Decimal(0)
    assert orderbook.ask_size(Decimal("0.2")) == Decimal(0)
    assert orderbook.ask_size(0.9) == Decimal(0)
    assert orderbook.ask_size(Decimal("0.9")) == Decimal(0)
    assert orderbook.ask_size(1) == Decimal(0)

    assert [Decimal(str(i)) for i in orderbook.bid_prices.tolist()] == bid_p
    assert orderbook.bid_sizes.tolist() == bid_s

    assert [Decimal(str(i)) for i in orderbook.ask_prices.tolist()] == [Decimal("0.12")]
    assert orderbook.ask_sizes.tolist() == [1000]

    assert orderbook.best_bid_price == 0.1
    assert orderbook.best_bid_size == Decimal(10)
    assert orderbook.best_ask_price == 0.12
    assert orderbook.best_ask_size == Decimal(1000)


def test_orderbook_set_decimal_raise_type():
    def arr_factory(x: int, *_):
        return np.array([Decimal(0)] * x, dtype=object)

    orderbook = OrderBook("test_token_id", 0.01, zeros_factory=arr_factory)
    orderbook.set_bids([0.5], [Decimal(200)])

    with pytest.raises(TypeError):
        orderbook.set_bids([0.5], [200])

    with pytest.raises(TypeError):
        orderbook.set_bids([0.5], [200.2])


# noinspection DuplicatedCode
def test_orderbook_book_hash(
    book_t0_ws_msg_hashed, book_t1_rest_msg_hashed, book_t2_rest_msg_hashed
):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    market_id = book_t0_ws_msg_hashed["market"]
    orderbook = OrderBook(asset_id, 0.01)

    orderbook, hash_status = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert hash_status is HASH_STATUS.VALID
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed["timestamp"])
        == book_t0_ws_msg_hashed["hash"]
    )

    # test updating
    orderbook, hash_status = message_to_orderbook(
        book_t1_rest_msg_hashed, orderbook, event_type="book"
    )
    assert hash_status is HASH_STATUS.VALID
    assert (
        orderbook.hash(market_id, book_t1_rest_msg_hashed["timestamp"])
        == book_t1_rest_msg_hashed["hash"]
    )

    orderbook, hash_status = message_to_orderbook(
        book_t2_rest_msg_hashed, orderbook, event_type="book"
    )
    assert hash_status is HASH_STATUS.VALID
    assert (
        orderbook.hash(market_id, book_t2_rest_msg_hashed["timestamp"])
        == book_t2_rest_msg_hashed["hash"]
    )


# noinspection DuplicatedCode
def test_orderbook_price_change_hash(
    book_t0_ws_msg_hashed,
    price_change_t1_ws_msg_hashed,
    book_t1_rest_msg_hashed,
    price_change_t2_ws_msg_hashed,
    book_t2_rest_msg_hashed,
):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    market_id = book_t0_ws_msg_hashed["market"]
    orderbook = OrderBook(asset_id, 0.01)

    # check hash against initial book message
    orderbook, hash_status = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert hash_status is HASH_STATUS.VALID
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed["timestamp"])
        == book_t0_ws_msg_hashed["hash"]
    )

    # check hash after updating first price change
    orderbook, hash_status = message_to_orderbook(
        price_change_t1_ws_msg_hashed, orderbook
    )
    _, _, hash_t1 = guess_check_orderbook_hash(
        price_change_t1_ws_msg_hashed["hash"],
        orderbook,
        market_id,
        [int(price_change_t1_ws_msg_hashed["timestamp"]) - i for i in range(15)],
    )
    assert hash_status is HASH_STATUS.UNKNOWN
    assert price_change_t1_ws_msg_hashed["hash"] == book_t1_rest_msg_hashed["hash"]
    assert hash_t1 == price_change_t1_ws_msg_hashed["hash"]

    # check hash after updating second price change
    orderbook, hash_status = message_to_orderbook(
        price_change_t2_ws_msg_hashed, orderbook
    )
    _, _, hash_t2 = guess_check_orderbook_hash(
        price_change_t2_ws_msg_hashed["hash"],
        orderbook,
        market_id,
        [int(price_change_t2_ws_msg_hashed["timestamp"]) - i for i in range(15)],
    )
    assert hash_status is HASH_STATUS.UNKNOWN
    assert price_change_t2_ws_msg_hashed["hash"] == book_t2_rest_msg_hashed["hash"]
    assert price_change_t2_ws_msg_hashed["hash"] != book_t1_rest_msg_hashed["hash"]
    assert hash_t2 == price_change_t2_ws_msg_hashed["hash"]


def test_orderbook_marketable_amount_buy(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    orderbook = OrderBook(asset_id, 0.001)
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    price, amount = orderbook.marketable_price("BUY", 135.26)
    assert math.isclose(price, 0.544)
    assert math.isclose(amount, 135.26)

    price, amount = orderbook.marketable_price("BUY", 135.27)
    assert math.isclose(price, 0.548)
    assert math.isclose(amount, 667.42828)


def test_orderbook_marketable_amount_sell(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    orderbook = OrderBook(asset_id, 0.001)
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    price, amount = orderbook.marketable_price("SELL", 1917.82)
    assert math.isclose(price, 0.53)
    assert math.isclose(amount, 1917.82)

    price, amount = orderbook.marketable_price("SELL", 1917.83)
    assert math.isclose(price, 0.523)
    assert math.isclose(amount, 2048.57)


def test_orderbook_marketable_amount_raises_unknown_side(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    orderbook = OrderBook(asset_id, 0.001)
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    with pytest.raises(OrderBookException) as e:
        orderbook.marketable_price("SOME", 135.26)

    assert "Unknown side" in str(e)


def test_orderbook_marketable_amount_raises_unknown_max_liquidity(
    book_t0_ws_msg_hashed,
):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    orderbook = OrderBook(asset_id, 0.001)
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    with pytest.raises(OrderBookException) as e:
        orderbook.marketable_price("BUY", 1e9)

    assert "No marketable price" in str(e)


def test_orderbook_from_dict(book_t0_ws_msg_hashed):
    book = OrderBook.from_dict(
        book_t0_ws_msg_hashed,
    )

    asset_id = book_t0_ws_msg_hashed["asset_id"]
    market_id = book_t0_ws_msg_hashed["market"]
    timestamp = book_t0_ws_msg_hashed["timestamp"]

    orderbook = OrderBook(asset_id, 0.001)
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    assert book.hash(market_id, timestamp) == orderbook.hash(market_id, timestamp)
    assert list(book.bid_prices) == list(orderbook.bid_prices)
    assert list(book.ask_prices) == list(orderbook.ask_prices)
    assert book.token_id == orderbook.token_id


# noinspection DuplicatedCode
def test_guess_orderbook_hash_int_timestamps(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    market_id = book_t0_ws_msg_hashed["market"]
    timestamp = int(book_t0_ws_msg_hashed["timestamp"])
    orderbook = OrderBook(asset_id, 0.01)

    # check hash against initial book message
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed["timestamp"])
        == book_t0_ws_msg_hashed["hash"]
    )

    assert (
        guess_check_orderbook_hash(
            book_t0_ws_msg_hashed["hash"],
            orderbook,
            market_id,
            [timestamp - i for i in range(15)],
        )[0]
        is True
    )


# noinspection DuplicatedCode
def test_guess_orderbook_hash_str_timestamps(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    market_id = book_t0_ws_msg_hashed["market"]
    timestamp = int(book_t0_ws_msg_hashed["timestamp"])
    orderbook = OrderBook(asset_id, 0.01)

    # check hash against initial book message
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed["timestamp"])
        == book_t0_ws_msg_hashed["hash"]
    )

    assert (
        guess_check_orderbook_hash(
            book_t0_ws_msg_hashed["hash"],
            orderbook,
            market_id,
            [str(timestamp - i) for i in range(15)],
        )[0]
        is True
    )


def test_guess_orderbook_hash_false(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    market_id = book_t0_ws_msg_hashed["market"]
    orderbook = OrderBook(asset_id, 0.01)

    # check hash against initial book message
    orderbook, _ = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed["timestamp"])
        == book_t0_ws_msg_hashed["hash"]
    )

    assert (
        guess_check_orderbook_hash(
            book_t0_ws_msg_hashed["hash"], orderbook, market_id, [0, 1, 2, 3, 4, 5, 6]
        )[0]
        is False
    )


def test_orderbook_to_dict(book_t1_rest_msg_hashed):
    client_orderbook = parse_raw_orderbook_summary(book_t1_rest_msg_hashed)

    orderbook, _ = message_to_orderbook(
        book_t1_rest_msg_hashed,
        OrderBook(book_t1_rest_msg_hashed["asset_id"], 0.01),
        event_type="book",
    )

    client_dict = client_orderbook.__dict__
    book_dict = orderbook.to_dict(
        book_t1_rest_msg_hashed["market"],
        book_t1_rest_msg_hashed["timestamp"],
        book_t1_rest_msg_hashed["hash"],
    )

    assert client_dict == book_dict


def test_orderbook_to_dict_int_timestamp(book_t1_rest_msg_hashed):
    client_orderbook = parse_raw_orderbook_summary(book_t1_rest_msg_hashed)

    orderbook, _ = message_to_orderbook(
        book_t1_rest_msg_hashed,
        OrderBook(book_t1_rest_msg_hashed["asset_id"], 0.01),
        event_type="book",
    )

    client_dict = client_orderbook.__dict__
    book_dict = orderbook.to_dict(
        book_t1_rest_msg_hashed["market"],
        int(book_t1_rest_msg_hashed["timestamp"]),
        book_t1_rest_msg_hashed["hash"],
    )

    assert client_dict == book_dict


# noinspection DuplicatedCode
def test_parse_ws_event_type_price_change_float():
    msg = {
        "changes": [
            {"price": "0.1", "side": "SELL", "size": "10"},
            {"price": "0.11", "side": "SELL", "size": "6.69"},
            {"price": "0.5", "side": "BUY", "size": "200101090.45"},
            {"price": "0.67", "side": "BUY", "size": "10.0"},
        ]
    }
    bid_prices, bid_sizes, ask_prices, ask_sizes = _quotes_event_type_price_change(
        msg, float
    )

    assert bid_prices == [0.5, 0.67]
    assert bid_sizes == [200101090.45, 10.0]
    assert ask_prices == [0.1, 0.11]
    assert ask_sizes == [10, 6.69]


# noinspection DuplicatedCode
def test_parse_ws_event_type_price_change_decimal():
    msg = {
        "changes": [
            {"price": "0.1", "side": "SELL", "size": "10"},
            {"price": "0.11", "side": "SELL", "size": "6.69"},
            {"price": "0.5", "side": "BUY", "size": "200101090.45"},
            {"price": "0.67", "side": "BUY", "size": "10.0"},
        ]
    }
    bid_prices, bid_sizes, ask_prices, ask_sizes = _quotes_event_type_price_change(
        msg, Decimal
    )

    assert bid_prices == [Decimal(str(i)) for i in [0.5, 0.67]]
    assert bid_sizes == [Decimal(str(i)) for i in [200101090.45, 10.0]]
    assert ask_prices == [Decimal(str(i)) for i in [0.1, 0.11]]
    assert ask_sizes == [Decimal(str(i)) for i in [10, 6.69]]


def test_parse_ws_event_type_price_change_float_empty_asks():
    msg = {
        "changes": [
            {"price": "0.5", "side": "BUY", "size": "200101090.45"},
            {"price": "0.67", "side": "BUY", "size": "10.0"},
        ]
    }
    bid_prices, bid_sizes, ask_prices, ask_sizes = _quotes_event_type_price_change(
        msg, Decimal
    )

    assert bid_prices == [Decimal(str(i)) for i in [0.5, 0.67]]
    assert bid_sizes == [Decimal(str(i)) for i in [200101090.45, 10.0]]
    assert len(ask_prices) == 0
    assert len(ask_sizes) == 0


# noinspection DuplicatedCode
def test_parse_ws_event_type_price_change_float_zeroing_asks():
    msg = {
        "changes": [
            {"price": "0.1", "side": "SELL", "size": "0"},
            {"price": "0.11", "side": "SELL", "size": "0"},
            {"price": "0.5", "side": "BUY", "size": "200101090.45"},
            {"price": "0.67", "side": "BUY", "size": "10.0"},
        ]
    }
    bid_prices, bid_sizes, ask_prices, ask_sizes = _quotes_event_type_price_change(
        msg, float
    )

    assert bid_prices == [0.5, 0.67]
    assert bid_sizes == [200101090.45, 10.0]
    assert ask_prices == [0.1, 0.11]
    assert ask_sizes == [0, 0]


def test_parse_ws_event_type_price_change_raise_side():
    msg = {
        "changes": [
            {"price": "0.1", "side": "SELL", "size": "10"},
            {"price": "0.11", "side": "SELL", "size": "6.69"},
            {"price": "0.5", "side": "SOME", "size": "200101090.45"},
            {"price": "0.67", "side": "BUY", "size": "10.0"},
        ]
    }

    with pytest.raises(ValueError):
        _quotes_event_type_price_change(msg, float)


def test_message_to_orderbook_empty_asks_book_msg(book_t0_ws_msg_hashed):
    book = OrderBook(book_t0_ws_msg_hashed["asset_id"], 0.01)

    book_t0_ws_msg_hashed["asks"] = []

    book, hash_status = message_to_orderbook(book_t0_ws_msg_hashed, book)

    assert hash_status is HASH_STATUS.VALID
    assert np.sum(book.asks[1]) == 0
    assert np.sum(book.bids[1]) > 0


# noinspection DuplicatedCode
def test_message_to_orderbook_empty_asks_price_change_msg(
    book_t0_ws_msg_hashed, price_change_t1_ws_msg_hashed
):
    book = OrderBook(book_t0_ws_msg_hashed["asset_id"], 0.01)
    book, hash_status = message_to_orderbook(book_t0_ws_msg_hashed, book)

    assert hash_status is HASH_STATUS.VALID
    assert np.sum(book.asks[1]) > 0
    assert np.sum(book.bids[1]) > 0

    bid_vol = np.sum(book.bids[1])
    ask_vol = np.sum(book.asks[1])

    price_change_t1_ws_msg_hashed["changes"] = [
        d for d in price_change_t1_ws_msg_hashed["changes"] if d["side"] == "BUY"
    ]
    book, hash_status = message_to_orderbook(price_change_t1_ws_msg_hashed, book)

    assert hash_status is HASH_STATUS.UNKNOWN
    assert np.sum(book.asks[1]) == ask_vol
    assert np.sum(book.bids[1]) != bid_vol
    assert np.sum(book.asks[1]) > 0
    assert np.sum(book.bids[1]) > 0


# noinspection DuplicatedCode
def test_message_to_orderbook_zeroing_asks_price_change_msg(
    book_t0_ws_msg_hashed, price_change_t1_ws_msg_hashed
):
    book = OrderBook(book_t0_ws_msg_hashed["asset_id"], 0.01)
    book, hash_status = message_to_orderbook(book_t0_ws_msg_hashed, book)

    assert hash_status is HASH_STATUS.VALID
    assert np.sum(book.asks[1]) > 0
    assert np.sum(book.bids[1]) > 0

    bid_vol = np.sum(book.bids[1])

    buy_changes = [
        d for d in price_change_t1_ws_msg_hashed["changes"] if d["side"] == "BUY"
    ]
    ask_changes = [
        {"price": str(i), "side": "SELL", "size": "0"} for i in book.ask_prices
    ]
    price_change_t1_ws_msg_hashed["changes"] = buy_changes + ask_changes

    book, hash_status = message_to_orderbook(price_change_t1_ws_msg_hashed, book)

    assert hash_status is HASH_STATUS.UNKNOWN
    assert np.sum(book.asks[1]) == 0
    assert np.sum(book.bids[1]) != bid_vol
    assert np.sum(book.bids[1]) > 0


def test_message_to_orderbook_raise_event_type_exists(book_t0_ws_msg_hashed):
    # ws response with event_type specified: fail
    # ws response without event_type: success
    with pytest.raises(ValueError):
        message_to_orderbook(
            book_t0_ws_msg_hashed,
            OrderBook(book_t0_ws_msg_hashed["asset_id"], 0.01),
            event_type="book",
        )

    message_to_orderbook(
        book_t0_ws_msg_hashed, OrderBook(book_t0_ws_msg_hashed["asset_id"], 0.01)
    )


def test_message_to_orderbook_raise_specify_event_type(book_t1_rest_msg_hashed):
    # rest response without event_type: fail
    # rest response with event_type specified: success
    with pytest.raises(KeyError) as e:
        message_to_orderbook(
            book_t1_rest_msg_hashed,
            OrderBook(book_t1_rest_msg_hashed["asset_id"], 0.01),
        )

        assert "REST" in str(e)

    message_to_orderbook(
        book_t1_rest_msg_hashed,
        OrderBook(book_t1_rest_msg_hashed["asset_id"], 0.01),
        event_type="book",
    )


def test_message_to_orderbook_raise_unknown_event_type(book_t0_ws_msg_hashed):
    book = OrderBook(book_t0_ws_msg_hashed["asset_id"], 0.01)
    book, _ = message_to_orderbook(book_t0_ws_msg_hashed, book)
    assert (
        book.hash(book_t0_ws_msg_hashed["market"], book_t0_ws_msg_hashed["timestamp"])
        == book_t0_ws_msg_hashed["hash"]
    )

    book_t0_ws_msg_hashed["event_type"] = "some"

    with pytest.raises(EventTypeException):
        message_to_orderbook(book_t0_ws_msg_hashed, book)


def test_message_to_orderbook_raise_silent_valid_asset_id(book_t0_ws_msg_hashed):
    book, _ = message_to_orderbook(
        book_t0_ws_msg_hashed,
        OrderBook("test_token_id", 0.01),
        mode="silent",
    )

    assert np.sum(book.bids[1]) == 0
    assert np.sum(book.asks[1]) == 0


def test_message_to_orderbook_tick_size_change():
    book = OrderBook("test_token_id", 0.01)

    msg = {
        "event_type": "tick_size_change",
        "asset_id": "test_token_id",
        "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
        "old_tick_size": "0.01",
        "new_tick_size": "0.001",
        "timestamp": "100000000",
    }

    book, hash_status = message_to_orderbook(msg, book)

    assert hash_status is HASH_STATUS.UNCHANGED
    assert book.tick_size == 0.001


def test_merge_quotes_to_order_summaries_dict():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]

    data_dict = merge_quotes_to_order_summaries(prices, sizes, False, dict)

    assert data_dict == [
        {"price": "0.1", "size": "20000001.45"},
        {"price": "0.5", "size": "50"},
        {"price": "0.75", "size": "75.5"},
        {"price": "0.2", "size": "20"},
    ]


def test_merge_quotes_to_order_summaries_dict_reverse():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]

    data_dict = merge_quotes_to_order_summaries(prices, sizes, True, dict)
    data_not_reversed = merge_quotes_to_order_summaries(prices, sizes, False, dict)

    assert data_dict == [
        {"price": "0.2", "size": "20"},
        {"price": "0.75", "size": "75.5"},
        {"price": "0.5", "size": "50"},
        {"price": "0.1", "size": "20000001.45"},
    ]
    assert data_dict != data_not_reversed


def test_merge_quotes_to_order_summaries_order_summary():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]

    data_dict = merge_quotes_to_order_summaries(prices, sizes, False, OrderSummary)

    assert data_dict == [
        OrderSummary(price="0.1", size="20000001.45"),
        OrderSummary(price="0.5", size="50"),
        OrderSummary(price="0.75", size="75.5"),
        OrderSummary(price="0.2", size="20"),
    ]


def test_merge_quotes_to_order_summaries_order_summaries_reverse():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]

    data_dict = merge_quotes_to_order_summaries(prices, sizes, True, OrderSummary)
    data_not_reversed = merge_quotes_to_order_summaries(
        prices, sizes, False, OrderSummary
    )

    assert data_dict == [
        OrderSummary(price="0.2", size="20"),
        OrderSummary(price="0.75", size="75.5"),
        OrderSummary(price="0.5", size="50"),
        OrderSummary(price="0.1", size="20000001.45"),
    ]
    assert data_dict != data_not_reversed


# noinspection DuplicatedCode
def test_split_order_summaries_to_quotes_dict_float():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]
    x = [
        {"price": "0.1", "size": "20000001.45"},
        {"price": "0.5", "size": "50"},
        {"price": "0.75", "size": "75.5"},
        {"price": "0.2", "size": "20"},
    ]

    p, s = split_order_summaries_to_quotes(x, float)

    assert p == prices
    assert s == sizes


# noinspection DuplicatedCode
def test_split_order_summaries_to_quotes_dict_decimal():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]
    x = [
        {"price": "0.1", "size": "20000001.45"},
        {"price": "0.5", "size": "50"},
        {"price": "0.75", "size": "75.5"},
        {"price": "0.2", "size": "20"},
    ]

    p, s = split_order_summaries_to_quotes(x, Decimal)

    assert p == [Decimal(str(i)) for i in prices]
    assert s == [Decimal(str(i)) for i in sizes]


def test_split_order_summaries_to_quotes_dict_float_maintain_order():
    prices = [0.1, 0.5, 0.75, 0.2][::-1]
    sizes = [20000001.45, 50, 75.5, 20.0][::-1]
    x = [
        {"price": "0.1", "size": "20000001.45"},
        {"price": "0.5", "size": "50"},
        {"price": "0.75", "size": "75.5"},
        {"price": "0.2", "size": "20"},
    ][::-1]

    p, s = split_order_summaries_to_quotes(x, float)

    assert p == prices
    assert s == sizes


# noinspection DuplicatedCode
def test_split_order_summaries_to_quotes_order_summary_float():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]
    x = [
        OrderSummary(price="0.1", size="20000001.45"),
        OrderSummary(price="0.5", size="50"),
        OrderSummary(price="0.75", size="75.5"),
        OrderSummary(price="0.2", size="20"),
    ]

    p, s = split_order_summaries_to_quotes(x, float)

    assert p == prices
    assert s == sizes


# noinspection DuplicatedCode
def test_split_order_summaries_to_quotes_order_summary_decimal():
    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]
    x = [
        OrderSummary(price="0.1", size="20000001.45"),
        OrderSummary(price="0.5", size="50"),
        OrderSummary(price="0.75", size="75.5"),
        OrderSummary(price="0.2", size="20"),
    ]

    p, s = split_order_summaries_to_quotes(x, Decimal)

    assert p == [Decimal(str(i)) for i in prices]
    assert s == [Decimal(str(i)) for i in sizes]


def test_split_order_summaries_to_quotes_named_tuple_decimal():
    S = collections.namedtuple("S", ["price", "size"])

    prices = [0.1, 0.5, 0.75, 0.2]
    sizes = [20000001.45, 50, 75.5, 20.0]
    x = [
        S(price="0.1", size="20000001.45"),
        S(price="0.5", size="50"),
        S(price="0.75", size="75.5"),
        S(price="0.2", size="20"),
    ]

    # noinspection PyTypeChecker
    p, s = split_order_summaries_to_quotes(x, Decimal)

    assert p == [Decimal(str(i)) for i in prices]
    assert s == [Decimal(str(i)) for i in sizes]


def test_split_order_summaries_to_quotes_raise_type():
    with pytest.raises(AttributeError):
        # noinspection PyTypeChecker
        split_order_summaries_to_quotes([("0.1", "20000001.45"), ("0.5", "50")], float)


def test_valid_message_id(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed["asset_id"]
    book = OrderBook(asset_id, 0.01)

    # test True case
    assert _valid_message_asset_id(book_t0_ws_msg_hashed, book) is True

    # test False case
    book_t0_ws_msg_hashed["asset_id"] = "test"

    with pytest.warns():
        _valid_message_asset_id(book_t0_ws_msg_hashed, book, "warn")

    with pytest.raises(OrderBookException):
        _valid_message_asset_id(book_t0_ws_msg_hashed, book, "except")

    assert _valid_message_asset_id(book_t0_ws_msg_hashed, book, "silent") is False


@pytest.mark.skip(reason="only for visualization purpose")
def test_visualize_orderbook(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = OrderBook("test_token", 0.001)

    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bidsp, bidsq = orderbook.bids
    asksp, asksq = orderbook.asks

    from matplotlib import pyplot as plt

    plt.figure()
    plt.bar(
        bidsp,
        np.log(np.cumsum(bidsq)),
        color="green",
        alpha=0.5,
        width=min(orderbook.allowed_tick_sizes),
    )
    plt.bar(
        asksp,
        np.log(np.cumsum(asksq)),
        color="red",
        alpha=0.5,
        width=min(orderbook.allowed_tick_sizes),
    )
    plt.show()