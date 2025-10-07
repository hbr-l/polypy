import copy
import json
import math
import pathlib
from decimal import Decimal
from fractions import Fraction

import msgspec
import numpy as np
import pytest
import responses
from py_clob_client.utilities import parse_raw_orderbook_summary

from polypy.book import message_to_orderbook
from polypy.book.hashing import guess_check_orderbook_hash
from polypy.book.order_book import SharedOrderBook
from polypy.exceptions import EventTypeException, OrderBookException
from polypy.order.common import SIDE
from polypy.structs import (
    BookEvent,
    PriceChangeEvent,
    PriceChangeSummary,
    TickSizeEvent,
)
from polypy.typing import NumericAlias, dec

test_pth = pathlib.Path(__file__).parent


@pytest.fixture
def unified_book_yes(json_book_to_arrays):
    return json_book_to_arrays(test_pth / "data/ws_msg_book_yes.json")


@pytest.fixture
def book_t0_ws_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return msgspec.convert(json.loads(f.readlines()[1]), BookEvent, strict=False)


@pytest.fixture
def price_change_t1_ws_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return msgspec.convert(
            json.loads(f.readlines()[4]), PriceChangeEvent, strict=False
        )


@pytest.fixture
def book_t1_rest_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return msgspec.convert(json.loads(f.readlines()[5]), BookEvent, strict=False)


@pytest.fixture
def price_change_t2_ws_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return msgspec.convert(
            json.loads(f.readlines()[8]), PriceChangeEvent, strict=False
        )


@pytest.fixture
def book_t2_rest_msg_hashed():
    with open(test_pth / "data/messages_hash.txt", "r") as f:
        return msgspec.convert(json.loads(f.readlines()[9]), BookEvent, strict=False)


def list_to_dec(x: list[NumericAlias]) -> list[Decimal]:
    return [dec(i) for i in x]


def test_orderbook_set():
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

    assert orderbook.dtype is Decimal

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

    assert orderbook.bid_prices.tolist() == list_to_dec(bid_p)
    assert orderbook.bid_sizes.tolist() == list_to_dec(bid_s)

    assert orderbook.ask_prices.tolist() == list_to_dec([0.12])
    assert orderbook.ask_sizes.tolist() == list_to_dec([1000])

    assert orderbook.best_bid_price == dec(0.1)
    assert orderbook.best_bid_size == dec(10)
    assert orderbook.best_ask_price == dec(0.12)
    assert orderbook.best_ask_size == dec(1000)

    assert type(orderbook.bids[1][0]) is orderbook.dtype


def test_orderbook_set_unordered():
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

    bid_p = [0.10, 0.05, 0.08, 0.09]
    bid_s = [10, 500, 8, 9]

    ask_p = [0.2, 0.12]
    ask_s = [0, 1000]

    orderbook.set_bids(bid_p, bid_s)
    orderbook.set_asks(ask_p, ask_s)

    assert orderbook.bid_prices.tolist() == list_to_dec([0.10, 0.09, 0.08, 0.05])
    assert orderbook.bid_sizes.tolist() == list_to_dec([10, 9, 8, 500])
    assert orderbook.ask_prices.tolist() == list_to_dec([0.12])
    assert orderbook.ask_sizes.tolist() == list_to_dec([1000])

    assert orderbook.best_bid_price == dec(0.1)
    assert orderbook.best_bid_size == dec(10)
    assert orderbook.best_ask_price == dec(0.12)
    assert orderbook.best_ask_size == dec(1000)


def test_orderbook_raise_price_gt_1():
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

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
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

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


def test_orderbook_set_fraction():
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

    assert orderbook.dtype is Decimal

    orderbook.set_bids([0.1], [Fraction(10)])

    assert orderbook.bid_prices.tolist() == list_to_dec([0.1])
    assert orderbook.bid_sizes.tolist() == list_to_dec([10])


def test_orderbook_tick_size():
    book = SharedOrderBook("test_token_id", 0.01, True)

    assert book.tick_size == dec(0.01)

    book.tick_size = 0.001

    assert book.tick_size == dec(0.001)


def test_orderbook_raise_tick_size_base10():
    with pytest.raises(OrderBookException):
        SharedOrderBook("test_token_id", 0.011, True)


def test_orderbook_raise_tick_size_allowed():
    with pytest.raises(OrderBookException):
        SharedOrderBook("test_token_id", 0.00001, True)

    book = SharedOrderBook("test_token_id", 0.01, True)
    with pytest.raises(OrderBookException):
        book.tick_size = 0.00001


@responses.activate
def test_orderbook_sync(book_t0_ws_msg_hashed):
    book = SharedOrderBook(book_t0_ws_msg_hashed.asset_id, 0.01, True)
    endpoint = "https://test_endpoint.com"

    assert book.tick_size == dec(0.01)
    assert np.sum(book.ask_prices) == 0
    assert np.sum(book.bid_prices) == 0

    # mock response
    rest_data = {"minimum_tick_size": 0.001}
    responses.get(f"{endpoint}/tick-size?token_id={book.token_id}", json=rest_data)
    book_data: dict = msgspec.to_builtins(copy.deepcopy(book_t0_ws_msg_hashed))
    del book_data["event_type"]
    book_data.update(
        {"min_order_size": "0.001", "neg_risk": False, "tick_size": "0.001"}
    )
    responses.get(f"{endpoint}/book?token_id={book.token_id}", json=book_data)

    book.sync(endpoint)
    assert book.tick_size == dec(0.001)
    assert np.sum(book.ask_prices) > 0
    assert np.sum(book.bid_prices) > 0


@responses.activate
def test_orderbook_update_tick_size():
    book = SharedOrderBook("test_token_id", 0.01, True)
    endpoint = "https://test_endpoint.com"

    # mock response
    rest_data = {"minimum_tick_size": 0.001}
    responses.get(f"{endpoint}/tick-size?token_id={book.token_id}", json=rest_data)

    book.update_tick_size(endpoint)
    assert book.tick_size == dec(0.001)


@responses.activate
def test_orderbook_update_tick_size_raise_allowed_tick_sizes():
    book = SharedOrderBook("test_token_id", 0.01, True)
    endpoint = "https://test_endpoint.com"

    # mock response
    rest_data = {"minimum_tick_size": 0.000001}
    responses.get(f"{endpoint}/tick-size?token_id={book.token_id}", json=rest_data)

    with pytest.raises(OrderBookException):
        book.update_tick_size(endpoint)


@responses.activate
def test_orderbook_update_tick_size_raise_base10():
    book = SharedOrderBook("test_token_id", 0.01, True)
    endpoint = "https://test_endpoint.com"

    # mock response
    rest_data = {"minimum_tick_size": 0.02}
    responses.get(f"{endpoint}/tick-size?token_id={book.token_id}", json=rest_data)

    with pytest.raises(OrderBookException):
        book.update_tick_size(endpoint)


def test_orderbook_bids_asks(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = SharedOrderBook("test_token", 0.001, True)

    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bidsp, bidsq = orderbook.bids
    asksp, asksq = orderbook.asks

    assert bidsp.tolist() == list_to_dec(
        np.round(np.linspace(0, 1, 1001), 3)[::-1].tolist()
    )
    assert asksp.tolist() == list_to_dec(np.round(np.linspace(0, 1, 1001), 3).tolist())

    cmp_bidsq, cmp_asksq = np.zeros(1001), np.zeros(1001)
    cmp_bidsq[(bid_p * 1000).astype(int)] = bid_q
    cmp_asksq[(ask_p * 1000).astype(int)] = ask_q

    assert bidsq.tolist() == list_to_dec(cmp_bidsq[::-1].tolist())
    assert asksq.tolist() == list_to_dec(cmp_asksq.tolist())


def test_orderbook_null(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = SharedOrderBook("test_token", 0.001, True)
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
    orderbook = SharedOrderBook("test_token", 0.001, True)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    assert len(orderbook.bid_prices) > dec(1)
    assert len(orderbook.ask_prices) > dec(1)
    assert np.sum(orderbook.bids[1]) != dec(2)
    assert np.sum(orderbook.asks[1]) != dec(6)

    bid_p, bid_q = [0.1], [2]
    ask_p, ask_q = [0.5], [6]

    orderbook.reset_bids(bid_p, bid_q)
    orderbook.reset_asks(ask_p, ask_q)

    assert orderbook.bid_size(0.1) == dec(2)
    assert orderbook.ask_size(0.5) == dec(6)
    assert len(orderbook.bid_prices) == dec(1)
    assert len(orderbook.ask_prices) == dec(1)
    assert np.sum(orderbook.bids[1]) == dec(2)
    assert np.sum(orderbook.asks[1]) == dec(6)

    orderbook.reset_bids()
    orderbook.reset_asks()

    assert len(orderbook.bid_sizes) == dec(0)
    assert len(orderbook.ask_sizes) == dec(0)
    assert np.sum(orderbook.bids[1]) == dec(0)
    assert np.sum(orderbook.asks[1]) == dec(0)

    assert type(orderbook.bids[1][0]) is orderbook.dtype


def test_orderbook_set_mixed_types(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = SharedOrderBook("test_token", 0.001, True)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bids_q, asks_q = orderbook.bids[1], orderbook.asks[1]

    assert bids_q.tolist() == list_to_dec(orderbook.bids[1].tolist())
    assert asks_q.tolist() == list_to_dec(orderbook.asks[1].tolist())

    orderbook.set_bids([Decimal(1)], [1])
    orderbook.set_asks([Decimal(1)], [1])
    assert orderbook.ask_size(1) == dec(1)
    assert orderbook.bid_size(1) == dec(1)


def test_orderbook_reset_mixed_type(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = SharedOrderBook("test_token", 0.001, True)
    orderbook.set_bids(bid_p, bid_q)
    orderbook.set_asks(ask_p, ask_q)

    bids_q, asks_q = orderbook.bids[1], orderbook.asks[1]

    assert bids_q.tolist() == list_to_dec(orderbook.bids[1].tolist())
    assert asks_q.tolist() == list_to_dec(orderbook.asks[1].tolist())

    orderbook.reset_bids([Decimal(1)], [1])
    orderbook.reset_asks([Decimal(1)], [1])

    assert np.sum(orderbook.bids[1]) == dec(1)
    assert np.sum(orderbook.asks[1]) == dec(1)


# noinspection DuplicatedCode
def test_orderbook_set_raise_empty(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = SharedOrderBook("test_token", 0.001, True)
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

    assert bid_quants.tolist() == list_to_dec(orderbook.bids[1].tolist())
    assert ask_quants.tolist() == list_to_dec(orderbook.asks[1].tolist())


# noinspection DuplicatedCode
def test_orderbook_reset_raise_empty(unified_book_yes):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes
    orderbook = SharedOrderBook("test_token", 0.001, True)
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

    assert bid_quants.tolist() == list_to_dec(orderbook.bids[1].tolist())
    assert ask_quants.tolist() == list_to_dec(orderbook.asks[1].tolist())


# noinspection PyPropertyAccess
def test_orderbook_frozen_attributes(unified_book_yes):
    orderbook = SharedOrderBook("test_token", 0.001, True)

    with pytest.raises(AttributeError):
        orderbook.min_tick_size = 0.00001


# noinspection DuplicatedCode
def test_orderbook_set_decimal():
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

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

    assert orderbook.best_bid_price == Decimal("0.1")
    assert orderbook.best_bid_size == Decimal(10)
    assert orderbook.best_ask_price == Decimal("0.12")
    assert orderbook.best_ask_size == Decimal(1000)

    assert orderbook.tick_size == Decimal("0.01")


def test_tick_size_decimal():
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

    orderbook.tick_size = 0.001
    assert orderbook.tick_size == Decimal("0.001")

    orderbook.tick_size = 0.01
    assert orderbook.tick_size == Decimal("0.01")


# noinspection DuplicatedCode
def test_orderbook_reset_decimal():
    orderbook = SharedOrderBook("test_token_id", 0.01, True)

    assert orderbook.dtype is Decimal

    orderbook.set_bids([Decimal("0.2")], [Decimal(10)])
    orderbook.set_asks([Decimal("0.5")], [Decimal(10)])
    assert orderbook.bid_size(0.2) == Decimal(10)
    assert orderbook.ask_size(0.5) == Decimal(10)
    assert np.sum(orderbook.bids[1]) == dec(10)
    assert np.sum(orderbook.asks[1]) == dec(10)

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

    assert orderbook.best_bid_price == Decimal("0.1")
    assert orderbook.best_bid_size == Decimal(10)
    assert orderbook.best_ask_price == Decimal("0.12")
    assert orderbook.best_ask_size == Decimal(1000)


# noinspection DuplicatedCode
def test_orderbook_book_hash(
    book_t0_ws_msg_hashed, book_t1_rest_msg_hashed, book_t2_rest_msg_hashed
):
    asset_id = book_t0_ws_msg_hashed.asset_id
    market_id = book_t0_ws_msg_hashed.market
    orderbook = SharedOrderBook(asset_id, 0.01, True)

    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert book_t0_ws_msg_hashed.event_type == "book"
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed.timestamp)
        == book_t0_ws_msg_hashed.hash
    )

    # test updating
    orderbook = message_to_orderbook(book_t1_rest_msg_hashed, orderbook)
    assert book_t1_rest_msg_hashed.event_type == "book"
    assert (
        orderbook.hash(market_id, book_t1_rest_msg_hashed.timestamp)
        == book_t1_rest_msg_hashed.hash
    )

    orderbook = message_to_orderbook(book_t2_rest_msg_hashed, orderbook)
    assert book_t2_rest_msg_hashed.event_type == "book"
    assert (
        orderbook.hash(market_id, book_t2_rest_msg_hashed.timestamp)
        == book_t2_rest_msg_hashed.hash
    )


# noinspection DuplicatedCode
def test_orderbook_price_change_hash(
    book_t0_ws_msg_hashed,
    price_change_t1_ws_msg_hashed,
    book_t1_rest_msg_hashed,
    price_change_t2_ws_msg_hashed,
    book_t2_rest_msg_hashed,
):
    asset_id = book_t0_ws_msg_hashed.asset_id
    market_id = book_t0_ws_msg_hashed.market
    orderbook = SharedOrderBook(asset_id, 0.01, True)

    # check hash against initial book message
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert book_t0_ws_msg_hashed.event_type == "book"
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed.timestamp)
        == book_t0_ws_msg_hashed.hash
    )

    # check hash after updating first price change
    orderbook = message_to_orderbook(price_change_t1_ws_msg_hashed, orderbook)
    _, _, hash_t1 = guess_check_orderbook_hash(
        price_change_t1_ws_msg_hashed.price_changes[0].hash,
        orderbook,
        market_id,
        [int(price_change_t1_ws_msg_hashed.timestamp) - i for i in range(15)],
    )
    assert price_change_t1_ws_msg_hashed.event_type == "price_change"
    assert (
        price_change_t1_ws_msg_hashed.price_changes[0].hash
        == book_t1_rest_msg_hashed.hash
    )
    assert hash_t1 == price_change_t1_ws_msg_hashed.price_changes[0].hash

    # check hash after updating second price change
    orderbook = message_to_orderbook(price_change_t2_ws_msg_hashed, orderbook)
    _, _, hash_t2 = guess_check_orderbook_hash(
        price_change_t2_ws_msg_hashed.price_changes[0].hash,
        orderbook,
        market_id,
        [int(price_change_t2_ws_msg_hashed.timestamp) - i for i in range(15)],
    )
    assert price_change_t2_ws_msg_hashed.event_type == "price_change"
    assert (
        price_change_t2_ws_msg_hashed.price_changes[0].hash
        == book_t2_rest_msg_hashed.hash
    )
    assert (
        price_change_t2_ws_msg_hashed.price_changes[0].hash
        != book_t1_rest_msg_hashed.hash
    )
    assert hash_t2 == price_change_t2_ws_msg_hashed.price_changes[0].hash


def test_orderbook_marketable_amount_buy(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    orderbook = SharedOrderBook(asset_id, 0.001, True)
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    price, amount = orderbook.marketable_price("BUY", 135.26)
    assert math.isclose(price, dec(0.544))
    assert math.isclose(amount, dec(135.26))

    price, amount = orderbook.marketable_price("BUY", 135.27)
    assert math.isclose(price, dec(0.548))
    assert math.isclose(amount, dec(667.42828))


def test_orderbook_marketable_amount_sell(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    orderbook = SharedOrderBook(asset_id, 0.001, True)
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    price, amount = orderbook.marketable_price("SELL", 1917.82)
    assert math.isclose(price, dec(0.53))
    assert math.isclose(amount, dec(1917.82))

    price, amount = orderbook.marketable_price("SELL", 1917.83)
    assert math.isclose(price, dec(0.523))
    assert math.isclose(amount, dec(2048.57))


def test_orderbook_marketable_amount_raises_unknown_side(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    orderbook = SharedOrderBook(asset_id, 0.001, True)
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    with pytest.raises(OrderBookException) as e:
        # noinspection PyTypeChecker
        orderbook.marketable_price("SOME", 135.26)

    assert "Unknown side" in str(e)


def test_orderbook_marketable_amount_raises_unknown_max_liquidity(
    book_t0_ws_msg_hashed,
):
    asset_id = book_t0_ws_msg_hashed.asset_id
    orderbook = SharedOrderBook(asset_id, 0.001, True)
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    with pytest.raises(OrderBookException) as e:
        orderbook.marketable_price("BUY", 1e9)

    assert "No marketable price" in str(e)
    # noinspection PyUnresolvedReferences
    orderbook.cleanup()


def test_orderbook_midpoint_price(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    orderbook = SharedOrderBook(asset_id, 0.001, True)
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)

    assert orderbook.midpoint_price == dec(0.5365)


# noinspection DuplicatedCode
def test_guess_orderbook_hash_int_timestamps(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    market_id = book_t0_ws_msg_hashed.market
    timestamp = int(book_t0_ws_msg_hashed.timestamp)
    orderbook = SharedOrderBook(asset_id, 0.01, True)

    # check hash against initial book message
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed.timestamp)
        == book_t0_ws_msg_hashed.hash
    )

    assert (
        guess_check_orderbook_hash(
            book_t0_ws_msg_hashed.hash,
            orderbook,
            market_id,
            [timestamp - i for i in range(15)],
        )[0]
        is True
    )


# noinspection DuplicatedCode
def test_guess_orderbook_hash_str_timestamps(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    market_id = book_t0_ws_msg_hashed.market
    timestamp = int(book_t0_ws_msg_hashed.timestamp)
    orderbook = SharedOrderBook(asset_id, 0.01, True)

    # check hash against initial book message
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed.timestamp)
        == book_t0_ws_msg_hashed.hash
    )

    assert (
        guess_check_orderbook_hash(
            book_t0_ws_msg_hashed.hash,
            orderbook,
            market_id,
            [str(timestamp - i) for i in range(15)],
        )[0]
        is True
    )


def test_guess_orderbook_hash_false(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    market_id = book_t0_ws_msg_hashed.market
    orderbook = SharedOrderBook(asset_id, 0.01, True)

    # check hash against initial book message
    orderbook = message_to_orderbook(book_t0_ws_msg_hashed, orderbook)
    assert (
        orderbook.hash(market_id, book_t0_ws_msg_hashed.timestamp)
        == book_t0_ws_msg_hashed.hash
    )

    assert (
        guess_check_orderbook_hash(
            book_t0_ws_msg_hashed.hash, orderbook, market_id, [0, 1, 2, 3, 4, 5, 6]
        )[0]
        is False
    )


def test_orderbook_to_dict(book_t1_rest_msg_hashed):
    msg = msgspec.to_builtins(book_t1_rest_msg_hashed)
    msg["timestamp"] = str(msg["timestamp"])
    client_orderbook = parse_raw_orderbook_summary(msg)

    orderbook = message_to_orderbook(
        book_t1_rest_msg_hashed,
        SharedOrderBook(book_t1_rest_msg_hashed.asset_id, 0.01, True),
    )

    client_dict = client_orderbook.__dict__
    book_dict = orderbook.to_dict(
        book_t1_rest_msg_hashed.market,
        book_t1_rest_msg_hashed.timestamp,
        book_t1_rest_msg_hashed.hash,
    )

    assert client_dict == book_dict


def test_orderbook_to_dict_int_timestamp(book_t1_rest_msg_hashed):
    msg = msgspec.to_builtins(book_t1_rest_msg_hashed)
    msg["timestamp"] = str(msg["timestamp"])
    client_orderbook = parse_raw_orderbook_summary(msg)

    orderbook = message_to_orderbook(
        book_t1_rest_msg_hashed,
        SharedOrderBook(book_t1_rest_msg_hashed.asset_id, 0.01, True),
    )

    client_dict = client_orderbook.__dict__
    book_dict = orderbook.to_dict(
        book_t1_rest_msg_hashed.market,
        int(book_t1_rest_msg_hashed.timestamp),
        book_t1_rest_msg_hashed.hash,
    )

    assert client_dict == book_dict

    # noinspection PyUnresolvedReferences
    orderbook.cleanup()


def test_message_to_orderbook_empty_asks_book_msg(book_t0_ws_msg_hashed):
    book = SharedOrderBook(book_t0_ws_msg_hashed.asset_id, 0.01, True)

    book_t0_ws_msg_hashed.asks = []

    book = message_to_orderbook(book_t0_ws_msg_hashed, book)

    assert book_t0_ws_msg_hashed.event_type == "book"
    assert np.sum(book.asks[1]) == dec(0)
    assert np.sum(book.bids[1]) > dec(0)


# noinspection DuplicatedCode
def test_message_to_orderbook_empty_asks_price_change_msg(
    book_t0_ws_msg_hashed, price_change_t1_ws_msg_hashed
):
    book = SharedOrderBook(book_t0_ws_msg_hashed.asset_id, 0.01, True)
    book = message_to_orderbook(book_t0_ws_msg_hashed, book)

    assert book_t0_ws_msg_hashed.event_type == "book"
    assert np.sum(book.asks[1]) > dec(0)
    assert np.sum(book.bids[1]) > dec(0)

    bid_vol = np.sum(book.bids[1])
    ask_vol = np.sum(book.asks[1])

    price_change_t1_ws_msg_hashed.price_changes = [
        d for d in price_change_t1_ws_msg_hashed.price_changes if d.side == "BUY"
    ]
    book = message_to_orderbook(price_change_t1_ws_msg_hashed, book)

    assert price_change_t1_ws_msg_hashed.event_type == "price_change"
    assert np.sum(book.asks[1]) == dec(ask_vol)
    assert np.sum(book.bids[1]) != dec(bid_vol)
    assert np.sum(book.asks[1]) > dec(0)
    assert np.sum(book.bids[1]) > dec(0)


# noinspection DuplicatedCode
def test_message_to_orderbook_zeroing_asks_price_change_msg(
    book_t0_ws_msg_hashed, price_change_t1_ws_msg_hashed
):
    book = SharedOrderBook(book_t0_ws_msg_hashed.asset_id, 0.01, True)
    book = message_to_orderbook(book_t0_ws_msg_hashed, book)

    assert book_t0_ws_msg_hashed.event_type == "book"
    assert np.sum(book.asks[1]) > dec(0)
    assert np.sum(book.bids[1]) > dec(0)

    bid_vol = np.sum(book.bids[1])

    buy_changes = [
        d for d in price_change_t1_ws_msg_hashed.price_changes if d.side == "BUY"
    ]
    ask_changes = [
        PriceChangeSummary(
            "72936048731589292555781174533757608024096898681344338816447372274344589246891",
            str(i),
            "0",
            SIDE.SELL,
            "",
            "0.0",
            "0.0",
        )
        for i in book.ask_prices
    ]
    price_change_t1_ws_msg_hashed.price_changes = buy_changes + ask_changes

    book = message_to_orderbook(price_change_t1_ws_msg_hashed, book)

    assert price_change_t1_ws_msg_hashed.event_type == "price_change"
    assert np.sum(book.asks[1]) == dec(0)
    assert np.sum(book.bids[1]) != dec(bid_vol)
    assert np.sum(book.bids[1]) > dec(0)


def test_message_to_orderbook_raise_wrong_event_type(
    book_t0_ws_msg_hashed, price_change_t1_ws_msg_hashed
):
    msg = msgspec.to_builtins(book_t0_ws_msg_hashed)
    msg["timestamp"] = str(msg["timestamp"])
    msg["event_type"] = "price_change"
    book = SharedOrderBook(book_t0_ws_msg_hashed.asset_id, 0.01, True)
    with pytest.raises(EventTypeException) as e2:
        message_to_orderbook(msg, book)
    assert "Unknown" in str(e2)

    msg["event_type"] = "tick_size_change"
    with pytest.raises(EventTypeException) as e2:
        message_to_orderbook(msg, book)
    assert "Unknown" in str(e2)

    msg = msgspec.to_builtins(price_change_t1_ws_msg_hashed)
    msg["timestamp"] = str(msg["timestamp"])
    msg["event_type"] = "book"
    with pytest.raises(EventTypeException) as e2:
        message_to_orderbook(msg, book)
    assert "Unknown" in str(e2)

    book_new = SharedOrderBook(book_t0_ws_msg_hashed.asset_id, 0.01, False)
    message_to_orderbook(book_t0_ws_msg_hashed, book_new)

    book_new.cleanup()
    book.cleanup()


def test_message_to_orderbook_raise_specify_event_type(book_t1_rest_msg_hashed):
    # rest response without event_type: fail
    # rest response with event_type specified: success
    msg = msgspec.to_builtins(book_t1_rest_msg_hashed)
    del msg["event_type"]
    msg["timestamp"] = str(msg["timestamp"])

    book = SharedOrderBook(book_t1_rest_msg_hashed.asset_id, 0.01, True)
    with pytest.raises(EventTypeException) as e:
        message_to_orderbook(msg, book)
    assert "Unknown" in str(e)

    book = SharedOrderBook(book_t1_rest_msg_hashed.asset_id, 0.01, False)
    message_to_orderbook(book_t1_rest_msg_hashed, book)
    book.cleanup()


def test_message_to_orderbook_raise_unknown_event_type(book_t0_ws_msg_hashed):
    book = SharedOrderBook(book_t0_ws_msg_hashed.asset_id, 0.01, True)
    book = message_to_orderbook(book_t0_ws_msg_hashed, book)

    assert (
        book.hash(book_t0_ws_msg_hashed.market, book_t0_ws_msg_hashed.timestamp)
        == book_t0_ws_msg_hashed.hash
    )

    msg = msgspec.to_builtins(book_t0_ws_msg_hashed)
    msg["timestamp"] = str(msg["timestamp"])
    msg["event_type"] = "some"

    with pytest.raises(EventTypeException):
        message_to_orderbook(msg, book)


def test_message_to_orderbook_raise_asset_id(book_t0_ws_msg_hashed):
    with pytest.raises(OrderBookException):
        message_to_orderbook(
            book_t0_ws_msg_hashed, SharedOrderBook("test_token_id", 0.01, True)
        )


def test_message_to_orderbook_tick_size_change():
    book = SharedOrderBook("test_token_id", 0.01, True)

    msg = {
        "event_type": "tick_size_change",
        "asset_id": "test_token_id",
        "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
        "old_tick_size": "0.01",
        "new_tick_size": "0.001",
        "timestamp": "100000000",
    }
    msg = msgspec.convert(msg, TickSizeEvent, strict=False)

    book = message_to_orderbook(msg, book)

    assert msg.event_type == "tick_size_change"
    assert book.tick_size == dec(0.001)


def test_valid_message_id(book_t0_ws_msg_hashed):
    asset_id = book_t0_ws_msg_hashed.asset_id
    book = SharedOrderBook(asset_id, 0.01, True)

    # test True case
    message_to_orderbook(book_t0_ws_msg_hashed, book)

    # test False case
    msg = msgspec.structs.replace(book_t0_ws_msg_hashed, asset_id="test")
    assert msg.asset_id == "test"
    assert book_t0_ws_msg_hashed.asset_id != "test"

    with pytest.raises(OrderBookException):
        message_to_orderbook(msg, book)


def test_coerce_inbound_msg():
    book = SharedOrderBook(
        "103709219220485385381402953404611086294052546893918388630950161985342301458022",
        0.001,
        True,
        coerce_inbound_prices=True,
    )

    msg = {
        "market": "0x4c48d72bee2347ee0a0802850be8a7f36d318b405fb4bacbba455eb8a68dc6a3",
        "asset_id": "103709219220485385381402953404611086294052546893918388630950161985342301458022",
        "timestamp": "1745946996141",
        "hash": "4f4f191d88bf886c5f8c10786430528717591a18",
        "bids": [
            {"price": "0.001", "size": "200000"},
            {"price": "0.25", "size": "5000"},
            {"price": "0.4", "size": "2200"},
            {"price": "0.401", "size": "1496"},
            {"price": "0.81", "size": "222"},
            {"price": "0.868", "size": "1982.88"},
            {"price": "0.96", "size": "1173.95"},
            {"price": "0.974", "size": "2000"},
            {"price": "0.98", "size": "2540"},
            {"price": "0.985", "size": "270"},
            {"price": "0.988", "size": "1978.99"},
            {"price": "0.989", "size": "1000"},
            {"price": "0.99", "size": "10691.38"},
            {"price": "0.991", "size": "1154.28"},
            {"price": "0.992", "size": "851.14"},
        ],
        "asks": [
            {"price": "1.01", "size": "500"},
            {"price": "0.999", "size": "13072.46"},
            {"price": "0.997", "size": "4000"},
            {"price": "0.995", "size": "478.33"},
        ],
        "event_type": "book",
    }
    msg = msgspec.convert(msg, BookEvent, strict=False)

    book = message_to_orderbook(msg, book)

    assert book.ask_size(1) == dec(500)
    assert book.ask_size(Decimal("0.999")) == Decimal("13072.46")
    assert book.ask_size(Decimal("0.997")) == Decimal(4000)
    assert book.ask_size(Decimal("0.995")) == Decimal("478.33")

    assert book.bid_size(Decimal(".001")) == 200000
    assert book.bid_size(Decimal(".25")) == 5000
    assert book.bid_size(Decimal(".401")) == 1496
    assert book.bid_size(Decimal(".974")) == 2000
    assert book.bid_size(Decimal(".868")) == Decimal("1982.88")
    assert book.bid_size(Decimal(".988")) == Decimal("1978.99")
    assert book.bid_size(Decimal(".99")) == Decimal("10691.38")
    assert book.bid_size(Decimal(".992")) == Decimal("851.14")
