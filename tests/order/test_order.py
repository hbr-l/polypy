import copy
import itertools
import json
import math
import pathlib
import random
from decimal import Decimal
from functools import lru_cache
from typing import Any, Callable

import pytest
from freezegun import freeze_time
from py_clob_client.clob_types import MarketOrderArgs, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.order_builder.helpers import decimal_places, round_down, round_up

from polypy.book import OrderBook, dict_to_sha1
from polypy.constants import CHAIN_ID, SIG_DIGITS_SIZE
from polypy.exceptions import OrderCreationException, OrderUpdateException
from polypy.order.base import Order
from polypy.order.common import (
    INSERT_STATUS,
    SIDE,
    TIME_IN_FORCE,
    frozen_order,
    update_order,
)
from polypy.order.limit import create_limit_order
from polypy.order.market import create_market_order, is_marketable_amount
from polypy.rounding import max_allowed_decimals, round_ceil
from polypy.rounding import round_down as poly_round_down
from polypy.rounding import (
    round_down_tenuis_up,
    round_floor,
    round_floor_tenuis_ceil,
    round_half_even,
)
from polypy.rounding import round_up as poly_round_up
from polypy.rounding import scale_1e06
from polypy.signing import SIGNATURE_TYPE, polymarket_domain

test_pth = pathlib.Path(__file__).parent
rng_rand_float_dec = random.Random(42)


@lru_cache(maxsize=4)
def get_token_ids(book_pth: str | pathlib.Path) -> tuple[str, str, float]:
    with open(book_pth, "r") as f:
        data = json.load(f)
        token_id = data["book"]["asset_id"]
        complement_token_id = data["complement_asset_id"]
        tick_size = float(data["tick_size"])

        return token_id, complement_token_id, tick_size


@lru_cache(maxsize=4)
def get_book(
    book_pth: str | pathlib.Path,
) -> tuple[str, dict[str, Any], dict[str, Any], str, str]:
    with open(book_pth, "r") as f:
        data = json.load(f)
        book = data["book"]
        token_id = data["book"]["asset_id"]
        complement_token_id = data["complement_asset_id"]

    # invert book
    #   we could make use of polypy.order_book.OrderBook, but we'll try to depend
    #   as little as possible on self-written code to not introduce test result interdependencies (i.e.
    #   test_order.py would only be valid if test_orderbook.py is valid as well) and not to invalidate
    #   or corrupt test data
    q = Decimal(book["bids"][0]["price"])
    complement_book = copy.deepcopy(book)
    complement_book["bids"] = [
        {
            "price": str((1 - Decimal(sub_dict["price"])).quantize(q)),
            "size": sub_dict["size"],
        }
        for sub_dict in book["asks"]
    ]
    complement_book["asks"] = [
        {
            "price": str((1 - Decimal(sub_dict["price"])).quantize(q)),
            "size": sub_dict["size"],
        }
        for sub_dict in book["bids"]
    ]
    complement_book["asset_id"] = complement_token_id
    complement_book["hash"] = ""
    complement_book["hash"] = dict_to_sha1(complement_book)

    return (
        data["book"]["market"],
        book,
        complement_book,
        token_id,
        complement_token_id,
    )


def gen_rand_float_dec(
    n_pots: int,
    n_decimals: int,
    rnd_int_fn: Callable[[int, int], float] | None = rng_rand_float_dec.randint,
) -> float:
    if rnd_int_fn is None:
        rnd_int_fn = random.randint

    while True:
        ret_str = (
            f"{rnd_int_fn(10**(n_pots-1), 10**n_pots - 1)}."
            f"{rnd_int_fn(10**(n_decimals-1), 10**n_decimals - 1)}"
        )
        if not ret_str.endswith("0"):
            return float(ret_str)


def gen_rand_price(
    n_decimals: int,
    rnd_int_fn: Callable[[int, int], float] | None = rng_rand_float_dec.randint,
) -> float:
    if rnd_int_fn is None:
        rnd_int_fn = random.randint

    while True:
        ret_str = f"0.{rnd_int_fn(10**(n_decimals-1), 10**n_decimals - 1)}"
        if not ret_str.endswith("0"):
            return float(ret_str)


######################
# Test polypy package
######################


def test_gen_rand_float_dec():
    floats_generated = set()
    prices_generated = set()
    fixed_floats_generated = set()
    fixed_prices_generated = set()
    nb_pots, nb_decs, iters = 4, 12, 100

    for n_pot, n_dec, _ in itertools.product(
        range(1, nb_pots), range(1, nb_decs), range(iters)
    ):
        # gen_rand_float_dec uses its own RNG s.t. it is not affected by global seeding
        # this behaviour is tested in the very last assert len(floats_generated) > (nb_pots - 1) * (nb_decs - 1)
        random.seed(0)

        x = gen_rand_float_dec(n_pot, n_dec)
        # 10 < 10.01 | 99.99 < 100
        assert 10 ** (n_pot - 1) < x < 10**n_pot
        assert decimal_places(x) == n_dec
        x_str = str(x)
        x_pot, x_dec = x_str.split(".")
        assert len(x_pot) == n_pot
        assert len(x_dec) == n_dec

        floats_generated.add(x)

        # perform same tests for gen_rand_price
        random.seed(0)
        y = gen_rand_price(n_dec)
        assert y < 1
        assert decimal_places(y) == n_dec
        y_str = str(y)
        y_pot, y_dec = y_str.split(".")
        assert len(y_dec) == n_dec
        assert y_pot == "0"
        prices_generated.add(y)

        # test seed behavior
        random.seed(0)
        fixed_floats_generated.add(gen_rand_float_dec(n_pot, n_dec, None))
        random.seed(0)
        fixed_prices_generated.add(gen_rand_price(n_dec, None))

    # test gen_rand_float_dec own RNG behavior
    assert len(floats_generated) > (nb_pots - 1) * (nb_decs - 1)
    assert len(prices_generated) > (nb_decs - 1)
    assert len(fixed_floats_generated) <= (nb_pots - 1) * (nb_decs - 1)
    assert len(fixed_prices_generated) <= nb_decs - 1


def test_max_allowed_decimals():
    # sourcery skip: extract-duplicate-method
    # checks on 300_000 samples (3 * 10 * 10_000)

    for n_pot, n_dec, _ in itertools.product(range(1, 4), range(1, 11), range(10_000)):
        x = gen_rand_float_dec(n_pot, n_dec)

        for n_false in range(max(0, n_dec - 5), n_dec):
            assert not max_allowed_decimals(x, n_false)
            assert decimal_places(x) > n_false
        for n_true in range(n_dec, n_dec + 5):
            assert max_allowed_decimals(x, n_true)
            assert decimal_places(x) <= n_true

    # manual cases: 0.58 is in fact equal to 0.57999999999999996... due to floating point imprecision
    #   so we test 0.58 and 0.5799999999999999 as two separate, distinct but very close numbers
    #   to see if we can differentiate them nonetheless
    assert not max_allowed_decimals(0.5799999999999999, 2, 1e-16)
    assert max_allowed_decimals(0.5799999999999999, 16)
    assert decimal_places(0.5799999999999999) > 2
    assert decimal_places(0.5799999999999999) <= 16

    assert not max_allowed_decimals(0.5799999, 2)
    assert max_allowed_decimals(0.5799999, 12)
    assert decimal_places(0.5799999) > 2
    assert decimal_places(0.5799999) <= 12

    assert max_allowed_decimals(0.58, 2)
    assert max_allowed_decimals(0.58, 16)
    assert decimal_places(0.58) <= 2
    assert decimal_places(0.58) <= 16

    assert not max_allowed_decimals(5.1863393351, 9)
    assert max_allowed_decimals(5.1863393351, 10)
    assert decimal_places(5.1863393351) == 10

    assert not max_allowed_decimals(0.30000000000000004, 1)
    assert max_allowed_decimals(0.30000000000000004, 1, 1e-12)


def test_regression_max_allowed_decimals():
    # old versions of this function should raise assertion due to floating point imprecision
    def allowed_decimals(x_, ndecimal_digits):
        return x_ * 10**ndecimal_digits % 1 == 0

    def allowed_decimals_v2(x_, ndecimal_digits):
        x_round = round(x_ * 10**ndecimal_digits, 15)
        return False if x_round == 0 else x_round % 1 == 0

    assert not allowed_decimals(0.02345, 1)
    assert not allowed_decimals(0.02345, 2)
    assert not allowed_decimals(0.02345, 3)
    assert allowed_decimals(0.02345, 5)
    assert allowed_decimals(0.02345, 7)
    assert allowed_decimals(0.02345, 8)

    assert not allowed_decimals_v2(0.02345, 1)
    assert not allowed_decimals_v2(0.02345, 2)
    assert not allowed_decimals_v2(0.02345, 3)
    assert allowed_decimals_v2(0.02345, 5)
    assert allowed_decimals_v2(0.02345, 7)
    assert allowed_decimals_v2(0.02345, 8)

    # test for regression
    with pytest.raises(AssertionError):
        for n_pot, n_dec, _ in itertools.product(
            range(1, 4), range(1, 11), range(50_000)
        ):
            x = gen_rand_float_dec(n_pot, n_dec)

            for n_false in range(max(0, n_dec - 5), n_dec):
                assert not allowed_decimals(x, n_false)
            for n_true in range(n_dec, n_dec + 5):
                assert allowed_decimals(x, n_true)

    with pytest.raises(AssertionError):
        for n_pot, n_dec, _ in itertools.product(
            range(1, 4), range(1, 11), range(50_000)
        ):
            x = gen_rand_float_dec(n_pot, n_dec)

            for n_false in range(max(0, n_dec - 5), n_dec):
                assert not allowed_decimals_v2(x, n_false)
            for n_true in range(n_dec, n_dec + 5):
                assert allowed_decimals_v2(x, n_true)


def round_clob(x_, sig_digits, extra_digits=4):
    if decimal_places(x_) > sig_digits:
        x_ = round_up(x_, sig_digits + extra_digits)
        if decimal_places(x_) > sig_digits:
            x_ = round_down(x_, sig_digits)

    return x_


def test_round_floor_tenuis_ceil(patch_py_clob_client_rounding):
    xs = [
        0.58,
        0.5799999,
        82.82,
        0.12399999,
        0.06134969325153374,
        0.1299999,
        5.1863393351,
    ]
    for x, ndigits in itertools.product(xs, range(1, 10)):
        assert round_clob(x, ndigits) == round_floor_tenuis_ceil(x, ndigits, 4)
        assert (decimal_places(x) <= ndigits) == max_allowed_decimals(x, ndigits)

    for n_pot, ndigits, _ in itertools.product(
        range(1, 4), range(1, 11), range(50_000)
    ):
        x = gen_rand_float_dec(n_pot, ndigits)

        assert (decimal_places(x) <= ndigits) == max_allowed_decimals(x, ndigits)
        assert round_clob(x, ndigits) == round_floor_tenuis_ceil(x, ndigits, 4)
        assert round_clob(x, ndigits - 1) == round_floor_tenuis_ceil(x, ndigits - 1, 4)


def test_round_down_tenuis_up(patch_py_clob_client_rounding):
    xs = [
        0.58,
        0.5799999,
        82.82,
        0.12399999,
        0.06134969325153374,
        0.1299999,
        5.1863393351,
    ]
    for x, ndigits in itertools.product(xs, range(1, 10)):
        assert round_clob(x, ndigits) == round_down_tenuis_up(x, ndigits, 4)
        assert (decimal_places(x) <= ndigits) == max_allowed_decimals(x, ndigits)

    for n_pot, ndigits, _ in itertools.product(
        range(1, 4), range(1, 11), range(50_000)
    ):
        x = gen_rand_float_dec(n_pot, ndigits)

        assert (decimal_places(x) <= ndigits) == max_allowed_decimals(x, ndigits)
        assert round_clob(x, ndigits) == round_down_tenuis_up(x, ndigits, 4)
        assert round_clob(x, ndigits - 1) == round_down_tenuis_up(x, ndigits - 1, 4)


def test_regression_round_floor_down_ceil_up():
    a = 5 - 7.4
    b = 7.4 - 5

    assert round_floor(a, 5) != poly_round_down(a, 5)
    assert round_floor(b, 5) == poly_round_down(b, 5)

    assert round_ceil(a, 5) != poly_round_up(a, 5)
    assert round_ceil(b, 5) == poly_round_up(b, 5)

    for n_pot, ndigits, _ in itertools.product(range(1, 2), range(1, 8), range(50_000)):
        x = gen_rand_float_dec(n_pot, ndigits)

        assert round_floor(x, ndigits - 1) == round_down(x, ndigits - 1)
        assert round_ceil(x, ndigits - 1) == round_up(x, ndigits - 1)
        assert round_floor(x, ndigits - 1) == -round_up(-x, ndigits - 1)
        assert round_ceil(x, ndigits - 1) == -round_down(-x, ndigits - 1)


# noinspection DuplicatedCode
@freeze_time("2024-12-19 17:05:55")
@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_limit_order_buy_against_clob(
    clob_client_factory,
    pth,
    patch_py_clob_client_rounding,
    local_host_addr,
    private_key,
    fix_seed,
):
    token_id, _, tick_size = get_token_ids(pth)

    sig_digits = -int(math.log10(tick_size))
    for n_pot, n_dec, i, neg_risk in itertools.product(
        range(1, 4), range(1, SIG_DIGITS_SIZE + 1), range(25), [False, True]
    ):
        price = round_half_even(gen_rand_price(n_dec), min(sig_digits, n_dec))
        size = round_floor(
            gen_rand_float_dec(n_pot, n_dec), min(SIG_DIGITS_SIZE, n_dec)
        )

        polypy_order = fix_seed(create_limit_order)(
            price,
            size,
            token_id,
            SIDE.BUY,
            tick_size,
            neg_risk,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )

        pyclob_order = fix_seed(
            clob_client_factory(
                tick_size,
                neg_risk,
                *get_book(pth),
                local_host_addr,
                private_key,
            ).create_order
        )(OrderArgs(token_id, price, size, BUY))

        assert polypy_order.to_dict() == pyclob_order.dict()


# noinspection DuplicatedCode
@freeze_time("2024-12-19 17:05:55")
@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_limit_order_sell_against_clob(
    clob_client_factory,
    pth,
    patch_py_clob_client_rounding,
    local_host_addr,
    private_key,
    fix_seed,
):
    token_id, _, tick_size = get_token_ids(pth)

    sig_digits = -int(math.log10(tick_size))
    for n_pot, n_dec, i, neg_risk in itertools.product(
        range(1, 4), range(1, SIG_DIGITS_SIZE + 1), range(25), [False, True]
    ):
        price = round_half_even(gen_rand_price(n_dec), min(sig_digits, n_dec))
        size = round_floor(
            gen_rand_float_dec(n_pot, n_dec), min(SIG_DIGITS_SIZE, n_dec)
        )

        polypy_order = fix_seed(create_limit_order)(
            price,
            size,
            token_id,
            SIDE.SELL,
            tick_size,
            neg_risk,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )

        pyclob_order = fix_seed(
            clob_client_factory(
                tick_size, neg_risk, *get_book(pth), local_host_addr, private_key
            ).create_order
        )(OrderArgs(token_id, price, size, SELL))

        assert polypy_order.to_dict() == pyclob_order.dict()


@freeze_time("2024-12-19 17:05:55")
@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_market_buy_against_clob(
    clob_client_factory,
    pth,
    patch_py_clob_client_rounding,
    local_host_addr,
    private_key,
    fix_seed,
):
    token_id, _, tick_size = get_token_ids(pth)

    for n_pot, n_dec, i, neg_risk in itertools.product(
        range(1, 4), range(1, SIG_DIGITS_SIZE + 1), range(25), [False, True]
    ):
        amount = round_floor(gen_rand_float_dec(n_pot, n_dec), SIG_DIGITS_SIZE)

        polypy_order = fix_seed(create_market_order)(
            amount,
            token_id,
            SIDE.BUY,
            tick_size,
            neg_risk,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )

        client = clob_client_factory(
            tick_size, neg_risk, *get_book(pth), local_host_addr, private_key
        )
        pyclob_order = fix_seed(client.create_market_order)(
            MarketOrderArgs(token_id, amount)
        )

        assert polypy_order.to_dict() == pyclob_order.dict()

    # test against regression bug when amount is int
    polypy_order = fix_seed(create_market_order)(
        100,
        token_id,
        SIDE.BUY,
        tick_size,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    client = clob_client_factory(
        tick_size, False, *get_book(pth), local_host_addr, private_key
    )
    pyclob_order = fix_seed(client.create_market_order)(MarketOrderArgs(token_id, 100))
    assert polypy_order.to_dict() == pyclob_order.dict()


def test_market_order_size_open(private_key):
    order = create_market_order(
        100,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    order.size_matched = 0.5

    assert order.numeric_type == float
    # amount = price * size -> size = amount / price
    assert order.size == round_floor(100 / 0.99, 4)
    assert order.size_open == order.size - 0.5


def test_limit_order_size_open(private_key):
    order = create_limit_order(
        0.5,
        100,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    order.size_matched = 0.5

    assert order.numeric_type == float
    assert order.size_open == 99.5


@freeze_time("2024-12-19 17:05:55")
@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_market_buy_quasi_equiv_deep_limit_buy(
    clob_client_factory, pth, private_key, fix_seed
):
    # Buy Order:
    #   makerAmount: total amount spent
    #   takerAmount: minimum size received
    # Sell Order:
    #   makerAmount: total size spent
    #   takerAmount: minimum amount received
    # Key takeaway:
    #   makerAmount: defines your spending limit or the number of tokens you're offering.
    #   takerAmount: defines the minimum price you'll accept for buying or selling tokens.
    # price = amount / size

    token_id, _, tick_size = get_token_ids(pth)

    sig_digits = -int(math.log10(tick_size))
    for n_pot, n_dec, i in itertools.product(
        range(1, 4), range(1, SIG_DIGITS_SIZE + 1), range(25)
    ):
        amount = round_floor(
            gen_rand_float_dec(n_pot, n_dec), min(n_dec, SIG_DIGITS_SIZE)
        )
        price = 1 - tick_size  # buy at the highest price to be marketable
        size = amount / price

        polypy_limit_buy = fix_seed(create_limit_order)(
            price,
            size,
            token_id,
            SIDE.BUY,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )
        polypy_market_buy = fix_seed(create_market_order)(
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

        # limit buy: exact size, rounding error in amount (taker vs maker)
        # market buy: exact in amount, rounding error in size  (maker vs taker)
        assert polypy_limit_buy.taker_amount == scale_1e06(
            round_floor(size, SIG_DIGITS_SIZE)
        )  # limit buy exact size
        assert polypy_limit_buy.maker_amount <= scale_1e06(
            round_floor_tenuis_ceil(amount, SIG_DIGITS_SIZE + sig_digits, 4)
        )  # limit buy amount not more spent than specified amount
        assert polypy_market_buy.maker_amount == scale_1e06(
            round_floor(amount, SIG_DIGITS_SIZE)
        )  # market buy exact amount
        assert polypy_market_buy.taker_amount >= scale_1e06(
            round_floor_tenuis_ceil(size, SIG_DIGITS_SIZE + sig_digits, 4)
        )  # market buy at least buy size

        # limit order spends less equal to the amount (prioritize exact size in limit order)
        assert polypy_limit_buy.maker_amount <= polypy_market_buy.maker_amount
        # # market order might receive more equal to the size (best size in market order)
        assert polypy_limit_buy.taker_amount <= polypy_market_buy.taker_amount

        # price can be approximated by makerAmount/takerAmount
        assert math.isclose(
            round_half_even(
                polypy_limit_buy.maker_amount / polypy_limit_buy.taker_amount,
                sig_digits,
            ),
            price,
        )
        assert math.isclose(
            round_half_even(
                polypy_market_buy.maker_amount / polypy_market_buy.taker_amount,
                sig_digits,
            ),
            price,
        )


@freeze_time("2024-12-19 17:05:55")
@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_market_sell_quasi_equiv_deep_limit_sell(pth, private_key):
    # py_clob_client does not have market sell order,
    # so to test market sell order, we  compare against limit sell order
    token_id, complement_token_id, tick_size = get_token_ids(pth)

    sig_digits = -int(math.log10(tick_size))

    for n_pot, n_dec, _ in itertools.product(
        range(1, 4), range(1, SIG_DIGITS_SIZE + 1), range(25)
    ):
        amount = round_floor(
            gen_rand_float_dec(n_pot, n_dec), min(SIG_DIGITS_SIZE, n_dec)
        )
        price = tick_size  # sell at the lowest price to be marketable
        size = round_half_even(amount / price, sig_digits)

        market_sell = create_market_order(
            amount,
            token_id,
            SIDE.SELL,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            max_size=size * 1.2,
        )

        limit_sell = create_limit_order(
            price,
            size,
            token_id,
            SIDE.SELL,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )

        # Sell Order:
        #   makerAmount: total size spent
        #   takerAmount: minimum amount received
        # spent token, receive cash

        # limit sell: exact size, rounding error in amount (maker vs taker)
        # market sell: exact in amount, rounding error in size  (taker vs maker)
        assert limit_sell.maker_amount == scale_1e06(
            round_floor(size, SIG_DIGITS_SIZE)
        )  # limit sell exact size
        assert limit_sell.taker_amount >= scale_1e06(
            round_floor_tenuis_ceil(amount, SIG_DIGITS_SIZE + sig_digits, 4)
        )  # limit receives at least or more than specified amount
        assert market_sell.taker_amount == scale_1e06(
            round_floor_tenuis_ceil(amount, SIG_DIGITS_SIZE + sig_digits, 4)
        )  # market sell exact amount
        assert market_sell.maker_amount <= scale_1e06(
            round_floor(size, SIG_DIGITS_SIZE)
        )  # market sell less or equal than size

        # limit sell receives at least or more the specified amount
        assert limit_sell.taker_amount >= market_sell.taker_amount
        # market sell spends less equal to available size
        assert limit_sell.maker_amount >= market_sell.maker_amount

        # price can be approximated by takerAmount/makerAmount
        assert math.isclose(
            round_half_even(
                limit_sell.taker_amount / limit_sell.maker_amount, sig_digits
            ),
            price,
        )
        assert math.isclose(
            round_half_even(
                market_sell.taker_amount / market_sell.maker_amount,
                sig_digits,
            ),
            price,
        )


@freeze_time("2024-12-19 17:05:55")
@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_market_sell_vs_clob_market_buy_complement(
    clob_client_factory,
    pth,
    patch_py_clob_client_rounding,
    local_host_addr,
    private_key,
    fix_seed,
):
    # py_clob_client does not have market sell order,
    # so to test market sell order, we have compare against market buy of complement token
    token_id, complement_token_id, tick_size = get_token_ids(pth)

    sig_digits = -int(math.log10(tick_size))

    for n_pot, n_dec, _ in itertools.product(
        range(1, 4), range(1, SIG_DIGITS_SIZE + 1), range(25)
    ):
        amount = round_floor(
            gen_rand_float_dec(n_pot, n_dec), min(SIG_DIGITS_SIZE, n_dec)
        )
        price = tick_size  # sell at the lowest price to be marketable
        size = round_half_even(amount / price, sig_digits)

        sell_order = fix_seed(create_market_order)(
            amount,
            token_id,
            SIDE.SELL,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            max_size=size * 1.2,
        )

        buy_order = fix_seed(
            clob_client_factory(
                tick_size, False, *get_book(pth), local_host_addr, private_key
            ).create_market_order
        )(
            MarketOrderArgs(
                token_id=complement_token_id, amount=amount, price=1 - tick_size
            )
        )

        buy_dict = buy_order.dict()

        # sell taker amount -> USDC
        # buy maker amount -> USDC
        assert sell_order.taker_amount == int(buy_dict["makerAmount"])

        # we sell at tick size and buy at 1 - tick size, which means a ratio of
        # (1 - tick size) / tick size of shares when orders have same amount
        ratio = (1 - tick_size) / tick_size
        # due to rounding within order creation, there will be a small offset, which
        # is quite tedious to determine analytical, so we just compare to an error threshold (0.01 %)
        assert (
            abs(1 - sell_order.maker_amount / (int(buy_dict["takerAmount"]) * ratio))
            < 0.0001
        )

        # because size will be round, we estimate against the two most extreme cases:
        # 1) round up denominator, round down numerator
        # 2) vice versa
        # the absolut max floor rounding error is 9.9999... * 10**-(nb_digits+1) ~ 10**-nb_digits,
        # which we add or subtract to the denominator and numerator
        eps = scale_1e06(10 ** -(SIG_DIGITS_SIZE + sig_digits))
        lower = (sell_order.maker_amount - eps) / (int(buy_dict["takerAmount"]) + eps)
        upper = (sell_order.maker_amount + eps) / (int(buy_dict["takerAmount"]) - eps)
        assert lower < ratio < upper

        # price can be approximated by takerAmount/makerAmount
        assert math.isclose(
            round_half_even(
                sell_order.taker_amount / sell_order.maker_amount, sig_digits
            ),
            price,
        )
        # complement price
        assert math.isclose(
            round_half_even(
                int(buy_dict["makerAmount"]) / int(buy_dict["takerAmount"]),
                sig_digits,
            ),
            1 - price,
        )


def test_market_sell_raise_overspending(private_key):
    # we sell at tick size:
    # size = amount / tick size
    with pytest.raises(OrderCreationException):
        # size = 100 / 0.01 = 10_000
        # undersize sell order to provoke exception
        create_market_order(
            100,
            "1234",
            SIDE.SELL,
            0.01,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            max_size=10_000 - SIG_DIGITS_SIZE,
        )


@freeze_time("2024-12-19 17:05:55")
def test_limit_order_decimals(private_key, fix_seed):
    buy = fix_seed(create_limit_order)(
        0.2,
        20,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    decimal_buy = fix_seed(create_limit_order)(
        Decimal("0.2"),
        Decimal("20"),
        "1234",
        SIDE.BUY,
        Decimal("0.01"),
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    assert buy.to_dict() == decimal_buy.to_dict()
    assert isinstance(decimal_buy.price, Decimal)
    assert isinstance(decimal_buy.size, Decimal)
    assert isinstance(decimal_buy.amount, Decimal)
    assert isinstance(decimal_buy.size_open, Decimal)
    assert decimal_buy.size_open == decimal_buy.size

    sell = fix_seed(create_limit_order)(
        0.6,
        84,
        "1234",
        SIDE.SELL,
        0.001,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    decimal_sell = fix_seed(create_limit_order)(
        Decimal("0.6"),
        Decimal("84"),
        "1234",
        SIDE.SELL,
        Decimal("0.001"),
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        size_matched=Decimal(20),
        price_matched=Decimal(0.5),
    )

    assert sell.to_dict() == decimal_sell.to_dict()
    assert isinstance(decimal_sell.price, Decimal)
    assert isinstance(decimal_sell.size, Decimal)
    assert isinstance(decimal_sell.amount, Decimal)
    assert isinstance(decimal_sell.size_open, Decimal)
    assert decimal_sell.size_open == Decimal(64)


def test_market_order_decimals(private_key, fix_seed):
    # sourcery skip: extract-duplicate-method
    buy = fix_seed(create_market_order)(
        22.5,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    decimal_buy = fix_seed(create_market_order)(
        Decimal("22.5"),
        "1234",
        SIDE.BUY,
        Decimal("0.01"),
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    assert buy.to_dict() == decimal_buy.to_dict()
    assert isinstance(decimal_buy.price, Decimal)
    assert isinstance(decimal_buy.size, Decimal)
    assert isinstance(decimal_buy.amount, Decimal)
    assert isinstance(decimal_buy.size_open, Decimal)
    assert decimal_buy.size_open == decimal_buy.size

    sell = fix_seed(create_market_order)(
        84.3,
        "1234",
        SIDE.SELL,
        0.001,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        max_size=math.inf,
    )

    decimal_sell = fix_seed(create_market_order)(
        Decimal("84.3"),
        "1234",
        SIDE.SELL,
        Decimal("0.001"),
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        max_size=math.inf,
    )

    assert sell.to_dict() == decimal_sell.to_dict()
    assert isinstance(decimal_sell.price, Decimal)
    assert isinstance(decimal_sell.size, Decimal)
    assert isinstance(decimal_sell.amount, Decimal)
    assert isinstance(decimal_sell.size_open, Decimal)
    assert decimal_sell.size_open == decimal_sell.size


@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_marketable(pth, private_key):
    book = OrderBook.from_dict(get_book(pth)[1])
    _, _, tick_size = get_token_ids(pth)

    # buy at ask, sell at bid
    total_buy_amount = sum(book.ask_prices * book.ask_sizes)
    total_sell_amount = sum(book.bid_prices * book.bid_sizes)

    buy_order = create_market_order(
        total_buy_amount * 0.99,
        book.token_id,
        SIDE.BUY,
        tick_size,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        book,
    )
    sell_order = create_market_order(
        total_sell_amount * 0.99,
        book.token_id,
        SIDE.SELL,
        tick_size,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        book,
        max_size=math.inf,
    )

    assert is_marketable_amount(book, buy_order.side, buy_order.amount)
    assert is_marketable_amount(book, sell_order.side, sell_order.amount)

    buy_order = create_market_order(
        total_buy_amount * 1.005,
        book.token_id,
        SIDE.BUY,
        tick_size,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    sell_order = create_market_order(
        total_sell_amount * 1.005,
        book.token_id,
        SIDE.SELL,
        tick_size,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        max_size=math.inf,
    )

    assert not is_marketable_amount(book, buy_order.side, buy_order.amount)
    assert not is_marketable_amount(book, sell_order.side, sell_order.amount)


@pytest.mark.parametrize(
    "pth",
    [test_pth / "data/book_data_a.txt", test_pth / "data/book_data_b.txt"],
)
def test_marketable_raise_not_enough_liquidity(pth, private_key):
    book = OrderBook.from_dict(get_book(pth)[1])
    _, _, tick_size = get_token_ids(pth)

    total_buy_amount = sum(book.ask_prices * book.ask_sizes)
    total_sell_amount = sum(book.bid_prices * book.bid_sizes)

    with pytest.raises(OrderCreationException):
        _ = create_market_order(
            total_buy_amount * 1.01,
            "1234",
            SIDE.BUY,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            book,
        )
    with pytest.raises(OrderCreationException) as e:
        _ = create_market_order(
            total_sell_amount * 1.01,
            "1234",
            SIDE.SELL,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            book,
            max_size=math.inf,
        )
        assert "max order size" not in str(e)


@pytest.mark.parametrize(
    "price, side",
    [
        (1, SIDE.BUY),
        (1, SIDE.SELL),
        (0, SIDE.BUY),
        (0, SIDE.SELL),
        (-0.1, SIDE.BUY),
        (-0.1, SIDE.SELL),
        (1.1, SIDE.BUY),
        (1.1, SIDE.SELL),
    ],
)
def test_order_raise_valid_price(price, side, private_key):
    with pytest.raises(OrderCreationException):
        _ = create_market_order(
            10,
            "1234",
            side,
            0.01,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            price=price,
        )

    with pytest.raises(OrderCreationException):
        _ = create_limit_order(
            price,
            3,
            "1234",
            side,
            0.01,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )


@pytest.mark.parametrize(
    "tick_size, side",
    [
        (1, SIDE.BUY),
        (1, SIDE.SELL),
        (0.2, SIDE.BUY),
        (0.2, SIDE.SELL),
        (-0.1, SIDE.BUY),
        (-0.1, SIDE.SELL),
    ],
)
def test_order_raise_tick_size(tick_size, side, private_key):
    with pytest.raises(OrderCreationException):
        create_market_order(
            10,
            "1234",
            side,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )

    with pytest.raises(OrderCreationException):
        create_limit_order(
            0.3,
            3,
            "1234",
            side,
            tick_size,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
        )


def test_limit_order_price_size_auto_rounding(private_key):
    limit_order = create_limit_order(
        0.231,
        2.345,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    assert limit_order.price == 0.23
    assert limit_order.size == 2.34


def test_market_order_amount_auto_rounding(private_key):
    market_order = create_market_order(
        21.345,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    assert market_order.amount == 21.34


def test_order_optional_frozen(private_key):
    order = create_market_order(
        21.345,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    assert order.id is None
    assert order.signature is not None
    assert order.created_at is None

    order.id = 4321
    assert order.id == "4321"

    order.created_at = 1234
    assert order.created_at == 1234

    with pytest.raises(OrderUpdateException):
        order.signature = "0"

    with pytest.raises(OrderUpdateException):
        order.created_at = 0

    with pytest.raises(OrderUpdateException):
        order.id = "0"


def test_frozen_order(private_key):
    order = create_market_order(
        21.345,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )

    assert order.strategy_id is None

    order.strategy_id = "some"
    order.strategy_id = "strategy"
    assert order.strategy_id == "strategy"

    frozen = frozen_order(order)

    with pytest.raises(OrderUpdateException):
        frozen.strategy_id = "some"

    with pytest.raises(OrderUpdateException):
        frozen.created_at = 1234

    with pytest.raises(OrderUpdateException):
        frozen.tif = TIME_IN_FORCE.GTD

    order.created_at = 1234
    assert order.created_at == 1234

    with pytest.raises(OrderUpdateException):
        order.created_at = 2468

    attr_names = [
        "token_id",
        "signature",
        "side",
        "price",
        "id",
        "tif",
        "status",
        "size",
        "salt",
    ]
    for attr in attr_names:
        assert getattr(frozen, attr) == getattr(order, attr)


def test_price_size_amount(private_key):
    order = create_limit_order(
        0.42,
        2.4,
        "123",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    assert order.price == 0.42
    assert order.size == 2.4
    assert order.amount == 1.008

    order = create_limit_order(
        0.42,
        2.4,
        "123",
        SIDE.SELL,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    assert order.price == 0.42
    assert order.size == 2.4
    assert order.amount == 1.008

    order = create_market_order(
        0.42,
        "234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    assert order.amount == 0.42
    assert round_half_even(order.price, 2) == 0.99
    assert round_floor(order.size, 2) == 0.42

    order = create_market_order(
        0.42,
        "234",
        SIDE.SELL,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        max_size=math.inf,
    )
    assert order.amount == 0.42
    assert order.price == 0.01
    assert order.size == 42


def test_update_order(private_key):  # sourcery skip: extract-duplicate-method
    order = create_limit_order(
        0.42,
        2.4,
        "123",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    assert order.created_at is None

    update_order(order, INSERT_STATUS.LIVE, created_at=1200)
    assert order.status is INSERT_STATUS.LIVE
    assert order.created_at == 1200
    assert order.strategy_id is None
    assert order.size_matched == 0
    assert order.price_matched is None

    update_order(order, created_at=4000)
    assert order.created_at == 1200
    assert order.strategy_id is None
    assert order.size_matched == 0
    assert order.price_matched is None


def test_numeric_type_create(private_key):
    order = Order.create(
        "123",
        SIDE.BUY,
        1000,
        1000,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        polymarket_domain(CHAIN_ID.POLYGON, False),
        TIME_IN_FORCE.GTC,
        size_matched=Decimal(0),
        price_matched=None,
        numeric_type=Decimal,
    )
    assert order.side is SIDE.BUY
    assert order.price == Decimal(1)
    assert order.size_matched == Decimal(0)
    assert order.price_matched is None

    with pytest.raises(OrderCreationException) as record:
        Order.create(
            "123",
            SIDE.BUY,
            1000,
            1000,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            polymarket_domain(CHAIN_ID.POLYGON, False),
            TIME_IN_FORCE.GTC,
            size_matched=10,
            price_matched=Decimal(0.5),
            numeric_type=Decimal,
        )
    assert "type" in str(record)
    assert "size_matched" in str(record)

    with pytest.raises(OrderCreationException) as record:
        Order.create(
            "123",
            SIDE.BUY,
            1000,
            1000,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            polymarket_domain(CHAIN_ID.POLYGON, False),
            TIME_IN_FORCE.GTC,
            size_matched=Decimal(10),
            price_matched=10,
            numeric_type=Decimal,
        )
    assert "type" in str(record)
    assert "price_matched" in str(record)


def test_create_size_matched_price_matched(private_key):
    # sourcery skip: extract-duplicate-method
    order = Order.create(
        "123",
        SIDE.BUY,
        1000,
        1000,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        polymarket_domain(CHAIN_ID.POLYGON, False),
        TIME_IN_FORCE.GTC,
        size_matched=Decimal(10),
        price_matched=Decimal(10),
        numeric_type=Decimal,
    )
    assert order.side is SIDE.BUY
    assert order.price == Decimal(1)
    assert order.size_matched == Decimal(10)
    assert order.price_matched == Decimal(10)

    order = Order.create(
        "123",
        SIDE.BUY,
        1000,
        1000,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
        polymarket_domain(CHAIN_ID.POLYGON, False),
        TIME_IN_FORCE.GTC,
        size_matched=None,
        price_matched=None,
        numeric_type=Decimal,
    )
    assert order.side is SIDE.BUY
    assert order.price == Decimal(1)
    assert order.size_matched == Decimal(0)
    assert order.price_matched is None

    with pytest.raises(OrderCreationException) as record:
        Order.create(
            "123",
            SIDE.BUY,
            1000,
            1000,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            polymarket_domain(CHAIN_ID.POLYGON, False),
            TIME_IN_FORCE.GTC,
            size_matched=Decimal(10),
            price_matched=None,
            numeric_type=Decimal,
        )
    assert "simultaneously" in str(record)

    with pytest.raises(OrderCreationException) as record:
        Order.create(
            "123",
            SIDE.BUY,
            1000,
            1000,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            polymarket_domain(CHAIN_ID.POLYGON, False),
            TIME_IN_FORCE.GTC,
            size_matched=None,
            price_matched=Decimal(10),
            numeric_type=Decimal,
        )
    assert "simultaneously" in str(record)
