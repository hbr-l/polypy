import json
import pathlib
from decimal import Decimal

import numpy as np
import pytest

from polypy.exceptions import RewardException
from polypy.reward import (
    _market_side_score,
    estimate_sampling_reward,
    filter_qualifying_orders,
    insert_merge,
    size_cutoff_adj_midpoint,
)

test_pth = pathlib.Path(__file__).parent


@pytest.fixture
def market_data():
    with open(test_pth / "data/ws_msg_market_data.json", "r") as f:
        return json.load(f)


@pytest.fixture
def unified_book_yes(json_book_to_arrays):
    return json_book_to_arrays(test_pth / "data/ws_msg_book_yes.json")


@pytest.mark.skip(reason="only for visualization purpose")
def test_visualize(unified_book_yes, market_data):
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(
        unified_book_yes[0], np.cumsum(unified_book_yes[1][::-1])[::-1], color="green"
    )
    plt.plot(
        unified_book_yes[2][::-1],
        np.cumsum(unified_book_yes[3][::-1]),
        color="red",
    )
    plt.show()


def test_midpoint(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    midpoint = size_cutoff_adj_midpoint(
        bid_p,
        bid_q,
        ask_p,
        ask_q,
        market_data["rewards"]["min_size"],
        int(-np.log10(market_data["minimum_tick_size"])),
    )

    assert bid_p[-1] <= midpoint <= ask_p[-1]
    assert midpoint == round(0.2345, 3)


def test_midpoint_fallback_best_quote(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    # force to take best bid instead of size cut-off
    bid_q[:-1] = 0

    with pytest.warns(UserWarning):
        midpoint = size_cutoff_adj_midpoint(
            bid_p,
            bid_q,
            ask_p,
            ask_q,
            market_data["rewards"]["min_size"],
            int(-np.log10(market_data["minimum_tick_size"])),
        )

    assert midpoint == 0.235


# noinspection DuplicatedCode
def test_midpoint_decimal(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    bid_p = np.array([Decimal(str(x)) for x in bid_p])
    bid_q = np.array([Decimal(x) for x in bid_q])
    ask_p = np.array([Decimal(str(x)) for x in ask_p])
    ask_q = np.array([Decimal(x) for x in ask_q])

    midpoint = size_cutoff_adj_midpoint(
        bid_p,
        bid_q,
        ask_p,
        ask_q,
        market_data["rewards"]["min_size"],
        int(-np.log10(market_data["minimum_tick_size"])),
    )

    assert bid_p[-1] <= midpoint <= ask_p[-1]
    assert midpoint == round(Decimal("0.2345"), 3)
    assert isinstance(midpoint, Decimal)


def test_insert_merge():
    book_p = [0.1, 0.3, 0.4, 0.5, 0.6]
    book_q = [1, 3, 4, 5, 6]

    p = [0.2, 0.4]
    q = [2, 4]

    new_p, new_q = insert_merge(p, q, book_p, book_q, False)

    assert new_p.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert new_q.tolist() == [1, 2, 3, 8, 5, 6]


def test_insert_merge_reverse_a():
    # input reversed as well
    book_p = [0.1, 0.3, 0.4, 0.5, 0.6][::-1]
    book_q = [1, 3, 4, 5, 6][::-1]

    p = [0.2, 0.4]
    q = [2, 4]

    # noinspection DuplicatedCode
    new_p, new_q = insert_merge(p, q, book_p, book_q, True)

    assert new_p.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][::-1]
    assert new_q.tolist() == [1, 2, 3, 8, 5, 6][::-1]


def test_insert_merge_reverse_b():
    # input not reversed
    book_p = [0.1, 0.3, 0.4, 0.5, 0.6]
    book_q = [1, 3, 4, 5, 6]

    p = [0.2, 0.5]
    q = [2, 5]

    new_p, new_q = insert_merge(p, q, book_p, book_q, True)

    assert new_p.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][::-1]
    assert new_q.tolist() == [1, 2, 3, 4, 10, 6][::-1]


def test_insert_merge_unordered():
    book_p = [0.3, 0.4, 0.1, 0.6, 0.5]
    book_q = [3, 4, 1, 6, 5]

    p = [0.4, 0.2]
    q = [4, 2]

    new_p, new_q = insert_merge(p, q, book_p, book_q, False)
    assert new_p.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert new_q.tolist() == [1, 2, 3, 8, 5, 6]

    # noinspection DuplicatedCode
    new_p, new_q = insert_merge(p, q, book_p, book_q, True)
    assert new_p.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][::-1]
    assert new_q.tolist() == [1, 2, 3, 8, 5, 6][::-1]


def test_insert_merge_decimal():
    book_p = [
        Decimal("0.1"),
        Decimal("0.3"),
        Decimal("0.4"),
        Decimal("0.5"),
        Decimal("0.6"),
    ]
    book_q = [Decimal(1), Decimal(3), Decimal(4), Decimal(5), Decimal(6)]

    p = [Decimal("0.2"), Decimal("0.4")]
    q = [Decimal(2), Decimal(4)]

    new_p, new_q = insert_merge(p, q, book_p, book_q, False)

    assert isinstance(new_p[0], Decimal)
    assert isinstance(new_q[0], Decimal)
    assert new_p.tolist() == [
        Decimal("0.1"),
        Decimal("0.2"),
        Decimal("0.3"),
        Decimal("0.4"),
        Decimal("0.5"),
        Decimal("0.6"),
    ]
    assert new_q.tolist() == [
        Decimal(1),
        Decimal(2),
        Decimal(3),
        Decimal(8),
        Decimal(5),
        Decimal(6),
    ]


def test_insert_merge_empty():
    book_p = [0.1, 0.3, 0.4, 0.5, 0.6]
    book_q = [1, 3, 4, 5, 6]

    p = np.array([])
    q = np.array([])

    new_p, new_q = insert_merge(p, q, book_p, book_q, False)

    assert new_p.tolist() == [0.1, 0.3, 0.4, 0.5, 0.6]
    assert new_q.tolist() == [1, 3, 4, 5, 6]


def test_filter_qualifying_orders(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    midpoint = size_cutoff_adj_midpoint(
        bid_p,
        bid_q,
        ask_p,
        ask_q,
        market_data["rewards"]["min_size"],
        int(-np.log10(market_data["minimum_tick_size"])),
    )

    # [0.199, 0.269], 50
    f_bid_p, f_bid_q, f_ask_p, f_ask_q = filter_qualifying_orders(
        bid_p,
        bid_q,
        ask_p,
        ask_q,
        midpoint,
        market_data["rewards"]["max_spread"] / 100,
        market_data["rewards"]["min_size"],
    )

    assert f_bid_p.tolist() == bid_p[-6:-1].tolist()
    assert f_bid_q.tolist() == bid_q[-6:-1].tolist()
    assert f_ask_p.tolist() == ask_p[-1:].tolist()
    assert f_ask_q.tolist() == ask_q[-1:].tolist()


def test_filter_qualifying_orders_except_unit(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    midpoint = size_cutoff_adj_midpoint(
        bid_p,
        bid_q,
        ask_p,
        ask_q,
        market_data["rewards"]["min_size"],
        int(-np.log10(market_data["minimum_tick_size"])),
    )

    # [0.199, 0.269], 50
    with pytest.raises(RewardException):
        filter_qualifying_orders(
            bid_p,
            bid_q,
            ask_p,
            ask_q,
            midpoint,
            market_data["rewards"]["max_spread"],
            market_data["rewards"]["min_size"],
        )


def test_market_side_score(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    # [0.199, 0.269], 50
    f_bid_p, f_bid_q, f_ask_p, f_ask_q = filter_qualifying_orders(
        bid_p, bid_q, ask_p, ask_q, 0.234, 3.5 / 100, 50
    )

    ask_score = _market_side_score(f_ask_p - 0.234, f_ask_q, 3.5 / 100)
    assert round(ask_score, 2) == 27.55

    bid_score = _market_side_score(0.234 - f_bid_p, f_bid_q, 3.5 / 100)
    assert round(bid_score, 2) == 79.74


def test_market_side_score_dec(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    bid_p = np.array([Decimal(str(x)) for x in bid_p])
    bid_q = np.array([Decimal(x) for x in bid_q])
    ask_p = np.array([Decimal(str(x)) for x in ask_p])
    ask_q = np.array([Decimal(x) for x in ask_q])

    # [0.199, 0.269], 50
    f_bid_p, f_bid_q, f_ask_p, f_ask_q = filter_qualifying_orders(
        bid_p, bid_q, ask_p, ask_q, Decimal("0.234"), Decimal("0.035"), 50
    )

    ask_score = _market_side_score(
        f_ask_p - Decimal("0.234"), f_ask_q, Decimal("0.035")
    )
    assert isinstance(ask_score, Decimal)
    assert round(ask_score, 2) == Decimal("27.55")

    bid_score = _market_side_score(
        Decimal("0.234") - f_bid_p, f_bid_q, Decimal("0.035")
    )
    assert isinstance(bid_score, Decimal)
    assert round(bid_score, 2) == Decimal("79.74")


def test_market_side_score_except_unit(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    with pytest.raises(RewardException):
        filter_qualifying_orders(bid_p, bid_q, ask_p, ask_q, 0.234, 3.5, 50)


def test_estimate_sampling_rewards(unified_book_yes, market_data):
    bid_p, bid_q, ask_p, ask_q = unified_book_yes

    rewards, _, _ = estimate_sampling_reward(
        [0.21],
        [50],
        None,
        None,
        bid_p,
        bid_q,
        ask_p,
        ask_q,
        market_data["rewards"]["min_size"],
        market_data["rewards"]["max_spread"] / 100,
        market_data["rewards"]["rates"][0]["rewards_daily_rate"] / 1440,
        int(-np.log10(market_data["minimum_tick_size"])),
    )

    assert 0 < rewards < market_data["rewards"]["rates"][0]["rewards_daily_rate"] / 1440
