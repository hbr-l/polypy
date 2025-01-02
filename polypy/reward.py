import warnings
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import numpy as np

from polypy.exceptions import RewardException
from polypy.typing import NumericAlias


def market_position_score(
    adj_spread: NumericAlias | np.ndarray,
    max_qualifying_spread: NumericAlias | np.ndarray,
) -> NumericAlias | np.ndarray:
    return ((max_qualifying_spread - adj_spread) / max_qualifying_spread) ** 2


def _market_side_score(
    adj_spread: NumericAlias | np.ndarray,
    order_size: NumericAlias | np.ndarray,
    max_qualifying_spread: NumericAlias | np.ndarray,
) -> NumericAlias:
    if max_qualifying_spread > 1:
        raise RewardException(
            f"'max_qualifying_spread' must be quoted in USDC. Got {max_qualifying_spread} > 1."
        )

    return np.sum(market_position_score(adj_spread, max_qualifying_spread) * order_size)


def _q_min(
    side_score_1: NumericAlias,
    side_score_2: NumericAlias,
    adj_midpoint: NumericAlias,
    scaling_factor: NumericAlias = 3,
) -> NumericAlias:
    if 0.1 <= adj_midpoint <= 0.9:
        # allow single sided liquidity to score
        return max(
            min(side_score_1, side_score_2),
            max(side_score_1 / scaling_factor, side_score_2 / scaling_factor),
        )

    # require liquidity to be double-sided to score
    return min(side_score_1, side_score_2)


def _q_normal(
    min_side_score: NumericAlias,
    comp_side_scores: np.ndarray | list[NumericAlias] | NumericAlias,
) -> NumericAlias:
    return min_side_score / np.sum(comp_side_scores)


# bids: sorted ascending, best (highest) bid at index -1
# asks: sorted descending, best (lowest) ask at index -1
def size_cutoff_adj_midpoint(
    u_bid_price: list[NumericAlias] | np.ndarray,
    u_bid_size: list[NumericAlias] | np.ndarray,
    u_ask_price: list[NumericAlias] | np.ndarray,
    u_ask_size: list[NumericAlias] | np.ndarray,
    min_shares: NumericAlias,
    tick_size_digits: int,
    sort: bool = True,
) -> NumericAlias:
    if sort:
        # sort bids descending
        bid_idx = np.argsort(u_bid_price)[::-1]
        u_bid_price, u_bid_size = u_bid_price[bid_idx], u_bid_size[bid_idx]

        ask_idx = np.argsort(u_ask_price)
        u_ask_price, u_ask_size = u_ask_price[ask_idx], u_ask_size[ask_idx]

    # bids: sorted descending, best (highest) bid at index 0
    # asks: sorted ascending, best (lowest) ask at index 0
    if u_bid_price[0] < u_bid_price[-1]:
        # best bid: highest
        raise RewardException(
            f"Bid prices are not sorted in descending order. Index 0 = {u_bid_price[0]}, index -1 = {u_bid_price[-1]}"
        )
    if u_ask_price[0] > u_ask_price[-1]:
        # best ask: lowest
        raise RewardException(
            f"Ask prices are not sorted in ascending order. Index 0 = {u_ask_price[0]}, index -1 = {u_ask_price[-1]}"
        )

    cum_bid_q = np.cumsum(u_bid_size)  # always ascending
    cum_ask_q = np.cumsum(u_ask_size)  # always ascending

    cutoff_bid = np.searchsorted(cum_bid_q, min_shares)
    cutoff_ask = np.searchsorted(cum_ask_q, min_shares)

    if cutoff_bid == len(u_bid_size):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Cumulative bid volume below min_shares. Taking best bid for midpoint calculation."
            )
        cutoff_bid = 0
    if cutoff_ask == len(u_ask_size):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Cumulative ask volume below min_shares. Taking best ask for midpoint calculation."
            )
        cutoff_ask = 0

    adj_best_bid = u_bid_price[cutoff_bid]
    adj_best_ask = u_ask_price[cutoff_ask]

    return round((adj_best_bid + adj_best_ask) / 2, tick_size_digits)


def bincount_decimal(
    x: Sequence[int] | np.ndarray, weights: Sequence[Decimal]
) -> np.ndarray:
    counts = np.zeros(max(x) + 1, dtype=object)

    for i, k in enumerate(x):
        counts[k] += weights[i]

    return counts


def insert_merge(
    price: list[NumericAlias] | np.ndarray,
    size: list[NumericAlias] | np.ndarray,
    book_price: list[NumericAlias] | np.ndarray,
    book_size: list[NumericAlias] | np.ndarray,
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    # noinspection PyTypeChecker
    new_p = np.concatenate([price, book_price])
    # noinspection PyTypeChecker
    new_q = np.concatenate([size, book_size])

    # automatically sorts in ascending order
    new_p, inv_idx = np.unique(new_p, return_inverse=True)

    if isinstance(new_q[0], Decimal):
        new_q = bincount_decimal(inv_idx, weights=new_q)
    else:
        new_q = np.bincount(inv_idx, weights=new_q)

    return (new_p[::-1], new_q[::-1]) if reverse else (new_p, new_q)


def _filter_min_shares(
    p: np.ndarray, q: np.ndarray, min_shares: NumericAlias
) -> tuple[np.ndarray, np.ndarray]:
    valid = q >= min_shares
    return p[valid], q[valid]


def _filter_in_spread(
    p: np.ndarray,
    q: np.ndarray,
    midpoint: NumericAlias,
    max_qualifying_spread: NumericAlias,
) -> tuple[np.ndarray, np.ndarray]:
    valid_below = p <= midpoint + max_qualifying_spread
    valid_above = midpoint - max_qualifying_spread <= p
    valid = np.logical_and(valid_below, valid_above)
    return p[valid], q[valid]


def insert_orderbook(
    u_order_bid_price: list[NumericAlias] | np.ndarray | None,
    u_order_bid_size: list[NumericAlias] | np.ndarray | None,
    u_order_ask_price: list[NumericAlias] | np.ndarray | None,
    u_order_ask_size: list[NumericAlias] | np.ndarray | None,
    u_book_bid_price: list[NumericAlias] | np.ndarray,
    u_book_bid_size: list[NumericAlias] | np.ndarray,
    u_book_ask_price: list[NumericAlias] | np.ndarray,
    u_book_ask_size: list[NumericAlias] | np.ndarray,
    bid_revers: bool = True,
    ask_revers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # bids: sorted descending, best (highest) bid at index 0
    # asks: sorted ascending, best (lowest) ask at index 0
    u_bid_p, u_bid_q = insert_merge(
        u_order_bid_price,
        u_order_bid_size,
        u_book_bid_price,
        u_book_bid_size,
        bid_revers,
    )
    u_ask_p, u_ask_q = insert_merge(
        u_order_ask_price,
        u_order_ask_size,
        u_book_ask_price,
        u_book_ask_size,
        ask_revers,
    )

    return u_bid_p, u_bid_q, u_ask_p, u_ask_q


def filter_qualifying_orders(
    u_bid_price: np.ndarray,
    u_bid_size: np.ndarray,
    u_ask_price: np.ndarray,
    u_ask_size: np.ndarray,
    midpoint: NumericAlias,
    max_qualifying_spread: NumericAlias,
    min_shares: NumericAlias | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_qualifying_spread > 1:
        raise RewardException(
            f"'max_qualifying_spread' must be quoted in USDC. Got {max_qualifying_spread} > 1."
        )

    # filter out orders which do not satisfy qualifying spread
    u_bid_price, u_bid_size = _filter_in_spread(
        u_bid_price, u_bid_size, midpoint, max_qualifying_spread
    )
    u_ask_price, u_ask_size = _filter_in_spread(
        u_ask_price, u_ask_size, midpoint, max_qualifying_spread
    )

    # filter out orders which do not satisfy min_shares
    u_bid_price, u_bid_size = _filter_min_shares(u_bid_price, u_bid_size, min_shares)
    u_ask_price, u_ask_size = _filter_min_shares(u_ask_price, u_ask_size, min_shares)

    return u_bid_price, u_bid_size, u_ask_price, u_ask_size


def _compute_score(
    u_bid_p: np.ndarray,
    u_bid_q: np.ndarray,
    u_ask_p: np.ndarray,
    u_ask_q: np.ndarray,
    midpoint: NumericAlias,
    max_qualifying_spread: NumericAlias,
    scaling_factor: NumericAlias,
) -> NumericAlias:
    bid_spread, ask_spread = midpoint - u_bid_p, u_ask_p - midpoint
    score_bid = _market_side_score(bid_spread, u_bid_q, max_qualifying_spread)
    score_ask = _market_side_score(ask_spread, u_ask_q, max_qualifying_spread)
    return _q_min(score_bid, score_ask, midpoint, scaling_factor)


def _discount_competitor_score(
    total_score_min: NumericAlias,
    order_score_min: NumericAlias,
    matching_discount: NumericAlias,
) -> NumericAlias:
    # equiv to: new_total = (total - order_score) * discount + order_score
    #   total - order_score == competitor_score

    try:
        return matching_discount * total_score_min + order_score_min * (
            1 - matching_discount
        )
    except TypeError as e:
        if "decimal.Decimal" in str(e):
            raise TypeError(
                "When using decimal.Decimal, argument `matching_discount` must be provided explicitly as Decimal type."
            ) from e
        else:
            raise e


def _parse_order_args(
    u_order_bid_price: list[NumericAlias] | np.ndarray | None,
    u_order_bid_size: list[NumericAlias] | np.ndarray | None,
    u_order_ask_price: list[NumericAlias] | np.ndarray | None,
    u_order_ask_size: list[NumericAlias] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if u_order_bid_price is None or u_order_bid_size is None:
        u_order_bid_price, u_order_bid_size = np.array([]), np.array([])
    if u_order_ask_price is None or u_order_ask_size is None:
        u_order_ask_price, u_order_ask_size = np.array([]), np.array([])

    u_order_bid_price, u_order_bid_size = np.array(u_order_bid_price), np.array(
        u_order_bid_size
    )
    u_order_ask_price, u_order_ask_size = np.array(u_order_ask_price), np.array(
        u_order_ask_size
    )
    return (
        u_order_bid_price,
        u_order_bid_size,
        u_order_ask_price,
        u_order_ask_size,
    )


def _is_qualified(
    u_order_bid_price: Sequence[Any],
    u_order_ask_price: Sequence[Any],
    midpoint: NumericAlias,
    lb_1sided_liq: NumericAlias | float,
    ub_1sided_liq: NumericAlias | float,
) -> bool:
    # check if we are still eligible for rewards

    # at least liquidity on one side
    if (len(u_order_bid_price) + len(u_order_ask_price)) == 0:
        # no qualifying orders at all
        return False

    # if we're below or above a certain threshold, we have to provide
    #   double-sided liquidity to be eligible for rewards
    return (
        lb_1sided_liq <= midpoint <= ub_1sided_liq
        or len(u_order_bid_price) * len(u_order_ask_price) != 0
    )


# todo axis args and broadcast (vectorization)
# todo BUG: approach for competitor is WRONG -> recheck entire module
#   approximate competitors via:
#   theta * S(v,s) * (alpha * ((V_bid_s + V_ask_s)/3) + (alpha - 1) * min(V_bid_s, V_ask_s))
#   where:
#       theta: ratio of participants with at least min_shares
#       alpha: ratio between extrema 1) all participants one-sided and 2) all participants two-sided
#   alpha and theta need to be fitted
def estimate_sampling_reward(
    u_order_bid_price: list[NumericAlias] | np.ndarray | None,
    u_order_bid_size: list[NumericAlias] | np.ndarray | None,
    u_order_ask_price: list[NumericAlias] | np.ndarray | None,
    u_order_ask_size: list[NumericAlias] | np.ndarray | None,
    u_book_bid_price: list[NumericAlias] | np.ndarray,
    u_book_bid_size: list[NumericAlias] | np.ndarray,
    u_book_ask_price: list[NumericAlias] | np.ndarray,
    u_book_ask_size: list[NumericAlias] | np.ndarray,
    min_shares: NumericAlias | int,
    max_qualifying_spread: NumericAlias,
    reward_per_sampling: NumericAlias,
    tick_size_digits: int,
    lb_1sided_liq: NumericAlias = 0.1,
    ub_1sided_liq: NumericAlias = 0.9,
    scaling_factor: NumericAlias = 3,
    matching_discount: NumericAlias = 1,
) -> tuple[NumericAlias, NumericAlias, NumericAlias]:
    """

    Parameters
    ----------
    u_order_bid_price
    u_order_bid_size
    u_order_ask_price
    u_order_ask_size
    u_book_bid_price
    u_book_bid_size
    u_book_ask_price
    u_book_ask_size
    min_shares
    max_qualifying_spread
    reward_per_sampling
    tick_size_digits
    lb_1sided_liq
    ub_1sided_liq
    scaling_factor
    matching_discount

    Returns
    -------
    reward: Numeric
        reward per sampling period, same unit as py:arg:`reward_per_sampling`, 0 if not qualified for rewards
    order_score: Numeric
        order score, 0 if not qualified for rewards
    total_score: Numeric
        total market score (discounted competitor score + order score), -1 if not qualified for rewards.

    Notes
    -----
    1) If decimal.Decimal is in use, default arguments should be provided explicitly as decimal.Decimal type when
    calling this function.
    2) order book data on level 2 is aggregated, such that the competitor score is overestimated and the own
    order score is underestimated (estimated towards safe-side). This is due to the fact, that we can not see
    single orders but only the sum of all orders at each price level, which makes filtering for min_shares per order
    (and in case of required double-sided liquidity filtering for double-sided placed order pairs) not possible.
    """
    (
        u_order_bid_price,
        u_order_bid_size,
        u_order_ask_price,
        u_order_ask_size,
    ) = _parse_order_args(
        u_order_bid_price, u_order_bid_size, u_order_ask_price, u_order_ask_size
    )

    u_bid_p, u_bid_q, u_ask_p, u_ask_q = insert_orderbook(
        u_order_bid_price,
        u_order_bid_size,
        u_order_ask_price,
        u_order_ask_size,
        u_book_bid_price,
        u_book_bid_size,
        u_book_ask_price,
        u_book_ask_size,
        True,
        False,
    )

    midpoint = size_cutoff_adj_midpoint(
        u_bid_p, u_bid_q, u_ask_p, u_ask_q, min_shares, tick_size_digits, False
    )

    u_bid_p, u_bid_q, u_ask_p, u_ask_q = filter_qualifying_orders(
        u_bid_p, u_bid_q, u_ask_p, u_ask_q, midpoint, max_qualifying_spread, min_shares
    )
    (
        u_order_bid_price,
        u_order_bid_size,
        u_order_ask_price,
        u_order_ask_size,
    ) = filter_qualifying_orders(
        u_order_bid_price,
        u_order_bid_size,
        u_order_ask_price,
        u_order_ask_size,
        midpoint,
        max_qualifying_spread,
        min_shares,
    )

    # check if we are still eligible for rewards
    if not _is_qualified(
        u_order_bid_price, u_order_ask_price, midpoint, lb_1sided_liq, ub_1sided_liq
    ):
        return type(u_order_bid_price[0])(0), 0, -1

    # compute scores
    # todo include self?
    total_score_min = _compute_score(
        u_bid_p,
        u_bid_q,
        u_ask_p,
        u_ask_q,
        midpoint,
        max_qualifying_spread,
        scaling_factor,
    )
    order_score_min = _compute_score(
        u_order_bid_price,
        u_order_bid_size,
        u_order_ask_price,
        u_order_ask_size,
        midpoint,
        max_qualifying_spread,
        scaling_factor,
    )

    total_score_min = _discount_competitor_score(
        total_score_min, order_score_min, matching_discount
    )

    # compute reward
    return (
        _q_normal(order_score_min, total_score_min) * reward_per_sampling,
        order_score_min,
        total_score_min,
    )
