import random
from decimal import Decimal
from typing import Callable

import pytest

from polypy import (
    CHAIN_ID,
    SIDE,
    TRADE_STATUS,
    AugmentedConversionCache,
    MarketIdQuintet,
    MarketIdTriplet,
    PositionManager,
    RPCSettings,
    dec,
)
from polypy.exceptions import (
    PositionException,
    PositionNegativeException,
    PositionTrackingException,
)

# noinspection PyProtectedMember
from polypy.manager.rpc_ops import (
    MTX,
    _tx_post_batch_operate_positions,
    _tx_pre_batch_operate_positions,
)

# noinspection PyProtectedMember
from polypy.manager.rpc_proc import (
    _tx_post_convert_positions,
    _tx_post_merge_positions,
    _tx_post_redeem_positions,
    _tx_post_split_positions,
    _tx_pre_merge_positions,
)

rng_rand_float_dec = random.Random(42)


# noinspection DuplicatedCode
@pytest.fixture(scope="function")
def position_manager() -> PositionManager:
    return PositionManager("", "", dec(100))


@pytest.fixture(scope="function")
def rpc_settings(private_key) -> RPCSettings:
    return RPCSettings(
        "",
        "",
        CHAIN_ID.AMOY,
        private_key,
        None,
        1,
        None,
        True,
        None,
        None,
        None,
        None,
        None,
    )


@pytest.fixture(scope="function")
def market_triplet() -> MarketIdTriplet:
    return MarketIdTriplet("0x000000", "000001", "000002", None)


def test_post_split_position(position_manager, market_triplet):
    _tx_post_split_positions(position_manager, market_triplet, dec(21.75))
    assert position_manager.balance_available == dec("78.25")
    assert position_manager.get_by_id(market_triplet[1]).size == dec("21.75")
    assert position_manager.get_by_id(market_triplet[2]).size == dec("21.75")

    # auto-type casting
    _tx_post_split_positions(position_manager, market_triplet, 21.75)
    assert position_manager.balance_available == dec("78.25") - dec("21.75")
    assert position_manager.get_by_id(market_triplet[1]).size == dec("21.75") * 2
    assert position_manager.get_by_id(market_triplet[2]).size == dec("21.75") * 2
    assert isinstance(position_manager.balance_available, Decimal)
    assert isinstance(position_manager.get_by_id(market_triplet[1]).size, Decimal)


def test_post_split_position_raises_overspending(position_manager, market_triplet):
    with pytest.raises(PositionException):
        _tx_post_split_positions(position_manager, market_triplet, dec(100.75))


def test_post_merge_positions(position_manager, market_triplet, rpc_settings):
    position_manager.transact(
        market_triplet[1], dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        market_triplet[2], dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    _tx_post_merge_positions(position_manager, market_triplet, dec(6))
    assert position_manager.balance_available == dec("106")
    assert position_manager.get_by_id(market_triplet[1]).size == dec("4")
    assert position_manager.get_by_id(market_triplet[2]).size == dec("6")

    _tx_post_merge_positions(position_manager, market_triplet, 1)
    assert position_manager.balance_available == dec("107")
    assert position_manager.get_by_id(market_triplet[1]).size == dec("3")
    assert position_manager.get_by_id(market_triplet[2]).size == dec("5")
    assert isinstance(position_manager.balance_available, Decimal)
    assert isinstance(position_manager.get_by_id(market_triplet[1]).size, Decimal)

    cond_id, size, neg_risk = _tx_pre_merge_positions(
        position_manager, market_triplet, None, False, ""
    )
    _tx_post_merge_positions(position_manager, market_triplet, size)
    assert position_manager.balance_available == dec("110")
    assert position_manager.get_by_id(market_triplet[1]).size == dec("0")
    assert position_manager.get_by_id(market_triplet[2]).size == dec("2")
    assert isinstance(position_manager.balance_available, Decimal)
    assert isinstance(position_manager.get_by_id(market_triplet[2]).size, Decimal)


def test_post_merge_positions_raises_overspending(position_manager, market_triplet):
    position_manager.transact(
        market_triplet[1], dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        market_triplet[2], dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    with pytest.raises(PositionException):
        _tx_post_merge_positions(position_manager, market_triplet, dec(12.001))


# noinspection PyPep8Naming,DuplicatedCode
def test_post_convert_positions_bookkeep_deposit_false_N_YY(position_manager):
    # (Y1, N1), (Y2, N2), (Y3, N3): N1 -> Y2, Y3

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(7)

    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0]],
        all_market_quintets,
        size,
        False,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size
    assert position_manager.get_by_id("Y2").size == size
    assert position_manager.get_by_id("Y3").size == size
    assert position_manager.balance == dec(100)
    assert conv_cache.caches["0x9"].cumulative_size == size
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3"}

    size2 = dec(3)
    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0]],
        all_market_quintets
        + [MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4", None)],
        size2,
        False,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size - size2
    assert position_manager.get_by_id("Y2").size == size + size2
    assert position_manager.get_by_id("Y3").size == size + size2
    assert position_manager.get_by_id("Y4").size == size + size2
    assert position_manager.balance == dec(100)
    assert conv_cache.caches["0x9"].cumulative_size == size + size2
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3", "0x4"}


# noinspection PyPep8Naming,DuplicatedCode
def test_post_convert_positions_bookkeep_deposit_true_N_YY(position_manager):
    # (Y1, N1), (Y2, N2), (Y3, N3): N1 -> Y2, Y3

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(7)

    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0]],
        all_market_quintets,
        size,
        True,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size
    assert position_manager.get_by_id("Y2").size == size
    assert position_manager.get_by_id("Y3").size == size
    assert position_manager.balance == dec(100)
    assert conv_cache.caches["0x9"].cumulative_size == size
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3"}

    size2 = dec(3)
    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0]],
        all_market_quintets
        + [MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4", None)],
        size2,
        True,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size - size2
    assert position_manager.get_by_id("Y2").size == size + size2
    assert position_manager.get_by_id("Y3").size == size + size2
    assert position_manager.get_by_id("Y4").size == size + size2
    assert position_manager.balance == dec(100)
    assert conv_cache.caches["0x9"].cumulative_size == size + size2
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3", "0x4"}


# noinspection PyPep8Naming,DuplicatedCode
def test_post_convert_positions_bookkeep_deposit_false_NN_Y(position_manager):
    # (Y1, N1), (Y2, N2), (Y3, N3): N1, N2 -> Y3

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(7)

    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets,
        size,
        False,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size
    assert position_manager.get_by_id("N2").size == dec(12) - size
    assert position_manager.get_by_id("Y3").size == size
    assert position_manager.balance == dec(100) + size
    assert conv_cache.caches["0x9"].cumulative_size == size
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3"}

    size2 = dec(3)
    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets
        + [MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4", None)],
        size2,
        False,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size - size2
    assert position_manager.get_by_id("N2").size == dec(12) - size - size2
    assert position_manager.get_by_id("Y3").size == size + size2
    assert position_manager.get_by_id("Y4").size == size + size2
    assert position_manager.balance == dec(100) + size + size2
    assert conv_cache.caches["0x9"].cumulative_size == size + size2
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3", "0x4"}


# noinspection PyPep8Naming,DuplicatedCode
def test_post_convert_positions_bookkeep_deposit_true_NN_Y(position_manager):
    # (Y1, N1), (Y2, N2), (Y3, N3): N1, N2 -> Y3

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(7)

    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets,
        size,
        True,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size
    assert position_manager.get_by_id("N2").size == dec(12) - size
    assert position_manager.get_by_id("Y3").size == size
    assert position_manager.balance == dec(100) + size
    assert conv_cache.caches["0x9"].cumulative_size == size
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3"}

    size2 = dec(3)
    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets
        + [MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4", None)],
        size2,
        True,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size - size2
    assert position_manager.get_by_id("N2").size == dec(12) - size - size2
    assert position_manager.get_by_id("Y3").size == size + size2
    assert position_manager.get_by_id("Y4").size == size + size2
    assert position_manager.balance == dec(100) + size + size2
    assert conv_cache.caches["0x9"].cumulative_size == size + size2
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3", "0x4"}


# noinspection PyPep8Naming,DuplicatedCode
def test_post_convert_positions_bookkeep_deposit_false_NN_YY(position_manager):
    # (Y1, N1), (Y2, N2), (Y3, N3), (Y4, N4): N1, N2 -> Y3, Y4
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
        MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(7)

    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets,
        size,
        False,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size
    assert position_manager.get_by_id("N2").size == dec(12) - size
    assert position_manager.get_by_id("Y3").size == size
    assert position_manager.get_by_id("Y4").size == size
    assert position_manager.balance == dec(100) + size
    assert conv_cache.caches["0x9"].cumulative_size == size
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3", "0x4"}

    size2 = dec(3)
    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets
        + [MarketIdQuintet("0x5", "0x9", "0x004", "Y5", "N5", None)],
        size2,
        False,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size - size2
    assert position_manager.get_by_id("N2").size == dec(12) - size - size2
    assert position_manager.get_by_id("Y3").size == size + size2
    assert position_manager.get_by_id("Y4").size == size + size2
    assert position_manager.get_by_id("Y5").size == size + size2
    assert position_manager.balance == dec(100) + size + size2
    assert conv_cache.caches["0x9"].cumulative_size == size + size2
    assert conv_cache.caches["0x9"].seen_condition_ids == {
        "0x1",
        "0x2",
        "0x3",
        "0x4",
        "0x5",
    }


# noinspection PyPep8Naming,DuplicatedCode
def test_post_convert_positions_bookkeep_deposit_true_NN_YY(position_manager):
    # (Y1, N1), (Y2, N2), (Y3, N3), (Y4, N4): N1, N2 -> Y3, Y4
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
        MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(12), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(7)

    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets,
        size,
        True,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size
    assert position_manager.get_by_id("N2").size == dec(12) - size
    assert position_manager.get_by_id("Y3").size == size
    assert position_manager.get_by_id("Y4").size == size
    assert position_manager.balance == dec(100) + size
    assert conv_cache.caches["0x9"].cumulative_size == size
    assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3", "0x4"}

    size2 = dec(3)
    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0], all_market_quintets[1]],
        all_market_quintets
        + [MarketIdQuintet("0x5", "0x9", "0x004", "Y5", "N5", None)],
        size2,
        True,
    )
    assert position_manager.get_by_id("N1").size == dec(10) - size - size2
    assert position_manager.get_by_id("N2").size == dec(12) - size - size2
    assert position_manager.get_by_id("Y3").size == size + size2
    assert position_manager.get_by_id("Y4").size == size + size2
    assert position_manager.get_by_id("Y5").size == size + size2
    assert position_manager.balance == dec(100) + size + size2
    assert conv_cache.caches["0x9"].cumulative_size == size + size2
    assert conv_cache.caches["0x9"].seen_condition_ids == {
        "0x1",
        "0x2",
        "0x3",
        "0x4",
        "0x5",
    }


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


# noinspection DuplicatedCode
def test_post_convert_positions_bookkeep_deposit_false_numerical_stability():
    # N_YY no bookkeeping random numbers -> test for numeric stability
    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
    ]

    def setup():
        pm = PositionManager("", "", dec(100))
        cache = AugmentedConversionCache("")
        pm.transact(
            "N1", dec(100), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
        )
        pm.transact(
            "N2", dec(120), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
        )
        return pm, cache

    for _ in range(350):
        size = dec(gen_rand_float_dec(2, 2))
        pos_mng, conv_cache = setup()

        _tx_post_convert_positions(
            pos_mng,
            conv_cache,
            [all_market_quintets[0], all_market_quintets[1]],
            all_market_quintets,
            size,
            False,
        )
        assert pos_mng.get_by_id("N1").size == dec(100) - size
        assert pos_mng.get_by_id("N2").size == dec(120) - size
        assert pos_mng.get_by_id("Y3").size == size
        assert pos_mng.balance == dec(100) + size
        assert conv_cache.caches["0x9"].cumulative_size == size
        assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3"}

    for _ in range(150):
        size = dec(gen_rand_float_dec(1, 1))
        pos_mng, conv_cache = setup()

        _tx_post_convert_positions(
            pos_mng,
            conv_cache,
            [all_market_quintets[0], all_market_quintets[1]],
            all_market_quintets,
            size,
            False,
        )
        assert pos_mng.get_by_id("N1").size == dec(100) - size
        assert pos_mng.get_by_id("N2").size == dec(120) - size
        assert pos_mng.get_by_id("Y3").size == size
        assert pos_mng.balance == dec(100) + size
        assert conv_cache.caches["0x9"].cumulative_size == size
        assert conv_cache.caches["0x9"].seen_condition_ids == {"0x1", "0x2", "0x3"}


# noinspection DuplicatedCode
def test_post_convert_positions_raises_act_conversion_cache_invalidate(
    position_manager,
):
    # N_YY invalidate position_manager through _act_conversion_cache

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(7)

    _tx_post_convert_positions(
        position_manager,
        conv_cache,
        [all_market_quintets[0]],
        all_market_quintets,
        size,
        False,
    )

    size2 = dec(3)
    position_manager.transact(
        "Y4", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    with pytest.raises(PositionTrackingException) as record:
        _tx_post_convert_positions(
            position_manager,
            conv_cache,
            [all_market_quintets[0]],
            all_market_quintets
            + [MarketIdQuintet("0x4", "0x9", "0x003", "Y4", "N4", None)],
            size2,
            False,
        )
    assert "already" in str(record)
    assert position_manager._invalid_token is True
    assert "already" in position_manager._invalid_reason


# noinspection DuplicatedCode
def test_post_convert_positions_raises_overspending_invalidate(position_manager):
    # N_YY invalidate position_manager through overspending

    all_market_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y1", "N1", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y2", "N2", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y3", "N3", None),
    ]
    conv_cache = AugmentedConversionCache("")
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    size = dec(-7)

    with pytest.raises(PositionNegativeException) as record:
        _tx_post_convert_positions(
            position_manager,
            conv_cache,
            [all_market_quintets[0]],
            all_market_quintets,
            size,
            False,
        )
    assert "negative" in str(record)
    assert position_manager._invalid_token is True
    assert "negative" in position_manager._invalid_reason


def test_post_redeem_positions(position_manager):
    triplet = MarketIdTriplet("0x01", "Y1", "N1", None)
    position_manager.transact(
        "Y1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    _tx_post_redeem_positions(position_manager, triplet, 5, 2, "YES")
    assert position_manager.get_by_id("Y1").size == dec(5)
    assert position_manager.get_by_id("N1").size == dec(8)
    assert position_manager.balance == dec(105)

    _tx_post_redeem_positions(position_manager, triplet, 2, 5, "NO")
    assert position_manager.get_by_id("Y1").size == dec(3)
    assert position_manager.get_by_id("N1").size == dec(3)
    assert position_manager.balance == dec(110)


def test_post_redeem_positions_one_side(position_manager):
    triplet = MarketIdTriplet("0x01", "Y1", "N1", None)
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )

    _tx_post_redeem_positions(position_manager, triplet, 0, 2, "YES")
    assert "Y1" not in position_manager
    assert position_manager.get_by_id("N1").size == dec(8)
    assert position_manager.balance == dec(100)


def test_post_redeem_positions_raises(position_manager):
    triplet = MarketIdTriplet("0x01", "Y1", "N1", None)
    # failure case: positions not in position manager, overspending

    # positions not in position manager
    with pytest.raises(PositionTrackingException):
        _tx_post_redeem_positions(position_manager, triplet, 5, 2, "YES")

    position_manager.transact(
        "Y1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N1", dec(10), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    with pytest.raises(PositionNegativeException):
        _tx_post_redeem_positions(position_manager, triplet, 50, 20, "YES")


def test_post_batch_operate_positions(position_manager):
    triplet1 = MarketIdTriplet("0x01", "Y1", "N1", None)
    triplet2 = MarketIdTriplet("0x02", "Y2", "N2", None)
    all_quintets = [
        MarketIdQuintet("0x1", "0x9", "0x000", "Y11", "N11", None),
        MarketIdQuintet("0x2", "0x9", "0x001", "Y12", "N12", None),
        MarketIdQuintet("0x3", "0x9", "0x002", "Y13", "N13", None),
    ]
    cvt_quintets = all_quintets[:2]
    position_manager.transact(
        "Y2", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N2", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N11", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    position_manager.transact(
        "N12", dec(4), dec(0), "some", SIDE.BUY, TRADE_STATUS.MATCHED, True
    )
    mtxs = [
        MTX.split(triplet1, dec(2), False),
        MTX.merge(triplet2, dec(4), False),
        MTX.convert(cvt_quintets, all_quintets, dec(2), False),
    ]

    cache = AugmentedConversionCache("")

    txns = _tx_pre_batch_operate_positions(position_manager, mtxs, "", "")

    _tx_post_batch_operate_positions(position_manager, cache, mtxs, txns)

    assert position_manager.get_by_id("Y1").size == dec(2)  # split
    assert position_manager.get_by_id("N1").size == dec(2)  # split
    assert position_manager.get_by_id("Y2").size == dec(0)  # merge
    assert position_manager.get_by_id("N2").size == dec(0)  # merge
    assert position_manager.get_by_id("N11").size == dec(2)  # convert
    assert position_manager.get_by_id("N12").size == dec(2)  # convert
    assert position_manager.get_by_id("Y13").size == dec(2)  # convert
    assert position_manager.balance == dec(104)

    # we don't test for failure cases, since any exceptions are either captured in _tx_pre_batch or in any of
    #   the _tx_posts functions
