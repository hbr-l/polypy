from decimal import Decimal

import pytest

from polypy.exceptions import PositionException, PositionNegativeException
from polypy.position import ACT_SIDE, Position, frozen_position
from polypy.trade import TRADE_STATUS

# todo test both for FAILED


def test_create_position():
    pos = Position.create("test", 1)

    assert pos.size == 1
    assert isinstance(pos.size, float)
    assert pos.size_available == 1
    assert pos.asset_id == "test"


# noinspection DuplicatedCode
def test_act_taker():
    pos = Position.create("test", 1)

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    assert pos.size == 2
    assert pos.size_available == 2
    assert pos.size_total == 2

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    assert pos.size == 3
    assert pos.size_available == 3
    assert pos.size_total == 3

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.size == 3
    assert pos.size_available == 3
    assert pos.size_total == 3

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 3
    assert pos.size_available == 3
    assert pos.size_total == 3

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 3
    assert pos.size_available == 3
    assert pos.size_total == 3

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 3
    assert pos.size_available == 3
    assert pos.size_total == 3

    assert isinstance(pos.size, float)
    assert isinstance(pos.size_available, float)
    assert isinstance(pos.size_total, float)


# noinspection DuplicatedCode
def test_act_maker():
    pos = Position.create("test", 10)

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert pos.size == 8
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.size == 8
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 8
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 8
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 8
    assert pos.size_available == 8
    assert pos.size_total == 8

    assert isinstance(pos.size, float)
    assert isinstance(pos.size_available, float)
    assert isinstance(pos.size_total, float)


# noinspection DuplicatedCode
def test_act_taker_missed_matched():
    # seq 1
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    # seq 2
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    # seq 3
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10


# noinspection DuplicatedCode
def test_act_maker_missed_matched():
    # seq 1
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    # seq 2
    pos = Position.create(
        "test",
        10,
    )
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    # seq 3
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 10


# noinspection DuplicatedCode
def test_act_maker_taker():
    pos = Position.create("test", 10)

    pos.act(1, "taker", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    pos.act(3, "taker2", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    assert pos.size == 12
    assert pos.size_available == 12
    assert pos.size_total == 12

    pos.act(1, "taker", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    pos.act(3, "taker2", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == 12
    assert pos.size_available == 12
    assert pos.size_total == 12

    pos.act(1, "taker", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == pos.size_available == pos.size_total == 12

    pos.act(2, "taker3", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == pos.size_available == pos.size_total == 12


def test_failed_taker():
    pos = Position.create("test", 10)
    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.FAILED)
    assert pos.size == 10

    # even though this is formally a bug:
    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.FAILED)
    assert pos.size == 5


def test_failed_maker():
    pos = Position.create("test", 10)
    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.FAILED)
    assert pos.size == 10

    # even though this is formally a bug:
    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.FAILED)
    assert pos.size == 15


def test_empty():
    pos = Position.create("test", 10)
    assert pos.empty is False

    pos.act(10, "test", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert pos.empty is True
    assert pos.size == 0

    pos.act(10, "test", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.empty is True
    assert pos.size == 0


def test_raise_act_allow_negative():
    pos = Position.create("test", 10)

    with pytest.raises(PositionNegativeException) as record:
        pos.act(12, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert "negative" in str(record)


def test_raise_frozen_attr():
    pos = Position.create("test", 10)

    with pytest.raises(PositionException):
        pos.allow_neg = True

    with pytest.raises(PositionException):
        pos.asset_id = "something"


# noinspection DuplicatedCode
def test_raise_frozen_position():
    pos = Position.create("test", 10)
    pos.size = 12
    assert pos.size == 12

    frozen = frozen_position(pos)

    with pytest.raises(PositionException):
        frozen.allow_neg = True

    with pytest.raises(PositionException):
        frozen.size = 12


def test_decimal():
    pos = Position.create("test", Decimal("1"))

    pos.act(Decimal("0.5"), "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    assert pos.size == Decimal("1.5")
