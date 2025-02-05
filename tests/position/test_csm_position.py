from decimal import Decimal

import pytest

from polypy.exceptions import (
    PositionException,
    PositionNegativeException,
    PositionTransactionException,
)
from polypy.position import ACT_SIDE
from polypy.position import CSMPosition as Position
from polypy.position import frozen_position
from polypy.trade import TRADE_STATUS


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
    assert pos.pending_taker == 1
    assert pos.pending_maker == 0
    assert pos.size == 1
    assert pos.size_available == 1
    assert pos.size_total == 2

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    assert pos.pending_taker == 2
    assert pos.pending_maker == 0
    assert pos.size == 1
    assert pos.size_available == 1
    assert pos.size_total == 3

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.pending_taker == 2
    assert pos.pending_maker == 0
    assert pos.size == 1
    assert pos.size_available == 1
    assert pos.size_total == 3

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_taker == 2
    assert pos.pending_maker == 0
    assert pos.size == 1
    assert pos.size_available == 1
    assert pos.size_total == 3

    pos.act(1, "trade2", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_taker == 1
    assert pos.pending_maker == 0
    assert pos.size == 2
    assert pos.size_available == 2
    assert pos.size_total == 3

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_taker == 0
    assert pos.pending_maker == 0
    assert pos.size == 3
    assert pos.size_available == 3
    assert pos.size_total == 3

    assert isinstance(pos.size, float)
    assert isinstance(pos.size_available, float)
    assert isinstance(pos.size_total, float)
    assert len(pos.pending_trade_ids) == 0


# noinspection DuplicatedCode
def test_act_maker():
    pos = Position.create("test", 10)

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert pos.pending_maker == -1
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert pos.pending_maker == -2
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == -2
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_maker == -2
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade2", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == -1
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 8
    assert pos.size_total == 8

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 8
    assert pos.size_available == 8
    assert pos.size_total == 8

    assert isinstance(pos.size, float)
    assert isinstance(pos.size_available, float)
    assert isinstance(pos.size_total, float)
    assert len(pos.pending_trade_ids) == 0


# noinspection DuplicatedCode
def test_act_taker_direct_settlement():
    # case: settlement at confirmed
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11

    # case: settlement at mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11

    # case: settlement at mined - missed mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    # missed yet
    assert pos.size == pos.size_total == 10
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11


def test_act_taker_retrying_confirmed():
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.CONFIRMED)
    assert pos.size == pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == pos.size_total == 11


# noinspection DuplicatedCode
def test_act_taker_missed_matched():
    # case: settlement at confirmed
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 1
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 1
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 1
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11

    # case: settlement at mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 1
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11

    # case: settlement at mined - missed mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 1
    assert pos.size == 10
    assert pos.size_available == 10
    assert pos.size_total == 11

    pos.act(1, "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 11


# noinspection DuplicatedCode
def test_act_maker_direct_settlement():
    # case: settlement at confirmed
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9

    # case: settlement at mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9

    # case: settlement at mined - missed mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    # missed yet
    assert pos.size == pos.size_total == 10
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9


def test_act_maker_retrying_confirmed():
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.CONFIRMED)
    assert pos.size == pos.size_total == 10

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.size == 10
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == pos.size_total == 9


# noinspection DuplicatedCode
def test_act_maker_missed_matched():
    # case: settlement at confirmed
    pos = Position.create("test", 10)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == -1
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == -1
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_maker == -1
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9

    # case: settlement at mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_maker == -1
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.MINED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9

    # case: settlement at mined - missed mined
    pos = Position.create("test", 10, settlement_status=TRADE_STATUS.MINED)
    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.RETRYING)
    assert pos.pending_maker == -1
    assert pos.pending_taker == 0
    assert pos.size == 10
    assert pos.size_available == 9
    assert pos.size_total == 9

    pos.act(1, "trade1", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == 0
    assert pos.pending_taker == 0
    assert pos.size == 9
    assert pos.size_available == 9
    assert pos.size_total == 9


def test_act_maker_taker():
    pos = Position.create("test", 10)

    pos.act(1, "taker", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    pos.act(3, "taker2", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    assert pos.pending_taker == 4
    assert pos.pending_maker == -2
    assert pos.size == 10
    assert pos.size_available == 8
    assert pos.size_total == 12

    pos.act(1, "taker", ACT_SIDE.TAKER, TRADE_STATUS.MINED)
    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    pos.act(3, "taker2", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_taker == 1
    assert pos.pending_maker == 0
    assert pos.size == 11
    assert pos.size_available == 11
    assert pos.size_total == 12

    pos.act(1, "taker", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.pending_maker == pos.pending_taker == 0
    assert pos.size == pos.size_available == pos.size_total == 12

    pos.act(2, "taker3", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == pos.size_available == pos.size_total == 14


def test_failed_taker():
    pos = Position.create("test", 10)
    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.FAILED)
    assert pos.size == 10
    assert pos.size_total == pos.size_available == 10

    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.FAILED)
    assert pos.size == 10
    assert pos.size_total == pos.size_available == 10

    # even though, this is formally a bug:
    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    pos.act(5, "taker", ACT_SIDE.TAKER, TRADE_STATUS.FAILED)
    assert pos.size == 15
    assert pos.size_total == pos.size_available == 15


def test_failed_maker():
    pos = Position.create("test", 10)
    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.FAILED)
    assert pos.size == 10
    assert pos.size_total == pos.size_available == 10

    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.FAILED)
    assert pos.size == 10
    assert pos.size_total == pos.size_available == 10

    # even though, this is formally a bug:
    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    pos.act(5, "maker", ACT_SIDE.MAKER, TRADE_STATUS.FAILED)
    assert pos.size == 5
    assert pos.size_total == pos.size_available == 5


def test_empty():
    pos = Position.create("test", 10)
    assert pos.empty is False

    pos.act(10, "test", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert pos.empty is False
    assert pos.size == 10

    pos.act(10, "test", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert pos.empty is True
    assert pos.size == 0


def test_raise_act_match():
    pos = Position.create("test", 10)
    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)

    with pytest.raises(PositionTransactionException):
        pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)

    assert pos.size == 10
    assert pos.size_total == pos.size_available == 8
    assert pos.pending_maker == -2
    assert pos.pending_taker == 0


def test_raise_act_settle():
    pos = Position.create("test", 10)
    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)

    with pytest.raises(PositionTransactionException):
        pos.act(1, "maker", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert "maker" in pos.pending_trade_ids
    assert pos.size == 10
    assert pos.size_total == pos.size_available == 8
    assert pos.pending_maker == -2
    assert pos.pending_taker == 0

    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert len(pos.pending_trade_ids) == 0
    assert pos.size == pos.size_total == pos.size_available == 8
    assert pos.pending_maker == pos.pending_taker == 0


def test_raise_act_allow_negative():
    pos = Position.create("test", 10)
    pos.act(12, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)

    with pytest.raises(PositionNegativeException) as record:
        pos.act(12, "maker", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    assert "negative" in str(record)


def test_raise_max_size_trade_ids():
    pos = Position.create("test", 10, max_size_trade_ids=1)

    pos.act(2, "maker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)

    with pytest.raises(PositionTransactionException) as record:
        pos.act(2, "taker", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert "max_size_trade_ids" in str(record)

    assert "taker" not in pos.pending_trade_ids
    assert "maker" in pos.pending_trade_ids
    assert pos.size == 10
    assert pos.size_total == pos.size_available == 8
    assert pos.pending_maker == -2
    assert pos.pending_taker == 0


def test_raise_frozen_attr():
    pos = Position.create("test", 10, max_size_trade_ids=1)

    with pytest.raises(PositionException):
        pos.allow_neg = True

    with pytest.raises(PositionException):
        pos.asset_id = "something"


def test_raise_frozen_position():
    pos = Position.create("test", 10)
    pos.size = 12
    assert pos.size == 12

    frozen = frozen_position(pos)

    with pytest.raises(PositionException):
        frozen.allow_neg = True

    with pytest.raises(PositionException):
        frozen.size = 12


def test_raise_settlement_status():
    with pytest.raises(PositionException):
        Position.create("test", 10, settlement_status=TRADE_STATUS.MATCHED)

    with pytest.raises(PositionException):
        Position.create("test", 10, settlement_status=TRADE_STATUS.FAILED)

    with pytest.raises(PositionException):
        Position.create("test", 10, settlement_status=TRADE_STATUS.RETRYING)


def test_decimal():
    pos = Position.create("test", Decimal("1"))

    pos.act(Decimal("0.5"), "trade1", ACT_SIDE.TAKER, TRADE_STATUS.MATCHED)
    assert pos.pending_maker == Decimal("0")
    assert pos.pending_taker == Decimal("0.5")
    assert pos.size == Decimal("1")
    assert pos.size_total == Decimal("1.5")
    assert pos.pending_trade_ids == {"trade1": Decimal("0.5")}

    pos.act(Decimal("0.5"), "trade1", ACT_SIDE.TAKER, TRADE_STATUS.CONFIRMED)
    assert pos.size == Decimal("1.5")
