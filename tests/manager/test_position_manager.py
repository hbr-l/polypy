from decimal import Decimal

import pytest
import responses

from polypy.book import OrderBook
from polypy.exceptions import (
    ManagerInvalidException,
    PositionTrackingException,
    PositionTransactionException,
)
from polypy.manager.position import PositionManager
from polypy.order.common import SIDE
from polypy.position import ACT_SIDE, USDC, CSMPosition, Position, frozen_position
from polypy.trade import TRADE_STATUS


def test_create_get_position(local_host_addr):
    pm = PositionManager(local_host_addr, None, 10)

    pm.create_position("test", 10, size_sig_digits=3, allow_neg=True)

    assert list(list(pm.asset_ids)) == [USDC, "test"]
    assert pm.get_by_id(USDC).size == 10
    assert pm.get_by_id(USDC).size_sig_digits == 5
    assert pm.get_by_id(USDC).allow_neg == False
    assert pm.get_by_id("test").size == 10
    assert pm.get_by_id("test").size_sig_digits == 3
    assert pm.get_by_id("test").allow_neg == True
    assert [p.asset_id for p in pm.get(size=10)] == [USDC, "test"]
    assert [p.asset_id for p in pm.get(size_sig_digits=3)] == ["test"]
    assert [p.asset_id for p in pm.get(allow_neg=True)] == ["test"]


def test_track():
    pm = PositionManager(None, None, 10)
    pos = Position.create("test", 12)

    pm.track(pos)

    assert list(list(pm.asset_ids)) == [USDC, "test"]
    assert pm.get_by_id("test").size == 12


def test_raise_track():  # sourcery skip: extract-duplicate-method
    pm = PositionManager(None, None, 10)

    pos = Position.create("", 12)
    with pytest.raises(PositionTrackingException) as record:
        pm.track(pos)
    assert "asset_id" in str(record)
    assert list(list(pm.asset_ids)) == [USDC]

    frozen = frozen_position(Position.create("test", 12))
    with pytest.raises(PositionTrackingException) as record:
        # noinspection PyTypeChecker
        pm.track(frozen)
    assert "frozen" in str(record)
    assert list(list(pm.asset_ids)) == [USDC]


def test_raise_max_size():
    pm = PositionManager(None, None, 10, max_size=1)
    pos = Position.create("test", 12)

    with pytest.raises(PositionTrackingException) as record:
        pm.track(pos)
    assert "max_size" in str(record)
    assert list(list(pm.asset_ids)) == [USDC]


def test_untrack():  # sourcery skip: extract-duplicate-method
    pm = PositionManager(None, None, 10)
    pos = Position.create("test", 12)

    pm.track(pos)
    pm.untrack("some")
    assert len(list(list(pm.asset_ids))) == 2

    pm.untrack("test")
    assert len(list(list(pm.asset_ids))) == 1

    pm.untrack("some")
    assert len(list(list(pm.asset_ids))) == 1


def test_total_nums():
    pm = PositionManager(None, None, 10)
    pm.track(Position.create("test", 12))

    total = pm.total({USDC: 1, "test": 0.5}, None)
    assert total == 16

    total = pm.total({"test": 0.5}, None)
    assert total == 16


def test_total_books():  # sourcery skip: extract-duplicate-method
    pm = PositionManager(None, None, 10)
    pm.track(Position.create("test", 12))
    pm.track(Position.create("test_2", 6))

    book_test = OrderBook("test", 0.01)
    book_test.set_bids([0.4], [10])
    book_test.set_asks([0.6], [1])

    book_test_2 = OrderBook("test_2", 0.001)
    book_test_2.set_bids([0.3], [10])
    book_test_2.set_asks([0.5], [1])

    total = pm.total({"test": book_test, "test_2": book_test_2}, None)
    assert total == 18.4


def test_total_rest(local_host_addr):
    pm = PositionManager(local_host_addr, None, 10)
    pm.track(Position.create("test", 12))
    pm.track(Position.create("test_2", 6))

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            f"{local_host_addr}/midpoints",
            json={"test": "0.5", "test_2": "0.4"},
        )

        total = pm.total(None, None)
        assert total == 18.4

        total = pm.total({"test": 0.5}, None)
        assert total == 18.4


def test_total_mixed(local_host_addr):
    pm = PositionManager(local_host_addr, None, 10)
    pm.track(Position.create("test", 12))
    pm.track(Position.create("test_2", 6))
    pm.track(Position.create("test_3", 4))

    book_test_2 = OrderBook("test_2", 0.001)
    book_test_2.set_bids([0.3], [10])
    book_test_2.set_asks([0.5], [1])

    with responses.RequestsMock() as rsps:
        # intentionally wrong data for test and test_2
        rsps.add(
            responses.POST,
            f"{local_host_addr}/midpoints",
            json={"test": "0.6", "test_2": "0.6", "test_3": "0.3"},
        )

        total = pm.total({"test": 0.5, "test_2": book_test_2}, None)
        assert total == 19.6


def test_clean():
    pos_usdc = CSMPosition.create("USDC", 10, 5)
    pos_1 = CSMPosition.create("test_1", 12)
    pos_2 = CSMPosition.create("test_2", 6)
    pos_3 = CSMPosition.create("test_3", 4)
    pm = PositionManager(None, None, pos_usdc, position_factory=CSMPosition)
    pm.track(pos_1)
    pm.track(pos_2)
    pm.track(pos_3)

    pos_usdc.size = 0
    pos_1.size = 0
    pos_2.act(6, "trade_id", ACT_SIDE.MAKER, TRADE_STATUS.MATCHED)
    assert list(pm.asset_ids) == [USDC, "test_1", "test_2", "test_3"]
    assert pos_2.size != 0

    pos_c = pm.clean()
    assert list(pm.asset_ids) == [USDC, "test_2", "test_3"]
    assert pos_c[0].asset_id == "test_1"

    pos_2.act(6, "trade_id", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    pos_3.act(4, "trade_id", ACT_SIDE.MAKER, TRADE_STATUS.CONFIRMED)
    pos_c = pm.clean()
    assert list(pm.asset_ids) == [USDC]
    assert [p.asset_id for p in pos_c] == ["test_2", "test_3"]

    pos_c = pm.clean()
    assert pos_c == []
    assert list(pm.asset_ids) == [USDC]


def test_deposit_withdraw():
    pm = PositionManager(None, None, 123.45)

    # prone to floating point error
    pm.deposit(0.01)
    assert pm.balance == 123.46

    pm.withdraw(0.2)
    assert pm.balance == 123.26


def test_raise_untrack_usdc():
    pm = PositionManager(None, None, 10)

    with pytest.raises(PositionTrackingException) as record:
        pm.untrack(USDC)
    assert "USDC" in str(record)


@pytest.mark.skip(
    reason="This is already implicitly tested above and in test_position.py."
)
def test_balances():
    ...


# noinspection DuplicatedCode
def test_transact():  # sourcery skip: extract-duplicate-method
    pm = PositionManager(None, None, 10, position_factory=CSMPosition)
    pm.track(CSMPosition.create("test_2", 5))

    pm.transact("test", 1, 0.3, "trade_id", SIDE.BUY, TRADE_STATUS.MATCHED, True)
    assert pm.balance == 10
    assert pm.balance_available == 9.7
    assert pm.get_by_id(USDC).pending_maker == -0.3
    assert pm.get_by_id(USDC).pending_trade_ids.keys() == {"trade_id"}
    assert list(pm.asset_ids) == [USDC, "test_2", "test"]
    assert pm.get_by_id("test").size == 0
    assert pm.get_by_id("test").pending_taker == 1
    assert pm.get_by_id("test").pending_trade_ids.keys() == {"trade_id"}

    pm.transact("test_2", 2, 0.5, "trade_id_2", SIDE.SELL, TRADE_STATUS.MATCHED, True)
    assert pm.balance == 10
    assert pm.balance_available == 9.7
    assert pm.get_by_id(USDC).pending_maker == -0.3
    assert pm.get_by_id(USDC).pending_taker == 1
    assert pm.get_by_id(USDC).pending_trade_ids.keys() == {"trade_id", "trade_id_2"}
    assert list(pm.asset_ids) == [USDC, "test_2", "test"]
    assert pm.get_by_id("test").size == 0
    assert pm.get_by_id("test").pending_taker == 1
    assert pm.get_by_id("test").pending_maker == 0
    assert pm.get_by_id("test").pending_trade_ids.keys() == {"trade_id"}
    assert pm.get_by_id("test_2").size == 5
    assert pm.get_by_id("test_2").pending_taker == 0
    assert pm.get_by_id("test_2").pending_maker == -2
    assert pm.get_by_id("test_2").pending_trade_ids.keys() == {"trade_id_2"}
    assert pm.get_by_id("test_2").size_available == 3

    pm.transact("test_2", 2, 0.5, "trade_id_2", SIDE.SELL, TRADE_STATUS.CONFIRMED, True)
    assert pm.balance == 11
    assert pm.balance_available == 10.7
    assert pm.get_by_id(USDC).pending_maker == -0.3
    assert pm.get_by_id(USDC).pending_taker == 0
    assert pm.get_by_id(USDC).pending_trade_ids.keys() == {"trade_id"}
    assert list(pm.asset_ids) == [USDC, "test_2", "test"]
    assert pm.get_by_id("test").size == 0
    assert pm.get_by_id("test").pending_taker == 1
    assert pm.get_by_id("test").pending_maker == 0
    assert pm.get_by_id("test").pending_trade_ids.keys() == {"trade_id"}
    assert pm.get_by_id("test_2").size == 3
    assert pm.get_by_id("test_2").pending_taker == 0
    assert pm.get_by_id("test_2").pending_maker == 0
    assert len(pm.get_by_id("test_2").pending_trade_ids.keys()) == 0
    assert pm.get_by_id("test_2").size_available == 3

    pm.transact("test", 1, 0.3, "trade_id", SIDE.BUY, TRADE_STATUS.CONFIRMED, True)
    assert pm.balance == 10.7
    assert pm.balance_available == 10.7
    assert pm.get_by_id(USDC).pending_maker == 0
    assert pm.get_by_id(USDC).pending_taker == 0
    assert len(pm.get_by_id(USDC).pending_trade_ids.keys()) == 0
    assert list(pm.asset_ids) == [USDC, "test_2", "test"]
    assert pm.get_by_id("test").size == 1
    assert pm.get_by_id("test").pending_taker == 0
    assert pm.get_by_id("test").pending_maker == 0
    assert len(pm.get_by_id("test").pending_trade_ids.keys()) == 0
    assert pm.get_by_id("test_2").size == 3
    assert pm.get_by_id("test_2").pending_taker == 0
    assert pm.get_by_id("test_2").pending_maker == 0
    assert len(pm.get_by_id("test_2").pending_trade_ids.keys()) == 0
    assert pm.get_by_id("test_2").size_available == 3


def test_raise_transact():
    pm = PositionManager(None, None, 10)

    with pytest.raises(PositionTrackingException) as record:
        pm.transact("test", 1, 0.3, "trade_id", SIDE.BUY, TRADE_STATUS.CONFIRMED, False)
    assert "not found" in str(record)

    with pytest.raises(PositionTransactionException) as record:
        pm.transact(USDC, 1, 0.3, "trade_id", SIDE.BUY, TRADE_STATUS.CONFIRMED, False)
    assert "USDC" in str(record)

    assert pm.balance == 10
    assert list(pm.asset_ids) == [USDC]


def test_decimal():
    # total, transact, balances, deposit, withdraw
    pm = PositionManager(None, None, Decimal("10"), position_factory=CSMPosition)
    pm.deposit(Decimal("5"))
    assert pm.balance == Decimal("15")

    pm.transact(
        "test",
        Decimal("1"),
        Decimal("0.3"),
        "trade_id",
        SIDE.BUY,
        TRADE_STATUS.MATCHED,
        True,
    )
    assert pm.balance == Decimal("15")
    assert pm.balance_available == Decimal("14.7")
    assert pm.get_by_id("test").size == Decimal("0")
    assert list(pm.asset_ids) == [USDC, "test"]

    pm.transact(
        "test",
        Decimal("1"),
        Decimal("0.3"),
        "trade_id",
        SIDE.BUY,
        TRADE_STATUS.CONFIRMED,
        True,
    )
    assert pm.balance == Decimal("14.7")
    assert pm.get_by_id("test").size == Decimal("1")
    assert list(pm.asset_ids) == [USDC, "test"]

    pm.withdraw(Decimal("0.7"))
    assert pm.balance == Decimal("14")
    assert pm.get_by_id("test").size == Decimal("1")
    assert list(pm.asset_ids) == [USDC, "test"]

    assert pm.total({"test": Decimal("0.5")}, None) == Decimal("14.5")


def test_invalidate():
    pm = PositionManager(None, None, 10)

    pm.invalidate()

    with pytest.raises(ManagerInvalidException) as record:
        pm.create_position("1234", 10)
    assert "invalid" in str(record)
    assert list(pm.asset_ids) == [USDC]

    with pytest.raises(ManagerInvalidException) as record:
        _ = pm.balance_total
    assert "invalid" in str(record)
    assert list(pm.asset_ids) == [USDC]

    pm._invalid_token = False
    pm.create_position("1234", 10)
    assert list(pm.asset_ids) == [USDC, "1234"]
    assert pm.balance_total == 10

    pm.invalidate("pencil broke")
    with pytest.raises(ManagerInvalidException) as record:
        pm.untrack("1234")
    assert "invalid" in str(record)
    assert "pencil broke" in str(record)
    assert list(pm.asset_ids) == [USDC, "1234"]

    with pytest.raises(ManagerInvalidException) as record:
        pm.transact("1234", 1, 0.5, "1234123", SIDE.BUY, TRADE_STATUS.MATCHED, False)
    assert "invalid" in str(record)
    assert list(pm.asset_ids) == [USDC, "1234"]

    with pytest.raises(ManagerInvalidException) as record:
        _ = pm.balance_total
    assert "invalid" in str(record)
    assert list(pm.asset_ids) == [USDC, "1234"]

    pm._invalid_token = False
    assert pm.balance_total == 10
    assert pm.get_by_id("1234").size == 10
