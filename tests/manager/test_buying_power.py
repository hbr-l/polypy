from decimal import Decimal

import pytest

from polypy.constants import CHAIN_ID
from polypy.manager import OrderManager, PositionManager
from polypy.order import INSERT_STATUS, SIDE, create_limit_order
from polypy.position import CSMPosition
from polypy.signing import SIGNATURE_TYPE
from polypy.trade import TRADE_STATUS


@pytest.fixture
def sample_order(private_key):
    def _closure(order_id, token_id, price, size, side, signature="signature"):
        order = create_limit_order(
            price,
            size,
            token_id,
            side,
            0.01,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            order_id=order_id,
            signature=signature,
        )
        return order

    return _closure


# noinspection DuplicatedCode
def test_buying_power(
    local_host_addr, private_key, secret, api_key, passphrase, sample_order
):
    pm = PositionManager(local_host_addr, None, 100)
    om = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        None,
        CHAIN_ID.POLYGON,
    )

    assert pm.buying_power(om) == 100

    pm.transact("1234", 10, 0.5, "test1", SIDE.BUY, TRADE_STATUS.MATCHED, True)
    pm.transact("2345", 12, 0.2, "test2", SIDE.BUY, TRADE_STATUS.MATCHED, True)
    assert pm.buying_power(om) == 92.6

    om.track(
        sample_order("1111", "1234", 0.3, 10, SIDE.BUY),
        False,
    )
    om.track(
        sample_order("2222", "4567", 0.3, 12, SIDE.BUY),
        False,
    )
    om.track(
        sample_order("3333", "1234", 0.3, 10, SIDE.SELL),
        False,
    )
    om.track(
        sample_order("4444", "5678", 0.5, 10, SIDE.SELL),
        False,
    )
    assert pm.buying_power(om) == 92.6

    om.update("1111", status=INSERT_STATUS.LIVE)
    om.update("3333", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == 89.6

    om.update("2222", status=INSERT_STATUS.LIVE)
    om.update("4444", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == 86

    pm.transact("1234", 10, 0.5, "test1", SIDE.BUY, TRADE_STATUS.CONFIRMED, True)
    pm.transact("2345", 12, 0.2, "test2", SIDE.BUY, TRADE_STATUS.CONFIRMED, True)
    assert pm.buying_power(om) == 86

    om.update("1111", status=INSERT_STATUS.MATCHED)
    om.update("3333", status=INSERT_STATUS.MATCHED)
    assert pm.buying_power(om) == 89


# noinspection DuplicatedCode
def test_buying_power_csm_position(
    local_host_addr, private_key, secret, api_key, passphrase, sample_order
):
    pm = PositionManager(local_host_addr, None, 100, position_factory=CSMPosition)
    om = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        None,
        CHAIN_ID.POLYGON,
    )

    assert pm.buying_power(om) == 100

    pm.transact("1234", 10, 0.5, "test1", SIDE.BUY, TRADE_STATUS.MATCHED, True)
    pm.transact("2345", 12, 0.2, "test2", SIDE.BUY, TRADE_STATUS.MATCHED, True)
    pm.transact("3456", 5, 0.1, "test3", SIDE.SELL, TRADE_STATUS.MATCHED, True)
    assert pm.buying_power(om) == 92.6

    om.track(
        sample_order("1111", "1234", 0.3, 10, SIDE.BUY),
        False,
    )
    om.track(
        sample_order("2222", "4567", 0.3, 12, SIDE.BUY),
        False,
    )
    om.track(
        sample_order("3333", "1234", 0.3, 10, SIDE.SELL),
        False,
    )
    om.track(
        sample_order("4444", "5678", 0.5, 10, SIDE.SELL),
        False,
    )
    assert pm.buying_power(om) == 92.6

    om.update("1111", status=INSERT_STATUS.LIVE)
    om.update("3333", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == 89.6

    om.update("2222", status=INSERT_STATUS.LIVE)
    om.update("4444", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == 86

    pm.transact("1234", 10, 0.5, "test1", SIDE.BUY, TRADE_STATUS.CONFIRMED, True)
    pm.transact("2345", 12, 0.2, "test2", SIDE.BUY, TRADE_STATUS.CONFIRMED, True)
    assert pm.buying_power(om) == 86

    om.update("1111", status=INSERT_STATUS.MATCHED)
    om.update("3333", status=INSERT_STATUS.MATCHED)
    assert pm.buying_power(om) == 89


# noinspection DuplicatedCode
def test_buying_power_decimal(
    local_host_addr, private_key, secret, api_key, passphrase, sample_order
):
    pm = PositionManager(local_host_addr, None, Decimal("100"))
    om = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        None,
        CHAIN_ID.POLYGON,
    )

    assert pm.buying_power(om) == Decimal("100")

    pm.transact(
        "1234",
        Decimal("10"),
        Decimal("0.5"),
        "test1",
        SIDE.BUY,
        TRADE_STATUS.MATCHED,
        True,
    )
    pm.transact(
        "2345",
        Decimal("12"),
        Decimal("0.2"),
        "test2",
        SIDE.BUY,
        TRADE_STATUS.MATCHED,
        True,
    )
    assert pm.buying_power(om) == Decimal("92.6")

    om.track(
        sample_order("1111", "1234", Decimal("0.3"), Decimal("10"), SIDE.BUY),
        False,
    )
    om.track(
        sample_order("2222", "4567", Decimal("0.3"), Decimal("12"), SIDE.BUY),
        False,
    )
    om.track(
        sample_order("3333", "1234", Decimal("0.3"), Decimal("10"), SIDE.SELL),
        False,
    )
    om.track(
        sample_order("4444", "5678", Decimal("0.5"), Decimal("10"), SIDE.SELL),
        False,
    )
    assert pm.buying_power(om) == Decimal("92.6")

    om.update("1111", status=INSERT_STATUS.LIVE)
    om.update("3333", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == Decimal("89.6")

    om.update("2222", status=INSERT_STATUS.LIVE)
    om.update("4444", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == Decimal("86")

    pm.transact(
        "1234",
        Decimal("10"),
        Decimal("0.5"),
        "test1",
        SIDE.BUY,
        TRADE_STATUS.CONFIRMED,
        True,
    )
    pm.transact(
        "2345",
        Decimal("12"),
        Decimal("0.2"),
        "test2",
        SIDE.BUY,
        TRADE_STATUS.CONFIRMED,
        True,
    )
    assert pm.buying_power(om) == Decimal("86")

    om.update("1111", status=INSERT_STATUS.MATCHED)
    om.update("3333", status=INSERT_STATUS.MATCHED)
    assert pm.buying_power(om) == Decimal("89")


# noinspection DuplicatedCode
def test_buying_power_decimal_csm_position(
    local_host_addr, private_key, secret, api_key, passphrase, sample_order
):
    pm = PositionManager(
        local_host_addr, None, Decimal("100"), position_factory=CSMPosition
    )
    om = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        None,
        CHAIN_ID.POLYGON,
    )

    assert pm.buying_power(om) == Decimal("100")

    pm.transact(
        "1234",
        Decimal("10"),
        Decimal("0.5"),
        "test1",
        SIDE.BUY,
        TRADE_STATUS.MATCHED,
        True,
    )
    pm.transact(
        "2345",
        Decimal("12"),
        Decimal("0.2"),
        "test2",
        SIDE.BUY,
        TRADE_STATUS.MATCHED,
        True,
    )
    pm.transact(
        "3456",
        Decimal("5"),
        Decimal("0.1"),
        "test3",
        SIDE.SELL,
        TRADE_STATUS.MATCHED,
        True,
    )
    assert pm.buying_power(om) == Decimal("92.6")

    om.track(
        sample_order("1111", "1234", Decimal("0.3"), Decimal("10"), SIDE.BUY),
        False,
    )
    om.track(
        sample_order("2222", "4567", Decimal("0.3"), Decimal("12"), SIDE.BUY),
        False,
    )
    om.track(
        sample_order("3333", "1234", Decimal("0.3"), Decimal("10"), SIDE.SELL),
        False,
    )
    om.track(
        sample_order("4444", "5678", Decimal("0.5"), Decimal("10"), SIDE.SELL),
        False,
    )
    assert pm.buying_power(om) == Decimal("92.6")

    om.update("1111", status=INSERT_STATUS.LIVE)
    om.update("3333", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == Decimal("89.6")

    om.update("2222", status=INSERT_STATUS.LIVE)
    om.update("4444", status=INSERT_STATUS.LIVE)
    assert pm.buying_power(om) == Decimal("86")

    pm.transact(
        "1234",
        Decimal("10"),
        Decimal("0.5"),
        "test1",
        SIDE.BUY,
        TRADE_STATUS.CONFIRMED,
        True,
    )
    pm.transact(
        "2345",
        Decimal("12"),
        Decimal("0.2"),
        "test2",
        SIDE.BUY,
        TRADE_STATUS.CONFIRMED,
        True,
    )
    assert pm.buying_power(om) == Decimal("86")

    om.update("1111", status=INSERT_STATUS.MATCHED)
    om.update("3333", status=INSERT_STATUS.MATCHED)
    assert pm.buying_power(om) == Decimal("89")
