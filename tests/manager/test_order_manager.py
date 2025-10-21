import math
from collections import namedtuple
from decimal import Decimal

import pytest
import responses

from polypy.book import OrderBook
from polypy.constants import CHAIN_ID
from polypy.exceptions import (
    ManagerInvalidException,
    OrderCreationException,
    OrderGetException,
    OrderPlacementFailure,
    OrderPlacementUnmatched,
    OrderTrackingException,
    OrderUpdateException,
)
from polypy.manager.order import OrderManager
from polypy.order import (
    INSERT_STATUS,
    SIDE,
    TIME_IN_FORCE,
    OrderProtocol,
    create_market_order,
    frozen_order,
)
from polypy.signing import SIGNATURE_TYPE

MarketInfo = namedtuple("MarketInfo", ["condition_id", "tokens"])
TokenInfo = namedtuple("TokenInfo", ["token_id", "outcome"])

ASSET_ID_YES = "1111"
ASSET_ID_NO = "0000"
MARKET_IDX = MarketInfo(
    "market_id", [TokenInfo(ASSET_ID_YES, "Yes"), TokenInfo(ASSET_ID_NO, "No")]
)
STRATEGY_ID = "strategy"

# todo add more tests w.r.t. server-side errors?


@pytest.fixture
def order_manager(
    local_host_addr, private_key, api_key, passphrase, secret
) -> OrderManager:
    return OrderManager(
        rest_endpoint=local_host_addr,
        private_key=private_key,
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        maker_funder=None,
        signature_type=None,
        chain_id=CHAIN_ID.POLYGON,
    )


@pytest.fixture
def sample_order(private_key):
    def _closure(
        order_id: str,
        strategy_id: str = STRATEGY_ID,
        token_id: int | str | None = None,
        signature: str | None = "some_signature",
    ) -> OrderProtocol:
        return create_market_order(
            21.345,
            ASSET_ID_YES if token_id is None else token_id,
            SIDE.BUY,
            0.01,
            False,
            CHAIN_ID.POLYGON,
            private_key,
            None,
            SIGNATURE_TYPE.EOA,
            strategy_id=strategy_id,
            order_id=order_id,
            signature=signature,
        )

    return _closure


@pytest.fixture
def mock_get_order(rsps, local_host_addr, api_key):
    def _closure(
        order_id: str,
        order_status: INSERT_STATUS,
        original_size: float | str,
        price: float | str,
        side: SIDE,
        size_matched: float | str,
        status=200,
    ):
        data = {
            "associate_trades": [],
            "id": order_id,
            "market": MARKET_IDX.condition_id,
            "original_size": str(original_size),
            "outcome": "Yes",
            "maker_address": "maker_address",
            "owner": api_key,
            "price": str(price),
            "status": str(order_status),
            "side": str(side),
            "size_matched": str(size_matched),
            "asset_id": ASSET_ID_YES,
            "expiration": 0,
            "order_type": str(TIME_IN_FORCE.GTC),
            "created_at": 100000,
        }
        rsps.upsert(
            responses.GET,
            f"{local_host_addr}/data/order/{order_id}",
            status=status,
            json=data,
        )

    return _closure


@pytest.fixture
def mock_get_order_none(rsps, local_host_addr):
    def _closure(order_id: str, status=200):
        rsps.upsert(
            responses.GET,
            f"{local_host_addr}/data/order/{order_id}",
            status=status,
            json=None,
        )

    return _closure


def test_track_no_sync(order_manager, sample_order):
    # sourcery skip: extract-duplicate-method
    assert len(order_manager.order_dict) == 0
    assert len(order_manager.token_dict) == 0

    order_manager.track(sample_order("some"), False)
    assert len(order_manager.order_dict) == 1
    assert len(order_manager.token_dict) == 1

    order_manager.track(sample_order("some"), False)
    assert len(order_manager.order_dict) == 1
    assert len(order_manager.token_dict) == 1

    order = sample_order("one", "strat")
    order_manager.track(order, False)
    assert len(order_manager.order_dict) == 2
    assert len(order_manager.token_dict) == 1

    assert list(order_manager.order_ids) == ["some", "one"]
    assert order_manager.token_dict == {ASSET_ID_YES: 2}


def test_raise_track_no_sync(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)

    # cannot track order without ID
    with pytest.raises(OrderTrackingException):
        # noinspection PyTypeChecker
        order = sample_order(None)
        object.__setattr__(order, "id", None)
        order_manager.track(order, False)

    # cannot track frozen order
    with pytest.raises(OrderTrackingException):
        # noinspection PyTypeChecker
        order_manager.track(frozen_order(sample_order("some")), False)

    assert len(order_manager.order_dict) == 1
    assert list(order_manager.order_ids) == ["some"]


def test_raise_track_on_sync(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)

    with pytest.raises(OrderUpdateException):
        order_manager.track(sample_order("some_order"), True)

    assert len(order_manager.order_dict) == 1
    assert list(order_manager.order_ids) == ["some"]


def test_untrack_no_sync(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)
    order_manager.track(sample_order("one"), False)

    ret_order = order_manager.untrack("one", False)
    assert ret_order.id == "one"
    assert list(order_manager.order_ids) == ["some"]
    ret_order.strategy_id = "some strategy"  # test for frozen

    ret_order = order_manager.untrack("arbitrary", False)
    assert ret_order is None
    assert list(order_manager.order_ids) == ["some"]


def test_raise_untrack_on_sync(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)
    order_manager.track(sample_order("one"), False)

    with pytest.raises(OrderUpdateException):
        order_manager.untrack("one", True)

    assert list(order_manager.order_ids) == ["some"]


def test_raise_track_changed_token_id(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)
    order_manager.track(sample_order("some"), False)
    assert order_manager.token_dict == {ASSET_ID_YES: 1}
    assert list(order_manager.order_ids) == ["some"]

    with pytest.raises(OrderTrackingException) as record:
        order_manager.track(sample_order("some", token_id="12345567"), False)
    assert "token_id" in str(record)

    assert order_manager.get_by_id("some").token_id == ASSET_ID_YES


def test_token_id(order_manager, sample_order):
    # sourcery skip: extract-duplicate-method
    order_manager.track(sample_order("some"), False)
    order_manager.track(sample_order("one"), False)
    assert list(order_manager.token_ids) == [ASSET_ID_YES]

    order_manager.track(sample_order("some"), False)
    assert order_manager.token_dict == {ASSET_ID_YES: 2}

    order_manager.track(sample_order("more", token_id="91823451"), False)
    order_manager.track(sample_order("more", token_id="91823451"), False)
    order_manager.track(sample_order("thing", token_id="91823451"), False)
    assert list(order_manager.token_ids) == [ASSET_ID_YES, "91823451"]
    assert order_manager.token_dict == {ASSET_ID_YES: 2, "91823451": 2}

    order_manager.untrack("some", False)
    order_manager.untrack("more", False)
    assert order_manager.token_dict == {ASSET_ID_YES: 1, "91823451": 1}

    order_manager.untrack("some", False)
    assert order_manager.token_dict == {ASSET_ID_YES: 1, "91823451": 1}

    order_manager.untrack("one", False)
    assert order_manager.token_dict == {"91823451": 1}
    assert list(order_manager.token_ids) == ["91823451"]

    order_manager.untrack("one", False)
    order_manager.untrack("some", False)
    assert order_manager.token_dict == {"91823451": 1}
    assert list(order_manager.token_ids) == ["91823451"]

    order_manager.untrack("thing", False)
    assert not list(order_manager.token_ids)
    assert order_manager.token_dict == {}


def test_retrieve_by_id(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)
    order_manager.track(sample_order("one"), False)

    order = order_manager.get_by_id("some")
    assert order.id == "some"

    order = order_manager.get_by_id("one")
    assert order.id == "one"

    order = order_manager.get_by_id("other")
    assert order is None


def test_retrieve(order_manager, sample_order):
    # sourcery skip: extract-duplicate-method
    order_manager.track(sample_order("some"), False)
    order = sample_order("one", "strat")
    order_manager.track(order, False)

    # multiple (single match)
    orders = order_manager.get(signature_type=SIGNATURE_TYPE.EOA)
    assert [x.id for x in orders] == list(order_manager.order_ids)

    # specific (single match)
    orders = order_manager.get(strategy_id="strat")
    assert len(orders) == 1
    assert orders[0].id == "one"
    with pytest.raises(OrderUpdateException):
        orders[0].status = INSERT_STATUS.LIVE

    # specific (multiple matches)
    orders = order_manager.get(id="some", signature_type=SIGNATURE_TYPE.EOA)
    assert len(orders) == 1
    assert orders[0].id == "some"
    with pytest.raises(OrderUpdateException):
        orders[0].status = INSERT_STATUS.LIVE

    # no match (single match)
    orders = order_manager.get(maker="abcdefghjiklmn")
    assert not orders

    # no match (multiple)
    orders = order_manager.get(maker="abcdefghjiklmn", expiration=1)
    assert not orders

    # no match (partial match)
    orders = order_manager.get(
        maker="abcdefghjiklmn", signature_type=SIGNATURE_TYPE.EOA
    )
    assert not orders

    # non existent attribute
    orders = order_manager.get(some_non_existent_attribute=123)
    assert not orders

    # mix non existent attribute and existent attribute
    orders = order_manager.get(
        some_non_existent_attribute=123, signature_type=SIGNATURE_TYPE.EOA
    )
    assert not orders

    # retrieve all
    orders = order_manager.get()
    assert [x.id for x in orders] == list(order_manager.order_ids)


def test_update(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)
    order = sample_order("one", "strat")
    order_manager.track(order, False)

    # change status: should be fine
    order_manager.update(order_id="some", status=INSERT_STATUS.LIVE)
    assert order_manager.get_by_id("some").status == INSERT_STATUS.LIVE
    assert order_manager.get_by_id("one").status == INSERT_STATUS.DEFINED

    # order non existent
    assert "other" not in list(order_manager.order_ids)
    with pytest.raises(OrderGetException):
        order_manager.update(order_id="other")

    # frozen attribute
    orig_signature = order_manager.get_by_id("one").signature
    with pytest.raises(OrderUpdateException) as record:
        order_manager.update(order_id="one", signature="1234")
    assert "signature" in str(record)
    assert order_manager.get_by_id(order_id="one").signature != "123"
    assert order_manager.get_by_id(order_id="one").signature == orig_signature

    # partially frozen attr (frozen after setting it for the first time)
    order_manager.update(order_id="one", created_at=1)
    assert order_manager.get_by_id("one").created_at == 1
    with pytest.warns() as record:
        order_manager.update(order_id="one", created_at=2)
    assert "created_at" in str(record[0].message)
    assert "Ignoring" in str(record[0].message)
    assert order_manager.get_by_id("one").created_at == 1

    # strategy_id can be changed
    order_manager.update(order_id="one", strategy_id="one")
    assert order_manager.get_by_id("one").strategy_id == "one"

    # regression will be ignored
    order_manager.update("some", status=INSERT_STATUS.MATCHED)
    with pytest.warns() as record:
        order_manager.update("some", status=INSERT_STATUS.DEFINED)
    assert "status" in str(record[0].message)
    assert "Ignoring" in str(record[0].message)
    assert order_manager.get_by_id("some").status is INSERT_STATUS.MATCHED


def test_modify(order_manager, sample_order):
    order_manager.track(sample_order("some"), False)
    order = sample_order("one", "strat")
    order_manager.track(order, False)

    # change status: should be fine
    order_manager.modify(order_id="some", status=INSERT_STATUS.LIVE)
    assert order_manager.get_by_id("some").status == INSERT_STATUS.LIVE
    assert order_manager.get_by_id("one").status == INSERT_STATUS.DEFINED

    # order non existent
    assert "other" not in list(order_manager.order_ids)
    with pytest.raises(OrderGetException):
        order_manager.modify(order_id="other")

    # frozen attribute
    orig_signature = order_manager.get_by_id("one").signature
    with pytest.raises(OrderUpdateException) as record:
        order_manager.modify(order_id="one", signature="1234")
    assert "signature" in str(record)
    assert order_manager.get_by_id(order_id="one").signature != "123"
    assert order_manager.get_by_id(order_id="one").signature == orig_signature

    # partially frozen attr (frozen after setting it for the first time)
    order_manager.modify(order_id="one", created_at=1)
    assert order_manager.get_by_id("one").created_at == 1
    with pytest.raises(OrderUpdateException):
        order_manager.modify(order_id="one", created_at=2)
    assert order_manager.get_by_id("one").created_at == 1

    # strategy_id can be changed
    order_manager.modify(order_id="one", strategy_id="one")
    assert order_manager.get_by_id("one").strategy_id == "one"

    # change size_matched
    order_manager.modify(order_id="one", size_matched=100)
    assert order_manager.get_by_id("one").size_matched == 100
    # regress
    order_manager.modify(order_id="one", size_matched=10)
    assert order_manager.get_by_id("one").size_matched == 10
    # auto-cast by order
    order_manager.modify(order_id="one", size_matched="12")
    assert order_manager.get_by_id("one").size_matched == 12


def test_order_id_response_raises_not_matching_id(
    order_manager,
    mock_get_order,
    mock_post_order,
    mock_tick_size,
    mock_neg_risk,
):
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_neg_risk(ASSET_ID_YES)
    mock_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0x0000000000000000000000000000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "0.5",
            "makingAmount": "0.25",
        },
    )

    with pytest.raises(OrderPlacementFailure) as record:
        order_manager.limit_order(
            Decimal("0.3"),
            Decimal(10),
            ASSET_ID_YES,
            SIDE.BUY,
            None,
            TIME_IN_FORCE.GTC,
            None,
            neg_risk=None,
        )
    assert "does not match" in str(record)


# noinspection DuplicatedCode
def test_market_order(
    order_manager,
    local_host_addr,
    mock_neg_risk,
    mock_tick_size,
    mock_post_order,
):
    mock_neg_risk(ASSET_ID_YES)
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "1234",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.MATCHED),
            "takingAmount": "100",  # receive size
            "makingAmount": "50",  # spent amount
        },
    )

    order_manager.market_order(
        100,
        ASSET_ID_YES,
        SIDE.BUY,
        None,
        TIME_IN_FORCE.FOK,
        None,
        None,
        strategy_id="strat",
        neg_risk=None,
        order_id="1234",
        signature="signature",
    )
    assert list(order_manager.order_ids) == ["1234"]
    assert order_manager.get_by_id("1234").id == "1234"
    assert order_manager.get(side=SIDE.BUY)[0].status == INSERT_STATUS.MATCHED
    assert order_manager.get_by_id("1234").size_matched == 100

    mock_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "2345",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.UNMATCHED),
            "takingAmount": "",
            "makingAmount": "",
        },
    )
    with pytest.raises(OrderPlacementUnmatched) as e:
        order_manager.market_order(
            20,
            ASSET_ID_YES,
            SIDE.SELL,
            None,
            TIME_IN_FORCE.FOK,
            None,
            math.inf,
            strategy_id="tactic",
            neg_risk=None,
            order_id="2345",
            signature="signature",
        )
    assert e.value.order.id == "2345"
    assert list(order_manager.order_ids) == ["1234", "2345"]
    assert order_manager.get_by_id("2345").status == INSERT_STATUS.UNMATCHED
    assert order_manager.get_by_id("2345").strategy_id == "tactic"
    assert order_manager.get_by_id("1234").strategy_id == "strat"
    assert order_manager.get_by_id("2345").side is SIDE.SELL
    assert order_manager.get_by_id("1234").side is SIDE.BUY

    mock_post_order({"error": "order 2345 is invalid. Duplicated."}, 400)
    with pytest.raises(OrderPlacementFailure) as e:
        order_manager.market_order(
            20,
            ASSET_ID_YES,
            SIDE.SELL,
            None,
            TIME_IN_FORCE.FOK,
            None,
            math.inf,
            neg_risk=None,
            order_id="3456",
            signature="signature",
        )
    assert e.value.order.id == "3456"
    assert list(order_manager.order_ids) == ["1234", "2345", "3456"]


def test_market_order_raise_book(
    order_manager, local_host_addr, mock_neg_risk, mock_tick_size
):
    mock_neg_risk(ASSET_ID_YES)
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_neg_risk("9876543212")
    mock_tick_size("9876543212", 0.01)

    # book.token_id does not match
    book = OrderBook("9876543212", 0.01)
    with pytest.raises(OrderCreationException) as record:
        order_manager.market_order(
            100,
            ASSET_ID_YES,
            SIDE.SELL,
            None,
            TIME_IN_FORCE.FOK,
            book,
            math.inf,
            neg_risk=None,
        )
    assert "book.token_id" in str(record)

    assert not list(order_manager.order_ids)


# noinspection DuplicatedCode
def test_limit_order(
    order_manager,
    local_host_addr,
    mock_neg_risk,
    mock_tick_size,
    mock_post_order,
):
    mock_neg_risk(ASSET_ID_YES)
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "1234",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "1",
            "makingAmount": "0.5",
        }
    )

    order_manager.limit_order(
        0.5,
        10,
        ASSET_ID_YES,
        SIDE.BUY,
        None,
        TIME_IN_FORCE.GTC,
        None,
        neg_risk=None,
        strategy_id="strat",
        order_id="1234",
        signature="signature",
    )
    assert list(order_manager.order_ids) == ["1234"]
    assert order_manager.get_by_id("1234").id == "1234"
    assert order_manager.get(side=SIDE.BUY)[0].status == INSERT_STATUS.LIVE
    assert order_manager.get_by_id("1234").size_matched == 1

    mock_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "2345",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.UNMATCHED),
            "takingAmount": "",
            "makingAmount": "",
        },
    )
    with pytest.raises(OrderPlacementUnmatched) as e:
        order_manager.limit_order(
            0.5,
            20,
            ASSET_ID_YES,
            SIDE.SELL,
            None,
            TIME_IN_FORCE.GTC,
            None,
            neg_risk=None,
            strategy_id="tactic",
            order_id="2345",
            signature="signature",
        )
    assert e.value.order.id == "2345"
    assert list(order_manager.order_ids) == ["1234", "2345"]
    assert order_manager.get_by_id("2345").status == INSERT_STATUS.UNMATCHED
    assert order_manager.get_by_id("2345").strategy_id == "tactic"
    assert order_manager.get_by_id("1234").strategy_id == "strat"
    assert order_manager.get_by_id("2345").side is SIDE.SELL
    assert order_manager.get_by_id("1234").side is SIDE.BUY
    assert order_manager.get_by_id("2345").size == 20
    assert order_manager.get_by_id("1234").size == 10

    mock_post_order({"error": "order 2345 is invalid. Duplicated."}, 400)
    with pytest.raises(OrderPlacementFailure) as e:
        order_manager.limit_order(
            0.2,
            50,
            ASSET_ID_YES,
            SIDE.SELL,
            None,
            TIME_IN_FORCE.GTC,
            None,
            neg_risk=None,
            order_id="3456",
            signature="signature",
        )
    assert e.value.order.id == "3456"
    assert list(order_manager.order_ids) == ["1234", "2345", "3456"]


@pytest.mark.skip(reason="Implicitly included in market order and limit order.")
def test_post():
    ...


def test_cancel(
    order_manager,
    sample_order,
    mock_tick_size,
    mock_neg_risk,
    private_key,
    mock_cancel_order,
):  # sourcery skip: extract-duplicate-method
    # todo split in multiple tests
    # input: order object, frozen order, str, list vs single
    # edge cases: order not in order manager, order not canceled
    # mode
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_neg_risk(ASSET_ID_YES)

    # --- order object ---
    # not yet tracked: track success
    order = sample_order("1234")
    order.status = INSERT_STATUS.LIVE
    mock_cancel_order(["1234"], None)
    order_manager.cancel(order)
    assert list(order_manager.order_ids) == ["1234"]
    assert order_manager.get_by_id("1234").status is INSERT_STATUS.CANCELED

    # not yet tracked: track fail
    order = sample_order("2345", strategy_id="some")
    order.status = INSERT_STATUS.LIVE
    order = frozen_order(order)
    with pytest.raises(OrderTrackingException) as record:
        order_manager.cancel(order)
    assert "Not all orders trackable" in str(record)
    assert list(order_manager.order_ids) == ["1234"]
    assert order.status is INSERT_STATUS.LIVE

    # tracked
    order = sample_order("2345")
    order.status = INSERT_STATUS.LIVE
    order_manager.track(order, False)
    mock_cancel_order(["2345"], None)
    order_manager.cancel(order)
    assert list(order_manager.order_ids) == ["1234", "2345"]
    assert order_manager.get_by_id("1234").status is INSERT_STATUS.CANCELED
    assert order_manager.get_by_id("2345").status is INSERT_STATUS.CANCELED

    # cancel fail
    order = sample_order("3456")
    order.status = INSERT_STATUS.LIVE
    order_manager.track(order, False)
    mock_cancel_order([], {"3456": "some reason"})
    with pytest.raises(OrderUpdateException) as record:
        order_manager.cancel(order)
    assert "Cancellation" in str(record)
    assert list(order_manager.order_ids) == ["1234", "2345", "3456"]
    assert order_manager.get_by_id("3456").status is INSERT_STATUS.LIVE

    # --- list order object ---
    # not yet tracked: track success
    orders = [sample_order("4567"), sample_order("5678")]
    for x in orders:
        x.status = INSERT_STATUS.LIVE
    mock_cancel_order(["4567", "5678"], None)
    order_manager.cancel(orders)
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
    ]
    assert order_manager.get_by_id("1234").status is INSERT_STATUS.CANCELED
    assert order_manager.get_by_id("3456").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("4567").status is INSERT_STATUS.CANCELED
    assert order_manager.get_by_id("5678").status is INSERT_STATUS.CANCELED

    # not yet tracked: track partial fail
    orders = [sample_order("6789"), sample_order("78910", "some")]
    for x in orders:
        x.status = INSERT_STATUS.LIVE
    orders[1] = frozen_order(orders[1])
    with pytest.raises(OrderTrackingException) as record:
        order_manager.cancel(orders)
    assert "Not all orders trackable" in str(record)
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
    ]
    assert order_manager.get_by_id("3456").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("4567").status is INSERT_STATUS.CANCELED

    # tracked
    orders = [sample_order("891011"), sample_order("9101112")]
    for x in orders:
        x.status = INSERT_STATUS.LIVE
        order_manager.track(x, False)
    mock_cancel_order(["891011", "9101112"], None)
    order_manager.cancel(orders)
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
    ]
    assert order_manager.get_by_id("3456").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("891011").status is INSERT_STATUS.CANCELED
    assert order_manager.get_by_id("9101112").status is INSERT_STATUS.CANCELED

    # tracked: cancel partial fail
    orders = [sample_order("10111213"), sample_order("11121314")]
    for x in orders:
        x.status = INSERT_STATUS.LIVE
        order_manager.track(x, False)
    mock_cancel_order(["11121314"], {"10111213": "some reason"})
    with pytest.raises(OrderUpdateException) as record:
        order_manager.cancel(orders)
    assert "Cancellation" in str(record)
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
        "10111213",
        "11121314",
    ]
    assert order_manager.get_by_id("3456").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("10111213").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("11121314").status is INSERT_STATUS.CANCELED

    # --- frozen order ---
    frozen = [
        order_manager.get_by_id("3456"),
        order_manager.get_by_id("10111213"),
    ]
    mock_cancel_order(["3456", "10111213"], None)
    order_manager.cancel(frozen)
    assert order_manager.get_by_id("3456").status is INSERT_STATUS.CANCELED
    assert order_manager.get_by_id("10111213").status is INSERT_STATUS.CANCELED

    # --- str ---
    # cancel success
    order = sample_order("11121314")
    order.status = INSERT_STATUS.LIVE
    order_manager.track(order, False)
    mock_cancel_order(["11121314"], None)
    order_manager.cancel("11121314")
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
        "10111213",
        "11121314",
    ]
    assert order_manager.get_by_id("11121314").status is INSERT_STATUS.CANCELED

    # cancel fail
    order = sample_order("12131415")
    order.status = INSERT_STATUS.DEFINED
    order_manager.track(order, False)
    mock_cancel_order([], {"12131415": "some reason"})
    with pytest.raises(OrderUpdateException) as record:
        order_manager.cancel("12131415")
    assert "Cancellation" in str(record)
    assert order_manager.get_by_id("12131415").status is INSERT_STATUS.DEFINED
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
        "10111213",
        "11121314",
        "12131415",
    ]

    # not yet tracked: fail
    order = sample_order("13141516")
    order.status = INSERT_STATUS.DEFINED
    with pytest.raises(OrderGetException) as record:
        order_manager.cancel("13141516")
    assert "Not all order" in str(record)
    assert order_manager.get_by_id("12131415").status is INSERT_STATUS.DEFINED
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
        "10111213",
        "11121314",
        "12131415",
    ]

    # --- list str ---
    # cancel success
    order = sample_order("14151617")
    order.status = INSERT_STATUS.LIVE
    order_manager.track(order, False)
    mock_cancel_order(["12131415", "14151617"], None)
    order_manager.cancel(["12131415", "14151617"])
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
        "10111213",
        "11121314",
        "12131415",
        "14151617",
    ]
    assert order_manager.get_by_id("12131415").status is INSERT_STATUS.CANCELED
    assert order_manager.get_by_id("14151617").status is INSERT_STATUS.CANCELED

    # cancel fail
    orders = [sample_order("15161718"), sample_order("16171819")]
    for x in orders:
        x.status = INSERT_STATUS.LIVE
        order_manager.track(x, False)
    mock_cancel_order([], {"15161718": "some reason", "16171819": "other reason"})
    ret_orders, ret_resp = order_manager.cancel(["15161718", "16171819"], "warn")
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
        "10111213",
        "11121314",
        "12131415",
        "14151617",
        "15161718",
        "16171819",
    ]
    assert order_manager.get_by_id("15161718").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("16171819").status is INSERT_STATUS.LIVE
    assert ret_orders[0].id == "15161718"
    assert list(ret_resp.not_canceled.keys()) == ["15161718", "16171819"]

    # partial not yet tracked: fail
    with pytest.raises(OrderGetException) as record:
        order_manager.cancel(
            ["15161718", "16171819", "17181920"], mode_not_canceled="ignore"
        )
    assert "Not all order ids" in str(record)
    assert list(order_manager.order_ids) == [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "891011",
        "9101112",
        "10111213",
        "11121314",
        "12131415",
        "14151617",
        "15161718",
        "16171819",
    ]
    assert order_manager.get_by_id("15161718").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("16171819").status is INSERT_STATUS.LIVE


def test_cancel_empty(order_manager):
    order_manager.cancel(orders=[])


def test_cancel_all(order_manager, mock_cancel_order, sample_order):
    orders = [sample_order("1"), sample_order("2"), sample_order("3")]
    for x in orders:
        x.status = INSERT_STATUS.LIVE
        order_manager.track(x, False)
    assert list(order_manager.order_ids) == ["1", "2", "3"]

    mock_cancel_order(["1", "2", "3"], None)
    order_manager.cancel_all()
    assert list(order_manager.order_ids) == ["1", "2", "3"]
    for x in ["1", "2", "3"]:
        assert order_manager.get_by_id(x).status == INSERT_STATUS.CANCELED
    assert len(order_manager.get(status=INSERT_STATUS.CANCELED)) == 3

    # partial server-side error
    orders = [sample_order("4"), sample_order("5")]
    for x in orders:
        x.status = INSERT_STATUS.LIVE
        order_manager.track(x, False)
    mock_cancel_order(["4"], {"5": "some reason"})
    with pytest.raises(OrderUpdateException) as record:
        order_manager.cancel_all()
    assert "Cancellation" in str(record)
    assert list(order_manager.order_ids) == ["1", "2", "3", "4", "5"]
    for x in ["1", "2", "3", "4"]:
        assert order_manager.get_by_id(x).status == INSERT_STATUS.CANCELED
    assert len(order_manager.get(status=INSERT_STATUS.CANCELED)) == 4
    assert order_manager.get_by_id("5").status == INSERT_STATUS.LIVE


def test_cancel_all_mixed_states(order_manager, mock_cancel_order, sample_order):
    order1 = sample_order("1")
    order2 = sample_order("2")
    order3 = sample_order("3")

    order1.status = INSERT_STATUS.DEFINED
    order2.status = INSERT_STATUS.MATCHED
    order3.status = INSERT_STATUS.LIVE

    order_manager.track(order1, False)
    order_manager.track(order2, False)
    order_manager.track(order3, False)

    mock_cancel_order(["3"], None)
    order_manager.cancel_all(statuses=[INSERT_STATUS.LIVE])

    assert order1.status is INSERT_STATUS.DEFINED
    assert order2.status is INSERT_STATUS.MATCHED
    assert order3.status is INSERT_STATUS.CANCELED


def test_cancel_all_empty(order_manager):
    order_manager.cancel_all()


# noinspection DuplicatedCode
def test_sync(
    order_manager, sample_order, mock_get_order, mock_get_order_none
):  # sourcery skip: extract-duplicate-method
    # todo split into multiple tests

    # --- order object ---
    # not yet tracked: success
    order = sample_order("1")
    assert order.status == INSERT_STATUS.DEFINED
    mock_get_order("1", INSERT_STATUS.LIVE, order.size, order.price, order.side, 0.5)
    order_manager.sync(order, "except")
    assert list(order_manager.order_ids) == ["1"]
    assert order_manager.get_by_id("1").status == INSERT_STATUS.LIVE
    assert order_manager.get_by_id("1").size_matched == 0.5

    # not yet tracked: fail except
    order = frozen_order(sample_order("2", "strat"))
    with pytest.raises(OrderTrackingException) as record:
        order_manager.sync(order, "except")
    assert "Not all orders trackable" in str(record)
    assert list(order_manager.order_ids) == ["1"]
    assert order_manager.get_by_id("1").size_matched == 0.5

    # not yet tracked: fail remove
    order = sample_order("2")
    mock_get_order_none("2")
    mod, fail = order_manager.sync(order, "remove")
    assert list(order_manager.order_ids) == ["1"]
    assert mod.id == "2"
    assert mod.status is INSERT_STATUS.DEFINED
    assert len(fail) == 1

    # tracked: success
    order = sample_order("3")
    order.status = INSERT_STATUS.LIVE
    order_manager.track(order, False)
    assert list(order_manager.order_ids) == ["1", "3"]
    mock_get_order("3", INSERT_STATUS.LIVE, order.size, order.price, order.side, 1.5)
    order_manager.sync(order, "remove")
    assert order_manager.get_by_id("1").size_matched == 0.5
    assert order_manager.get_by_id("3").size_matched == 1.5
    assert len(order_manager.get(status=INSERT_STATUS.LIVE)) == 2

    # tracked: fail (server-side)
    order = sample_order("4")
    order_manager.track(order, False)
    assert list(order_manager.order_ids) == ["1", "3", "4"]
    mock_get_order_none("4")
    with pytest.raises(OrderUpdateException) as record:
        order_manager.sync(order, "except")
    assert "could not be updated" in str(record)
    assert order_manager.get_by_id("1").size_matched == 0.5
    assert order_manager.get_by_id("3").size_matched == 1.5
    assert len(order_manager.get(status=INSERT_STATUS.LIVE)) == 2
    assert len(order_manager.get(status=INSERT_STATUS.DEFINED)) == 1

    # --- list order objects ---
    # not yet tracked: success
    orders = [sample_order("5"), sample_order("6")]
    mock_get_order(
        "5", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 1
    )
    mock_get_order(
        "6", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 2
    )
    order_manager.sync(orders, "except")
    assert list(order_manager.order_ids) == ["1", "3", "4", "5", "6"]
    assert order_manager.get_by_id("1").size_matched == 0.5
    assert order_manager.get_by_id("3").size_matched == 1.5
    assert order_manager.get_by_id("5").size_matched == 1
    assert order_manager.get_by_id("6").size_matched == 2

    # not yet tracked: fail server-side
    orders = [sample_order("7"), sample_order("8")]
    mock_get_order(
        "7", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 1
    )
    mock_get_order_none("8")
    with pytest.raises(OrderUpdateException) as record:
        order_manager.sync(orders, "except")
    assert "could not be updated" in str(record)
    assert list(order_manager.order_ids) == ["1", "3", "4", "5", "6", "7", "8"]
    assert order_manager.get_by_id("7").size_matched == 1
    assert order_manager.get_by_id("7").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("8").status is INSERT_STATUS.DEFINED

    # not yet tracked: fail client-side
    orders = [sample_order("9"), frozen_order(sample_order("10", "strat"))]
    mock_get_order(
        "9", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 1
    )
    with pytest.raises(OrderTrackingException) as record:
        order_manager.sync(orders, "except")
    assert "Not all orders trackable" in str(record)
    assert list(order_manager.order_ids) == ["1", "3", "4", "5", "6", "7", "8"]

    # tracked: success
    orders = [sample_order("11"), sample_order("12")]
    for x in orders:
        order_manager.track(x, False)
    mock_get_order(
        "11", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 11
    )
    mock_get_order(
        "12", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 12
    )
    order_manager.sync(orders, "except")
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "11",
        "12",
    ]
    assert order_manager.get_by_id("8").status is INSERT_STATUS.DEFINED
    assert order_manager.get_by_id("11").size_matched == 11
    assert order_manager.get_by_id("12").size_matched == 12

    # tracked: fail (server-side)
    orders = [sample_order("13"), sample_order("14")]
    for x in orders:
        order_manager.track(x, False)
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "11",
        "12",
        "13",
        "14",
    ]
    mock_get_order_none("13")
    mock_get_order(
        "14", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 14
    )
    order_manager.sync(orders, "remove")
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "11",
        "12",
        "14",
    ]
    assert order_manager.get_by_id("14").size_matched == 14

    # --- str ---
    # not yet tracked: fail
    with pytest.raises(OrderGetException) as record:
        order_manager.sync("15", "except")
    assert "Not all order ids" in str(record)
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "11",
        "12",
        "14",
    ]

    # tracked: success
    mock_get_order(
        "8", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 12
    )
    order_manager.sync("8", "except")
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "11",
        "12",
        "14",
    ]
    assert order_manager.get_by_id("8").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("8").size_matched == 12

    # tracked: fail (server-side)
    mock_get_order_none("8")
    order_manager.sync("8", "warn")
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "11",
        "12",
        "14",
    ]
    assert order_manager.get_by_id("8").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("8").size_matched == 12

    order_manager.sync("8", "remove")
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "11",
        "12",
        "14",
    ]

    # --- list str ---
    # not yet tracked: fail
    with pytest.raises(OrderGetException) as record:
        order_manager.sync(["15", "16"], "except")
    assert "Not all order ids" in str(record)
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "11",
        "12",
        "14",
    ]

    # tracked: success
    mock_get_order(
        "1", INSERT_STATUS.DELAYED, orders[0].size, orders[0].price, orders[0].side, 20
    )
    mock_get_order(
        "3", INSERT_STATUS.DELAYED, orders[0].size, orders[0].price, orders[0].side, 20
    )
    order_manager.sync(["1", "3"], "except")
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "11",
        "12",
        "14",
    ]
    assert order_manager.get_by_id("1").status is INSERT_STATUS.DELAYED
    assert order_manager.get_by_id("1").size_matched == 20
    assert order_manager.get_by_id("3").status is INSERT_STATUS.DELAYED
    assert order_manager.get_by_id("3").size_matched == 20

    # tracked: fail (server-side)
    mock_get_order_none("1")
    mock_get_order_none("3")
    order_manager.sync(["1", "3"], "warn")
    assert list(order_manager.order_ids) == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
        "11",
        "12",
        "14",
    ]

    order_manager.sync(["1", "3"], "remove")
    assert list(order_manager.order_ids) == ["4", "5", "6", "7", "11", "12", "14"]

    # --- frozen ---
    frozen = order_manager.get_by_id("4")
    mock_get_order(
        "4", INSERT_STATUS.DELAYED, orders[0].size, orders[0].price, orders[0].side, 20
    )
    order_manager.sync(frozen, "except")
    assert list(order_manager.order_ids) == ["4", "5", "6", "7", "11", "12", "14"]
    assert order_manager.get_by_id("4").status is INSERT_STATUS.DELAYED
    assert order_manager.get_by_id("4").size_matched == 20

    mock_get_order(
        "4", INSERT_STATUS.MATCHED, orders[0].size, orders[0].price, orders[0].side, 21
    )
    mock_get_order(
        "5", INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 20
    )
    frozen = [
        order_manager.get_by_id("4"),
        order_manager.get_by_id("5"),
    ]
    order_manager.sync(frozen, "except")
    assert list(order_manager.order_ids) == ["4", "5", "6", "7", "11", "12", "14"]
    assert order_manager.get_by_id("4").status is INSERT_STATUS.MATCHED
    assert order_manager.get_by_id("5").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("4").size_matched == 21
    assert order_manager.get_by_id("5").size_matched == 20

    # --- None (all) ---
    # success
    for x in ["4", "5", "6", "7", "11", "12", "14"]:
        mock_get_order(
            x, INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 50
        )
    order_manager.sync(None, "except")
    assert list(order_manager.order_ids) == ["4", "5", "6", "7", "11", "12", "14"]
    for x in ["5", "6", "7", "11", "12", "14"]:
        assert order_manager.get_by_id(x).status is INSERT_STATUS.LIVE
        assert order_manager.get_by_id(x).size_matched == 50
    assert order_manager.get_by_id("4").size_matched == 21
    assert order_manager.get_by_id("4").status == INSERT_STATUS.MATCHED

    # fail (server-side)
    for x in ["4", "5", "6", "7", "14"]:
        mock_get_order(
            x, INSERT_STATUS.LIVE, orders[0].size, orders[0].price, orders[0].side, 51
        )
    mock_get_order_none("11")
    mock_get_order_none("12")
    order_manager.sync(None, "remove")
    assert list(order_manager.order_ids) == ["4", "5", "6", "7", "14"]
    for x in ["5", "6", "7", "14"]:
        assert order_manager.get_by_id(x).status is INSERT_STATUS.LIVE
        assert order_manager.get_by_id(x).size_matched == 51
    assert order_manager.get_by_id("4").size_matched == 21
    assert order_manager.get_by_id("4").status == INSERT_STATUS.MATCHED


def test_sync_decimal(
    order_manager,
    mock_get_order,
    mock_post_order,
    mock_tick_size,
    mock_neg_risk,
):
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_neg_risk(ASSET_ID_YES)
    mock_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "1234",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "0.5",
            "makingAmount": "0.25",
        },
    )
    order_manager.limit_order(
        Decimal("0.3"),
        Decimal(10),
        ASSET_ID_YES,
        SIDE.BUY,
        None,
        TIME_IN_FORCE.GTC,
        None,
        neg_risk=None,
        order_id="1234",
        signature="signature",
    )
    assert order_manager.get_by_id("1234").size_matched == 0.5

    mock_get_order("1234", INSERT_STATUS.LIVE, 10, "0.3", SIDE.BUY, 1)
    order_manager.sync("1234", "except")

    assert order_manager.get_by_id("1234").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("1234").size_matched == Decimal(1)
    assert order_manager.get_by_id("1234").size == Decimal(10)
    assert order_manager.get_by_id("1234").price == Decimal("0.3")
    assert order_manager.get_by_id("1234").side is SIDE.BUY


# noinspection DuplicatedCode
def test_sync_valid_states(order_manager, sample_order, mock_get_order):
    orders = [
        sample_order("1"),
        sample_order("2"),
        sample_order("3"),
        sample_order("4"),
        sample_order("5"),
        sample_order("6"),
    ]
    orders[1].status = INSERT_STATUS.LIVE
    orders[2].status = INSERT_STATUS.DELAYED
    orders[3].status = INSERT_STATUS.MATCHED
    orders[4].status = INSERT_STATUS.UNMATCHED
    orders[5].status = INSERT_STATUS.CANCELED

    for x in range(1, 7):
        mock_get_order(
            str(x),
            INSERT_STATUS.LIVE,
            orders[0].size,
            orders[0].price,
            orders[0].side,
            10,
        )

    order_manager.sync(orders, "remove")

    assert list(order_manager.order_ids) == ["1", "2", "3", "4", "5", "6"]
    assert order_manager.get_by_id("1").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("2").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("3").status is INSERT_STATUS.LIVE
    assert order_manager.get_by_id("4").status is INSERT_STATUS.MATCHED
    assert order_manager.get_by_id("5").status is INSERT_STATUS.UNMATCHED
    assert order_manager.get_by_id("6").status is INSERT_STATUS.CANCELED

    assert order_manager.get_by_id("1").size_matched == 10
    assert order_manager.get_by_id("2").size_matched == 10
    assert order_manager.get_by_id("3").size_matched == 10
    assert order_manager.get_by_id("4").size_matched != 10
    assert order_manager.get_by_id("5").size_matched != 10
    assert order_manager.get_by_id("6").size_matched != 10


def test_market_sell_max_size(order_manager, mock_tick_size, mock_neg_risk):
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_neg_risk(ASSET_ID_YES)

    with pytest.raises(OrderCreationException) as record:
        order_manager.market_order(
            100,
            ASSET_ID_YES,
            SIDE.SELL,
            None,
            TIME_IN_FORCE.FOK,
            None,
            0.01,
            neg_risk=None,
        )

    assert "exceeds" in str(record)


# noinspection DuplicatedCode
def test_clean(
    order_manager, sample_order, mock_post_order, mock_tick_size, mock_neg_risk
):
    mock_tick_size(ASSET_ID_YES, 0.01)
    mock_neg_risk(ASSET_ID_YES)

    # empty order manager
    order_manager.clean()

    # success: states
    orders = [
        sample_order("1"),
        sample_order("2"),
        sample_order("3"),
        sample_order("4"),
        sample_order("5"),
        sample_order("6"),
    ]
    orders[1].status = INSERT_STATUS.LIVE
    orders[2].status = INSERT_STATUS.DELAYED
    orders[3].status = INSERT_STATUS.MATCHED
    orders[4].status = INSERT_STATUS.UNMATCHED
    orders[5].status = INSERT_STATUS.CANCELED
    for x in orders:
        order_manager.track(x, False)

    ret = order_manager.clean()
    assert list(order_manager.order_ids) == ["1", "2", "3"]
    assert len(ret) == 3

    # success: expiration
    mock_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "1234",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "",
            "makingAmount": "",
        },
    )
    order_manager.limit_order(
        0.5,
        10,
        ASSET_ID_YES,
        SIDE.BUY,
        None,
        TIME_IN_FORCE.GTC,
        1000,
        neg_risk=None,
        order_id="1234",
        signature="signature",
    )
    assert list(order_manager.order_ids) == ["1", "2", "3", "1234"]
    assert order_manager.get_by_id("1234").size_matched == 0
    order_manager.clean(expiration=1001)
    assert list(order_manager.order_ids) == ["1", "2", "3"]

    # custom input
    order_manager.clean(INSERT_STATUS.DEFINED)
    assert list(order_manager.order_ids) == ["2", "3"]

    # empty input
    order_manager.clean([], 0)
    assert list(order_manager.order_ids) == ["2", "3"]


def test_clean_expiration(order_manager, sample_order):
    order = sample_order("123")
    order.status = INSERT_STATUS.MATCHED
    order_manager.track(order, False)

    order2 = sample_order("456")
    order2.status = INSERT_STATUS.UNMATCHED
    order2.eip712order.expiration = 1000
    order_manager.track(order2, False)

    assert order_manager.order_ids == {"123", "456"}

    # only clean expired
    order_manager.clean(None, 1001)
    assert order_manager.order_ids == {"123"}

    # restore
    order_manager.track(order2, False)
    assert order_manager.order_ids == {"123", "456"}

    # only clean status
    order_manager.clean(INSERT_STATUS.MATCHED)
    assert order_manager.order_ids == {"456"}

    # restore
    order_manager.track(order, False)
    assert order_manager.order_ids == {"123", "456"}

    # only clean status with expiration set
    order_manager.clean(INSERT_STATUS.MATCHED, 500)
    assert order_manager.order_ids == {"456"}

    # restore
    order_manager.track(order, False)
    assert order_manager.order_ids == {"123", "456"}

    # clean expired and status
    order_manager.clean(expiration=1001)
    assert not order_manager.order_ids


def test_clean_empty(order_manager, sample_order):
    order = sample_order("123")
    order.status = INSERT_STATUS.UNMATCHED
    order_manager.track(order, False)

    order_manager.clean(None)
    assert "123" in order_manager.order_ids


def test_contains(order_manager, sample_order):
    orders = [sample_order("1"), sample_order("2")]
    for order in orders:
        order_manager.track(order, False)

    assert "1" in order_manager
    assert "3" not in order_manager


def test_max_size(
    sample_order, local_host_addr, private_key, api_key, secret, passphrase
):
    orders = [sample_order("1"), sample_order("2")]
    # noinspection PyTypeChecker
    om = OrderManager(
        rest_endpoint=local_host_addr,
        private_key=private_key,
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        maker_funder=None,
        signature_type=None,
        chain_id=CHAIN_ID.POLYGON,
        max_size=1,
    )

    om.track(orders[0], False)

    with pytest.raises(OrderTrackingException) as record:
        om.track(orders[1], False)
    assert "max_size" in str(record)
    assert list(om.order_ids) == ["1"]


def test_invalidate(sample_order, order_manager):
    # sourcery skip: extract-duplicate-method
    order_manager.invalidate()

    with pytest.raises(ManagerInvalidException) as record:
        order_manager.track(sample_order("1234"), False)
    assert "invalid" in str(record)
    assert not order_manager.order_ids
    assert not order_manager.token_ids

    order_manager._invalid_token = False
    order_manager.track(sample_order("2345"), False)
    assert list(order_manager.order_ids) == ["2345"]
    assert list(order_manager.token_ids) == [ASSET_ID_YES]

    order_manager.invalidate("pencil broke")
    with pytest.raises(ManagerInvalidException) as record:
        order_manager.untrack("2345", False)
    assert "invalid" in str(record)
    assert "pencil broke" in str(record)
    assert list(order_manager.order_ids) == ["2345"]
    assert list(order_manager.token_ids) == [ASSET_ID_YES]

    with pytest.raises(ManagerInvalidException) as record:
        order_manager.market_order(
            12,
            ASSET_ID_YES,
            SIDE.BUY,
            0.01,
            TIME_IN_FORCE.FOK,
            None,
            math.inf,
            neg_risk=None,
        )
    assert "invalid" in str(record)
    assert list(order_manager.order_ids) == ["2345"]
    assert list(order_manager.token_ids) == [ASSET_ID_YES]
