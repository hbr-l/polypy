import json
import pathlib
import time
from decimal import Decimal
from typing import Callable, Literal

import msgspec
import pytest

from polypy.constants import CHAIN_ID
from polypy.exceptions import (
    ManagerInvalidException,
    PositionNegativeException,
    StreamException,
)
from polypy.manager import OrderManager, PositionManager
from polypy.order import (
    INSERT_STATUS,
    SIDE,
    TIME_IN_FORCE,
    OrderProtocol,
    create_limit_order,
)
from polypy.position import USDC, Position, PositionProtocol
from polypy.signing import SIGNATURE_TYPE
from polypy.stream.user import CHANNEL, BufferSettings, UserStream
from polypy.structs import OrderWSInfo, TradeWSInfo
from polypy.typing import dec

test_pth = pathlib.Path(__file__).parent


@pytest.fixture
def market_id():
    return "0x84c0ffe3f56cb357ff5ff8bc5d2182ae90be4dd6718e8403a6af472b452dbfa8"


@pytest.fixture
def yes_asset_id():
    return (
        "72936048731589292555781174533757608024096898681344338816447372274344589246891"
    )


@pytest.fixture
def no_asset_id():
    return (
        "101669189743438912873361127612589311253202068943959811456820079057046819967115"
    )


@pytest.fixture
def order_manager(local_host_addr, private_key, api_key, secret, passphrase):
    return OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )


@pytest.fixture
def position_manager(local_host_addr):
    return PositionManager(local_host_addr, 100)


@pytest.fixture
def streamer(
    order_manager,
    position_manager,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
):
    streams = []

    def _closure(
        untrack_order_terminal: bool = True,
        clean_terminal_sates_s: float | None = None,
        monitor_order_assets_s: float | None = 0.1,
        buffer_settings: BufferSettings | None = (5, 0.01, 5_000),
        update_mode: Literal["explicit", "implicit"] = "explicit",
        invalidate_on_exc: bool = True,
        callback_msg: Callable[[UserStream, TradeWSInfo | OrderWSInfo], None]
        | None = None,
        callback_exc: Callable[[UserStream, Exception], None] | None = None,
        callback_clean: Callable[[list[OrderProtocol], list[PositionProtocol]], None]
        | None = None,
    ):
        _stream = UserStream(
            "ws://localhost:8002/",
            (order_manager, position_manager),
            (market_id, yes_asset_id, no_asset_id),
            api_key,
            secret,
            passphrase,
            untrack_order_terminal,
            clean_terminal_sates_s,
            monitor_order_assets_s,
            buffer_settings,
            5,
            update_mode,
            invalidate_on_exc,
            callback_msg,
            callback_exc,
            callback_clean,
            CHANNEL.USER,
        )
        _stream._stop_token.clear()
        _stream.pre_start()

        streams.append(_stream)

        return (
            _stream,
            order_manager,
            position_manager,
        )

    yield _closure

    stream = streams[0]
    stream._stop_token.set()
    stream.post_stop()


@pytest.fixture
def mock_std_post_order(mock_tick_size, mock_neg_risk, mock_post_order, yes_asset_id):
    def _closure(data: dict, asset_id=yes_asset_id):
        mock_tick_size(asset_id, 0.01)
        mock_neg_risk(asset_id)
        mock_post_order(data)

    yield _closure


@pytest.fixture
def trade_info_taker_buy_matched(mock_std_post_order):
    with open(test_pth / "data/test_trade_info_taker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )
    mock_std_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "5",  # receive size
            "makingAmount": "3",  # spent amount
        }
    )

    yield data


@pytest.fixture
def trade_info_taker_sell_matched(mock_std_post_order):
    with open(test_pth / "data/test_trade_info_taker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )
    data = msgspec.structs.replace(data, side=SIDE.SELL)

    mock_std_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "3",
            "makingAmount": "5",
        }
    )

    yield data


@pytest.fixture
def trade_info_maker_buy_matched(mock_std_post_order):
    with open(test_pth / "data/test_trade_info_maker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )

    mock_std_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "",  # receive size
            "makingAmount": "",  # spent amount
        }
    )

    yield data


@pytest.fixture
def trade_info_maker_buy_confirmed(mock_std_post_order):
    with open(test_pth / "data/test_trade_info_maker_buy_confirmed.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )

    mock_std_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "",  # receive size
            "makingAmount": "",  # spent amount
        },
    )

    yield data


@pytest.fixture
def order_info_maker_buy_placement_update(mock_std_post_order):
    with open(test_pth / "data/test_order_info_maker_buy_placement.json", "r") as f:
        data_placement = msgspec.json.decode(
            json.dumps(json.load(f)), type=OrderWSInfo, strict=False
        )

    with open(test_pth / "data/test_order_info_maker_buy_update.json", "r") as f:
        data_update = msgspec.json.decode(
            json.dumps(json.load(f)), type=OrderWSInfo, strict=False
        )

    mock_std_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "",  # receive size
            "makingAmount": "",  # spent amount
        },
    )

    yield data_placement, data_update


@pytest.fixture
def order_info_maker_buy_unmatched(mock_std_post_order):
    with open(test_pth / "data/test_order_info_maker_buy_unmatched.json", "r") as f:
        data_unmatched = msgspec.json.decode(
            json.dumps(json.load(f)), type=OrderWSInfo, strict=False
        )

    mock_std_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "",  # receive size
            "makingAmount": "",  # spent amount
        },
    )

    yield data_unmatched


@pytest.fixture
def callback_clean():
    class Callback:
        def __init__(self):
            self.cleaned_orders = []
            self.cleaned_positions = []

        def __call__(self, cleaned_orders, cleaned_positions):
            self.cleaned_orders.extend(cleaned_orders)
            self.cleaned_positions.extend(cleaned_positions)

    return Callback()


# test:
# - buffer (in and out)
# - trade info parsing
# - order info parsing
# - order manager invalidation: buffer thread, monitor thread, clean thread, main thread
# - untracking actions
# - cleaning thread
# - monitoring order manager assets thread
# - None in tuple managers
# - order only assigned to one order manager (raise)


def test_buffer_trade_info_taker_order(
    streamer, yes_asset_id, trade_info_taker_buy_matched
):
    stream, om, pm = streamer(False)

    stream.on_msg(trade_info_taker_buy_matched)  # this simulates websocket msg recv
    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    time.sleep(0.1)

    assert om.valid is True
    assert (
        om.get_by_id(
            "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000"
        ).size_matched
        == 5
    )
    assert (
        pm.get_by_id(
            "25742247876332768458781360292043764039507900813404980298479194684402595556451"
        ).size
        == 5
    )
    assert pm.balance == 100 - 3


def test_buffer_trade_info_maker_order(
    streamer, yes_asset_id, trade_info_maker_buy_matched
):
    stream, om, pm = streamer(False)

    stream.on_msg(trade_info_maker_buy_matched)
    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    time.sleep(0.1)

    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).size_matched
        == 0
    )
    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.LIVE
    )
    assert pm.balance_total == 100 - 3
    assert (
        pm.get_by_id(
            "95786924372760057572092804419385993470890190892343223404877167501659835222533"
        ).size
        == 5
    )


def test_buffer_order_info_maker_order(
    streamer, yes_asset_id, order_info_maker_buy_placement_update
):
    stream, om, pm = streamer(False)

    data_placement, data_update = order_info_maker_buy_placement_update

    # this simulates websocket msg recv which first will be buffered until rest call returns
    stream.on_msg(data_placement)
    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    time.sleep(0.1)

    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).size_matched
        == 0
    )
    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.LIVE
    )
    assert pm.balance_total == 100
    assert (
        "25742247876332768458781360292043764039507900813404980298479194684402595556451"
        not in pm
    )

    stream.on_msg(data_update)  # this simulates websocket msg recv
    time.sleep(0.1)

    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).size_matched
        == 5
    )
    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.MATCHED
    )
    assert pm.balance_total == 100
    assert (
        "25742247876332768458781360292043764039507900813404980298479194684402595556451"
        not in pm
    )


def test_raise_no_buffer_trade_info_taker_order(
    streamer, yes_asset_id, trade_info_taker_buy_matched
):
    stream, om, pm = streamer(buffer_settings=None)

    with pytest.raises(StreamException):
        stream.on_msg(trade_info_taker_buy_matched)  # this simulates websocket msg recv
        time.sleep(0.1)

    assert om.valid is False
    assert pm.balance_total == 100


def test_no_buffer_trade_info_taker_order(
    streamer, yes_asset_id, trade_info_taker_buy_matched
):
    stream, om, pm = streamer(buffer_settings=None)

    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    stream.on_msg(trade_info_taker_buy_matched)  # this simulates websocket msg recv
    time.sleep(0.1)

    assert om.valid is True
    assert (
        om.get_by_id(
            "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000"
        ).size_matched
        == 5
    )
    assert (
        pm.get_by_id(
            "25742247876332768458781360292043764039507900813404980298479194684402595556451"
        ).size
        == 5
    )
    assert pm.balance == 100 - 3


def test_raise_no_buffer_trade_info_maker_order(
    streamer, yes_asset_id, trade_info_maker_buy_matched
):
    stream, om, pm = streamer(buffer_settings=None)

    with pytest.raises(StreamException):
        stream.on_msg(trade_info_maker_buy_matched)  # this simulates websocket msg recv
        time.sleep(0.1)

    assert om.valid is False
    assert pm.balance_total == 100


def test_no_buffer_trade_info_maker_order(
    streamer, yes_asset_id, trade_info_maker_buy_matched
):
    stream, om, pm = streamer(buffer_settings=None)

    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    stream.on_msg(trade_info_maker_buy_matched)
    time.sleep(0.1)

    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).size_matched
        == 0
    )
    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.LIVE
    )
    assert pm.balance_total == 100 - 3
    assert (
        pm.get_by_id(
            "95786924372760057572092804419385993470890190892343223404877167501659835222533"
        ).size
        == 5
    )


def test_raise_no_buffer_order_info_maker_order(
    streamer, yes_asset_id, order_info_maker_buy_placement_update
):
    stream, om, pm = streamer(buffer_settings=None)

    data_placement, _ = order_info_maker_buy_placement_update

    with pytest.raises(StreamException):
        stream.on_msg(data_placement)
        time.sleep(0.1)

    assert om.valid is False
    assert pm.balance_total == 100


def test_no_buffer_order_info_maker_order(
    streamer, yes_asset_id, order_info_maker_buy_placement_update
):
    stream, om, pm = streamer(buffer_settings=None)

    data_placement, data_update = order_info_maker_buy_placement_update

    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    stream.on_msg(data_placement)
    time.sleep(0.1)
    stream.on_msg(data_update)  # this simulates websocket msg recv
    time.sleep(0.1)

    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).size_matched
        == 5
    )
    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.MATCHED
    )
    assert pm.balance_total == 100
    assert (
        "25742247876332768458781360292043764039507900813404980298479194684402595556451"
        not in pm
    )


def test_untrack_order_at_terminal_trade_info(
    streamer,
    yes_asset_id,
    trade_info_maker_buy_confirmed,
    order_info_maker_buy_placement_update,
    callback_clean,
):  # sourcery skip: extract-duplicate-method
    stream, om, pm = streamer(
        clean_terminal_sates_s=None, callback_clean=callback_clean
    )
    pm.track(Position("1234567", 0, False, 2))
    data_trade = trade_info_maker_buy_confirmed
    data_placement, data_update = order_info_maker_buy_placement_update

    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    stream.on_msg(data_placement)
    stream.on_msg(data_update)
    time.sleep(0.1)

    assert len(om.token_ids) == 1
    assert len(om.get(status=INSERT_STATUS.MATCHED)) == 1
    assert len(callback_clean.cleaned_orders) == 0
    assert len(callback_clean.cleaned_positions) == 0

    stream.on_msg(data_trade)
    time.sleep(0.1)

    assert len(om.token_ids) == 0
    assert len(callback_clean.cleaned_orders) == 1
    assert len(callback_clean.cleaned_positions) == 0
    assert callback_clean.cleaned_orders[0].status is INSERT_STATUS.MATCHED
    assert callback_clean.cleaned_orders[0].size_matched == 5


def test_raise_untrack_order_not_at_terminal_trade_info(
    streamer,
    yes_asset_id,
    trade_info_maker_buy_confirmed,
    order_info_maker_buy_placement_update,
    callback_clean,
):
    stream, om, pm = streamer(
        clean_terminal_sates_s=None, callback_clean=callback_clean
    )
    data_trade = trade_info_maker_buy_confirmed
    data_placement, _ = order_info_maker_buy_placement_update

    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    stream.on_msg(data_placement)
    time.sleep(0.1)

    with pytest.raises(StreamException) as record:
        stream.on_msg(data_trade)
        time.sleep(0.1)
    assert "TERMINAL_INSERT_STATI" in str(record)


def test_untrack_order_at_unmatched_order_info(
    streamer, yes_asset_id, order_info_maker_buy_unmatched, callback_clean
):
    stream, om, pm = streamer(
        clean_terminal_sates_s=None, callback_clean=callback_clean
    )
    pm.track(Position("1234567", 0, False, 2))
    data = order_info_maker_buy_unmatched

    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    assert len(om.token_ids) == 1
    assert len(om.get(status=INSERT_STATUS.LIVE)) == 1
    assert len(callback_clean.cleaned_orders) == 0
    assert len(callback_clean.cleaned_positions) == 0

    stream.on_msg(data)
    time.sleep(0.1)

    assert len(om.token_ids) == 0
    assert len(callback_clean.cleaned_orders) == 1
    assert len(callback_clean.cleaned_positions) == 0
    assert callback_clean.cleaned_orders[0].size_matched == 0
    assert callback_clean.cleaned_orders[0].status is INSERT_STATUS.UNMATCHED
    assert pm.balance_total == 100


def test_clean_terminal(
    streamer, yes_asset_id, order_info_maker_buy_unmatched, callback_clean
):
    stream, om, pm = streamer(clean_terminal_sates_s=0.2, callback_clean=callback_clean)
    pm.track(Position("1234567", 0, False, 2))
    data = order_info_maker_buy_unmatched

    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    assert len(om.token_ids) == 1
    assert len(pm.asset_ids) == 2
    assert len(om.get(status=INSERT_STATUS.LIVE)) == 1
    assert len(callback_clean.cleaned_orders) == 0
    assert len(callback_clean.cleaned_positions) == 0

    stream.on_msg(data)
    time.sleep(0.25)

    assert len(om.token_ids) == 0
    assert len(pm.asset_ids) == 1
    assert len(callback_clean.cleaned_orders) == 1
    assert len(callback_clean.cleaned_positions) == 1
    assert callback_clean.cleaned_orders[0].size_matched == 0
    assert callback_clean.cleaned_orders[0].status is INSERT_STATUS.UNMATCHED
    assert callback_clean.cleaned_positions[0].asset_id == "1234567"
    assert callback_clean.cleaned_positions[0].size == 0
    assert pm.balance_total == 100


def test_raise_monitor_order_assets(streamer, mock_std_post_order):
    stream, om, pm = streamer(monitor_order_assets_s=0.01)
    mock_std_post_order(
        {
            "success": True,
            "errorMsg": "",
            "orderID": "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000",
            "transactionsHashes": None,
            "status": str(INSERT_STATUS.LIVE),
            "takingAmount": "",  # receive size
            "makingAmount": "",  # spent amount
        },
        "1234",
    )
    assert om.valid is True

    om.limit_order(0.99, 5, "1234", SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    time.sleep(0.1)

    assert om.valid is False


def test_raise_multiple_updates_order_info(
    local_host_addr,
    private_key,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
):  # sourcery skip: extract-duplicate-method
    om1 = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )
    om2 = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )
    pm = PositionManager(local_host_addr, 100)

    stream = UserStream(
        "ws://localhost:8002/",
        [(om1, pm), (om2, pm)],
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        True,
        None,
    )
    stream._stop_token.clear()
    stream.pre_start()

    # unique order
    unique_order = create_limit_order(
        0.5,
        10,
        yes_asset_id,
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    unique_order.id = (
        "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
    )
    om1.track(unique_order, False)

    # common order
    common_order = create_limit_order(
        0.5,
        10,
        yes_asset_id,
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    common_order.id = "1234"
    om1.track(common_order, False)
    om2.track(common_order, False)

    # allowed update
    with open(test_pth / "data/test_order_info_maker_buy_placement.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=OrderWSInfo, strict=False
        )
    stream.on_msg(data)
    time.sleep(0.1)
    assert len(om1.token_ids) == 1
    assert len(om1.order_ids) == 2
    assert len(om2.token_ids) == 1
    assert len(om2.order_ids) == 1
    assert (
        om1.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.LIVE
    )
    assert om1.get_by_id("1234").status is INSERT_STATUS.DEFINED
    assert om2.get_by_id("1234").status is INSERT_STATUS.DEFINED
    assert om1.valid is True
    assert om2.valid is True

    # exception
    data = msgspec.structs.replace(data, id="1234")
    with pytest.raises(StreamException) as record:
        stream.on_msg(data)
        time.sleep(0.1)
    assert "not be assigned to more than one" in str(record)
    assert common_order.status is INSERT_STATUS.LIVE
    assert common_order.status is INSERT_STATUS.LIVE
    assert om1.valid is False
    assert om2.valid is False

    stream._stop_token.set()
    stream.post_stop()


def test_raise_multiple_updates_trade_info(
    local_host_addr,
    private_key,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
):
    om1 = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )
    om2 = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )
    pm = PositionManager(local_host_addr, 100)

    stream = UserStream(
        "ws://localhost:8002/",
        [(om1, pm), (om2, pm)],
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        True,
        None,
    )
    stream._stop_token.clear()
    stream.pre_start()

    # unique order
    unique_order = create_limit_order(
        0.5,
        10,
        yes_asset_id,
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    unique_order.id = (
        "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
    )
    om1.track(unique_order, False)

    # common order
    common_order = create_limit_order(
        0.5,
        10,
        yes_asset_id,
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    common_order.id = "1234"
    om1.track(common_order, False)
    om2.track(common_order, False)

    # allowed update
    with open(test_pth / "data/test_trade_info_maker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )
    stream.on_msg(data)
    time.sleep(0.1)
    assert om1.valid is True
    assert om2.valid is True
    assert pm.balance_total == 100 - 3
    assert (
        pm.get_by_id(
            "95786924372760057572092804419385993470890190892343223404877167501659835222533"
        ).size
        == 5
    )

    # exception
    data = msgspec.structs.replace(
        data,
        maker_orders=(msgspec.structs.replace(data.maker_orders[0], order_id="1234"),),
    )
    with pytest.raises(StreamException) as record:
        stream.on_msg(data)
        time.sleep(0.1)
    assert "not be assigned to more than one" in str(record)
    assert om1.valid is False
    assert om2.valid is False
    assert pm.balance_total == 100 - 9
    assert (
        pm.get_by_id(
            "95786924372760057572092804419385993470890190892343223404877167501659835222533"
        ).size
        == 15
    )

    stream._stop_token.set()
    stream.post_stop()


def test_order_manager_none(
    local_host_addr,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
    order_info_maker_buy_placement_update,
    trade_info_maker_buy_matched,
):
    pm = PositionManager(local_host_addr, 100)

    stream = UserStream(
        "ws://localhost:8002/",
        (None, pm),
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        True,
        None,
        buffer_settings=(0.01, 0.001, 10),
        update_mode="implicit",
    )
    stream._stop_token.clear()
    stream.pre_start()

    order_placement, order_update = order_info_maker_buy_placement_update

    stream.on_msg(order_placement)
    stream.on_msg(order_update)
    time.sleep(0.1)

    assert pm.balance_total == 100
    assert len(pm.asset_ids) == 1

    stream.on_msg(trade_info_maker_buy_matched)
    time.sleep(0.1)

    assert pm.balance_total == 100 - 3
    assert (
        pm.get_by_id(
            "95786924372760057572092804419385993470890190892343223404877167501659835222533"
        ).size
        == 5
    )

    stream._stop_token.set()
    stream.post_stop()


def test_position_manager_none(
    local_host_addr,
    private_key,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
    order_info_maker_buy_placement_update,
    trade_info_maker_buy_matched,
):
    om = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )

    stream = UserStream(
        "ws://localhost:8002/",
        (om, None),
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        True,
        None,
        update_mode="implicit",
    )
    stream._stop_token.clear()
    stream.pre_start()

    order_placement, order_update = order_info_maker_buy_placement_update

    stream.on_msg(order_placement)
    om.limit_order(0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None)
    stream.on_msg(trade_info_maker_buy_matched)
    time.sleep(0.0001)

    assert len(om.token_ids) == 1
    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.LIVE
    )
    assert om.get(status=INSERT_STATUS.LIVE)[0].size_matched == 0
    assert om.valid is True

    stream.on_msg(order_update)
    time.sleep(0.1)

    assert len(om.token_ids) == 1
    assert (
        om.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.MATCHED
    )
    assert om.get(status=INSERT_STATUS.MATCHED)[0].size_matched == 5
    assert om.valid is True

    stream._stop_token.set()
    stream.post_stop()


def test_multi_order_manager(
    local_host_addr,
    private_key,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
    order_info_maker_buy_placement_update,
):  # sourcery skip: extract-duplicate-method
    om1 = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )
    om2 = OrderManager(
        local_host_addr,
        private_key,
        api_key,
        secret,
        passphrase,
        None,
        SIGNATURE_TYPE.EOA,
        CHAIN_ID.POLYGON,
    )
    pm = PositionManager(local_host_addr, 100)

    order_placement, order_update = order_info_maker_buy_placement_update

    stream = UserStream(
        "ws://localhost:8002/",
        [(om1, pm), (om2, pm)],
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        True,
        None,
    )
    stream._stop_token.clear()
    stream.pre_start()

    order1 = create_limit_order(
        0.5,
        10,
        yes_asset_id,
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    order1.id = "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
    om1.track(order1, False)

    order2 = create_limit_order(
        0.5,
        10,
        yes_asset_id,
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    order2.id = "12345"
    om2.track(order2, False)

    assert list(om1.order_ids) == [
        "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
    ]
    assert list(om2.order_ids) == ["12345"]

    stream.on_msg(order_placement)
    order_update = msgspec.structs.replace(order_update, id="12345")
    stream.on_msg(order_update)
    time.sleep(0.1)

    assert om1.valid is True
    assert om2.valid is True
    assert (
        om1.get_by_id(
            "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
        ).status
        is INSERT_STATUS.LIVE
    )
    assert om2.get_by_id("12345").status is INSERT_STATUS.MATCHED

    stream._stop_token.set()
    stream.post_stop()


def test_raise_explicit_sell_market_position_not_yet_created(
    streamer, yes_asset_id, trade_info_taker_sell_matched
):
    # trade info
    stream, om, pm = streamer()

    data = trade_info_taker_sell_matched

    om.limit_order(0.99, 5, yes_asset_id, SIDE.SELL, 0.01, TIME_IN_FORCE.GTC, None)

    with pytest.raises(PositionNegativeException):
        stream.on_msg(data)
        time.sleep(0.1)


def test_raise_explicit_buffer_trade_info(streamer):
    # order not in order manager
    class Callback:
        def __init__(self):
            self.x = False

        def __call__(self, *args, **kwargs):
            self.x = True

    callback = Callback()
    stream, om, _ = streamer(buffer_settings=(0.1, 0.001, 10), callback_exc=callback)

    with open(test_pth / "data/test_trade_info_taker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )

    time.sleep(0.1)
    assert callback.x is False
    assert not om.get(strategy_id="some")  # order manager not yet invalidated

    stream.on_msg(data)
    time.sleep(0.2)
    assert callback.x is True
    with pytest.raises(ManagerInvalidException):
        om.get(strategy_id="some")


def test_raise_explicit_no_buffer_trade_info(streamer):
    # order not in order manager
    stream, om, _ = streamer(buffer_settings=None)

    with open(test_pth / "data/test_trade_info_taker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )

    assert not om.get(strategy_id="some")  # order manager not yet invalidated

    with pytest.raises(StreamException):
        stream.on_msg(data)

    with pytest.raises(ManagerInvalidException):
        om.get(strategy_id="some")


def test_raise_explicit_buffer_order_info(streamer):
    # order not in order manager
    class Callback:
        def __init__(self):
            self.x = False

        def __call__(self, *args, **kwargs):
            self.x = True

    callback = Callback()
    stream, om, _ = streamer(buffer_settings=(0.1, 0.001, 10), callback_exc=callback)

    with open(test_pth / "data/test_order_info_maker_buy_placement.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=OrderWSInfo, strict=False
        )

    time.sleep(0.1)
    assert callback.x is False
    assert not om.get(strategy_id="some")  # order manager not yet invalidated

    stream.on_msg(data)
    time.sleep(0.2)
    assert callback.x is True
    with pytest.raises(ManagerInvalidException):
        om.get(strategy_id="some")


def test_raise_explicit_no_buffer_order_info(streamer):
    # order not in order manager
    stream, om, _ = streamer(buffer_settings=None)

    with open(test_pth / "data/test_order_info_maker_buy_placement.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=OrderWSInfo, strict=False
        )

    assert not om.get(strategy_id="some")  # order manager not yet invalidated

    with pytest.raises(StreamException):
        stream.on_msg(data)

    with pytest.raises(ManagerInvalidException):
        om.get(strategy_id="some")


def test_implicit_buffer_order_info(streamer, order_info_maker_buy_placement_update):
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=(0.01, 0.001, 10))
    data_placement, _ = order_info_maker_buy_placement_update

    stream.on_msg(data_placement)
    time.sleep(0.1)

    assert om.valid is True


def test_implicit_no_buffer_order_info(streamer, order_info_maker_buy_placement_update):
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=None)
    data_placement, _ = order_info_maker_buy_placement_update

    stream.on_msg(data_placement)
    time.sleep(0.1)

    assert om.valid is True


def test_implicit_buffer_trade_info(streamer, trade_info_taker_buy_matched):
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=(0.01, 0.001, 10))

    stream.on_msg(trade_info_taker_buy_matched)
    time.sleep(0.1)

    assert om.valid is True


def test_implicit_no_buffer_trade_info(streamer, trade_info_taker_buy_matched):
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=None)

    stream.on_msg(trade_info_taker_buy_matched)
    time.sleep(0.1)

    assert om.valid is True


def test_raise_no_market_id(
    order_manager,
    position_manager,
    api_key,
    secret,
    passphrase,
    yes_asset_id,
    no_asset_id,
):
    with pytest.raises(StreamException):
        UserStream(
            "ws://localhost:8002/",
            (order_manager, position_manager),
            (None, yes_asset_id, no_asset_id),
            api_key,
            secret,
            passphrase,
        )

    with pytest.raises(StreamException):
        UserStream(
            "ws://localhost:8002/",
            (order_manager, position_manager),
            [],
            api_key,
            secret,
            passphrase,
        )


def test_raise_non_unique_order_manager(
    order_manager,
    position_manager,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
):
    with pytest.raises(StreamException) as record:
        UserStream(
            "ws://localhost:8002/",
            [(order_manager, position_manager), (order_manager, None)],
            (market_id, yes_asset_id, no_asset_id),
            api_key,
            secret,
            passphrase,
        )
    assert "unique" in str(record)


def test_raise_order_manager_pre_loaded_token_ids(
    order_manager,
    position_manager,
    private_key,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
):
    order = create_limit_order(
        0.99,
        10,
        "1234",
        SIDE.BUY,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    order.id = "5678"
    order_manager.track(order, False)

    with pytest.raises(StreamException) as record:
        UserStream(
            "ws://localhost:8002/",
            (order_manager, position_manager),
            (market_id, yes_asset_id, no_asset_id),
            api_key,
            secret,
            passphrase,
        )
    assert "contained" in str(record)


def test_decimals(
    order_manager,
    position_manager,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
    trade_info_taker_buy_matched,
):
    position_manager.track(Position(USDC, dec(100), False, 2))

    stream = UserStream(
        "ws://localhost:8002/",
        (order_manager, position_manager),
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
    )
    stream._stop_token.clear()
    stream.pre_start()
    time.sleep(0.1)

    stream.on_msg(trade_info_taker_buy_matched)  # this simulates websocket msg recv
    order_manager.limit_order(
        dec(0.99), dec(5), yes_asset_id, SIDE.BUY, dec(0.01), TIME_IN_FORCE.GTC, None
    )
    time.sleep(0.1)

    assert order_manager.valid is True
    assert order_manager.get_by_id(
        "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000"
    ).size_matched == dec(5)
    assert position_manager.get_by_id(
        "25742247876332768458781360292043764039507900813404980298479194684402595556451"
    ).size == dec(5)
    assert position_manager.balance == dec(100 - 3)

    assert isinstance(position_manager.balance_total, Decimal)
    assert isinstance(
        order_manager.get_by_id(
            "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000"
        ).size_matched,
        Decimal,
    )

    stream._stop_token.set()
    stream.post_stop()
