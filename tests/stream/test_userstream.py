import json
import pathlib
import time
from decimal import Decimal
from typing import Callable, Literal

import msgspec
import pytest

from polypy import TRADE_STATUS
from polypy.constants import CHAIN_ID
from polypy.exceptions import (
    ManagerInvalidException,
    PositionNegativeException,
    StreamException,
    SubscriptionException,
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
from polypy.stream.user import CHANNEL, BufferThreadSettings, UserStream
from polypy.structs import OrderWSInfo, TradeWSInfo
from polypy.typing import dec

test_pth = pathlib.Path(__file__).parent


@pytest.fixture
def market_id():
    # can be any ID because we do not have any server in the background but instead simulate websocket
    # via directly calling on_msg
    return "0x0000000000000000000000000000000000000000000000000000000000000000"


@pytest.fixture
def yes_asset_id():
    # can be any ID because we do not have any server in the background but instead simulate websocket
    #   via directly calling on_msg
    # asset_ids are only used within monitor thread, which we turn off for regular testing and test
    #   specifically in a dedicated test
    return (
        "72000000000000000000000000000000000000000000000000000000000000000000000000000"
    )


@pytest.fixture
def no_asset_id():
    # can be any ID because we do not have any server in the background but instead simulate websocket
    #   via directly calling on_msg
    # asset_ids are only used within monitor thread, which we turn off for regular testing and test
    #   specifically in a dedicated test
    return (
        "101000000000000000000000000000000000000000000000000000000000000000000000000000"
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
    return PositionManager(local_host_addr, None, 100)


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

    # noinspection PyProtectedMember
    def _closure(
        untrack_insert_status: INSERT_STATUS | list[INSERT_STATUS] | None = None,
        untrack_trade_status: TRADE_STATUS | list[TRADE_STATUS] | None = None,
        monitor_assets_thread_s: float | None = None,
        buffer_settings: BufferThreadSettings | None = (5, 5_000),
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
            untrack_insert_status,
            untrack_trade_status,
            monitor_assets_thread_s,
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

    if streams:
        stream = streams[0]
        # noinspection PyProtectedMember
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
def trade_info_maker_sell_matched(mock_std_post_order):
    with open(test_pth / "data/test_trade_info_maker_buy_matched.json", "r") as f:
        data = msgspec.json.decode(
            json.dumps(json.load(f)), type=TradeWSInfo, strict=False
        )
    data = msgspec.structs.replace(data, side=SIDE.SELL)

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


# noinspection DuplicatedCode
def test_buffer_trade_info_taker_order(
    streamer, yes_asset_id, trade_info_taker_buy_matched
):
    stream, om, pm = streamer(None)

    stream.on_msg(trade_info_taker_buy_matched)  # this simulates websocket msg recv
    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
    time.sleep(0.1)

    assert om.valid is True
    assert pm.valid is True
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
    stream, om, pm = streamer(None)

    stream.on_msg(trade_info_maker_buy_matched)
    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
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
    stream, om, pm = streamer(None)

    data_placement, data_update = order_info_maker_buy_placement_update

    # this simulates websocket msg recv which first will be buffered until rest call returns
    stream.on_msg(data_placement)
    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
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
    assert pm.valid is False
    assert pm.position_dict[USDC].size_total == 100

    with pytest.raises(ManagerInvalidException):
        _ = pm.balance_total


def test_no_buffer_trade_info_taker_order(
    streamer, yes_asset_id, trade_info_taker_buy_matched
):
    stream, om, pm = streamer(buffer_settings=None)

    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
    stream.on_msg(trade_info_taker_buy_matched)  # this simulates websocket msg recv
    time.sleep(0.1)

    assert om.valid is True
    assert pm.valid is True
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
    assert pm.valid is False
    assert pm.position_dict[USDC].size_total == 100

    with pytest.raises(ManagerInvalidException):
        _ = pm.balance_total


def test_no_buffer_trade_info_maker_order(
    streamer, yes_asset_id, trade_info_maker_buy_matched
):
    stream, om, pm = streamer(buffer_settings=None)

    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
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
    assert pm.valid is False
    assert pm.position_dict[USDC].size_total == 100

    with pytest.raises(ManagerInvalidException):
        _ = pm.balance_total


def test_no_buffer_order_info_maker_order(
    streamer, yes_asset_id, order_info_maker_buy_placement_update
):
    stream, om, pm = streamer(buffer_settings=None)

    data_placement, data_update = order_info_maker_buy_placement_update

    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
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


def test_untrack_trade_status_matched(
    streamer,
    yes_asset_id,
    trade_info_maker_sell_matched,
    trade_info_taker_sell_matched,
    callback_clean,
    private_key,
):
    # empty vs non-empty
    # other position empty not getting untracked
    # maker vs tacker

    stream, om, pm = streamer(
        callback_clean=callback_clean, untrack_trade_status=TRADE_STATUS.MATCHED
    )

    pm.track(Position.create("1234", 0))
    # maker position
    pos_maker = Position.create(
        "95786924372760057572092804419385993470890190892343223404877167501659835222533",
        100,
    )
    pm.track(pos_maker)
    # taker position
    pos_taker = Position.create(
        "25742247876332768458781360292043764039507900813404980298479194684402595556451",
        100,
    )
    pm.track(pos_taker)

    # we only need some fake orders s.t. trade messages will be assigned to the correct position manager
    order_maker = create_limit_order(
        0.99,
        10,
        "95786924372760057572092804419385993470890190892343223404877167501659835222533",
        SIDE.SELL,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    order_taker = create_limit_order(
        0.99,
        10,
        "25742247876332768458781360292043764039507900813404980298479194684402595556451",
        SIDE.SELL,
        0.01,
        False,
        CHAIN_ID.POLYGON,
        private_key,
        None,
        SIGNATURE_TYPE.EOA,
    )
    order_maker.id = (
        "0xc512c86c90ce3b4f657808cb6000000000000000000000000000000000000000"
    )
    order_taker.id = (
        "0xe9f3d896fba10ed3600000000000000000000000000000000000000000000000"
    )
    om.track(order_maker, False)
    om.track(order_taker, False)

    stream.on_msg(trade_info_maker_sell_matched)
    stream.on_msg(trade_info_taker_sell_matched)
    time.sleep(0.2)

    assert len(om.token_ids) == 2
    assert callback_clean.cleaned_orders == []
    assert callback_clean.cleaned_positions == []

    print(om.valid, pm.valid)
    print(om._invalid_reason)

    assert (
        pm.get_by_id(
            "25742247876332768458781360292043764039507900813404980298479194684402595556451"
        ).size_total
        == 95
    )
    assert (
        pm.get_by_id(
            "95786924372760057572092804419385993470890190892343223404877167501659835222533"
        ).size_total
        == 95
    )
    assert pm.get_by_id("1234").size_total == 0
    assert pm.balance_total == 106
    assert len(pm.asset_ids) == 4

    pos_taker.size = 5
    pos_maker.size = 5
    stream.on_msg(trade_info_maker_sell_matched)
    stream.on_msg(trade_info_taker_sell_matched)
    time.sleep(0.2)

    assert len(om.token_ids) == 2
    assert callback_clean.cleaned_orders == []
    assert len(callback_clean.cleaned_positions) == 2
    assert callback_clean.cleaned_positions[0].empty
    assert callback_clean.cleaned_positions[1].empty

    assert list(pm.asset_ids) == [USDC, "1234"]
    assert pm.balance_total == 112
    assert pm.get_by_id("1234").size_total == 0


def test_raise_untrack_trade_status_wrong_init(streamer):
    with pytest.raises(StreamException) as record:
        streamer(untrack_trade_status=TRADE_STATUS.RETRYING)
    assert "TRADE_STATUS.RETRYING" in str(record)

    with pytest.warns() as record:
        streamer(untrack_trade_status=TRADE_STATUS.MATCHED)
    assert "TERMINAL_TRADE_STATI" in str(record[0].message)


def test_untrack_insert_status_at_unmatched_order_info(
    streamer, yes_asset_id, order_info_maker_buy_unmatched, callback_clean
):
    stream, om, pm = streamer(
        untrack_insert_status=[INSERT_STATUS.UNMATCHED],
        callback_clean=callback_clean,
    )
    data = order_info_maker_buy_unmatched

    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
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


def test_untrack_insert_status_at_matched_order_info(
    streamer,
    yes_asset_id,
    order_info_maker_buy_placement_update,
    callback_clean,
    trade_info_maker_buy_matched,
):
    stream, om, pm = streamer(
        untrack_insert_status=[INSERT_STATUS.MATCHED],
        callback_clean=callback_clean,
    )
    data_placement, data_update = order_info_maker_buy_placement_update

    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
    assert len(om.token_ids) == 1
    assert len(om.get(status=INSERT_STATUS.LIVE)) == 1
    assert len(callback_clean.cleaned_orders) == 0
    assert len(callback_clean.cleaned_positions) == 0

    stream.on_msg(data_placement)
    time.sleep(0.5)
    assert len(om.token_ids) == 1
    assert len(om.get(status=INSERT_STATUS.LIVE)) == 1
    assert len(callback_clean.cleaned_orders) == 0
    assert len(callback_clean.cleaned_positions) == 0

    stream.on_msg(data_update)
    time.sleep(0.5)

    assert len(om.token_ids) == 0
    assert len(callback_clean.cleaned_orders) == 1
    assert len(callback_clean.cleaned_positions) == 0
    assert callback_clean.cleaned_orders[0].size_matched == 5
    assert callback_clean.cleaned_orders[0].status is INSERT_STATUS.MATCHED
    assert pm.balance_total == 100  # no trade message, only order infos

    stream.on_msg(trade_info_maker_buy_matched)
    time.sleep(0.5)

    assert pm.balance_total == 100 - 3


def test_raises_untrack_instert_status_wrong_init(streamer):
    with pytest.raises(StreamException) as record:
        streamer(untrack_insert_status=[INSERT_STATUS.LIVE])
    assert "TERMINAL_INSERT_STATI" in str(record)

    with pytest.raises(StreamException) as record:
        streamer(untrack_insert_status=INSERT_STATUS.DEFINED)
    assert "TERMINAL_INSERT_STATI" in str(record)

    with pytest.warns() as record:
        streamer(untrack_insert_status=INSERT_STATUS.MATCHED)
    assert "exceptions" in str(record[0].message)


def test_untrack_insert_status_unmatched_trade_status_confirmed(
    streamer,
    yes_asset_id,
    order_info_maker_buy_unmatched,
    trade_info_maker_buy_confirmed,
    callback_clean,
):
    stream, om, pm = streamer(
        callback_clean=callback_clean,
        untrack_insert_status=INSERT_STATUS.UNMATCHED,
        untrack_trade_status=TRADE_STATUS.CONFIRMED,
    )
    pm.track(
        Position(
            "95786924372760057572092804419385993470890190892343223404877167501659835222533",
            0,
            True,
            2,
        )
    )

    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
    assert len(om.token_ids) == 1
    assert len(pm.asset_ids) == 2
    assert len(om.get(status=INSERT_STATUS.LIVE)) == 1
    assert len(callback_clean.cleaned_orders) == 0
    assert len(callback_clean.cleaned_positions) == 0

    stream.on_msg(order_info_maker_buy_unmatched)
    stream.on_msg(trade_info_maker_buy_confirmed)
    # because trade info is in status CONFIRMED, no transactions will happen (only at MATCHED)
    time.sleep(0.25)

    assert len(om.token_ids) == 0
    assert len(pm.asset_ids) == 1
    assert len(callback_clean.cleaned_orders) == 1
    assert len(callback_clean.cleaned_positions) == 1
    assert callback_clean.cleaned_orders[0].size_matched == 0
    assert callback_clean.cleaned_orders[0].status is INSERT_STATUS.UNMATCHED
    assert (
        callback_clean.cleaned_positions[0].asset_id
        == "95786924372760057572092804419385993470890190892343223404877167501659835222533"
    )
    assert callback_clean.cleaned_positions[0].size == 0
    assert pm.balance_total == 100


def test_raise_monitor_order_assets(streamer, mock_std_post_order):
    stream, om, pm = streamer(monitor_assets_thread_s=0.01)
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
    assert pm.valid is True

    om.limit_order(
        0.99, 5, "1234", SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
    time.sleep(0.1)

    assert om.valid is False
    assert pm.valid is False


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
    pm = PositionManager(local_host_addr, None, 100)

    stream = UserStream(
        "ws://localhost:8002/",
        [(om1, pm), (om2, pm)],
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        [INSERT_STATUS.UNMATCHED, INSERT_STATUS.CANCELED],
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
    assert pm.valid is True

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
    assert pm.valid is False

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
    pm = PositionManager(local_host_addr, None, 100)

    stream = UserStream(
        "ws://localhost:8002/",
        [(om1, pm), (om2, pm)],
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        [INSERT_STATUS.CANCELED, INSERT_STATUS.UNMATCHED],
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
    assert pm.valid is True
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
    assert pm.valid is False
    assert pm.position_dict[USDC].size_total == 100 - 9
    assert (
        pm.position_dict[
            "95786924372760057572092804419385993470890190892343223404877167501659835222533"
        ].size_total
        == 15
    )

    with pytest.raises(ManagerInvalidException):
        _ = pm.balance_total

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
    pm = PositionManager(local_host_addr, None, 100)

    stream = UserStream(
        "ws://localhost:8002/",
        (None, pm),
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        [INSERT_STATUS.CANCELED, INSERT_STATUS.UNMATCHED],
        None,
        buffer_thread_settings=(0.01, 10),
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
        [INSERT_STATUS.CANCELED, INSERT_STATUS.UNMATCHED],
        None,
        update_mode="implicit",
    )
    stream._stop_token.clear()
    stream.pre_start()

    order_placement, order_update = order_info_maker_buy_placement_update

    stream.on_msg(order_placement)
    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.BUY, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )
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
    pm = PositionManager(local_host_addr, None, 100)

    order_placement, order_update = order_info_maker_buy_placement_update

    stream = UserStream(
        "ws://localhost:8002/",
        [(om1, pm), (om2, pm)],
        (market_id, yes_asset_id, no_asset_id),
        api_key,
        secret,
        passphrase,
        [INSERT_STATUS.CANCELED, INSERT_STATUS.UNMATCHED],
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
    assert pm.valid is True
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

    om.limit_order(
        0.99, 5, yes_asset_id, SIDE.SELL, 0.01, TIME_IN_FORCE.GTC, None, neg_risk=None
    )

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
    stream, om, _ = streamer(buffer_settings=(0.1, 10), callback_exc=callback)

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
    stream, om, _ = streamer(buffer_settings=(0.1, 10), callback_exc=callback)

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
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=(0.01, 10))
    data_placement, _ = order_info_maker_buy_placement_update

    stream.on_msg(data_placement)
    time.sleep(0.1)

    assert om.valid is True
    assert pm.valid is True


def test_implicit_no_buffer_order_info(streamer, order_info_maker_buy_placement_update):
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=None)
    data_placement, _ = order_info_maker_buy_placement_update

    stream.on_msg(data_placement)
    time.sleep(0.1)

    assert om.valid is True
    assert pm.valid is True


def test_implicit_buffer_trade_info(streamer, trade_info_taker_buy_matched):
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=(0.01, 10))

    stream.on_msg(trade_info_taker_buy_matched)
    time.sleep(0.1)

    assert om.valid is True
    assert pm.valid is True


def test_implicit_no_buffer_trade_info(streamer, trade_info_taker_buy_matched):
    stream, om, pm = streamer(update_mode="implicit", buffer_settings=None)

    stream.on_msg(trade_info_taker_buy_matched)
    time.sleep(0.1)

    assert om.valid is True
    assert pm.valid is True


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
        # noinspection PyTypeChecker
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
        dec(0.99),
        dec(5),
        yes_asset_id,
        SIDE.BUY,
        dec(0.01),
        TIME_IN_FORCE.GTC,
        None,
        neg_risk=None,
    )
    time.sleep(0.1)

    assert order_manager.valid is True
    assert position_manager.valid is True
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


def test_raises_max_subscriptions(
    local_host_addr,
    private_key,
    api_key,
    secret,
    passphrase,
    market_id,
    yes_asset_id,
    no_asset_id,
):  # sourcery skip: extract-duplicate-method
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
    pm = PositionManager(local_host_addr, None, 100)

    with pytest.raises(SubscriptionException) as record:
        UserStream(
            "ws://localhost:8002/",
            (om, pm),
            [(market_id, yes_asset_id, no_asset_id), ("123", "456", "789")],
            api_key,
            secret,
            passphrase,
            [INSERT_STATUS.UNMATCHED, INSERT_STATUS.CANCELED],
            None,
            max_subscriptions=1,
        )
    assert "Exceeding" in str(record)
