import copy
import pathlib
import threading
import time
import warnings
from typing import Any

import numpy as np
import pytest
import responses
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from polypy.book import OrderBook, guess_check_orderbook_hash
from polypy.exceptions import EventTypeException, OrderBookException
from polypy.stream.market import STATUS_ORDERBOOK, CheckHashParams, MarketStream

# cases
# 1. delay randomly
# 2. drop one socket
# 3. drop all socket - without REST fetch
# 4. tick_size_change
# 5. drop all socket - with REST fetch (separate test)
# 6. reconnect (separate test)
# 7. REST denial (separate test)
# 8. unknown event_type (separate test)
# 9. callback (separate test) (use to verify different thread_ids called)
# 10. multiple books (separate test)
# 11. wrong asset_id (separate test)
# 12. multiple last_trade_price (separate test)

# 1 - 4: done
# 5: done
# 6: done
# 7: done
# 8: done
# 9: done
# 10: done
# 11: done
# 12: done

# assert
# 1. order book hash
# 2. status_orderbook
# 3. last_traded_price
# 4. tick_size
# 5. counter_dict
# 6. warnings (invalid hash after skips)


test_pth = pathlib.Path(__file__).parent


@pytest.fixture
def setup_streamer():
    class Storage:
        def __init__(self):
            self.val = None

    stream_storage = Storage()

    def closure(
        asset_id: str,
        check_hash_params: CheckHashParams | None,
        endpoint: str | None,
        ping_time: float | None,
        nb_redundant_skt: int,
        url: str = "ws://localhost:8002/",
        callback_msg=None,
        callback_exception=None,
        buffer_size=10,
    ):
        book = OrderBook(asset_id, 0.001)
        streamer = MarketStream(
            url,
            book,
            check_hash_params=check_hash_params,
            rest_endpoint=endpoint,
            ws_channel="",
            buffer_size=buffer_size,
            ping_time=ping_time,
            nb_redundant_skt=nb_redundant_skt,
            callback_msg=callback_msg,
            callback_exception=callback_exception,
        )
        stream_storage.val = streamer
        streamer.start()
        time.sleep(2)
        # todo: strangely enough, we cannot reduce the sleep time here, don't know why though, but else tests fail...
        return streamer, book

    yield closure
    stream_storage.val.stop(True, 1)
    time.sleep(0.05)


@pytest.fixture
def callback_exception_storage():
    class ExcStorage:
        def __init__(self):
            self.val = []

        def __call__(self, exc: Exception, *_) -> None:
            self.val.append(exc)

    return ExcStorage()


@pytest.fixture
def callback_thread_id_storage():
    class Storage:
        def __init__(self):
            self.val = set()

        def __call__(self, msg: dict[str, Any], *_) -> None:
            self.val.add(threading.get_ident())

    return Storage()


def test_test_server(mock_server_click_on):
    # sourcery skip: extract-method

    port = 8004
    click_on, asset_id, _ = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt", port=port
    )

    with connect(f"ws://localhost:{port}") as ws:
        start_t = time.time()
        while len(click_on.click_value) != 1 and (time.time() - start_t < 5):
            time.sleep(0.01)
        assert time.time() - start_t < 5.01

        click_on.send()
        msg_1 = ws.recv(5)
        assert len(msg_1) > 1

        click_on.send()
        msg_2 = ws.recv(5)
        assert msg_1 != msg_2
        assert len(msg_2) > 1


def test_test_server_delay(mock_server_click_on):
    port = 8002
    click_on, asset_id, _ = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt", port=port
    )

    def _ws(resp, port_nb):
        with connect(f"ws://localhost:{port_nb}") as ws:
            while True:
                try:
                    ws.recv(1)
                    resp.append(time.time())
                except TimeoutError:
                    pass
                except ConnectionClosedError:
                    break

    resp1 = []
    resp2 = []

    thread1 = threading.Thread(target=_ws, args=(resp1, port))
    thread2 = threading.Thread(target=_ws, args=(resp2, port))
    thread1.start()
    time.sleep(0.1)  # force thread1 to connect first
    thread2.start()

    start_t = time.time()
    while len(click_on.click_value) != 2 and (time.time() - start_t < 5):
        time.sleep(0.01)
    assert time.time() - start_t < 5.01

    click_on.delay_send({0: 0.1}, 0.25)
    click_on.drop_send([1])

    start_t = time.time()
    while any(val is True for val in click_on.click_value.items()) and (
        time.time() - start_t < 5
    ):
        time.sleep(0.01)
    assert time.time() - start_t < 5.01

    click_on.stop(True, 5)
    thread1.join(1)
    thread2.join(1)

    assert resp1[0] > resp2[0]
    assert len(resp1) == 2
    assert len(resp2) == 1


def assert_market_streamer_state(
    streamer: MarketStream,
    book: OrderBook,
    market_id: str,
    expected_book_status: int,
    expected_ltp_price: str,
    expected_ltp_size: str,
    expected_ltp_side: str,
    expected_tick_size: float,
    expected_target_hash: str | None,
    expected_timestamp: str | int | float,
    expected_count: int,
):
    assert streamer.status_orderbook(book.token_id) == expected_book_status
    assert streamer.last_traded_price(book.token_id)["price"] == expected_ltp_price
    assert streamer.last_traded_price(book.token_id)["size"] == expected_ltp_size
    assert streamer.last_traded_price(book.token_id)["side"] == expected_ltp_side
    assert streamer.last_traded_price(book.token_id)["event_type"] == "last_trade_price"
    assert streamer.last_traded_price(book.token_id)["asset_id"] == book.token_id
    assert book.tick_size == expected_tick_size
    assert streamer.counter_dict[book.token_id] == expected_count

    if not expected_ltp_size:
        assert streamer.last_traded_price(book.token_id)["market"] == ""
    else:
        assert streamer.last_traded_price(book.token_id)["market"] == market_id

    if expected_target_hash is not None:
        assert (
            guess_check_orderbook_hash(
                expected_target_hash,
                book,
                market_id,
                [int(expected_timestamp) - i for i in range(10)],
            )[0]
            is True
        )


# noinspection DuplicatedCode
def test_marketstream_delay_partial_drop_skip_no_request(
    mock_server_click_on, setup_streamer
):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt"
    )
    streamer, book = setup_streamer(
        asset_id, CheckHashParams(3, 15), None, None, 2, buffer_size=5
    )

    market_id = data[0]["market"]

    # order of clicks:
    # 1. send - book
    # 2. delay - price_change
    # 3. delay- price_change
    # 4. partial drop - price_change
    # 5. partial drop - price_change
    # 6. send - last_trade_price
    # 7. skip - price_change
    # 8. send - price_change  (this triggers false hash)
    # 9. send - book
    # 10. send - tick_size_change

    # assert:
    # orderbook_status
    # warnings (skip-send)
    # last_trade_price
    # tick_size
    # order book hash
    # counter_dict

    assert np.sum(book.bids[1]) == np.sum(book.asks[1]) == 0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warnings expected

        click_on.send()  # 1. send - book
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        click_on.delay_send({1: 0.1}, 0.5)  # 2. delay - price_change
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            0,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            1,
        )

        click_on.delay_send({1: 0.1}, 0.5)  # 3. delay- price_change
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            0,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            2,
        )

        # 4. partial drop - price_change + hash check should trigger
        click_on.drop_send([0])
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        click_on.drop_send([0])  # 5. partial drop - price_change
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            0,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            1,
        )

        click_on.send()  # 6. send - last_trade_price
        # last_trade_price does not change book status or count_dict
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            0,
            data[click_on.counter]["price"],
            data[click_on.counter]["size"],
            data[click_on.counter]["side"],
            0.001,
            None,  # last_trade_price has no hash
            data[click_on.counter]["timestamp"],
            1,
        )

    click_on.skip()  # 7. skip - price_change

    # 8. send - price_change  (this triggers false hash)
    with pytest.warns() as record:
        # we expect a warning

        # we artificially force hash check internally
        streamer.counter_dict[book.token_id] = 2
        click_on.send(0.5)
    assert len(record) == 1  # emits warning
    assert "Invalid order book hash" in str(record[0].message)
    # use assert_args, because we double-check later on with similar settings against valid hash
    assert_args = [
        streamer,
        book,
        market_id,
        -1,
        "0.52",
        "25",
        "BUY",
        0.001,
        None,  # we know that hash is invalid at this point
        data[click_on.counter]["timestamp"],
        3,  # we artificially increased to 2, click_on.click's next price_change message further increases +1
    ]
    assert_market_streamer_state(*assert_args)
    with pytest.raises(AssertionError):
        # double check hash is invalid
        # we expect an AssertionError
        assert_args[-3] = data[click_on.counter]["hash"]
        assert_market_streamer_state(*assert_args)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warnings expected

        click_on.send()  # 9. send - book
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "0.52",
            "25",
            "BUY",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        click_on.send()  # 10. send - tick_size_change
        # tick_size_change does not change book status or counter_dict
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "0.52",
            "25",
            "BUY",
            0.01,
            None,  # tick_size_change has no hash
            data[click_on.counter]["timestamp"],
            0,
        )

    assert np.sum(book.bids[1]) > 0
    assert np.sum(book.asks[1]) > 0

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[asset_id])))
        == len(streamer.buffer_dict[asset_id])
        == streamer.buffer_size
    )


@responses.activate
def test_marketstream_fallback_mock_request(mock_server_click_on, setup_streamer):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt"
    )
    streamer, book = setup_streamer(
        asset_id, CheckHashParams(1, 15), "https://test_endpoint.com", None, 2
    )
    assert np.sum(book.bids[1]) == np.sum(book.asks[1]) == 0

    market_id = data[0]["market"]

    # order of clicks:
    # 1. send - book
    # 2. skip - price_change
    # 3. skip - price_change
    # 4. skip - price_change
    # 5. skip - price_change
    # 6. skip - last_trade_price
    # 7. skip - price_change
    # 8. send - price_change  (this triggers false hash) -> request
    # 9. send - book

    # mock request
    rest_data = copy.deepcopy(data[8])
    del rest_data["event_type"]
    responses.get(f"https://test_endpoint.com/book?token_id={asset_id}", json=rest_data)

    with warnings.catch_warnings():
        # no warnings expected
        warnings.simplefilter("error")
        click_on.send()  # 1. send - book
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        click_on.skip()  # 2
        click_on.skip()  # 3
        click_on.skip()  # 4
        click_on.skip()  # 5
        click_on.skip()  # 6
        click_on.skip()  # 7

    with pytest.warns() as record:
        click_on.send(2)
        # this should force a REST GET to manually fetch order book

    assert len(record) == 2  # emits warning
    assert "Invalid order book hash" in str(record[0].message)
    assert "REST fetch" in str(record[1].message)
    # after manually fetching, order book should be OK again
    assert_market_streamer_state(
        streamer,
        book,
        market_id,
        1,
        "",
        "",
        "",
        0.001,
        rest_data["hash"],  # check against fetched rest data
        rest_data["timestamp"],
        0,
    )
    fetched_bids = copy.deepcopy(book.bids[1])
    fetched_asks = copy.deepcopy(book.asks[1])

    with warnings.catch_warnings():
        # no warnings expected
        warnings.simplefilter("error")
        click_on.send()
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

    assert fetched_bids.tolist() == book.bids[1].tolist()
    assert fetched_asks.tolist() == book.asks[1].tolist()

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[asset_id])))
        == len(streamer.buffer_dict[asset_id])
        == 3
    )


def test_marketstream_wrong_asset_id_and_callback_exception(
    mock_server_click_on, setup_streamer, callback_exception_storage
):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_wrong_asset_id.txt"
    )
    streamer, book = setup_streamer(
        asset_id,
        CheckHashParams(3, 15),
        None,
        None,
        2,
        callback_exception=callback_exception_storage,
    )

    assert np.sum(book.bids[1]) == np.sum(book.asks[1]) == 0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warnings expected
        click_on.send()  # 1. send - book

    with pytest.warns() as record:
        # warning expected
        click_on.send()

    assert len(record.list) > 0
    assert "Exception in" in str(record[0].message)
    assert -2 in list(streamer.status_arr)
    assert streamer.counter_dict[asset_id] == streamer.nb_redundant_skt
    assert all(isinstance(exc, KeyError) for exc in callback_exception_storage.val)

    with pytest.raises(OrderBookException):
        streamer.status_orderbook(asset_id)

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[asset_id])))
        == len(streamer.buffer_dict[asset_id])
        == 1
    )


def test_marketstream_unknown_event_type_and_callback_exception(
    mock_server_click_on, setup_streamer, callback_exception_storage
):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_unknown_event_type.txt"
    )
    streamer, book = setup_streamer(
        asset_id,
        CheckHashParams(3, 15),
        None,
        None,
        2,
        callback_exception=callback_exception_storage,
    )

    click_on.send()  # 1. send - book
    with pytest.warns() as record:
        # warning expected
        click_on.send()

    assert "Exception in" in str(record[0].message)
    assert -2 in list(streamer.status_arr)
    assert all(
        isinstance(exc, EventTypeException) for exc in callback_exception_storage.val
    )

    with pytest.raises(OrderBookException):
        streamer.status_orderbook(asset_id)

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[asset_id])))
        == len(streamer.buffer_dict[asset_id])
        == 2
    )


def test_marketstream_status_orderbook_mode(
    mock_server_click_on, setup_streamer, callback_exception_storage
):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_unknown_event_type.txt"
    )
    streamer, book = setup_streamer(
        asset_id,
        None,
        None,
        None,
        2,
        callback_exception=callback_exception_storage,
    )

    click_on.send()  # 1. send - book
    with pytest.warns() as record:
        # warning expected: wrong event_type causes internal exception
        click_on.send()

    assert "Exception in" in str(record[0].message)

    assert streamer.status_orderbook(asset_id, mode="silent") is STATUS_ORDERBOOK.ERROR

    with pytest.warns() as record:
        streamer.status_orderbook(asset_id, mode="warn")
    assert "Orderbook status invalidated" in str(record[0].message)

    with pytest.raises(OrderBookException):
        streamer.status_orderbook(asset_id, mode="except")


@responses.activate
def test_marketstream_rest_denial_and_callback_exception(
    mock_server_click_on, setup_streamer, callback_exception_storage
):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt"
    )
    streamer, book = setup_streamer(
        asset_id,
        CheckHashParams(1, 15),
        "https://test_endpoint.com",
        None,
        2,
        callback_exception=callback_exception_storage,
    )

    # order of clicks:
    # 1. send - book
    # 2. skip - price_change
    # 3. skip - price_change (this triggers false hash) -> request -> denial

    # mock request
    responses.get(
        f"https://test_endpoint.com/book?token_id={asset_id}",
        json={"error": "not found"},
        status=404,
    )

    click_on.send()  # 1. send - book
    click_on.skip()  # 2. skip - price_change

    with pytest.warns() as record:
        click_on.send(2)
        # this should force a REST GET to manually fetch order book -> denial

    assert len(record) == 3  # emitted warning
    assert "Invalid order book hash" in str(record[0].message)
    assert "REST fetch" in str(record[1].message)
    assert "Exception in" in str(record[2].message)
    assert -2 in list(streamer.status_arr)
    assert all(isinstance(exc, Exception) for exc in callback_exception_storage.val)
    assert len(callback_exception_storage.val) > 0

    with pytest.raises(OrderBookException):
        streamer.status_orderbook(asset_id)

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[asset_id])))
        == len(streamer.buffer_dict[asset_id])
        == 2
    )


@responses.activate
def test_marketstream_rest_denial_no_recursion(
    mock_server_click_on, setup_streamer, callback_exception_storage
):
    """
    Test if we run into recursion if hash in book message is wrong (which never will be the case).
    Note: despite order book being corrupted, we do not expect any exception, because we do not
        check hash after book message for performance reason and thereby there is no trigger for any exception
    """
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt"
    )
    streamer, book = setup_streamer(
        asset_id,
        CheckHashParams(1, 15),
        "https://test_endpoint.com",
        None,
        2,
        callback_exception=callback_exception_storage,
    )

    # order of clicks:
    # 1. send - book
    # 2. skip - price_change
    # 3. skip - price_change (this triggers false hash) -> request -> wrong hash

    # mock request
    rest_data = copy.deepcopy(data[8])
    rest_data["hash"] = "1234"
    del rest_data["event_type"]
    responses.get(
        f"https://test_endpoint.com/book?token_id={asset_id}",
        json=rest_data,
    )

    click_on.send()  # 1. send - book
    click_on.skip()  # 2. skip - price_change

    with pytest.warns() as record:
        click_on.send(2)
        # this should force a REST GET to manually fetch order book -> wrong hash

    assert len(record) == 2  # emitted warning
    assert "Invalid order book hash" in str(record[0].message)
    assert "REST fetch" in str(record[1].message)

    # despite order book being corrupted, we do not expect any exceptions because
    # we do not check hash after book messages (unlikely that hash is incorrect after book message)
    assert len(callback_exception_storage.val) == 0
    streamer.status_orderbook(asset_id)

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[asset_id])))
        == len(streamer.buffer_dict[asset_id])
        == 3
    )
    assert "1234" in streamer.buffer_dict[asset_id]


def test_marketstream_callback_message(
    mock_server_click_on, setup_streamer, callback_thread_id_storage
):
    nb_skts = 2
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt"
    )
    _, _ = setup_streamer(
        asset_id,
        None,
        None,
        None,
        nb_skts,
        callback_msg=callback_thread_id_storage,
    )
    # force other thread to take over via DELAY
    click_on.send()
    click_on.delay_send({0: 0.1})
    click_on.send()
    click_on.delay_send({0: 0.1})
    click_on.send(0.5)

    time.sleep(0.2)

    assert len(callback_thread_id_storage.val) == nb_skts


# noinspection DuplicatedCode
def test_marketstream_multi_last_trade_price(mock_server_click_on, setup_streamer):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_multiple_last_trade_price.txt",
    )
    streamer, book = setup_streamer(
        asset_id, CheckHashParams(3, 15), None, None, 2, buffer_size=5
    )

    market_id = data[0]["market"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warnings expected

        click_on.send()  # 1. send - book
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        click_on.send()  # 2. send - last_trade_price
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "0.52",
            "25",
            "BUY",
            0.001,
            None,  # last_trade_price has no hash
            data[click_on.counter]["timestamp"],
            0,
        )
        assert "hash" not in data[click_on.counter].keys()

        click_on.send()  # 3. send - last_trade_price
        # -> no change because wrong timestamp
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "0.52",
            "25",
            "BUY",
            0.001,
            None,  # last_trade_price has no hash
            data[click_on.counter]["timestamp"],
            0,
        )
        assert "hash" not in data[click_on.counter].keys()

        click_on.send()  # 4. send - last_trade_price -> change
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            1,
            "0.52",
            "50",
            "BUY",
            0.001,
            None,  # last_trade_price has no hash
            data[click_on.counter]["timestamp"],
            0,
        )
        assert "hash" not in data[click_on.counter].keys()

        click_on.send()  # 5. send - price_change
        assert_market_streamer_state(
            streamer,
            book,
            market_id,
            0,
            "0.52",
            "50",
            "BUY",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            1,
        )

        assert np.sum(book.bids[1]) > 0
        assert np.sum(book.asks[1]) > 0

        # check deque max size not exceeded and only unique hashes
        assert (
            len(set((streamer.buffer_dict[asset_id])))
            == len(streamer.buffer_dict[asset_id])
            == 2
        )


# noinspection DuplicatedCode
def test_marketstream_multi_book(mock_server_click_on):
    click_on, _, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_multi_book.txt",
    )

    book1 = OrderBook(
        "72936048731589292555781174533757608024096898681344338816447372274344589246891",
        0.001,
    )
    book2 = OrderBook(
        "101669189743438912873361127612589311253202068943959811456820079057046819967115",
        0.001,
    )
    market_id1 = "0x84c0ffe3f56cb357ff5ff8bc5d2182ae90be4dd6718e8403a6af472b452dbfa8"
    market_id2 = "0x9d1f0296f3a016727193d2b45704e0debc3b8048fe0715f8bb1e91550d321872"
    streamer = MarketStream(
        "ws://localhost:8002/",
        [book1, book2],
        check_hash_params=CheckHashParams(3, 15),
        rest_endpoint=None,
        ws_channel="",
        ping_time=None,
        nb_redundant_skt=3,
    )
    streamer.start()
    time.sleep(2)
    # todo: strangely enough, we cannot reduce the sleep time here, don't know why though, but else tests fail...

    assert np.sum(book1.bids[1]) == np.sum(book1.asks[1]) == 0
    assert np.sum(book2.bids[1]) == np.sum(book2.asks[1]) == 0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warnings expected

        click_on.send()  # 1. send - book - book1
        assert_market_streamer_state(
            streamer,
            book1,
            market_id1,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        click_on.send()  # 2. send - book - book2
        assert_market_streamer_state(
            streamer,
            book2,
            market_id2,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        click_on.delay_send({1: 0.1}, 0.5)  # 3. delay - price_change - book1
        assert_market_streamer_state(
            streamer,
            book1,
            market_id1,
            0,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            1,
        )

        click_on.delay_send({1: 0.1}, 0.2)  # 4. delay- price_change - book1
        assert_market_streamer_state(
            streamer,
            book1,
            market_id1,
            0,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            2,
        )

        click_on.delay_send({1: 0.1}, 0.2)  # 5. delay - price_change - book2
        assert_market_streamer_state(
            streamer,
            book2,
            market_id2,
            0,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            1,
        )

        click_on.delay_send({1: 0.1}, 0.2)  # 6. delay- price_change - book2
        assert_market_streamer_state(
            streamer,
            book2,
            market_id2,
            0,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            2,
        )

        click_on.drop_send([0, 2])  # 7. partial drop - book - book2
        book2_counter = click_on.counter
        assert_market_streamer_state(
            streamer,
            book2,
            market_id2,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

        # 8. partial drop - price_change -> should trigger hash check - book1
        click_on.drop_send([0, 2], 0.1)
        assert_market_streamer_state(
            streamer,
            book1,
            market_id1,
            1,
            "",
            "",
            "",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

    click_on.skip()  # 9. skip - book1

    # 10. send - last_trade_price - book1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warnings expected
        click_on.send()
        # last_trade_price performs no hash checking and no increase in counter
        assert_market_streamer_state(
            streamer,
            book1,
            market_id1,
            1,
            "0.52",
            "25",
            "BUY",
            0.001,
            None,  # no hash for last_trade_price
            data[click_on.counter]["timestamp"],
            0,
        )

    # 11. send - price_change (this triggers false hash) - book1
    with pytest.warns() as record:
        # we expect a warning
        # we artificially force hash check internally
        streamer.counter_dict[book1.token_id] = 2
        click_on.send(0.5)
    assert len(record) == 1  # emits warning
    assert "Invalid order book hash" in str(record[0].message)
    # use assert_args, because we double-check later on with similar settings against valid hash
    assert_args = [
        streamer,
        book1,
        market_id1,
        -1,
        "0.52",
        "25",
        "BUY",
        0.001,
        None,  # we know that hash is invalid at this point
        data[click_on.counter]["timestamp"],
        3,  # we artificially increased to 2, click_on.click's next price_change message further increases +1
    ]
    assert_market_streamer_state(*assert_args)
    with pytest.raises(AssertionError):
        # double check hash is invalid
        # we expect an AssertionError
        assert_args[-3] = data[click_on.counter]["hash"]
        assert_market_streamer_state(*assert_args)

    # check that nothing has change for book2
    assert_market_streamer_state(
        streamer,
        book2,
        market_id2,
        1,
        "",
        "",
        "",
        0.001,
        data[book2_counter]["hash"],
        data[book2_counter]["timestamp"],
        0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # no warnings expected

        click_on.send()  # 12. send - book - book1
        assert_market_streamer_state(
            streamer,
            book1,
            market_id1,
            1,
            "0.52",
            "25",
            "BUY",
            0.001,
            data[click_on.counter]["hash"],
            data[click_on.counter]["timestamp"],
            0,
        )

    assert np.sum(book1.bids[1]) > 0
    assert np.sum(book1.asks[1]) > 0
    assert np.sum(book2.bids[1]) > 0
    assert np.sum(book2.asks[1]) > 0

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[book1.token_id])))
        == len(streamer.buffer_dict[book1.token_id])
        == 6
    )
    assert (
        len(set((streamer.buffer_dict[book2.token_id])))
        == len(streamer.buffer_dict[book2.token_id])
        == 4
    )

    streamer.stop(True, 5)


def test_marketstream_reconnect(
    mock_server_click_on, setup_streamer, callback_thread_id_storage
):  # sourcery skip: extract-duplicate-method
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt"
    )
    streamer, book = setup_streamer(
        asset_id,
        None,
        None,
        None,
        2,
        callback_msg=callback_thread_id_storage,
        buffer_size=2,
    )

    # force other thread to take over via DELAY
    click_on.send()
    click_on.delay_send({0: 0.1})
    click_on.send()
    click_on.delay_send({0: 0.1}, 0.5)

    time.sleep(0.5)

    assert len(callback_thread_id_storage.val) == 2
    assert len(click_on.click_value.keys()) == 2

    callback_thread_id_storage.val = set()
    prev_click_value_keys = set(click_on.click_value.keys())

    with pytest.warns() as record:
        click_on.disconnect(0, 0.5)
    assert len(record.list) == 1
    assert "Re-connecting" in str(record[0].message)

    # Give some time to re-connect
    time.sleep(2)

    # clean cache
    callback_thread_id_storage.val = set()

    # force other thread to take over via DELAY
    click_on.send()
    click_on.delay_send({0: 0.1})
    click_on.send()
    click_on.delay_send({0: 0.1}, 0.5)

    # on streamer side, we continue to use the same thread for re-connection
    #   (just opening new websocket from within the same thread) -> 2 threads
    # however, on the server side, we open a new thread for each new
    #   connection object per client re-connect -> different set of keys
    assert len(callback_thread_id_storage.val) == 2
    assert prev_click_value_keys != set(click_on.click_value.keys())
    assert len(click_on.click_value.keys()) == 2

    assert_market_streamer_state(
        streamer,
        book,
        data[0]["market"],
        0,
        "0.52",
        "25",
        "BUY",
        0.001,
        data[click_on.counter]["hash"],
        data[click_on.counter]["timestamp"],
        0,
    )

    assert np.sum(book.bids[1]) > 0
    assert np.sum(book.asks[1]) > 0

    # check deque max size not exceeded and only unique hashes
    assert (
        len(set((streamer.buffer_dict[asset_id])))
        == len(streamer.buffer_dict[asset_id])
        == streamer.buffer_size
    )


def test_marketstream_nb_sockets_and_buffer_protection(
    mock_server_click_on, setup_streamer, callback_thread_id_storage
):
    click_on, asset_id, data = mock_server_click_on(
        data_pth=test_pth / "data/test_server_messages.txt"
    )
    streamer, book = setup_streamer(
        asset_id,
        None,
        None,
        None,
        4,
        callback_msg=callback_thread_id_storage,
        buffer_size=20,
    )

    click_on.delay_send({0: 0.3, 1: 0.2, 2: 0.1}, 0.2)  # nb 3 receives immediately
    click_on.delay_send({0: 0.1, 1: 0.2, 3: 0.3}, 0.2)  # nb 2 receives immediately
    click_on.delay_send({0: 0.1, 2: 0.3, 3: 0.2}, 0.2)  # nb 1 receives immediately
    click_on.delay_send({1: 0.1, 2: 0.1, 3: 0.1}, 0.2)  # nb 0 receives immediately

    assert len(callback_thread_id_storage.val) == 4
    assert len(streamer.buffer_dict[book.token_id]) == 4
    assert len(streamer.buffer_dict[book.token_id]) == len(
        set(streamer.buffer_dict[book.token_id])
    )

    assert_market_streamer_state(
        streamer,
        book,
        data[0]["market"],
        0,
        "",
        "",
        "",
        0.001,
        data[click_on.counter]["hash"],
        data[click_on.counter]["timestamp"],
        0,
    )
