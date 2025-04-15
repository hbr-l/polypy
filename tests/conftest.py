import contextlib
import functools
import hashlib
import json
import pathlib
import random
import threading
import time
import traceback
from enum import Enum, auto
from typing import Any, Callable, Protocol, Sequence, TypeAlias, TypeVar

import msgspec.json
import numpy as np
import pytest
import responses
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from websockets.sync.server import ServerConnection, serve

from polypy.rounding import round_floor, round_half_even


@pytest.fixture
def private_key() -> str:
    """Publicly known private key for testing."""
    return "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"


@pytest.fixture
def api_key() -> str:
    """Empty API key for testing."""
    return "000000000-0000-0000-0000-000000000000"


@pytest.fixture
def passphrase() -> str:
    """Empty passphrase for testing."""
    return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture
def secret() -> str:
    """Empty secret for testing."""
    return "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


@pytest.fixture
def local_host_addr() -> str:
    return "http://localhost:8080"


@pytest.fixture
def rsps():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


@pytest.fixture
def mock_neg_risk(rsps, local_host_addr):
    def _closure(token_id):
        rsps.add(
            responses.GET,
            f"{local_host_addr}/neg-risk?token_id={token_id}",
            json={"neg_risk": False},
        )

    return _closure


@pytest.fixture
def mock_tick_size(rsps, local_host_addr):
    def _closure(token_id, tick_size=0.01):
        rsps.add(
            responses.GET,
            f"{local_host_addr}/tick-size?token_id={token_id}",
            json={"minimum_tick_size": tick_size},
        )

    return _closure


@pytest.fixture
def mock_post_order(rsps, local_host_addr):
    def _closure(data, status=200):
        rsps.upsert(
            responses.POST, f"{local_host_addr}/order", status=status, json=data
        )

    return _closure


@pytest.fixture
def mock_cancel_order(rsps, local_host_addr):
    def _closure(canceled: list[str], not_canceled: dict[str, str] | None, status=200):
        if not_canceled is None:
            not_canceled = {}

        data = {"canceled": canceled, "not_canceled": not_canceled}
        rsps.upsert(
            responses.DELETE, f"{local_host_addr}/orders", status=status, json=data
        )

    return _closure


@pytest.fixture
def json_book_to_arrays():
    def _closure(
        pth: str | pathlib.Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with open(pth, "r") as f:
            book_token = json.load(f)

        bids, asks = book_token["bids"], book_token["asks"]

        bid_p = np.array([d["price"] for d in bids], dtype=float)
        bid_q = np.array([d["size"] for d in bids], dtype=float)

        ask_p = np.array([d["price"] for d in asks], dtype=float)
        ask_q = np.array([d["size"] for d in asks], dtype=float)

        return bid_p, bid_q, ask_p, ask_q

    return _closure


@pytest.fixture
def patch_py_clob_client_rounding(monkeypatch):
    from py_clob_client.order_builder import helpers

    monkeypatch.setattr("polypy.order.market.round_floor", helpers.round_down)
    monkeypatch.setattr("polypy.order.market.round_half_even", helpers.round_normal)

    monkeypatch.setattr("polypy.order.limit.round_floor", helpers.round_down)
    monkeypatch.setattr("polypy.order.limit.round_half_even", helpers.round_normal)

    # we're yielding, because then fixture can be deactivated within test via `patch_clob_rounding.stop`
    # if at some given point bug-free rounding shall be applied instead of py_clob_client rounding routines
    yield

    monkeypatch.setattr("polypy.order.market.round_floor", round_floor)
    monkeypatch.setattr("polypy.order.market.round_half_even", round_half_even)

    monkeypatch.setattr("polypy.order.limit.round_floor", round_floor)
    monkeypatch.setattr("polypy.order.limit.round_half_even", round_half_even)


class HashDict(dict):
    def __hash__(self):
        return int(hashlib.sha1(msgspec.json.encode(self)).hexdigest(), 16)


def hash_dict(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = (HashDict(**arg) if isinstance(arg, dict) else arg for arg in args)
        kwargs = {
            k: HashDict(**v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


@pytest.fixture(scope="function")
def clob_client_factory():
    # mock http requests

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:

        @hash_dict
        @functools.lru_cache(maxsize=4)
        def _closure(
            tick_size,
            neg_risk,
            market_id,
            book,
            complement_book,
            token_id,
            complement_token_id,
            host,
            private_key,
        ):
            if not isinstance(book, dict):
                book = book.to_dict(
                    market_id=market_id,
                    timestamp=int(time.time()),
                    hash_str=book.hash(market_id=market_id, timestamp=int(time.time())),
                )
            if not isinstance(complement_book, dict):
                complement_book = complement_book.to_dict(
                    market_id=market_id,
                    timestamp=int(time.time()),
                    hash_str=complement_book.hash(
                        market_id=market_id, timestamp=int(time.time())
                    ),
                )

            # todo complete http REST request mocking for other functionalities
            rsps.add(
                responses.GET,
                f"{host}/tick-size?token_id={token_id}",
                json={"minimum_tick_size": tick_size},
            )
            rsps.add(
                responses.GET,
                f"{host}/neg-risk?token_id={token_id}",
                json={"neg_risk": neg_risk},
            )
            rsps.add(responses.GET, f"{host}/book?token_id={token_id}", json=book)

            rsps.add(
                responses.GET,
                f"{host}/tick-size?token_id={complement_token_id}",
                json={"minimum_tick_size": tick_size},
            )
            rsps.add(
                responses.GET,
                f"{host}/neg-risk?token_id={complement_token_id}",
                json={"neg_risk": neg_risk},
            )
            rsps.add(
                responses.GET,
                f"{host}/book?token_id={complement_token_id}",
                json=complement_book,
            )

            return ClobClient(host, POLYGON, private_key)

        yield _closure


F = TypeVar("F", bound=Callable)


class FixSeedCallable(Protocol):
    def __call__(self, fn: F, seed: int = 0) -> F:
        ...


@pytest.fixture
def fix_seed() -> FixSeedCallable:
    def _closure(fn: F, seed: int = 0) -> F:
        random.seed(seed)
        return fn

    return _closure


# noinspection PyPep8Naming
class CLICK_ACTION(Enum):
    SEND = auto()
    DELAY = auto()
    PARTIAL_DROP = auto()
    SKIP = auto()
    DISCONNECT = auto()


DelayId: TypeAlias = int
DelayTime: TypeAlias = float


class MockServerClickOn:
    def __init__(self):
        self.click_value = {}
        self.stop_token = False
        self.counter = -1
        self.server = None
        self.thread = None
        self.action = None
        self.exception = None

        self._lock = threading.Lock()

        self.drop_idx = None
        self.delay_params = None

    def _click(self, action: CLICK_ACTION, sleep: float | None = 0.08):
        self.action = action

        for k in self.click_value.keys():
            self.click_value[k] = True

        self.counter += 1

        if sleep:
            time.sleep(sleep)

    def send(self, sleep: float | None = 0.08):
        return self._click(CLICK_ACTION.SEND, sleep)

    def delay_send(
        self, delay_params: dict[DelayId, DelayTime], sleep: float | None = 0.08
    ):
        """Send message to all connections, but delay message to certain connections."""
        self.delay_params = delay_params
        return self._click(CLICK_ACTION.DELAY, sleep)

    def drop_send(self, drop_idx: Sequence[int], sleep: float | None = 0.08):
        """Send message to all connections except drop_idx."""
        self.drop_idx = drop_idx
        return self._click(CLICK_ACTION.PARTIAL_DROP, sleep)

    def skip(self, sleep: float | None = 0.08):
        return self._click(CLICK_ACTION.SKIP, sleep)

    def disconnect(self, conn_idx: int, sleep: float | None = 0.08):
        """Disconnects websocket connection, conn_idx is in order in which connections were made.
        If client reconnects, connection ids will rotate (oldest active
        connection at index 0, newest reconnected connection at highest index).
        Disconnected connection will also be removed from .click_value dictionary."""
        self.action = CLICK_ACTION.DISCONNECT

        self.click_value[list(self.click_value.keys())[conn_idx]] = True

        if sleep:
            time.sleep(sleep)

    def remove_conn(self, conn):
        with self._lock:
            del self.click_value[conn]

    def register_conn(self, conn):
        with self._lock:
            self.click_value[conn] = self.action != CLICK_ACTION.DISCONNECT

    def halt(self, conn):
        self.click_value[conn] = False

    def stop(self, join: bool, timeout: int | None):
        self.stop_token = True
        time.sleep(0.01)
        self.server.shutdown()
        if join:
            self.thread.join(timeout)


@pytest.fixture(scope="function")
def mock_server_click_on():
    click_on = MockServerClickOn()

    def closure(
        data_pth: str | pathlib.Path,
        port: int = 8002,
    ) -> tuple[MockServerClickOn, str, list[dict[str, Any]]]:
        with open(data_pth) as f:
            txt = f.readlines()
            data = [json.loads(line.replace("'", '"')) for line in txt]

        connections = []

        def handler(conn: ServerConnection):
            click_on.register_conn(conn)
            connections.append(conn)
            while True:
                with contextlib.suppress(TimeoutError):
                    conn.recv(0)

                conn_idx = connections.index(conn)

                if click_on.click_value[conn] is True:
                    item = data[click_on.counter]
                    action = click_on.action

                    if action == CLICK_ACTION.SEND:
                        conn.send(json.dumps([item]))
                    elif action == CLICK_ACTION.SKIP:
                        pass
                    elif action == CLICK_ACTION.DELAY:
                        if conn_idx in click_on.delay_params:
                            time.sleep(click_on.delay_params[conn_idx])
                        conn.send(json.dumps([item]))
                    elif action == CLICK_ACTION.PARTIAL_DROP:
                        if conn_idx not in click_on.drop_idx:
                            conn.send(json.dumps([item]))
                    elif action == CLICK_ACTION.DISCONNECT:
                        connections.remove(conn)
                        click_on.remove_conn(conn)
                        break
                    elif action is None:
                        # that is the case, if connection is freshly registered but no click action has
                        #   happened yet. We have to set click_value to True when registering connections s.t.
                        #   socket can catch up to the last action performed.
                        pass
                    else:
                        # raising an exception here wouldn't have any effect because this
                        #   runs in a child thread and does not propagate the exception
                        # instead we attach an exception to click_on object and assert at teardown time
                        click_on.exception = ValueError(f"Unknown action: {action}.")
                        print(
                            f"\033[91m"
                            + f"Exception in mock_server_click_on! Unknown action: {action}."
                            + "\033[0m"
                        )  # print red warning text to console
                        click_on.stop(False, None)

                    click_on.halt(conn)

                if click_on.stop_token is True:
                    conn.close()
                    connections.remove(conn)
                    break

                time.sleep(0.00001)

        def _test_serve():
            try:
                with serve(handler, "localhost", port) as server:
                    click_on.server = server
                    server.serve_forever()
            except Exception as e:
                print(
                    f"\033[91m" + f"Exception in mock_server_click_on: {e}."
                    f"Full Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}."
                    + "\033[0m"
                )  # print red warning text to console
                click_on.exception = e
                raise e

        thread = threading.Thread(target=_test_serve)
        click_on.thread = thread
        thread.start()

        return click_on, data[0]["asset_id"], data

    yield closure
    click_on.stop(True, 2)
    assert click_on.exception is None
