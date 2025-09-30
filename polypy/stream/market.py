import datetime
import socket
import warnings
from collections import namedtuple
from enum import IntEnum
from functools import lru_cache
from typing import Any, Callable, Literal, Self

import numpy as np

from polypy import BookEvent
from polypy.book.hashing import guess_check_orderbook_hash
from polypy.book.order_book import OrderBookProtocol
from polypy.book.parsing import message_to_orderbook
from polypy.constants import ENDPOINT
from polypy.exceptions import (
    EventTypeException,
    OrderBookException,
    PolyPyException,
    StreamException,
)
from polypy.ipc.shm import SharedArray
from polypy.rest.api import get_book_summaries
from polypy.stream.common import CHANNEL, MessageStreamer
from polypy.structs import LastTradePriceEvent, MarketEvent, PriceChangeEvent
from polypy.typing import ZerosFactoryFunc, ZerosProtocol

CheckHashParams = namedtuple(
    "CheckHashParams", ["nth_price_change", "max_emission_delay"]
)


# noinspection PyPep8Naming
class STATUS_ORDERBOOK(IntEnum):
    ERROR = -2
    CORRUPTED = -1
    UNKNOWN = 0
    VERIFIED = 1


# todo decouple in OrderBookStream message type from procedure
# todo factor out BookHashChecker
class MarketStream(MessageStreamer):
    def __init__(
        self,
        ws_endpoint: ENDPOINT | str,
        books: list[OrderBookProtocol] | OrderBookProtocol,
        check_hash_params: CheckHashParams | None,
        rest_endpoint: ENDPOINT | str | None,
        ws_channel: CHANNEL | str = CHANNEL.MARKET,
        ping_time: float | None = 5,
        status_arr_factory: type[ZerosProtocol] | ZerosFactoryFunc = np.zeros,
        last_traded_arr_factory: type[ZerosProtocol] | ZerosFactoryFunc = np.zeros,
        callback_msg: Callable[[Self, dict[str, Any]], None] | None = None,
        callback_exc: Callable[[Self, Exception], None] | None = None,
        sock: socket.socket | None = None,
        max_size: int | None = 2**20,
        max_queue: int | None | tuple[int | None, int | None] = 16,
    ):
        """

        Parameters
        ----------
        ws_endpoint: ENDPOINT
            usually ENDPOINT.WS
        books
        check_hash_params
        rest_endpoint: str | None,
            if None, no REST request to fetch order book in case order book hash cannot be confirmed
        ws_channel
        ping_time
        status_arr_factory: type[ZerosProtocol] | ZerosFactoryFunc
            zeros-array factory (either constructor or factory function)
            with dtype=int, used to store status per each order book
            (same ordering along dim=0 as `books`).
            If using shared memory (multiprocessing), use SharedArray.factory(...) with `fill_value=0`, and
            use `status_orderbook_shared(...)` to read back if `MarketStream` is defined in child process
            (use shm_name).
        last_traded_arr_factory: type[ZerosProtocol] | ZerosFactoryFunc
            zeros-array factory (either constructor or factory function)
            with kwargs dtype=np.dtype("U80"), used to store last traded price per each order book
            (same ordering along dim=0 as `books`).
            If using shared memory (multiprocessing), use SharedArray.factory(...) with `fill_value=""`, and
            use `last_traded_price_shared(...)` to read back if `MarketStream` is defined in child process
            (use shm_name).
        callback_msg: Callable[[str | bytes, Self], None] | None
            callback, takes message (JSON parsed dict) and self as input and has no output.
            This callback is only invoked on messages that are processed to change the order book
            or last traded price, and which are not filtered out (i.e., unknown event_type, etc.)
        callback_exc: Callable[[Exception, Self], None] | None
            callback, takes exception and self as input and has no output. Only invoked if any exception happens.
            Can be used for, e.g. observer pattern.
        sock: socket.socket | None, default=None
            custom socket to pass to `websockets.sync.client.connect()`

        Notes
        -----
        If used in separate process:
            - zeros_factory: instantiate/ return sharedMemory array for dtype=np.dtype("U80") and dtype=int
            - books: OrderBookProtocol must implement sharedMemory array as
              quantity buffers (see argument `zeros_factory` of OrderBookProtocol)

        Locking, if necessary, is at the user's responsibility:
            - zeros_factory: return array class, that locks __getitem__ and __setitem__
            - books: OrderBookProtocol must implement lock-based array access for quantity buffers (either by
              wrapping/ patching or via argument 'zeros_factory')
        Locking is only necessary, if external write operations occur, and most likely is only necessary
        for OrderBookProtocol (if at all...).
        """
        # todo logging?
        # todo implement __getstate__ and __setstate__ (pickling during multiprocessing)
        # todo refactor: more functional pattern
        # todo self._update_last_traded_price(): factor out event_type to decouple (and use Enum for event_type)

        if not isinstance(books, list):
            books = [books]

        if ws_endpoint[-1] != "/":
            ws_endpoint = f"{ws_endpoint}/"

        self.endpoint = rest_endpoint
        self.status_arr_factory = status_arr_factory
        self.last_traded_arr_factory = last_traded_arr_factory

        self.book_dict: dict[str, OrderBookProtocol] = {
            book.token_id: book for book in books
        }
        self._book_idx = {book.token_id: i for i, book in enumerate(books)}

        # will be initialized at pre_start()
        self.counter_dict: dict[str, int] | None = None
        # we use an array instead of a dict[book_id, LastTradePriceEvent] or dict[book_id, STATUS_ORDERBOOK]
        #   s.t. shared memory can be used for multiprocessing if necessary
        self.last_traded_price_arr: ZerosProtocol | None = None
        self.status_arr: ZerosProtocol | None = None

        if check_hash_params is not None:
            self.nth_price_change = check_hash_params[0]
            self.max_emission_delay = check_hash_params[1]
        else:
            self.max_emission_delay = None
            self.nth_price_change = None

        params = {
            "auth": {},
            "markets": [],
            "assets_ids": list(self.book_dict.keys()),
            "type": "market",
        }

        def _closure_callback_exc(_, exc: Exception) -> None:
            # we cannot propagate exceptions from child thread to parent thread,
            #   but at least we can invalidate status_orderbook, which at least
            #   has the chance to be noticed from parent thread
            self.status_arr[:] = -2
            for k in self.counter_dict.keys():
                self.counter_dict[k] += 1
            callback_exc(self, exc)

        super().__init__(
            f"{ws_endpoint}{ws_channel}",
            params,
            ping_time,
            callback_msg,
            _closure_callback_exc,
            MarketEvent | list[BookEvent],
            False,
            sock,
            max_size,
            max_queue,
        )

    def pre_start(self) -> None:
        self.counter_dict = {book.token_id: 0 for book in self.book_dict.values()}

        # not implemented as dicts: enable sharedMemory if necessary
        self.last_traded_price_arr = self.last_traded_arr_factory(
            (len(self._book_idx), 6), dtype=np.dtype("U80")
        )
        self.last_traded_price_arr[:] = ""  # just to be sure...
        self.status_arr = self.status_arr_factory(len(self._book_idx), dtype=int)

    def post_stop(self) -> None:
        return

    def on_msg(self, msg: MarketEvent) -> None:
        if msg.event_type == "last_trade_price":
            self._update_last_traded_price(msg)
        elif msg.event_type == "tick_size_change" or msg.event_type == "book":
            self._update_book(msg)
        elif msg.event_type == "price_change":
            asset_ids = set(price_change.asset_id for price_change in msg.price_changes)

            has_updated = False
            for asset_id in asset_ids:
                if asset_id not in self.book_dict:
                    continue

                self._update_book(msg, asset_id)
                has_updated = True

            if not has_updated:
                raise StreamException(
                    f"No books with matching asset_ids found. Message: {msg}."
                )
        else:
            raise EventTypeException(f"Unknown event_type: {msg.event_type}")

    def _update_book(self, msg: MarketEvent, asset_id: str | None = None) -> None:
        if asset_id is None:
            asset_id = msg.asset_id

        self.book_dict[asset_id] = message_to_orderbook(msg, self.book_dict[asset_id])
        if not self._check_hash(msg, asset_id) and self.endpoint is not None:
            warnings.warn(
                f"{datetime.datetime.now()}: REST fetch order book due to invalid order book hash."
            )

            resp = get_book_summaries(self.endpoint, asset_id)

            self._update_book(resp)

    def _check_hash(self, msg: MarketEvent, asset_id: str | None = None) -> bool:
        # sourcery skip: class-extract-method
        if self.nth_price_change is None:
            return True

        if asset_id is None:
            asset_id = msg.asset_id
        arr_id = self._book_idx[asset_id]

        if msg.event_type == "book":
            # fresh order book, no need to check anything
            self.counter_dict[asset_id] = 0
            self.status_arr[arr_id] = 1
            return True

        if msg.event_type == "tick_size_change" or msg.event_type == "last_trade_price":
            # this is tick_size_change
            return True

        self.counter_dict[asset_id] += 1
        self.status_arr[arr_id] = 0
        # all other event_type would already have thrown an exception in _update_book

        if self.counter_dict[asset_id] < self.nth_price_change:
            # counter not reached yet, no need to check hash
            return True

        # time to check order book hash
        return self._guess_hash(asset_id, arr_id, msg)

    def _guess_hash(self, asset_id: str, book_id: int, msg: PriceChangeEvent) -> bool:
        timestamps = [int(msg.timestamp) - i for i in range(self.max_emission_delay)]
        if guess_check_orderbook_hash(
            msg.price_changes[-1].hash, self.book_dict[asset_id], msg.market, timestamps
        )[0]:
            # good to go, reset counter
            self.counter_dict[asset_id] = 0
            self.status_arr[book_id] = 1
            return True

        warnings.warn(
            f"{datetime.datetime.now()}: Invalid order book hash. Message: {msg}."
        )
        self.status_arr[book_id] = -1
        return False

    def _write_last_traded_price_arr(self, book_id: int, msg: MarketEvent) -> None:
        # we use an array instead of a dict[book_id, LastTradePriceEvent] s.t.
        #   shared memory can be used for multiprocessing if necessary
        self.last_traded_price_arr[book_id, 0] = msg.price
        self.last_traded_price_arr[book_id, 1] = msg.side
        self.last_traded_price_arr[book_id, 2] = msg.size
        self.last_traded_price_arr[book_id, 3] = msg.timestamp
        self.last_traded_price_arr[book_id, 4] = msg.fee_rate_bps
        self.last_traded_price_arr[book_id, 5] = msg.market

    def _update_last_traded_price(self, msg: LastTradePriceEvent) -> None:
        asset_id = msg.asset_id
        arr_id = self._book_idx[asset_id]

        if self.last_traded_price_arr[arr_id, 0] == "":
            # initial entry
            self._write_last_traded_price_arr(arr_id, msg)
            return

        if int(self.last_traded_price_arr[arr_id, 3]) < int(msg.timestamp):
            # only update, if msg is newer than last update
            self._write_last_traded_price_arr(arr_id, msg)

    def last_traded_price(self, token_id: str) -> LastTradePriceEvent:
        if len(token_id) > 80:
            raise PolyPyException(
                "Internal exception: len(token_id) > 80. Array dtype `U80` not sufficient."
            )

        arr_id = self._book_idx[token_id]
        return LastTradePriceEvent(
            market=self.last_traded_price_arr[arr_id, 5],
            asset_id=token_id,
            fee_rate_bps=self.last_traded_price_arr[arr_id, 4],
            price=self.last_traded_price_arr[arr_id, 0],
            side=self.last_traded_price_arr[arr_id, 1],
            size=self.last_traded_price_arr[arr_id, 2],
            timestamp=self.last_traded_price_arr[arr_id, 3],
        )

    def status_orderbook(
        self, token_id: str, mode: Literal["except", "silent", "warn"] = "except"
    ) -> STATUS_ORDERBOOK:
        """Get current order book hash status.

        Parameters
        ----------
        token_id: str
            token ID (asset ID) of corresponding order book.
        mode: Literal["except", "silent"], default="except"
            if "except", raises OrderBookException if STATUS_ORDERBOOK.ERROR,
            if "warn", emits warning,
            if "silent", just returns STATUS_ORDERBOOK

        Returns
        -------
        STATUS_ORDERBOOK:
            -2: internal error -> raises OrderBookException
            -1: corrupted order book hash (needs to fetch REST request /book)
            0: no statement (not checked yet)
            1: verified order book hash

        Raises
        ------
        OrderBookException: if STATUS_ORDERBOOK.ERROR
        """
        status = STATUS_ORDERBOOK(self.status_arr[self._book_idx[token_id]])
        # we use an array instead of a dict[book_id, STATUS_ORDERBOOK] s.t.
        #   shared memory can be used for multiprocessing if necessary

        if status is STATUS_ORDERBOOK.ERROR:
            if mode == "except":
                raise OrderBookException(
                    f"Orderbook status invalidated. This indicates an exception "
                    f"within internal threads of {self.__class__.__name__}."
                )
            elif mode == "warn":
                warnings.warn(
                    f"Orderbook status invalidated. This indicates an exception "
                    f"within internal threads of {self.__class__.__name__}."
                )
            elif mode != "silent":
                raise ValueError(f"Unknown mode: {mode}.")

        return status


@lru_cache
def _book_idx(token_id: str, books: list[OrderBookProtocol]) -> int:
    book_ids = [book.token_id for book in books]
    return book_ids.index(token_id)


def status_orderbook_shared(
    token_id: str, books: list[OrderBookProtocol], shared_array: SharedArray | str
) -> STATUS_ORDERBOOK:
    idx = _book_idx(token_id, books)

    if isinstance(shared_array, str):
        shared_array = SharedArray(len(books), shared_array, False, int, 0)

    return STATUS_ORDERBOOK(shared_array[idx])


def last_traded_price_shared(
    token_id: str, books: list[OrderBookProtocol], shared_array: SharedArray | str
) -> LastTradePriceEvent:
    idx = _book_idx(token_id, books)

    if isinstance(shared_array, str):
        shared_array = SharedArray(
            (len(books), 6), shared_array, False, np.dtype("U80"), ""
        )

    return LastTradePriceEvent(
        market=shared_array[idx, 5],
        asset_id=token_id,
        fee_rate_bps=shared_array[idx, 4],
        price=shared_array[idx, 0],
        side=shared_array[idx, 1],
        size=shared_array[idx, 2],
        timestamp=shared_array[idx, 3],
    )
