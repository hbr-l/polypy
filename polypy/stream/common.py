import contextlib
import datetime
import threading
import traceback
import warnings
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Callable, NoReturn, Self

import msgspec
from websockets import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from polypy.exceptions import StreamException


class CHANNEL(StrEnum):
    MARKET = "market"
    USER = "user"


# todo port to async
class AbstractStreamer(ABC):
    def __init__(
        self,
        url: str,
        subscribe_params: dict[str:Any],
        ping_time: float | None = 5,
        callback_msg: Callable[[Self, dict[str, Any]], None] | None = None,
        callback_exc: Callable[[Self, Exception], None] | None = None,
        msgspec_type: msgspec.Struct | Any | type = Any,
        msgspec_strict: bool = True,
    ) -> None:
        self.url = url
        self.subscribe_params = subscribe_params
        self.ping_time = ping_time

        self.callback_exc = callback_exc
        self.callback_msg = callback_msg
        self.msgspec_type = msgspec_type
        self.msgspec_strict = msgspec_strict

        self._stop_token = threading.Event()
        self.thread: threading.Thread | None = None
        self.ws: ClientConnection | None = None
        self.exc: list[Exception] = []

    def _run(self) -> None:
        while not self._stop_token.is_set():
            try:
                self._loop()
            except Exception as e:
                self.register_exception(e)

    def register_exception(self, exc: Exception) -> None | NoReturn:
        if self._stop_token.is_set():
            # no need to bother, we're exiting anyway
            warnings.warn(
                f"{datetime.datetime.now()} | Exception in {self.__class__.__name__}._main_thread suppressed,"
                f"since {self.__class__.__name__}.stop() was called. "
                f"Full Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )

        warnings.warn(
            f"{datetime.datetime.now()} | Exception in {self.__class__.__name__}."
            f"Traceback: {exc.__class__.__name__}({exc})."
            f"Full Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}."
        )

        with contextlib.suppress(Exception):
            if self.callback_exc:
                self.callback_exc(self, exc)

        self.exc.append(exc)

        # force shutdown
        self._stop_token.set()

        raise exc

    def _loop(self) -> None:
        with connect(self.url) as self.ws:
            self.ws.send(msgspec.json.encode(self.subscribe_params))

            while not self._stop_token.is_set():
                try:
                    bytes_msg = self.ws.recv(self.ping_time, decode=False)
                    self._handle_msg(bytes_msg)
                except TimeoutError:
                    with contextlib.suppress(ConnectionClosed, AttributeError):
                        self.ws.ping("PING")
                    continue
                except ConnectionClosed as e:
                    if self._stop_token.is_set():
                        # no need to emit warning, because we stopped the websocket
                        return

                    warnings.warn(
                        f"{datetime.datetime.now()} | Re-connecting subscription: {self.subscribe_params}. "
                        f"Traceback: {e.__class__.__name__}({e})."
                        f"Full Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                    )
                    return

    def _handle_msg(self, bytes_msg: bytes) -> None:
        # somehow, received JSON dict is wrapped in a list
        msgs = msgspec.json.decode(
            bytes_msg, type=self.msgspec_type, strict=self.msgspec_strict
        )
        for msg in msgs:
            self.on_msg(msg)

            if self.callback_msg:
                self.callback_msg(self, msg)

    @abstractmethod
    def on_msg(self, msg: dict[Any, Any] | msgspec.Struct) -> None:
        ...

    @abstractmethod
    def pre_start(self) -> None:
        ...

    @abstractmethod
    def post_stop(self) -> None:
        ...

    def start(self) -> None:
        self.pre_start()
        self._stop_token.clear()

        if self.thread is not None:
            raise StreamException(
                "Internal error: self.thread is not None. There might have been an exception "
                "after which the stream was not closed properly."
            )
        if self.ws is not None:
            raise StreamException(
                "Internal error: self.ws is not None. There might have been an exception "
                "after which the stream was not closed properly."
            )

        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self, join: bool, timeout: float | None = None) -> None:
        self._stop_token.set()
        self.ws.close()

        if join:
            self.thread.join(timeout)

        self.thread = None
        self.ws = None
        self.post_stop()

        if self.exc:
            raise ExceptionGroup("StreamerExceptionGroup", self.exc)
