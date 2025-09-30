import math
import threading
import warnings
from contextlib import contextmanager
from types import UnionType
from typing import Any, Literal, Protocol, Self, Sequence, get_args

import numpy as np
from _decimal import Decimal
from numpy.typing import NDArray

from polypy.book.hashing import dict_to_sha1
from polypy.book.parsing import (
    _number_to_str,
    _set_book_event,
    dict_to_market_event_struct,
    guess_tick_size,
    merge_quotes_to_order_summaries,
    message_to_orderbook,
)
from polypy.book.tick import SharedTickSize, TickSize, TickSizeFactory, TickSizeProtocol
from polypy.constants import ENDPOINT, N_DIGITS_SIZE
from polypy.exceptions import OrderBookException
from polypy.ipc.lock import SharedProcRLock
from polypy.ipc.shm import FinalizedSharedMemory, SharedDecimalArray
from polypy.order.common import SIDE
from polypy.rest.api import get_book_summaries, get_tick_size
from polypy.rounding import round_half_even
from polypy.structs import BookEvent
from polypy.typing import ArrayCoercible, NumericAlias, ZerosFactoryFunc, ZerosProtocol


class OrderBookProtocol(Protocol):
    token_id: str
    tick_size: NumericAlias

    @property
    def min_tick_size(self) -> float:
        ...

    def update_tick_size(self, endpoint: str | ENDPOINT) -> NumericAlias:
        ...

    def sync(self, endpoint: str, include_tick_size: bool) -> None:
        ...

    @property
    def dtype(self) -> type:
        ...

    @property
    def bids(self) -> tuple[NDArray, ArrayCoercible]:
        ...

    @property
    def asks(self) -> tuple[NDArray, ArrayCoercible]:
        ...

    @property
    def bid_sizes(self) -> ArrayCoercible:
        ...

    @property
    def ask_sizes(self) -> ArrayCoercible:
        ...

    @property
    def bid_prices(self) -> ArrayCoercible:
        ...

    @property
    def ask_prices(self) -> ArrayCoercible:
        ...

    @property
    def best_bid_price(self) -> NumericAlias:
        ...

    @property
    def best_ask_price(self) -> NumericAlias:
        ...

    @property
    def midpoint_price(self) -> NumericAlias:
        ...

    @property
    def best_bid_size(self) -> NumericAlias:
        ...

    @property
    def best_ask_size(self) -> NumericAlias:
        ...

    def bid_size(self, price: NumericAlias) -> NumericAlias:
        ...

    def ask_size(self, price: NumericAlias) -> NumericAlias:
        ...

    def set_bids(
        self,
        bid_prices: list[NumericAlias] | ArrayCoercible,
        bid_sizes: list[NumericAlias] | ArrayCoercible,
    ) -> None:
        ...

    def set_asks(
        self,
        ask_prices: list[NumericAlias] | ArrayCoercible,
        ask_sizes: list[NumericAlias] | ArrayCoercible,
    ) -> None:
        ...

    def null_bids(self) -> None:
        ...

    def null_asks(self) -> None:
        ...

    def reset_bids(
        self,
        bid_prices: list[NumericAlias] | NDArray[NumericAlias] | None = None,
        bid_sizes: list[NumericAlias] | NDArray[NumericAlias] | None = None,
    ) -> None:
        ...

    def reset_asks(
        self,
        ask_prices: list[NumericAlias] | NDArray[NumericAlias] | None = None,
        ask_sizes: list[NumericAlias] | NDArray[NumericAlias] | None = None,
    ) -> None:
        ...

    @property
    def bids_summary(self) -> list[dict[str, str]]:
        ...

    @property
    def asks_summary(self) -> list[dict[str, str]]:
        ...

    def to_dict(
        self,
        market_id: str | None,
        timestamp: int | float | str | None,
        hash_str: str | None,
    ) -> dict[str, str]:
        ...

    def hash(self, market_id: str, timestamp: int | float | str) -> str:
        ...

    def marketable_price(
        self, side: Literal["BUY", "SELL"] | SIDE, amount: NumericAlias
    ) -> tuple[NumericAlias, NumericAlias]:
        ...


def calculate_marketable_price(
    price_levels: ArrayCoercible | Sequence[NumericAlias],
    size_levels: ArrayCoercible | Sequence[NumericAlias],
    amount: NumericAlias,
) -> tuple[NumericAlias, NumericAlias]:
    amount_levels = np.asarray(price_levels) * np.asarray(size_levels)
    cum_amount = np.cumsum(amount_levels)

    idx = np.searchsorted(cum_amount, amount)

    if idx == len(price_levels):
        raise OrderBookException("No marketable price at set target amount was found.")

    return price_levels[idx], cum_amount[idx]


def _zeroing_array(x: ArrayCoercible) -> ArrayCoercible:
    # need for this odd function:
    #   np.zeros(1, "U16") -> np.array([""]) instead of np.array(["0"])
    # so we manually force array to be all zeros

    dtype = type(x[0])
    # noinspection PyTypeChecker
    x[:] = dtype("0")

    return x


def _linspace_array(
    start: float, stop: float, num: int, nb_digits: int, dtype: type
) -> NDArray:
    arr = np.round(np.linspace(start, stop, num), nb_digits)

    if not isinstance(arr[0], dtype):
        arr = np.array([round_half_even(dtype(x), nb_digits) for x in arr])

    return arr


def _union_type(dtype_arr: type, dtype_item: type) -> UnionType:
    dtype_q = dtype_arr | dtype_item

    if float in get_args(dtype_q | None):
        # '(...| None)': ensures get_args always returns type of dtype
        # (if not Union yet (i.e. if not numpy dtype) get_args would return emtpy tuple)
        dtype_q |= int  # allow int upcast to float

    return dtype_q


def _check_quotes_shape(
    prices: list[NumericAlias] | ArrayCoercible,
    sizes: list[NumericAlias] | ArrayCoercible,
) -> None:
    if len(prices) != len(sizes) or len(sizes) == 0:
        raise IndexError(
            f"Arguments not allowed (len(prices)==len(sizes)!= 0). "
            f"Got: len(prices)={len(prices)}, len(sizes)={len(sizes)}."
        )


def _coerce_inbound_idx(
    level_idx: NDArray[int], sizes: list[NumericAlias] | ArrayCoercible, max_len: int
) -> tuple[NDArray[int], NDArray[NumericAlias]]:
    # todo could be more efficient if integrated into split_order_summaries_to_quotes() directly (?)

    sizes = np.asarray(sizes)

    underflow_idx = level_idx < 0
    underflow_sum = np.sum(sizes[underflow_idx])

    overflow_idx = level_idx > (max_len - 1)
    overflow_sum = np.sum(sizes[overflow_idx])

    out_of_bound_idx = underflow_idx | overflow_idx
    in_bound_idx = ~out_of_bound_idx

    level_idx = level_idx[in_bound_idx]
    sizes = sizes[in_bound_idx]

    if underflow_sum > 0:
        zero_idx = np.where(level_idx == 0)[0]
        if len(zero_idx):
            sizes[zero_idx[0]] += underflow_sum
        else:
            # todo optimize: pre-allocate instead of append
            level_idx = np.append(level_idx, 0)
            sizes = np.append(sizes, underflow_sum)

    if overflow_sum > 0:
        last_idx = np.where(level_idx == max_len - 1)[0]
        if len(last_idx):
            sizes[last_idx[0]] += overflow_sum
        else:
            # todo optimize: pre-allocate instead of append
            level_idx = np.append(level_idx, max_len - 1)
            sizes = np.append(sizes, overflow_sum)

    if np.any(out_of_bound_idx):
        warnings.warn(
            "Coercing level indices and sizes for out-of-bound prices (< 0 or > 1)."
        )

    return level_idx, sizes


class OrderBook:
    """Order Book class.

    Notes
    -----
    Threading:
        - this implementation is NOT multiprocessing-safe
        - though this implementation can be used in threaded applications (internal RLock)

    Dtype of sizes:
        - dtype of 'sizes' is determined by zeros_factory returned array. If, e.g. Decimals are required,
          then return zeros-array with Decimals in it, i.e.:
          >>> def factory(x: int):
          >>>   return np.array([0] * x, dtype=float)

    Returned prices:
        - returned 'prices' have same dtype as sizes and are rounded according to min_tick_size.
          Though, internally prices are stored as positions in an array, so de-facto no loss of precision.
    """

    def __init__(
        self,
        token_id: str,
        tick_size: NumericAlias,
        zeros_factory_bid: type[ZerosProtocol] | ZerosFactoryFunc = np.zeros,
        zeros_factory_ask: type[ZerosProtocol] | ZerosFactoryFunc | None = None,
        tick_size_factory: type[TickSizeProtocol] | TickSizeFactory | None = TickSize,
        min_tick_size: float = 0.001,
        coerce_inbound_prices: bool = False,
        strict_type_check: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        token_id
        tick_size: NumericAlias,
            Use get_tick_size(...) (REST) or MarketInfo.minimum_tick_size.
        zeros_factory_bid: type[ZerosProtocol] | ZerosFactoryFunc,
            For decimal type use `polypy.typing.zeros_dec` as factory.
        zeros_factory_ask: type[ZerosProtocol] | ZerosFactoryFunc | None,
            If None, then `zeros_factory_bid` will be used.
            For decimal type use `polypy.typing.zeros_dec` as factory.
        tick_size_factory: type[TickSizeProtocol] | TickSizeFactory | None, default = TickSize
            Container for storing tick size.
        min_tick_size: float, default = 0.001
            minimum allowed tick size in USDC terms.
            Will be passed to `tick_size_factory`
        coerce_inbound_prices
        strict_type_check
        """
        self.token_id = token_id
        self.coerce_inbound_prices = coerce_inbound_prices
        self.strict_type_check = strict_type_check

        self._inv_min_tick_size = int(1 / min_tick_size)

        len_quantities = self._inv_min_tick_size + 1
        self._bid_quantities = _zeroing_array(zeros_factory_bid(len_quantities))
        self._ask_quantities = _zeroing_array(
            (zeros_factory_ask or zeros_factory_bid)(len_quantities)
        )

        self._dtype_item = type(self._bid_quantities.item(0))
        # delegate heavy lifting of type inference to numpy
        self._dtype_quantities = _union_type(
            type(self._bid_quantities[0]), self._dtype_item
        )

        min_tick_digits = int(math.log10(self._inv_min_tick_size))
        self._bid_quote_levels = _linspace_array(
            1, 0, len_quantities, min_tick_digits, self._dtype_item
        )
        self._ask_quote_levels = _linspace_array(
            0, 1, len_quantities, min_tick_digits, self._dtype_item
        )

        self._tick_size = tick_size_factory(tick_size, self._dtype_item, min_tick_size)

        self.lock = threading.RLock()  # todo use read-write lock instead

    @property
    def tick_size(self) -> NumericAlias:
        with self.lock:
            return self._tick_size.get()

    @tick_size.setter
    def tick_size(self, val: NumericAlias) -> None:
        with self.lock:
            self._tick_size.set(val)

    @property
    def min_tick_size(self) -> float:
        return self._tick_size.min_tick_size

    @classmethod
    def from_dict(
        cls,
        book_msg_dict: dict[str, Any] | BookEvent,
        zeros_factory_bid: ZerosProtocol | ZerosFactoryFunc = np.zeros,
        zeros_factory_ask: ZerosProtocol | ZerosFactoryFunc = None,
        tick_size_factory: type[TickSizeProtocol] | TickSizeFactory | None = TickSize,
        min_tick_size: float = 0.001,
        coerce_inbound_prices: bool = False,
    ) -> Self:
        if isinstance(book_msg_dict, dict):
            book_msg_dict = dict_to_market_event_struct(book_msg_dict, "book")

        tick_size = guess_tick_size(book_msg_dict, 12)

        order_book = cls(
            book_msg_dict.asset_id,
            tick_size,
            zeros_factory_bid=zeros_factory_bid,
            zeros_factory_ask=zeros_factory_ask,
            tick_size_factory=tick_size_factory,
            min_tick_size=min_tick_size,
            coerce_inbound_prices=coerce_inbound_prices,
        )

        return _set_book_event(book_msg_dict, order_book, order_book.dtype)

    def update_tick_size(self, endpoint: str) -> NumericAlias:
        with self.lock:
            self.tick_size = get_tick_size(endpoint, self.token_id)
            return self.tick_size

    def sync(self, endpoint: str, include_tick_size: bool) -> None:
        with self.lock:
            if include_tick_size:
                self.update_tick_size(endpoint)
            response = get_book_summaries(endpoint, self.token_id)
            message_to_orderbook(response, self, None, "book")

    @property
    def dtype(self) -> type:
        """Native Python type of quantities."""
        # only dtype of quantities are relevant
        return self._dtype_item

    @property
    def bids(self) -> tuple[NDArray, ArrayCoercible]:
        with self.lock:
            return self._bid_quote_levels, self._bid_quantities[::-1]

    @property
    def asks(self) -> tuple[NDArray, ArrayCoercible]:
        with self.lock:
            return self._ask_quote_levels, self._ask_quantities[:]

    @property
    def bid_sizes(self) -> ArrayCoercible:
        # bids sorted in reverse order: best bid (highest) at index 0
        with self.lock:
            return self._bid_quantities[self._bid_quantities != 0][::-1]

    @property
    def ask_sizes(self) -> ArrayCoercible:
        # asks sorted in ascending order: best ask (lowest) at index 0
        with self.lock:
            return self._ask_quantities[self._ask_quantities != 0]

    @property
    def bid_prices(self) -> ArrayCoercible:
        # bids sorted in reverse order: best bid (highest) at index 0
        # return (np.nonzero(self._bid_quantities)[0] / self._inv_min_tick_size)[::-1]
        with self.lock:
            return self._bid_quote_levels[np.nonzero(self._bid_quantities[::-1])[0]]

    @property
    def ask_prices(self) -> ArrayCoercible:
        # asks sorted in ascending order: best ask (lowest) at index 0
        # return np.nonzero(self._ask_quantities)[0] / self._inv_min_tick_size
        with self.lock:
            return self._ask_quote_levels[np.nonzero(self._ask_quantities)[0]]

    @property
    def best_bid_price(self) -> NumericAlias:
        # return np.nonzero(self._bid_quantities)[0][-1] / self._inv_min_tick_size
        with self.lock:
            return self._bid_quote_levels[np.nonzero(self._bid_quantities[::-1])[0][0]]

    @property
    def best_ask_price(self) -> NumericAlias:
        # return np.nonzero(self._ask_quantities)[0][0] / self._inv_min_tick_size
        with self.lock:
            return self._ask_quote_levels[np.nonzero(self._ask_quantities)[0][0]]

    @property
    def midpoint_price(self) -> NumericAlias:
        with self.lock:
            return (self.best_bid_price + self.best_ask_price) / 2

    @property
    def best_bid_size(self) -> NumericAlias:
        with self.lock:
            return self._bid_quantities[np.nonzero(self._bid_quantities)[0][-1]]

    @property
    def best_ask_size(self) -> NumericAlias:
        with self.lock:
            return self._ask_quantities[np.nonzero(self._ask_quantities)[0][0]]

    def bid_size(self, price: NumericAlias) -> NumericAlias:
        if price < 0:
            raise OrderBookException("Only non-negative prices allowed.")

        with self.lock:
            return self._bid_quantities[int(price * self._inv_min_tick_size)]

    def ask_size(self, price: NumericAlias) -> NumericAlias:
        if price < 0:
            raise OrderBookException("Only non-negative prices allowed.")

        with self.lock:
            return self._ask_quantities[int(price * self._inv_min_tick_size)]

    def _prices_to_indices(
        self, price: list[NumericAlias] | NDArray[NumericAlias]
    ) -> NDArray[int]:
        idx = (np.asarray(price) * self._inv_min_tick_size).astype(int)

        if np.min(idx) < 0:
            raise OrderBookException("Only non-negative prices allowed.")

        return idx

    def _check_args_set_quantities(
        self,
        prices: list[NumericAlias] | ArrayCoercible,
        sizes: list[NumericAlias] | ArrayCoercible,
    ) -> None:
        _check_quotes_shape(prices, sizes)

        if not self.strict_type_check:
            return

        if not isinstance(sizes[0], self._dtype_quantities):
            raise TypeError(f"(Expected:){self._dtype_quantities}!={type(sizes[0])}.")

    def set_bids(
        self,
        bid_prices: list[NumericAlias] | ArrayCoercible,
        bid_sizes: list[NumericAlias] | ArrayCoercible,
    ) -> None:
        """

        Parameters
        ----------
        bid_prices: list[NumericAlias] | ArrayType
            prices are converted to position indices (internal buffer array), so doesn't really matter if
            float or any type, as long as it can be cast to int.
        bid_sizes

        Returns
        -------

        """
        self._check_args_set_quantities(bid_prices, bid_sizes)
        bid_idx = self._prices_to_indices(bid_prices)

        if self.coerce_inbound_prices:
            bid_idx, bid_sizes = _coerce_inbound_idx(
                bid_idx, bid_sizes, self._inv_min_tick_size + 1
            )

        with self.lock:
            self._bid_quantities[bid_idx] = bid_sizes

    def set_asks(
        self,
        ask_prices: list[NumericAlias] | ArrayCoercible,
        ask_sizes: list[NumericAlias] | ArrayCoercible,
    ) -> None:
        """

        Parameters
        ----------
        ask_prices: list[NumericAlias] | ArrayType
            prices are converted to position indices (internal buffer array), so doesn't really matter if
            float or any type, as long as it can be cast to int.
        ask_sizes

        Returns
        -------

        """
        self._check_args_set_quantities(ask_prices, ask_sizes)
        ask_idx = self._prices_to_indices(ask_prices)

        if self.coerce_inbound_prices:
            ask_idx, ask_sizes = _coerce_inbound_idx(
                ask_idx, ask_sizes, self._inv_min_tick_size + 1
            )

        with self.lock:
            self._ask_quantities[ask_idx] = ask_sizes

    def null_bids(self) -> None:
        with self.lock:
            # noinspection PyTypeChecker
            self._bid_quantities[:] = self._dtype_item("0")

    def null_asks(self) -> None:
        with self.lock:
            # noinspection PyTypeChecker
            self._ask_quantities[:] = self._dtype_item("0")

    def reset_bids(
        self,
        bid_prices: list[NumericAlias] | NDArray[NumericAlias] | None = None,
        bid_sizes: list[NumericAlias] | NDArray[NumericAlias] | None = None,
    ) -> None:
        """

        Parameters
        ----------
        bid_prices: list[NumericAlias] | NDArray[NumericAlias] | None
            prices are converted to position indices (internal buffer array), so doesn't really matter if
            float or any type, as long as it can be cast to int.
        bid_sizes

        Returns
        -------

        """
        if bid_prices is not None:
            self._check_args_set_quantities(bid_prices, bid_sizes)

        with self.lock:
            self.null_bids()

            if bid_prices is not None:
                self.set_bids(bid_prices, bid_sizes)

    def reset_asks(
        self,
        ask_prices: list[NumericAlias] | NDArray[NumericAlias] | None = None,
        ask_sizes: list[NumericAlias] | NDArray[NumericAlias] | None = None,
    ) -> None:
        """

        Parameters
        ----------
        ask_prices: list[NumericAlias] | NDArray[NumericAlias] | None
            prices are converted to position indices (internal buffer array), so doesn't really matter if
            float or any type, as long as it can be cast to int.
        ask_sizes

        Returns
        -------

        """
        if ask_prices is not None:
            self._check_args_set_quantities(ask_prices, ask_sizes)

        with self.lock:
            self.null_asks()

            if ask_prices is not None:
                self.set_asks(ask_prices, ask_sizes)

    @property
    def bids_summary(self) -> list[dict[str, str]]:
        with self.lock:
            bid_prices, bid_sizes = self.bid_prices, self.bid_sizes
        return merge_quotes_to_order_summaries(bid_prices, bid_sizes, True, dict)

    @property
    def asks_summary(self) -> list[dict[str, str]]:
        with self.lock:
            ask_prices, ask_sizes = self.ask_prices, self.ask_sizes
        return merge_quotes_to_order_summaries(ask_prices, ask_sizes, True, dict)

    def to_dict(
        self,
        market_id: str | None,
        timestamp: int | float | str | None,
        hash_str: str | None,
    ) -> dict[str, str]:
        if isinstance(timestamp, (int, float)):
            timestamp = f"{timestamp:.0f}"

        with self.lock:
            return {
                "market": market_id,
                "asset_id": self.token_id,
                "timestamp": timestamp,
                "hash": hash_str,
                "bids": self.bids_summary,
                "asks": self.asks_summary,
            }

    def hash(self, market_id: str, timestamp: int | float | str) -> str:
        return dict_to_sha1(self.to_dict(market_id, timestamp, ""))

    def marketable_price(
        self, side: Literal["BUY", "SELL"] | "SIDE", amount: NumericAlias
    ) -> tuple[NumericAlias, NumericAlias]:
        if side == "BUY":
            return calculate_marketable_price(*self.asks, amount)
        elif side == "SELL":
            return calculate_marketable_price(*self.bids, amount)
        else:
            raise OrderBookException(f"Unknown side: {side}")


# noinspection PyProtectedMember
class SharedOrderBook:
    def __init__(
        self,
        token_id: str,
        tick_size: NumericAlias | None,
        create: bool,
        min_tick_size: float | None = 0.001,
        coerce_inbound_prices: bool = False,
        shm_tick_size: str | None = None,
        shm_bid_q: str | None = None,
        shm_ask_q: str | None = None,
        shm_bid_p: str | None = None,
        shm_ask_p: str | None = None,
        shm_lock: str | None = None,
        shm_args: str | None = None,
    ):
        """Multiprocessing-safe OrderBookProtocol implementation.

        Currently, only supported dtype is Decimal.

        Notes
        -----
        No strict_type_check, since SharedDecZeros will always convert to Decimal.
        """

        self.token_id = token_id

        shm_tick_size = shm_tick_size or f"_shm_{token_id}_tick_size"
        shm_bid_q = shm_bid_q or f"_shm_{token_id}_bid_q"
        shm_ask_q = shm_ask_q or f"_shm_{token_id}_ask_q"
        shm_bid_p = shm_bid_p or f"_shm_{token_id}_bid_p"  # todo no shared mem needed?
        shm_ask_p = shm_ask_p or f"_shm_{token_id}_ask_p"
        shm_lock = shm_lock or f"_shm_{token_id}_lock"
        shm_args = shm_args or f"_shm_{token_id}_args"

        # self.state.buf[0] -> coerce_inbound_prices
        self.state = FinalizedSharedMemory(shm_args, create, 1)
        self.state.buf[0] = int(coerce_inbound_prices)

        self._tick_size = SharedTickSize(
            tick_size,
            min_tick_size=min_tick_size,
            shm_name=shm_tick_size,
            create=create,
        )

        self._inv_min_tick = int(1 / self._tick_size.min_tick_size)
        _len_q = self._inv_min_tick + 1
        _min_tick_digits = int(math.log10(self._inv_min_tick))

        # FIXME: check N_DIGITS_SIZE ?
        self._bid_q = SharedDecimalArray(
            _len_q, shm_bid_q, create=create, n_decimals=N_DIGITS_SIZE
        )  # quantity
        self._ask_q = SharedDecimalArray(
            _len_q, shm_ask_q, create=create, n_decimals=N_DIGITS_SIZE
        )

        self.lock = SharedProcRLock(shm_lock, create)  # todo reader-write lock instead

        self._bid_p = SharedDecimalArray(
            _len_q, shm_bid_p, create=create, n_decimals=_min_tick_digits
        )  # price
        self._ask_p = SharedDecimalArray(
            _len_q, shm_ask_p, create=create, n_decimals=_min_tick_digits
        )
        if create:
            self._bid_p[:] = _linspace_array(1, 0, _len_q, _min_tick_digits, Decimal)
            self._ask_p[:] = _linspace_array(0, 1, _len_q, _min_tick_digits, Decimal)

    def close(self) -> None:
        self.state.close()
        self.lock.close()

        self._tick_size.close()

        self._bid_q.close()
        self._ask_q.close()
        self._bid_p.close()
        self._ask_p.close()

    def unlink(self) -> None:
        self.state.unlink()
        self.lock.unlink()

        self._tick_size.unlink()

        self._bid_q.unlink()
        self._ask_q.unlink()
        self._bid_p.unlink()
        self._ask_p.unlink()

    def cleanup(self) -> None:
        self.state.cleanup()
        self.lock.cleanup()

        self._tick_size.cleanup()

        self._bid_q.cleanup()
        self._ask_q.cleanup()
        self._bid_p.cleanup()
        self._ask_p.cleanup()

    def __del__(self):
        self.cleanup()

    @property
    def auto_cleanup(self):
        @contextmanager
        def _func():
            try:
                with self.lock:
                    yield self
            finally:
                self.cleanup()

        return _func()

    @property
    def auto_cleanup_non_blocking(self):
        @contextmanager
        def _func():
            try:
                yield self
            finally:
                self.cleanup()

        return _func()

    @property
    def tick_size(self) -> Decimal:
        with self.lock:
            return self._tick_size.get()

    @tick_size.setter
    def tick_size(self, val: NumericAlias | str) -> None:
        with self.lock:
            self._tick_size.set(val)

    @property
    def min_tick_size(self) -> float:
        return self._tick_size.min_tick_size

    def update_tick_size(self, endpoint: str) -> Decimal:
        with self.lock:
            tick_size = get_tick_size(endpoint, self.token_id)
            self._tick_size.set(tick_size)
            return self._tick_size.get()

    def sync(self, endpoint: str, include_tick_size: bool) -> None:
        with self.lock:
            if include_tick_size:
                self.update_tick_size(endpoint)
            response = get_book_summaries(endpoint, self.token_id)
            message_to_orderbook(response, self, None)

    @property
    def dtype(self) -> type:
        return Decimal

    @property
    def bids(self) -> tuple[NDArray[Decimal], NDArray[Decimal]]:
        with self.lock:
            return self._bid_p[:], self._bid_q[::-1]

    @property
    def asks(self) -> tuple[NDArray[Decimal], NDArray[Decimal]]:
        with self.lock:
            return self._ask_p[:], self._ask_q[:]

    @property
    def bid_sizes(self) -> NDArray[Decimal]:
        with self.lock:
            return self._bid_q[self._bid_q._arr != 0][::-1]

    @property
    def ask_sizes(self) -> NDArray[Decimal]:
        with self.lock:
            return self._ask_q[self._ask_q._arr != 0]

    @property
    def bid_prices(self) -> NDArray[Decimal]:
        with self.lock:
            return self._bid_p[np.nonzero(self._bid_q._arr[::-1])[0]]

    @property
    def ask_prices(self) -> NDArray[Decimal]:
        with self.lock:
            return self._ask_p[np.nonzero(self._ask_q._arr)[0]]

    # noinspection PyTypeChecker
    @property
    def best_bid_price(self) -> Decimal:
        with self.lock:
            return self._bid_p[np.nonzero(self._bid_q._arr[::-1])[0][0]]

    # noinspection PyTypeChecker
    @property
    def best_ask_price(self) -> Decimal:
        with self.lock:
            return self._ask_p[np.nonzero(self._ask_q._arr)[0][0]]

    @property
    def midpoint_price(self) -> Decimal:
        with self.lock:
            return (self.best_bid_price + self.best_ask_price) / 2

    @property
    def best_bid_size(self) -> Decimal:
        with self.lock:
            return self._bid_q[np.nonzero(self._bid_q._arr)[0][-1]]

    @property
    def best_ask_size(self) -> Decimal:
        with self.lock:
            return self._ask_q[np.nonzero(self._ask_q._arr)[0][0]]

    def bid_size(self, price: NumericAlias) -> Decimal:
        if price < 0:
            raise OrderBookException("Only non-negative prices allowed.")

        with self.lock:
            return self._bid_q[int(price * self._inv_min_tick)]

    def ask_size(self, price: NumericAlias) -> Decimal:
        if price < 0:
            raise OrderBookException("Only non-negative prices allowed.")

        with self.lock:
            return self._ask_q[int(price * self._inv_min_tick)]

    def _prices_to_indices(
        self, price: list[NumericAlias] | NDArray[NumericAlias]
    ) -> NDArray[int]:
        idx = (np.asarray(price) * self._inv_min_tick).astype(int)

        if np.min(idx) < 0:
            raise OrderBookException("Only non-negative prices allowed.")

        return idx

    def set_bids(
        self,
        bid_prices: list[NumericAlias] | ArrayCoercible,
        bid_sizes: list[NumericAlias] | ArrayCoercible,
    ) -> None:
        _check_quotes_shape(bid_prices, bid_sizes)

        bid_idx = self._prices_to_indices(bid_prices)

        if self.state.buf[0]:
            bid_idx, bid_sizes = _coerce_inbound_idx(
                bid_idx, bid_sizes, self._inv_min_tick + 1
            )

        with self.lock:
            self._bid_q[bid_idx] = bid_sizes

    def set_asks(
        self,
        ask_prices: list[NumericAlias] | ArrayCoercible,
        ask_sizes: list[NumericAlias] | ArrayCoercible,
    ) -> None:
        _check_quotes_shape(ask_prices, ask_sizes)

        ask_idx = self._prices_to_indices(ask_prices)

        if self.state.buf[0]:
            ask_idx, ask_sizes = _coerce_inbound_idx(
                ask_idx, ask_sizes, self._inv_min_tick + 1
            )

        with self.lock:
            self._ask_q[ask_idx] = ask_sizes

    def null_bids(self) -> None:
        with self.lock:
            self._bid_q[:] = 0

    def null_asks(self) -> None:
        with self.lock:
            self._ask_q[:] = 0

    def reset_bids(
        self,
        bid_prices: list[NumericAlias] | NDArray[NumericAlias] | None = None,
        bid_sizes: list[NumericAlias] | NDArray[NumericAlias] | None = None,
    ) -> None:
        if bid_prices is not None:
            _check_quotes_shape(bid_prices, bid_sizes)
            bid_idx = self._prices_to_indices(bid_prices)

            if self.state.buf[0]:
                bid_idx, bid_sizes = _coerce_inbound_idx(
                    bid_idx, bid_sizes, self._inv_min_tick + 1
                )
        else:
            bid_idx = None

        with self.lock:
            self.null_bids()

            if bid_idx is not None:
                self._bid_q[bid_idx] = bid_sizes

    def reset_asks(
        self,
        ask_prices: list[NumericAlias] | NDArray[NumericAlias] | None = None,
        ask_sizes: list[NumericAlias] | NDArray[NumericAlias] | None = None,
    ) -> None:
        if ask_prices is not None:
            _check_quotes_shape(ask_prices, ask_sizes)
            ask_idx = self._prices_to_indices(ask_prices)

            if self.state.buf[0]:
                ask_idx, ask_sizes = _coerce_inbound_idx(
                    ask_idx, ask_sizes, self._inv_min_tick + 1
                )
        else:
            ask_idx = None

        with self.lock:
            self.null_asks()

            if ask_idx is not None:
                self._ask_q[ask_idx] = ask_sizes

    @property
    def bids_summary(self) -> list[dict[str, str]]:
        with self.lock:
            bid_q_int = self._bid_q._arr
            bid_s = bid_q_int[bid_q_int != 0].astype(np.float64) / self._bid_q.scale

            bid_p = self._bid_p._arr[np.nonzero(bid_q_int[::-1])[0]] / self._bid_p.scale

            return [
                dict(price=_number_to_str(p), size=_number_to_str(s))
                for p, s in zip(bid_p[::-1], bid_s)
            ]

    @property
    def asks_summary(self) -> list[dict[str, str]]:
        with self.lock:
            ask_q_int = self._ask_q._arr
            ask_s = ask_q_int[ask_q_int != 0].astype(np.float64) / self._ask_q.scale

            ask_p = self._ask_p._arr[np.nonzero(ask_q_int)[0]] / self._ask_p.scale

            return [
                dict(price=_number_to_str(p), size=_number_to_str(s))
                for p, s in zip(ask_p[::-1], ask_s[::-1])
            ]

    def to_dict(
        self,
        market_id: str | None,
        timestamp: int | float | str | None,
        hash_str: str | None,
    ) -> dict[str, str]:
        if isinstance(timestamp, (int, float)):
            timestamp = f"{timestamp:.0f}"

        with self.lock:
            return {
                "market": market_id,
                "asset_id": self.token_id,
                "timestamp": timestamp,
                "hash": hash_str,
                "bids": self.bids_summary,
                "asks": self.asks_summary,
            }

    def hash(self, market_id: str, timestamp: int | float | str) -> str:
        return dict_to_sha1(self.to_dict(market_id, timestamp, ""))

    def marketable_price(
        self, side: Literal["BUY", "SELL"] | "SIDE", amount: NumericAlias
    ) -> tuple[NumericAlias, NumericAlias]:
        if side == "BUY":
            return calculate_marketable_price(*self.asks, amount)
        elif side == "SELL":
            return calculate_marketable_price(*self.bids, amount)
        else:
            raise OrderBookException(f"Unknown side: {side}")
