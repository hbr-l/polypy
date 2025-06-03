import hashlib
import math
import warnings
from decimal import Decimal
from enum import Enum, auto
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    Self,
    Sequence,
    get_args,
)

import numpy as np
from msgspec import json as msgspec_json
from numpy.typing import NDArray

from polypy.exceptions import EventTypeException, OrderBookException
from polypy.rest.api import get_book_summaries, get_tick_size
from polypy.rounding import round_half_even
from polypy.typing import ArrayCoercible, NumericAlias, ZerosFactoryFunc, ZerosProtocol

if TYPE_CHECKING:
    from polypy.order.common import SIDE


def _validate_base10(x: float) -> None:
    exponent = np.log10(x)
    if exponent != int(exponent):
        raise OrderBookException(
            f"`tick_size` has to be base 10. Got: {x} with exponent {exponent}."
        )


def _validate_tick_size(val: Any, allowed_tick_sizes: set[float]) -> None:
    _validate_base10(val)
    if val not in allowed_tick_sizes:
        raise OrderBookException(
            f"`tick_size`={val} not in `allowed_tick_sizes`={allowed_tick_sizes}."
        )


def _validate_allowed_tick_size(x: set[float] | tuple[float]) -> None:
    for x_i in x:
        _validate_base10(x_i)


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


def dict_to_sha1(x: dict) -> str:
    return hashlib.sha1(
        # json.dumps(x, separators=(",", ":"), sort_keys=False).encode("utf-8")
        msgspec_json.encode(x)
    ).hexdigest()


def _conv_set_float(x: Sequence[Any]) -> set[float]:
    return {float(x_i) for x_i in x}


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


class TickSizeProtocol(Protocol):
    # noinspection PyUnusedLocal
    def __init__(
        self, x: Any, dtype: type, allowed_tick_sizes: set[float] | tuple[float, ...]
    ) -> None:
        ...

    def get(self) -> Any:
        ...

    def set(self, x: Any) -> None:
        ...


class TickSizeFactory(Protocol):
    def __call__(
        self, x: Any, dtype: type, allowed_tick_sizes: set[float] | tuple[float, ...]
    ) -> TickSizeProtocol:
        ...


class TickSize:
    def __init__(
        self, x: Any, dtype: type, allowed_tick_sizes: set[float] | tuple[float, ...]
    ) -> None:
        self.dtype = dtype
        self.allowed_tick_sizes = _conv_set_float(allowed_tick_sizes)

        self.x = None
        self.set(x)

    def get(self) -> Any:
        return self.x

    def set(self, x: Any) -> None:
        _validate_tick_size(x, self.allowed_tick_sizes)
        self.x = self.dtype(str(x))


class OrderBook:
    """Order Book class.

    Notes
    -----
    Multiprocessing:
        - zeros_factory has to return sharedMemory array
        - zeros_factory returned array has to handle locking of getitem and setitem if necessary

    Threading:
        - zeros_factory returned array has to handle locking of getitem and setitem if necessary

    Dtype of sizes:
        - dtype of 'sizes' is determined by zeros_factory returned array. If, e.g. Decimals are required,
          then return zeros-array with Decimals in it, i.e.:
          >>> def factory(x: int):
          >>>   return np.array([0] * x, dtype=float)

    Returned prices:
        - returned 'prices' have same dtype as sizes and are rounded according to min(allowed_tick_sizes).
          Though, internally prices are stored as positions in an array, so de-facto no loss of precision.
    """

    # todo implement __getstate__ and __setstate__ (pickling during multiprocessing)
    # todo TickSizeProtocol

    def __init__(
        self,
        token_id: str,
        tick_size: NumericAlias,
        zeros_factory_bid: type[ZerosProtocol] | ZerosFactoryFunc = np.zeros,
        zeros_factory_ask: type[ZerosProtocol] | ZerosFactoryFunc | None = None,
        tick_size_factory: type[TickSizeProtocol] | TickSizeFactory | None = TickSize,
        allowed_tick_sizes: tuple[float] | set[float] = (0.01, 0.001),
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
        allowed_tick_sizes
        coerce_inbound_prices
        strict_type_check
        """
        self.token_id = token_id
        self.coerce_inbound_prices = coerce_inbound_prices
        self.strict_type_check = strict_type_check

        allowed_tick_sizes = _conv_set_float(allowed_tick_sizes)
        _validate_allowed_tick_size(allowed_tick_sizes)
        self._allowed_tick_sizes = allowed_tick_sizes

        self._inv_min_tick_size = int(1 / min(self._allowed_tick_sizes))
        self._min_tick_digits = int(math.log10(self._inv_min_tick_size))

        len_quantities = self._inv_min_tick_size + 1
        self._bid_quantities = _zeroing_array(zeros_factory_bid(len_quantities))
        self._ask_quantities = _zeroing_array(
            (zeros_factory_ask or zeros_factory_bid)(len_quantities)
        )

        self._dtype_item = type(np.array([self._bid_quantities[0]]).item(0))
        # delegate heavy lifting of type inference to numpy
        self._dtype_quantities = _union_type(
            type(self._bid_quantities[0]), self._dtype_item
        )

        self._bid_quote_levels = _linspace_array(
            1, 0, len_quantities, self._min_tick_digits, self._dtype_item
        )
        self._ask_quote_levels = _linspace_array(
            0, 1, len_quantities, self._min_tick_digits, self._dtype_item
        )

        self._tick_size = tick_size_factory(
            tick_size, self._dtype_item, self._allowed_tick_sizes
        )

    @property
    def tick_size(self) -> NumericAlias:
        return self._tick_size.get()

    @tick_size.setter
    def tick_size(self, val: NumericAlias) -> None:
        self._tick_size.set(val)

    @property
    def allowed_tick_sizes(self) -> set[float]:
        return self._allowed_tick_sizes

    @classmethod
    def from_dict(
        cls,
        book_msg_dict: dict[str, Any],
        zeros_factory_bid: ZerosProtocol | ZerosFactoryFunc = np.zeros,
        zeros_factory_ask: ZerosProtocol | ZerosFactoryFunc = None,
        coerce_inbound_prices: bool = False,
        allowed_tick_sizes: tuple[float] = (0.01, 0.001),
    ) -> Self:
        # in case bids is empty, we might consider
        min_exponent = min(
            [
                Decimal(i["price"]).as_tuple().exponent
                for i in book_msg_dict["bids"][:12]
            ]
            + [
                Decimal(i["price"]).as_tuple().exponent
                for i in book_msg_dict["asks"][:12]
            ],
        )
        tick_size = 10**min_exponent

        order_book = cls(
            book_msg_dict["asset_id"],
            tick_size,
            zeros_factory_bid=zeros_factory_bid,
            zeros_factory_ask=zeros_factory_ask,
            allowed_tick_sizes=allowed_tick_sizes,
            coerce_inbound_prices=coerce_inbound_prices,
        )

        return _reset_event_type_book(book_msg_dict, order_book, order_book.dtype)

    def update_tick_size(self, endpoint: str) -> NumericAlias:
        self.tick_size = get_tick_size(endpoint, self.token_id)
        return self.tick_size

    def sync(self, endpoint: str) -> None:
        self.update_tick_size(endpoint)
        response = get_book_summaries(endpoint, self.token_id)
        message_to_orderbook(response, self, None, "book", "except")

    @property
    def dtype(self) -> type:
        """Native Python type of quantities."""
        # only dtype of quantities are relevant
        return self._dtype_item

    @property
    def dtype_quantities(self) -> type:
        """Dtype of elements in the quantities container (e.g., float vs np.float63)."""
        return type(self._bid_quantities[0])

    @property
    def bids(self) -> tuple[NDArray, ArrayCoercible]:
        return self._bid_quote_levels, self._bid_quantities[::-1]

    @property
    def asks(self) -> tuple[NDArray, ArrayCoercible]:
        return self._ask_quote_levels, self._ask_quantities[:]

    @property
    def bid_sizes(self) -> ArrayCoercible:
        # bids sorted in reverse order: best bid (highest) at index 0
        return self._bid_quantities[self._bid_quantities != 0][::-1]

    @property
    def ask_sizes(self) -> ArrayCoercible:
        # asks sorted in ascending order: best ask (lowest) at index 0
        return self._ask_quantities[self._ask_quantities != 0]

    @property
    def bid_prices(self) -> ArrayCoercible:
        # bids sorted in reverse order: best bid (highest) at index 0
        # return (np.nonzero(self._bid_quantities)[0] / self._inv_min_tick_size)[::-1]
        return self._bid_quote_levels[np.nonzero(self._bid_quantities[::-1])[0]]

    @property
    def ask_prices(self) -> ArrayCoercible:
        # asks sorted in ascending order: best ask (lowest) at index 0
        # return np.nonzero(self._ask_quantities)[0] / self._inv_min_tick_size
        return self._ask_quote_levels[np.nonzero(self._ask_quantities)[0]]

    @property
    def best_bid_price(self) -> NumericAlias:
        # return np.nonzero(self._bid_quantities)[0][-1] / self._inv_min_tick_size
        return self._bid_quote_levels[np.nonzero(self._bid_quantities[::-1])[0][0]]

    @property
    def best_ask_price(self) -> NumericAlias:
        # return np.nonzero(self._ask_quantities)[0][0] / self._inv_min_tick_size
        return self._ask_quote_levels[np.nonzero(self._ask_quantities)[0][0]]

    @property
    def midpoint_price(self) -> NumericAlias:
        return (self.best_bid_price + self.best_ask_price) / 2

    @property
    def best_bid_size(self) -> NumericAlias:
        return self._bid_quantities[np.nonzero(self._bid_quantities)[0][-1]]

    @property
    def best_ask_size(self) -> NumericAlias:
        return self._ask_quantities[np.nonzero(self._ask_quantities)[0][0]]

    def bid_size(self, price: NumericAlias) -> NumericAlias:
        if price < 0:
            raise OrderBookException("Only non-negative prices allowed.")
        return self._bid_quantities[int(price * self._inv_min_tick_size)]

    def ask_size(self, price: NumericAlias) -> NumericAlias:
        if price < 0:
            raise OrderBookException("Only non-negative prices allowed.")
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
        if not self.strict_type_check:
            return

        if len(prices) != len(sizes) or len(sizes) == 0:
            raise IndexError(
                f"Arguments not allowed (len(prices)==len(sizes)!= 0). "
                f"Got: len(prices)={len(prices)}, len(sizes)={len(sizes)}."
            )

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
                bid_idx, bid_sizes, len(self._bid_quantities)
            )

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
                ask_idx, ask_sizes, len(self._ask_quantities)
            )

        self._ask_quantities[ask_idx] = ask_sizes

    def null_bids(self) -> None:
        # noinspection PyTypeChecker
        self._bid_quantities[:] = self._dtype_item("0")

    def null_asks(self) -> None:
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

        self.null_asks()

        if ask_prices is not None:
            self.set_asks(ask_prices, ask_sizes)

    @property
    def bids_summary(self) -> list[dict[str, str]]:
        bid_prices, bid_sizes = self.bid_prices, self.bid_sizes
        return merge_quotes_to_order_summaries(bid_prices, bid_sizes, True, dict)

    @property
    def asks_summary(self) -> list[dict[str, str]]:
        ask_prices, ask_sizes = self.ask_prices, self.ask_sizes
        return merge_quotes_to_order_summaries(ask_prices, ask_sizes, True, dict)

    def to_dict(
        self,
        market_id: str | None,
        timestamp: int | float | str | None,
        hash_str: str | None,
    ):
        if isinstance(timestamp, (int, float)):
            timestamp = f"{timestamp:.0f}"

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


def guess_check_orderbook_hash(
    target_hash: str,
    book: OrderBook,
    market_id: str,
    timestamps: Sequence[int | float | str],
) -> tuple[bool, int | None, str | None]:
    book_dict = book.to_dict(market_id, None, "")

    # todo suboptimal approach (brute-forcing through timestamps), but currently no known alternative
    for i, ts in enumerate(timestamps):
        if not isinstance(ts, str):
            ts = f"{ts:.0f}"

        book_dict["timestamp"] = ts

        book_hash = dict_to_sha1(book_dict)
        if book_hash == target_hash:
            return True, i, book_hash

    return False, None, None


def _number_to_str(x: NumericAlias) -> str:
    # todo optimize performance

    # f"{x:.16g} breaks (i.e. truncation) if more than 16 chars:
    #  for $xxx$.¢¢ notation this is somewhere greater than around 100 tn
    #  AND is wrong for e.g. x=0.691 -> '0.6909999999999999'

    nb = str(x)
    return nb.rstrip("0").rstrip(".") if "." in nb else nb


class OrderSummaryProtocol(Protocol):
    price: str
    size: str


def merge_quotes_to_order_summaries(
    prices: list[NumericAlias] | ArrayCoercible,
    sizes: list[NumericAlias] | ArrayCoercible,
    reverse: bool,
    order_summary_factory: type[dict] | type[OrderSummaryProtocol] = dict,
) -> list[dict[str, str]] | list[OrderSummaryProtocol]:
    # Polymarket orders price level s.t. best bid is at index -1,
    #   whilst we order in reverse (i.e. to apply np.cumsum more easily)
    # remove trailing 0s and if necessary decimal point

    # todo optimize for performance:
    #   - prices could be cached (either cache manually or use python's mem_cache)
    #   - if sizes are stored as scaled ints: insert decimal point char manually

    if reverse:
        prices = reversed(prices)
        sizes = reversed(sizes)

    return [
        order_summary_factory(
            price=_number_to_str(price),
            size=_number_to_str(size),
        )
        for price, size in zip(prices, sizes)
    ]


def split_order_summaries_to_quotes(
    x: list[dict[str, str] | OrderSummaryProtocol],
    dtype_factory: Callable[[str], Any],
) -> tuple[list[NumericAlias], list[NumericAlias]]:
    if not len(x):
        return [], []
    elif isinstance(x[0], dict):
        return [dtype_factory(d["price"]) for d in x], [
            dtype_factory(d["size"]) for d in x
        ]
    else:
        return [dtype_factory(getattr(d, "price")) for d in x], [
            dtype_factory(getattr(d, "size")) for d in x
        ]


def _quotes_event_type_book(
    msg: dict[str, Any], dtype_factory: Callable[[str], Any]
) -> tuple[
    list[NumericAlias], list[NumericAlias], list[NumericAlias], list[NumericAlias]
]:
    bid_prices, bid_sizes = split_order_summaries_to_quotes(msg["bids"], dtype_factory)
    ask_prices, ask_sizes = split_order_summaries_to_quotes(msg["asks"], dtype_factory)
    return bid_prices, bid_sizes, ask_prices, ask_sizes


def _quotes_event_type_price_change(
    msg: dict[str, Any], dtype_factory: Callable[[str], Any]
) -> tuple[list, list, list, list]:
    bid_prices, bid_sizes, ask_prices, ask_sizes = [], [], [], []

    for change in msg["changes"]:
        if change["side"] == "BUY":
            bid_prices.append(dtype_factory(change["price"]))
            bid_sizes.append(dtype_factory(change["size"]))
        elif change["side"] == "SELL":
            ask_prices.append(dtype_factory(change["price"]))
            ask_sizes.append(dtype_factory(change["size"]))
        else:
            raise ValueError(f"Unknown side: {change['side']}")

    return bid_prices, bid_sizes, ask_prices, ask_sizes


def _valid_message_asset_id(
    msg: dict[str, Any],
    book: OrderBook,
    mode: Literal["silent", "warn", "except"] = "except",
) -> bool:
    if msg["asset_id"] == book.token_id:
        return True

    if mode == "except":
        raise OrderBookException(
            f"asset_id != token_id. Got asset_id={msg['asset_id']} and token_id={book.token_id}"
        )
    if mode == "warn":
        warnings.warn(
            f"asset_id != token_id. Got asset_id={msg['asset_id']} and token_id={book.token_id}"
        )

    return False


def _reset_event_type_book(
    msg: dict[str, Any],
    book: OrderBook,
    dtype_factory: Callable[[str], Any] | None = None,
) -> OrderBook:
    bid_prices, bid_sizes, ask_prices, ask_sizes = _quotes_event_type_book(
        msg, dtype_factory
    )

    if len(bid_prices):
        book.reset_bids(bid_prices, bid_sizes)
    else:
        book.null_bids()

    if len(ask_prices):
        book.reset_asks(ask_prices, ask_sizes)
    else:
        book.null_asks()

    return book


def _set_event_type_price_change(
    msg: dict[str, Any],
    book: OrderBook,
    dtype_factory: Callable[[str], Any] | None = None,
) -> OrderBook:
    (
        bid_prices,
        bid_sizes,
        ask_prices,
        ask_sizes,
    ) = _quotes_event_type_price_change(msg, dtype_factory)

    if len(bid_prices):
        book.set_bids(bid_prices, bid_sizes)
    if len(ask_prices):
        book.set_asks(ask_prices, ask_sizes)
    return book


def _parse_event_type_msg(
    msg: dict[str, Any],
    event_type: Literal["book", "price_change", "tick_size_change"] | None,
) -> str:
    if event_type is None:
        try:
            event_type = msg["event_type"]
        except KeyError as e:
            e.add_note(
                "If parsing a REST response, please specify 'event_type' argument."
            )
            raise e
    elif "event_type" in msg:
        raise ValueError("Key 'event_type' already exists in msg.")

    return event_type


# noinspection PyPep8Naming
class HASH_STATUS(Enum):
    UNKNOWN = auto()
    """Order book has changed, so no statement possible regarding validity of hash"""

    UNCHANGED = auto()
    """Order book has NOT changed, hash status has NOT changed (either UNKNOWN or VALID)"""

    VALID = auto()
    """Order book was completely reset, hash is valid"""


# todo use Enum for event_type? (also in MarketStream._update_last_traded_price)
def message_to_orderbook(
    msg: dict[str, Any],
    book: OrderBook,
    dtype_factory: Callable[[str], Any] | type | None = None,
    event_type: Literal["book", "price_change", "tick_size_change"] | None = None,
    mode: Literal["silent", "warn", "except"] = "except",
) -> tuple[OrderBook, HASH_STATUS]:
    if not _valid_message_asset_id(msg, book, mode):
        return book, HASH_STATUS.UNCHANGED

    event_type = _parse_event_type_msg(msg, event_type)

    dtype_factory = book.dtype if dtype_factory is None else dtype_factory

    if event_type == "book":
        return (
            _reset_event_type_book(msg, book, dtype_factory),
            HASH_STATUS.VALID,
        )

    if event_type == "price_change":
        return (
            _set_event_type_price_change(msg, book, dtype_factory),
            HASH_STATUS.UNKNOWN,
        )

    if event_type == "tick_size_change":
        book.tick_size = float(msg["new_tick_size"])
        return book, HASH_STATUS.UNCHANGED

    raise EventTypeException(f"Unknown event_type: {msg['event_type']}.")


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
