import copy
import itertools
import threading
import time
import traceback
import warnings
from collections import OrderedDict, deque, namedtuple
from typing import Any, Callable, Literal, NoReturn, Self, TypeAlias, Union

from polypy.constants import ENDPOINT
from polypy.exceptions import (
    EventTypeException,
    OrderGetException,
    StreamException,
    SubscriptionException,
)
from polypy.manager.order import OrderManagerProtocol
from polypy.manager.position import PositionManagerProtocol
from polypy.order import INSERT_STATUS, SIDE, OrderProtocol
from polypy.order.common import TERMINAL_INSERT_STATI
from polypy.position import PositionProtocol
from polypy.stream.common import CHANNEL, AbstractStreamer
from polypy.structs import MakerOrder, OrderWSInfo, TradeWSInfo
from polypy.trade import TRADER_SIDE


def lru_cache_non_empty(max_size: int, copy_mode: Literal["deep", "copy"] | None):
    copy_func = (
        lambda x: x
        if copy_mode is None
        else copy.copy
        if copy_mode == "copy"
        else copy.deepcopy
    )  # precompute copy function

    def decorator(func: Callable):
        cache = OrderedDict()

        def wrapper(*args, **kwargs) -> Any:
            # must be hashable
            key = (args, tuple(kwargs.items())) if kwargs else args

            cached_result = cache.get(key)
            if cached_result is not None:
                cache.move_to_end(key)  # recently used
                return copy_func(cached_result)

            result = func(*args, **kwargs)
            if result:
                if len(cache) >= max_size:
                    cache.popitem(last=False)  # remove the least recent
                cache[key] = copy_func(result)
            return result

        return wrapper

    return decorator


MarketAssetsInfo = namedtuple(
    "MarketAssetsInfo", ["condition_id", "token_id_yes", "token_id_no"]
)

BufferThreadSettings = namedtuple("BufferSettings", ["keep_s", "max_len"])

CleaningThreadSettings = namedtuple(
    "CleaningThreadSettings",
    ["interval_s", "insert_statuses", "expired_since_s", "bool_positions"],
)

TupleManager: TypeAlias = tuple[
    OrderManagerProtocol | None, PositionManagerProtocol | None
]


def _split_market_assets(
    market_assets: MarketAssetsInfo | list[MarketAssetsInfo],
) -> tuple[set[str], set[str]]:
    if isinstance(market_assets, list):
        market_ids = {mas[0] for mas in market_assets}
        asset_ids = {asset_id for mas in market_assets for asset_id in mas[1:]}
    else:
        market_ids = {market_assets[0]}
        asset_ids = {market_assets[1], market_assets[2]}

    if not market_ids:
        raise StreamException("No markets (condition_id) to subscribe to.")

    if None in market_ids:
        raise StreamException("None not allowed in market ids (condition_id).")

    if None in asset_ids:
        raise StreamException("None not allowed in asset ids (token_id).")

    return market_ids, asset_ids


def _parse_tuple_manager_list(
    tuple_manager: TupleManager | list[TupleManager] | None,
) -> list[TupleManager]:
    if tuple_manager is None:
        return []

    try:
        tuple_manager = list(map(tuple, tuple_manager))
    except TypeError:
        tuple_manager = [tuple(tuple_manager)]

    # noinspection PyTypeChecker
    return tuple_manager


def _assert_api_key_order_mng(tuple_manager: list[TupleManager], api_key: str) -> None:
    for tm in tuple_manager:
        if tm[0] is not None and tm[0].api_key != api_key:
            raise StreamException(
                f"OrderManager.api_key={tm[0].api_key} does not match specified api_key={api_key}. "
                f"Cannot receive stream data for that api_key."
            )


def _assert_unique_order_mngs(
    tuple_manager: list[TupleManager],
) -> None:
    # all order managers unique
    valid_tpl_mngs = [tm for tm in tuple_manager if tm[0] is not None]
    order_mngs = {id(tm[0]) for tm in valid_tpl_mngs}
    if len(order_mngs) != len(valid_tpl_mngs):
        raise StreamException("All Order Managers in `tuple_manager` must be unique.")


def _assert_token_ids_order_mngs(
    tuple_manager: list[TupleManager], asset_ids: set[str]
) -> None:
    token_ids = set(
        itertools.chain.from_iterable(
            tm[0].token_ids for tm in tuple_manager if tm[0] is not None
        )
    )

    if not token_ids.issubset(asset_ids):
        raise StreamException(
            "Not all token_ids of all Order Managers are contained in `market_assets`."
        )


def _check_update_mode(
    nb_updates: int,
    msg: TradeWSInfo | OrderWSInfo,
    update_mode: Literal["explicit", "implicit"],
    callback_register_raise: Callable[[Exception], NoReturn],
) -> None:
    if nb_updates == 0 and update_mode == "explicit":
        callback_register_raise(
            StreamException(
                f"Order not contained in any OrderManager. Original message: {msg}."
            )
        )

    if nb_updates > 1:
        callback_register_raise(
            StreamException(
                f"Order should not be assigned to more than one OrderManager. "
                f"Transactions and updates have been performed nonetheless, "
                f"Order and Position Manager states might be compromised. "
                f"Original message: {msg}."
            )
        )


def _maker_order_side(maker_order: MakerOrder, msg: TradeWSInfo) -> SIDE:
    if maker_order.asset_id == msg.asset_id:
        return SIDE.SELL if msg.side is SIDE.BUY else SIDE.BUY
    else:
        return msg.side


def _numeric_type_balance_total(x: PositionManagerProtocol) -> type | None:
    if x is None:
        return None

    numeric_type = type(x.balance_total)
    return numeric_type if numeric_type is not int else float


def _parse_buffer_settings(
    x: BufferThreadSettings | tuple[float, int] | None
) -> BufferThreadSettings:
    return BufferThreadSettings(0, 0) if x is None or None in x else x


def _parse_cleaning_settings(
    x: CleaningThreadSettings
    | tuple[float, list[INSERT_STATUS] | INSERT_STATUS, float]
    | None
) -> CleaningThreadSettings:
    # ["interval_s", "insert_statuses", "expired_since_s", "bool_positions"]

    if x is None or None in x:
        x = CleaningThreadSettings(0, (), 0, False)

    if isinstance(x[1], INSERT_STATUS):
        x = CleaningThreadSettings(x[0], [x[1]], x[2], x[3])

    if any(
        x_i
        not in [INSERT_STATUS.MATCHED, INSERT_STATUS.UNMATCHED, INSERT_STATUS.DEFINED]
        for x_i in x[1]
    ):
        raise StreamException(
            "Only INSERT_STATUS.MATCHED, INSERT_STATUS.UNMATCHED and INSERT_STATUS.DEFINED "
            "allowed in CleaningThreadSettings."
        )

    return x


def _parse_untrack_stati(
    x: INSERT_STATUS | list[INSERT_STATUS] | None,
) -> list[INSERT_STATUS]:
    if x is None:
        x = []

    if isinstance(x, INSERT_STATUS):
        x = [x]

    if None in x:
        x = []

    if any(x_i not in TERMINAL_INSERT_STATI for x_i in x):
        raise StreamException(
            "Only INSERT_STATUS in TERMINAL_INSERT_STATI allowed for untrack_insert_status."
        )

    if any(x_i is INSERT_STATUS.MATCHED for x_i in x):
        warnings.warn(
            "INSERT_STATUS.MATCHED via untrack_insert_status might lead to exceptions "
            "(if more than 128 new orders are submitted between receiving `order` websocket message "
            "and `trade` websocket message - which is extremely unlikely to happen)."
        )

    return x


_TradeOrderInfo = namedtuple(
    "_TradeOrderInfo", ["order_id", "size", "price", "asset_id", "side"]
)


# todo spec untrack pos bool
# todo order_manager see new todo necessary
# todo clean really without canceled? -> Tracking Bug could also occur for others -> delete and rename callback


# todo allow_untracked_sell? -> allow_neg or allow_untracked_sell -> feature + doc: allow_neg in factory
# todo if multiple updates raise exception: transaction will be conducted, so manager states might be compromised -> doc
class UserStream(AbstractStreamer):
    def __init__(
        self,
        ws_endpoint: ENDPOINT | str,
        tuple_manager: TupleManager | list[TupleManager] | None,
        market_assets: MarketAssetsInfo | list[MarketAssetsInfo],
        api_key: str,
        secret: str,
        passphrase: str,
        untrack_insert_status: INSERT_STATUS | list[INSERT_STATUS] | None = None,
        cleaning_thread_settings: CleaningThreadSettings | None = None,
        monitor_assets_thread_s: float | None = 0.1,
        buffer_thread_settings: BufferThreadSettings | None = (2, 5_000),
        ping_time: float | None = 5,
        update_mode: Literal["explicit", "implicit"] = "explicit",
        invalidate_on_exc: bool = True,
        callback_msg: Callable[[Self, TradeWSInfo | OrderWSInfo], None] | None = None,
        callback_exc: Callable[[Self, Exception], None] | None = None,
        callback_clean: Callable[[list[OrderProtocol], list[PositionProtocol]], None]
        | None = None,
        ws_channel: CHANNEL | str = CHANNEL.USER,
    ) -> None:
        market_ids, asset_ids = _split_market_assets(market_assets)

        tuple_manager = _parse_tuple_manager_list(tuple_manager)
        _assert_api_key_order_mng(tuple_manager, api_key)
        _assert_unique_order_mngs(tuple_manager)
        _assert_token_ids_order_mngs(tuple_manager, asset_ids)
        # we do not check position manager asset_ids (=token_ids), because a single position manager
        #   can be coupled with multiple order managers

        self._market_ids = market_ids  # does not get modified
        self._asset_ids = asset_ids
        self.tuple_mngs: list[TupleManager] = tuple_manager
        self._default_pos_mngs_idx = {
            idx
            for idx, tm in enumerate(self.tuple_mngs)
            if tm[0] is None and tm[1] is not None
        }
        self._numeric_type_pos_mng = {
            i: _numeric_type_balance_total(tm[1])
            for i, tm in enumerate(self.tuple_mngs)
        }  # we use balance_total as proxy to determine the numeric type

        self.untrack_insert_status = _parse_untrack_stati(untrack_insert_status)
        self.callback_clean = callback_clean

        if ws_endpoint[-1] != "/":
            ws_endpoint = f"{ws_endpoint}/"

        self.update_mode = update_mode
        self.invalidate_on_exc = invalidate_on_exc

        self.buffer_thread_settings = _parse_buffer_settings(buffer_thread_settings)
        self.buffer = deque(maxlen=self.buffer_thread_settings[1])
        self.buffer_thread: threading.Thread | None = None
        self.buffer_event = threading.Event()

        self.monitor_thread: threading.Thread | None = None
        self.monitor_assets_thread_s = monitor_assets_thread_s

        self.cleaning_thread: threading.Thread | None = None
        self.cleaning_thread_settings = _parse_cleaning_settings(
            cleaning_thread_settings
        )

        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase

        def _callback_invalidate_exc(stream: Self, exc: Exception) -> None:
            if self.invalidate_on_exc:
                for tm in self.tuple_mngs:
                    if tm is None:
                        continue

                    tm[0].invalidate(
                        f"Exception in {self.__class__.__name__}. "
                        f"Original exception: {exc.__class__.__name__}({exc})."
                        f"Full Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                    )
                warnings.warn(
                    f"Exception in {self.__class__.__name__}. "
                    f"Invalidated all Order Managers."
                )
            callback_exc(stream, exc)

        super().__init__(
            url=f"{ws_endpoint}{ws_channel}",
            subscribe_params={
                "auth": {
                    "apiKey": self.api_key,
                    "secret": self.secret,
                    "passphrase": self.passphrase,
                },
                "markets": list(self._market_ids),
                "asset_ids": [],
                "type": ws_channel,
            },
            ping_time=ping_time,
            callback_msg=callback_msg,
            callback_exc=_callback_invalidate_exc,
            msgspec_type=list[Union[TradeWSInfo, OrderWSInfo]],
            msgspec_strict=False,
        )

    @property
    def market_ids(self) -> list[str]:
        return self.subscribe_params.markets

    @property
    def asset_ids(self) -> list[str]:
        return list(self._asset_ids)

    def _start_aux_threads(self) -> None:
        if self.monitor_thread is not None:
            self.register_exception(
                StreamException("Internal exception: self.monitor_thread is not None.")
            )

        if self.buffer_thread is not None:
            self.register_exception(
                StreamException("Internal exception: self.buffer_thread is not None.")
            )

        if self.cleaning_thread is not None:
            self.register_exception(
                StreamException("Internal exception: self.clean_thread is not None.")
            )

        self.monitor_thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self.monitor_thread.start()

        self.cleaning_thread = threading.Thread(target=self._clean_thread, daemon=True)
        self.cleaning_thread.start()

        self.buffer_event.clear()

        self.buffer_thread = threading.Thread(target=self._buffer_thread, daemon=True)
        self.buffer_thread.start()

    def pre_start(self) -> None:
        self.buffer = deque(maxlen=self.buffer_thread_settings[1])
        self._start_aux_threads()

    def _stop_aux_threads(self) -> None:
        self.buffer_event.set()

        self.monitor_thread.join(self.ping_time + 0.01)
        self.buffer_thread.join(self.ping_time + 0.01)
        self.cleaning_thread.join(self.ping_time + 0.01)

        if self.monitor_thread.is_alive():
            self.register_exception(StreamException("Cannot join() monitoring thread."))

        if self.buffer_thread.is_alive():
            self.register_exception(StreamException("Cannot join() buffer thread."))

        if self.cleaning_thread.is_alive():
            self.register_exception(StreamException("Cannot join() cleaning thread."))

        self.monitor_thread = None
        self.buffer_thread = None
        self.buffer_event.clear()
        self.cleaning_thread = None

    def post_stop(self) -> None:
        self._stop_aux_threads()
        self.buffer = deque(maxlen=self.buffer_thread_settings[1])

    def _monitor_thread(self) -> None:
        if not self.tuple_mngs:
            # if no order managers assigned, we exit immediately
            return

        if self.monitor_assets_thread_s is None or self.monitor_assets_thread_s <= 0:
            return

        while not self._stop_token.wait(self.monitor_assets_thread_s):
            for i, tm in enumerate(self.tuple_mngs):
                if tm[0] is not None and not tm[0].token_ids <= self._asset_ids:
                    self.register_exception(
                        SubscriptionException(
                            f"Stream not subscribed to all token_ids of Order Managers. "
                            f"Token_ids Order Manager at index {i}: {tm[0].token_ids}. "
                            f"Asset_ids of stream: {self._asset_ids}."
                        )
                    )

    def _clean_order_managers(self) -> list[OrderProtocol]:
        # ["interval_s", "insert_statuses", "expired_since_s", "bool_positions"]

        clean_orders = []
        for tm in self.tuple_mngs:
            if tm[0] is not None:
                clean_orders.extend(
                    tm[0].clean(
                        self.cleaning_thread_settings[1],
                        int((time.time() - self.cleaning_thread_settings[2]) * 1_000),
                    )
                )
        return clean_orders

    def _clean_position_managers(self) -> list[PositionProtocol]:
        # ["interval_s", "insert_statuses", "expired_since_s", "bool_positions"]

        clean_positions = []
        if self.cleaning_thread_settings[3]:
            for tm in self.tuple_mngs:
                if tm[1] is not None:
                    clean_positions.extend(tm[1].clean())
        return clean_positions

    def _clean_thread(self) -> None:
        if not self.tuple_mngs:
            # if no tuple managers assigned, we exit immediately
            return

        # ["interval_s", "insert_statuses", "expired_since_s", "bool_positions"]
        if self.cleaning_thread_settings[0] <= 0:
            return

        while not self._stop_token.wait(self.cleaning_thread_settings[0]):
            clean_orders = self._clean_order_managers()
            clean_positions = self._clean_position_managers()

            if self.callback_clean is not None and (clean_orders or clean_positions):
                self.callback_clean(clean_orders, clean_positions)

    def _buffer_thread(self) -> None:
        if self.buffer.maxlen <= 0:
            return

        while not self._stop_token.is_set():
            if not self.buffer:
                self.buffer_event.wait()
                self.buffer_event.clear()

            while self.buffer and not self._stop_token.is_set():
                msg, exp_time = self.buffer.popleft()
                nb_updates = self._parse_msg(msg)

                if exp_time >= time.time() and nb_updates == 0:
                    # re-append: message not yet expired and no updates yet
                    self.buffer.append((msg, exp_time))
                else:
                    # had updates or message expired
                    self._evict_trade_message(msg)
                    _check_update_mode(
                        nb_updates, msg, self.update_mode, self.register_exception
                    )

                # we do not use _stop_token.wait() because it has considerable jitter compared to time.sleep()
                # therefore we sleep at max ping_time in order to be able to join thread when stop is called
                time.sleep(0)

    @lru_cache_non_empty(max_size=128, copy_mode=None)
    def _filter_tuple_mngs(self, order_id: str) -> set[int]:
        return {
            i
            for i, tm in enumerate(self.tuple_mngs)
            if tm[0] is not None and order_id in tm[0]
        }

    # todo caching only more efficient if large number of calls (break-even at about 5000 calls)
    # @lru_cache_non_empty(max_size=64, copy_mode=None)
    def _filter_orders_trade_info(self, msg: TradeWSInfo) -> list[_TradeOrderInfo]:
        if msg.trader_side is TRADER_SIDE.TAKER and msg.owner == self.api_key:
            return [
                _TradeOrderInfo(
                    msg.taker_order_id,
                    msg.size,
                    msg.price,
                    msg.asset_id,
                    msg.side,
                )
            ]
        elif msg.trader_side is TRADER_SIDE.MAKER:
            return [
                _TradeOrderInfo(
                    maker_order.order_id,
                    maker_order.matched_amount,
                    maker_order.price,
                    maker_order.asset_id,
                    _maker_order_side(maker_order, msg),
                )
                for maker_order in msg.maker_orders
                if maker_order.owner == self.api_key
            ]
        return []

    def _untrack_order_id(
        self,
        status: INSERT_STATUS,
        order_id: str,
        ord_mngs_idx: set[int],
    ) -> None:
        if not self.untrack_insert_status:
            return

        if not ord_mngs_idx:
            return

        if status not in self.untrack_insert_status:
            return

        untracked_orders = [
            self.tuple_mngs[tm_idx][0].untrack(order_id, False)
            for tm_idx in ord_mngs_idx
        ]
        untracked_orders = [order for order in untracked_orders if order is not None]

        if self.callback_clean is not None and untracked_orders:
            self.callback_clean(untracked_orders, [])

    def _trade_message(self, msg: TradeWSInfo) -> int:
        trade_order_info = self._filter_orders_trade_info(msg)

        if not trade_order_info:
            return 0

        nb_updates = []

        for trade_order in trade_order_info:
            ord_mngs_idx = self._filter_tuple_mngs(trade_order.order_id)

            # each order should only be updated by exactly one order manager
            # more (order tracked by multiple order managers) or fewer (order not tracked by any order manager)
            #  updates indicates an error
            for tm_idx in ord_mngs_idx:
                if self.tuple_mngs[tm_idx][1] is None:
                    continue

                numeric_type = self._numeric_type_pos_mng[tm_idx]
                self.tuple_mngs[tm_idx][1].transact(
                    asset_id=trade_order.asset_id,
                    delta_size=numeric_type(trade_order.size),
                    price=numeric_type(trade_order.price),
                    trade_id=msg.id,
                    side=trade_order.side,
                    trade_status=msg.status,
                    allow_create=True,
                )  # will raise exception if transact fails

            # should ideally be either be 1 or 0
            nb_updates.append(len(ord_mngs_idx))

        min_updates, max_update = min(nb_updates), max(nb_updates)
        # ideally, both should be 1
        return max_update if max_update > 1 else min_updates

    def _evict_trade_message(self, msg: TradeWSInfo) -> None:
        if msg.event_type != "trade":
            return

        if not self._default_pos_mngs_idx:
            return

        trade_order_info = self._filter_orders_trade_info(msg)

        for trade_order in trade_order_info:
            for tm_idx in self._default_pos_mngs_idx:
                # we know, that none of the position managers in _default_pos_mngs_idx is None,
                #   so we do not need to check again
                numeric_type = self._numeric_type_pos_mng[tm_idx]
                self.tuple_mngs[tm_idx][1].transact(
                    asset_id=trade_order.asset_id,
                    delta_size=numeric_type(trade_order.size),
                    price=numeric_type(trade_order.price),
                    trade_id=msg.id,
                    side=trade_order.side,
                    trade_status=msg.status,
                    allow_create=True,
                )

    def _order_message(self, msg: OrderWSInfo) -> int:
        tpl_mngs_idx = self._filter_tuple_mngs(msg.id)

        for tm_idx in tpl_mngs_idx:
            # in some cases, order_id still might be cached
            #   that's wyh we use a try except
            try:
                self.tuple_mngs[tm_idx][0].update(
                    order_id=msg.id,
                    status=msg.status,
                    size_matched=msg.size_matched,
                    created_at=msg.created_at,
                )
            except OrderGetException:
                # order_id still was in cache
                # OrderGetException indicates, that order_id no longer in order manager
                #   so we remove it manually from the list
                tpl_mngs_idx.remove(tm_idx)

        # only untrack specific order id, but do not clean specific stati,
        #   because order objects might still be needed for self._trade_message
        self._untrack_order_id(msg.status, msg.id, tpl_mngs_idx)

        return len(tpl_mngs_idx)

    def _parse_msg(self, msg: TradeWSInfo | OrderWSInfo) -> int:
        try:
            if msg.event_type == "order":
                return self._order_message(msg)
            elif msg.event_type == "trade":
                return self._trade_message(msg)
            self.register_exception(
                EventTypeException(f"Unknown event_type. Message: {msg}.")
            )
        except Exception as e:
            e.add_note(f"Exception note: websocket received msg={msg}.")
            self.register_exception(e)

    def on_msg(self, msg: TradeWSInfo | OrderWSInfo) -> None:
        if len(self.tuple_mngs) == 0:
            return

        nb_updates = self._parse_msg(msg)

        if nb_updates == 0 and self.buffer.maxlen > 0:
            # in this case we have not yet performed any update, and we have an active buffer
            # no updates were performed so we push to buffer thread to handle yet unprocessed message
            self.buffer.append((msg, time.time() + self.buffer_thread_settings[0]))
            self.buffer_event.set()
        else:
            # in this case we either have no active buffer or updates were performed
            # so we do not need to push to buffer but check directly
            self._evict_trade_message(msg)
            _check_update_mode(
                nb_updates, msg, self.update_mode, self.register_exception
            )
