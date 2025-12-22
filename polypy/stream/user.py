import copy
import itertools
import socket
import threading
import time
import traceback
import warnings
from collections import OrderedDict, deque, namedtuple
from typing import Any, Callable, Literal, NoReturn, Self, TypeAlias, Union

from polypy.constants import ENDPOINT
from polypy.ctf import MarketIdTriplet
from polypy.exceptions import (
    EventTypeException,
    OrderGetException,
    StreamException,
    SubscriptionException,
)
from polypy.manager.order import OrderManagerProtocol
from polypy.manager.position import PositionManagerProtocol
from polypy.order.common import (
    INSERT_STATUS,
    SIDE,
    TERMINAL_INSERT_STATI,
    OrderProtocol,
)
from polypy.position import PositionProtocol
from polypy.stream.common import CHANNEL, MessageStreamer
from polypy.structs import MakerOrder, OrderWSInfo, TradeWSInfo
from polypy.trade import TERMINAL_TRADE_STATI, TRADE_STATUS, TRADER_SIDE
from polypy.typing import infer_numeric_type


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


BufferThreadSettings = namedtuple("BufferSettings", ["keep_s", "max_len"])

TupleManager: TypeAlias = tuple[
    OrderManagerProtocol | None, PositionManagerProtocol | None
]

MarketTriplet: TypeAlias = MarketIdTriplet | tuple[str, str, str]


def _split_market_assets(
    market_triplets: MarketIdTriplet | list[MarketIdTriplet],
) -> tuple[set[str], set[str]]:
    if isinstance(market_triplets, list):
        market_ids = {mas[0] for mas in market_triplets}
        asset_ids = {asset_id for mas in market_triplets for asset_id in mas[1:]}
    elif isinstance(market_triplets, (tuple, MarketIdTriplet)) and isinstance(
        market_triplets[0], str
    ):
        market_ids = {market_triplets[0]}
        asset_ids = {market_triplets[1], market_triplets[2]}
    else:
        raise SubscriptionException(
            f"`market_triplets` must either be MarketTriplet or list[MarketTriplet]. "
            f"Got: {type(market_triplets)}"
        )

    # if not market_ids:
    #     raise SubscriptionException("No markets (condition_id) to subscribe to.")

    if None in market_ids:
        raise SubscriptionException("None not allowed in market ids (condition_id).")

    if None in asset_ids:
        raise SubscriptionException("None not allowed in asset ids (token_id).")

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
            raise SubscriptionException(
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
        raise SubscriptionException(
            "All Order Managers in `tuple_manager` must be unique."
        )


def _assert_token_ids_order_mngs(
    tuple_manager: list[TupleManager],
    asset_ids: set[str],
    coverage_mode: Literal["except", "warn", "ignore"],
) -> None:
    token_ids = set(
        itertools.chain.from_iterable(
            tm[0].token_ids for tm in tuple_manager if tm[0] is not None
        )
    )

    if not token_ids.issubset(asset_ids):
        if coverage_mode == "except":
            raise SubscriptionException(
                f"Not all token_ids of all Order Managers are contained in `market_assets`: "
                f"{token_ids - asset_ids}"
            )
        elif coverage_mode == "warn":
            warnings.warn(
                f"Not all token_ids of all Order Managers are contained in `market_assets`: "
                f"{token_ids - asset_ids}"
            )


def _check_max_subscriptions(market_ids: set[str], max_sub: int) -> None:
    if len(market_ids) > max_sub:
        raise SubscriptionException(
            f"Exceeding max number of subscriptions (={max_sub}). Got: {len(market_ids)}"
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


def _taker_price(maker_order: MakerOrder, asset_id: str) -> str:
    if maker_order.asset_id == asset_id:
        # same asset, so we return price as is: p* = p
        return maker_order.price
    else:
        # complimentary asset -> p* = 1 - p
        # since we do not know the numeric type yet, we return: -p
        return f"-{maker_order.price}"
    # we then can reconstruct via p_ret + (p_ret < 0)


def _numeric_type_balance_total(x: PositionManagerProtocol) -> type | None:
    return None if x is None else infer_numeric_type(x.balance_total)


def _parse_buffer_settings(
    x: BufferThreadSettings | tuple[float, int] | None
) -> BufferThreadSettings:
    return BufferThreadSettings(0, 0) if x is None or None in x else x


def _parse_untrack_insert_stati(
    x: INSERT_STATUS | list[INSERT_STATUS] | None,
) -> list[INSERT_STATUS]:
    if x is None:
        x = []

    if isinstance(x, INSERT_STATUS):
        x = [x]

    if None in x:
        x = []

    if any(x_i not in TERMINAL_INSERT_STATI for x_i in x):
        raise SubscriptionException(
            "Only INSERT_STATUS in TERMINAL_INSERT_STATI allowed for untrack_insert_status."
        )

    if any(x_i is INSERT_STATUS.MATCHED for x_i in x):
        warnings.warn(
            "INSERT_STATUS.MATCHED via untrack_insert_status might lead to exceptions "
            "(if more than 128 new orders are submitted between receiving `order` websocket message "
            "and `trade` websocket message - which is extremely unlikely to happen)."
        )

    return x


def _parse_untrack_trade_stati(
    x: TRADE_STATUS | list[TRADE_STATUS] | None,
) -> list[TRADE_STATUS]:
    if x is None:
        x = []

    if isinstance(x, TRADE_STATUS):
        x = [x]

    if None in x:
        x = []

    if any(x_i is TRADE_STATUS.RETRYING for x_i in x):
        raise SubscriptionException(
            "TRADE_STATUS.RETRYING must not be in untrack_trade_status."
        )

    if any(x_i not in TERMINAL_TRADE_STATI for x_i in x):
        warnings.warn(
            "Depending on the PositionProtocol implementation used by Position Managers, "
            "it is absolutely recommended to only use TRADE_STATUS contained in TERMINAL_TRADE_STATI "
            "for untrack_trade_status (i.e., if position is settled at .CONFIRMED, using "
            ".MATCHED in untrack_trade_status makes no sense and costs performance)."
        )

    return x


_TradeOrderInfo = namedtuple(
    "_TradeOrderInfo", ["order_id", "size", "price", "asset_id", "side"]
)


# todo allow_untracked_sell? -> allow_neg or allow_untracked_sell -> feature + doc: allow_neg in factory
# todo if one of multiple updates on_msg raise exception: transaction will be conducted, so manager states might be compromised -> if order_manager is invalidate, then it really is -> doc
# todo doc: if clean manually, potential danger of position not being adjusted/transacted if order not anymore in order manager (in explicit mode: will raise exception)
#   -> only manually clean if no websocket traffic is expected, especially cancel
# todo doc: in notes: untrack_insert_status only works because of lru_cache, especially if cancel -> see point above
#   (if order is removed from order manager before websocket message has arrived, message cannot be assigned to the correct position manager anymore,
#    resulting in lost position transaction)
# todo doc: to above points -> regular manually cleaning can make sense, because untracking is not the most sound approach especially for positions
# todo doc: use TERMINAL_X_STATI for untrack_x_status
# todo doc: untrack_trade_status should comply with settlement status of position (i.e. CMSPosition: MATCHED is not efficient -> use TERMINAL_TRADE_STATI as safe arg)
# todo doc: multiple default position manager use case: pm A only tracks one wallet (this user stream) whilst second pm tracks this wallet and an other wallet (other user stream) at once
# todo doc: untrack_order_by_trade_terminal -> this will fail if "CONFIRMED" is missed
class UserStream(MessageStreamer):
    def __init__(
        self,
        ws_endpoint: ENDPOINT | str,
        tuple_manager: TupleManager | list[TupleManager] | None,
        market_triplets: MarketTriplet | list[MarketTriplet],
        api_key: str,
        secret: str,
        passphrase: str,
        untrack_insert_status: INSERT_STATUS | list[INSERT_STATUS] | None = None,
        untrack_trade_status: TRADE_STATUS | list[TRADE_STATUS] | None = None,
        untrack_order_by_trade_terminal: bool = True,
        monitor_assets_thread_s: float | None = 0.1,
        buffer_thread_settings: BufferThreadSettings | None = (2, 5_000),
        pull_aug_conversion_s: float | None = None,
        ping_time: float | None = 5,
        update_mode: Literal["explicit", "implicit"] = "explicit",
        coverage_mode: Literal["except", "warn", "ignore"] = "except",
        invalidate_on_exc: bool = True,
        callback_msg: Callable[[Self, TradeWSInfo | OrderWSInfo], None] | None = None,
        callback_exc: Callable[[Self, Exception], None] | None = None,
        callback_untrack: Callable[[list[OrderProtocol], list[PositionProtocol]], None]
        | None = None,
        ws_channel: CHANNEL | str = CHANNEL.USER,
        max_subscriptions: int = 500,
        sock: socket.socket | None = None,
    ) -> None:
        market_ids, asset_ids = _split_market_assets(market_triplets)

        tuple_manager = _parse_tuple_manager_list(tuple_manager)
        _check_max_subscriptions(market_ids, max_subscriptions)
        _assert_api_key_order_mng(tuple_manager, api_key)
        _assert_unique_order_mngs(tuple_manager)
        _assert_token_ids_order_mngs(tuple_manager, asset_ids, coverage_mode)
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

        self.untrack_insert_status = _parse_untrack_insert_stati(untrack_insert_status)
        self.untrack_trade_status = _parse_untrack_trade_stati(untrack_trade_status)
        self.untrack_order_by_trade_terminal = untrack_order_by_trade_terminal
        self.callback_untrack = callback_untrack

        if ws_endpoint[-1] != "/":
            ws_endpoint = f"{ws_endpoint}/"

        self.update_mode = update_mode
        self.coverage_mode = coverage_mode
        self.invalidate_on_exc = invalidate_on_exc

        self.buffer_thread_settings = _parse_buffer_settings(buffer_thread_settings)
        self.buffer = deque(maxlen=self.buffer_thread_settings[1])
        self.buffer_thread: threading.Thread | None = None
        self.buffer_event = threading.Event()

        self.monitor_thread: threading.Thread | None = None
        self.monitor_assets_thread_s = monitor_assets_thread_s

        self.aug_conv_thread: threading.Thread | None = None
        self.pull_aug_conversion_s = pull_aug_conversion_s

        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase

        def _callback_invalidate_exc(stream: Self, exc: Exception) -> None:
            reason = (
                f"Exception in {self.__class__.__name__}. "
                f"Original exception: {exc.__class__.__name__}({exc})."
                f"Full Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )

            if self.invalidate_on_exc:
                for tm in self.tuple_mngs:
                    if tm[0] is not None:
                        tm[0].invalidate(reason)

                    if tm[1] is not None:
                        tm[1].invalidate(reason)

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
            msgspec_type=list[Union[TradeWSInfo, OrderWSInfo]]
            | Union[TradeWSInfo, OrderWSInfo],
            msgspec_strict=False,
            sock=sock,
        )

    @property
    def market_ids(self) -> list[str]:
        return list(self._market_ids)

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

        if self.aug_conv_thread is not None:
            self.register_exception(
                StreamException("Internal exception: self.aug_conv_thread is not None.")
            )

        self.monitor_thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self.monitor_thread.start()

        self.buffer_event.clear()

        self.buffer_thread = threading.Thread(target=self._buffer_thread, daemon=True)
        self.buffer_thread.start()

        self.aug_conv_thread = threading.Thread(
            target=self._aug_conv_thread, daemon=True
        )
        self.aug_conv_thread.start()

    def pre_start(self) -> None:
        self.buffer = deque(maxlen=self.buffer_thread_settings[1])
        self._start_aux_threads()

    def _stop_aux_threads(self) -> None:
        self.buffer_event.set()

        self.monitor_thread.join(self.ping_time + 0.01)
        self.aug_conv_thread.join(self.ping_time + 0.01)
        self.buffer_thread.join(self.ping_time + 0.01)

        if self.monitor_thread.is_alive():
            self.register_exception(StreamException("Cannot join() monitoring thread."))

        if self.buffer_thread.is_alive():
            self.register_exception(StreamException("Cannot join() buffer thread."))

        if self.aug_conv_thread.is_alive():
            self.register_exception(
                StreamException("Cannot join() augmented conversions thread.")
            )

        self.monitor_thread = None
        self.aug_conv_thread = None
        self.buffer_thread = None
        self.buffer_event.clear()

    def post_stop(self) -> None:
        self._stop_aux_threads()
        self.buffer = deque(maxlen=self.buffer_thread_settings[1])

    def _monitor_thread(self) -> None:
        if not self.tuple_mngs:
            # if no managers assigned, we exit immediately
            return

        if self.monitor_assets_thread_s is None or self.monitor_assets_thread_s <= 0:
            return

        if not self._market_ids:
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

    def _aug_conv_thread(self) -> None:
        if not self.tuple_mngs:
            # if no managers assigned, we exit immediately:
            return

        if self.pull_aug_conversion_s is None or self.pull_aug_conversion_s <= 0:
            warnings.warn(
                "PositionManagerProtocol.pull_augmented_conversions(...) is "
                "not pulled automatically (pull_aug_conversion_s=None). "
                "Make sure to manually call it periodically after converting positions in "
                "negative risk markets, else ignore this message."
            )
            return

        warnings.warn(
            "PositionManagerProtocol.pull_augmented_conversions(...) will be called automatically. "
            "Underlying Gamma API REST calls might impede performance if pull time is set too short."
        )

        while not self._stop_token.wait(self.pull_aug_conversion_s):
            try:
                for tm in self.tuple_mngs:
                    if tm[1] is not None:
                        tm[1].pull_augmented_conversions()
            except Exception as e:
                self.register_exception(e)

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
                    maker_order.matched_amount,
                    _taker_price(maker_order, msg.asset_id),
                    msg.asset_id,
                    msg.side,
                )
                for maker_order in msg.maker_orders
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
        self, status: INSERT_STATUS, order_id: str, ord_mngs_idx: set[int]
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

        if self.callback_untrack is not None and untracked_orders:
            self.callback_untrack(untracked_orders, [])

    def _untrack_order_trade_terminal(
        self, trade_status: TRADE_STATUS, order_id: str, ord_mngs_id: int
    ) -> None:
        # if OrderWSInfo arrives before TradeWSInfo:
        #   order is already updated and untracked (if applicable), in which case,
        #   this condition will be skipped and we are all good

        # if OrderWSInfo arrives before TradeWSInfo, we have three cases:
        #   1) maker order: OrderWSInfo will arrive later and will update and untrack order (if applicable)
        #   2) partial taker order with resting remainder: OrderWSInfo will be emitted as soon as resting
        #       remainder is matched, which then is case 1)
        #   3) (partial) taker order with no resting remainder:
        #       this can either be FOK (fully filled) or FAK (partially filled),
        #       in this case no (!) OrderWSInfo, we have two subcases:
        #           a) REST response arrives before TradeWSInfo: order is updated
        #               and untracked (if applicable), we are all good
        #           b) REST response arrives after TradeWSInfo:
        #               order status is not updated yet, but at latest when "CONFIRMED" arrives,
        #               order status will be in terminal state, such that we can untrack (if applicable)

        # todo
        # if CONFIRMED was missed, this will always fail

        if not self.untrack_order_by_trade_terminal:
            return

        if self.tuple_mngs[ord_mngs_id][0] is None:
            return

        if trade_status not in TERMINAL_TRADE_STATI:
            return

        if (order := self.tuple_mngs[ord_mngs_id][0].get_by_id(order_id)) is None:
            return

        if order.status not in TERMINAL_INSERT_STATI:
            return

        untracked_order = self.tuple_mngs[ord_mngs_id][0].untrack(order_id, False)

        if self.callback_untrack is not None and untracked_order:
            self.callback_untrack([untracked_order], [])

    def _untrack_position(
        self, status: TRADE_STATUS, asset_id: str, pos_mngs_id: int
    ) -> None:
        if not self.untrack_trade_status:
            return

        # we actually know that position manager is guaranteed not to be None
        # if self.tuple_mngs[pos_mngs_id][1] is None:
        #     return

        if status not in self.untrack_trade_status:
            return

        if (
            frozen_position := self.tuple_mngs[pos_mngs_id][1].get_by_id(asset_id)
        ) is None:
            return

        if not frozen_position.empty:
            return

        untracked_positions = self.tuple_mngs[pos_mngs_id][1].untrack(asset_id)

        if self.callback_untrack is not None and untracked_positions:
            self.callback_untrack([], [untracked_positions])

    def _trade_message(self, msg: TradeWSInfo) -> int:
        trade_order_info = self._filter_orders_trade_info(msg)

        if not trade_order_info:
            return 0

        nb_updates = []

        for trade_order in trade_order_info:
            tpl_mngs_idx = self._filter_tuple_mngs(trade_order.order_id)

            # each order should only be updated by exactly one order manager
            # more (order tracked by multiple order managers) or fewer (order not tracked by any order manager)
            #  updates indicates an error
            for tm_idx in tpl_mngs_idx:
                # noinspection DuplicatedCode
                if self.tuple_mngs[tm_idx][1] is not None:
                    # noinspection DuplicatedCode
                    numeric_type = self._numeric_type_pos_mng[tm_idx]
                    price = numeric_type(trade_order.price)
                    # if price is negative, we have to properly invert the price: 1 + p
                    # if price is positive, we take it as is: p + 0
                    self.tuple_mngs[tm_idx][1].transact(
                        asset_id=trade_order.asset_id,
                        delta_size=numeric_type(trade_order.size),
                        price=price + (price < 0),
                        trade_id=msg.id,
                        side=trade_order.side,
                        trade_status=msg.status,
                        allow_create=True,
                    )  # will raise exception if transact fails

                self._untrack_position(msg.status, trade_order.asset_id, tm_idx)

                # evict order only after position has been processed since we need the order to
                #   infer the correct position manager
                self._untrack_order_trade_terminal(
                    msg.status, trade_order.order_id, tm_idx
                )

            # should ideally be either 1 or 0
            nb_updates.append(len(tpl_mngs_idx))

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
            # noinspection DuplicatedCode
            for tm_idx in self._default_pos_mngs_idx:
                # we know, that none of the position managers in _default_pos_mngs_idx is None,
                #   so we do not need to check again
                # noinspection DuplicatedCode
                numeric_type = self._numeric_type_pos_mng[tm_idx]
                price = numeric_type(trade_order.price)
                self.tuple_mngs[tm_idx][1].transact(
                    asset_id=trade_order.asset_id,
                    delta_size=numeric_type(trade_order.size),
                    price=price + (price < 0),
                    trade_id=msg.id,
                    side=trade_order.side,
                    trade_status=msg.status,
                    allow_create=True,
                )

                # if we evict a trade, we know, that there is no order manager assigned,
                # so we do not need to untrack orders
                # we only untrack from position manager
                self._untrack_position(msg.status, trade_order.asset_id, tm_idx)

    def _order_message(self, msg: OrderWSInfo) -> int:
        tpl_mngs_idx = self._filter_tuple_mngs(msg.id)

        for tm_idx in tpl_mngs_idx.copy():
            # in some cases, order_id still might be cached
            #   that's wyh we use a try except
            try:
                self.tuple_mngs[tm_idx][0].update(
                    order_id=msg.id,
                    status=msg.status,
                    size_matched=msg.size_matched or "0",
                    created_at=msg.created_at,
                )
            except OrderGetException:
                # order_id was still in cache
                # OrderGetException indicates, that order_id no longer in order manager
                #   so we remove it manually from the list
                #   we do not remove it from the cache because there might be a 'trade' message still be incoming
                #   (and also we do not clean the cache if we untrack orders, so no need in general)
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
            # so we do not push to buffer but check directly
            self._evict_trade_message(msg)
            _check_update_mode(
                nb_updates, msg, self.update_mode, self.register_exception
            )
