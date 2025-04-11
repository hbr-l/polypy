import contextlib
import datetime
import math
import threading
import warnings
from typing import Any, KeysView, Literal, Protocol, TypeAlias

from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey

from polypy.book import OrderBook
from polypy.constants import CHAIN_ID, ENDPOINT
from polypy.exceptions import (
    ManagerInvalidException,
    OrderCreationException,
    OrderGetException,
    OrderTrackingException,
    OrderUpdateException,
    PolyPyException,
)
from polypy.order.common import (
    CANCELABLE_INSERT_STATI,
    INSERT_STATUS,
    SIDE,
    TERMINAL_INSERT_STATI,
    TIME_IN_FORCE,
    FrozenOrder,
    OrderProtocol,
    frozen_order,
    update_order,
)
from polypy.order.limit import create_limit_order
from polypy.order.market import create_market_order
from polypy.rest.api import (
    cancel_orders,
    get_neg_risk,
    get_order_updates,
    get_tick_size,
    post_order,
)
from polypy.signing import SIGNATURE_TYPE, parse_private_key
from polypy.structs import CancelOrdersResponse, PostOrderResponse
from polypy.typing import NumericAlias

AnyOrderProtocol: TypeAlias = OrderProtocol | FrozenOrder


class OrderManagerProtocol(Protocol):
    """Interface layer between bare OrderProtocol and UserStream.
    Must implement locking if necessary (e.g. sharedMem order).

    Notes
    -----
    __It is highly recommended, to use self.post, self.market_order or self.limit_order, instead of
    manually posting orders (via polypy.rest.api.post)!__ (or call self.sync after self.track).

    - If an already posted open order shall be tracked:
        >>> order = polypy.rest.api.get_order()
        >>> order_manager.track(order)
    - 'cancel_market' is not implemented on purpose, because other order managers may manage orders on
      the same market. Instead, use:
        >>> polypy.rest.api.cancel_market(...)
      Managed order will be updated automatically if Order Manager is assigned to a stream.
    """

    api_key: str
    secret: str
    passphrase: str

    def __contains__(self, order_id: str) -> bool:
        ...

    @property
    def order_ids(self) -> list[str] | KeysView[str]:
        ...

    @property
    def token_ids(self) -> list[str] | KeysView[str]:
        """Get all token IDs currently involved with any order (any state) managed by the Order Manager.
        This property is essential for validating the User Stream."""
        ...

    @property
    def valid(self) -> bool:
        """True, if .invalidate() has not been called. False, if Order Manager has been invalidated."""
        ...

    def invalidate(self, reason: str | None = None) -> None:
        """Invalidate the Order Manager s.t. any successive call will raise an exception.
        This method is mainly used for loose coupling of the Order Manager and the User Stream.
        """
        ...

    def track(self, order: OrderProtocol, sync: bool) -> None:
        """Add order to the Order Manager to be tracked. Orders are updated automatically,
        if Order Manager is assigned to a stream. Orders are updated whenever messages are received in the stream.
        Order must have an ID already assigned. If not, use self.post.

        Notes
        -----
        It is highly advised to use self.market_order or self.limit_order to create and track orders automatically,
        instead of creating orders manually and then track them.
        Call self.sync() subsequently if already posted (submitted) orders are added to the Order Manager to catch
        up with any changes happened in the meantime.

        If the timespan between posting the order and tracking it (adding it to the Order Manager) is relatively
        small, and the Order Manager is assigned to a stream, that uses an internal buffer, then calling
        self.sync() might in fact not be necessary at all.
        """
        ...

    def untrack(self, order_id: str, sync: bool) -> OrderProtocol | None:
        """Remove order from tracking by the Order Manager. Returns original (editable) order object.

        Notes
        -----
        Call self.sync() beforehand if already posted (submitted) orders are untracked from the Order Manager.
        """
        ...

    def get(self, **kwargs) -> list[FrozenOrder]:
        """Get orders tracked by the Order Manager via search criteria. Returned orders are frozen and read-only as
        they are still tracked and updated by the Order Manager.

        Notes
        -----
        Call self.sync() beforehand if Order Manager is not assigned to a stream or if already posted orders
        were added to the Order Manager.
        """
        ...

    def get_by_id(self, order_id: str) -> FrozenOrder | None:
        """Get order by id. Returned order is frozen."""

    def update(
        self,
        order_id: str,
        status: INSERT_STATUS | None = None,
        size_matched: NumericAlias | None = None,
        strategy_id: str | None = None,
        created_at: int | None = None,
        **kwargs,
    ) -> None:
        """Update an order within the Order Manager regarding status, size_matched, strategy_id and signature.
        `order_id` is necessary to identify the order within the Order Manager.
        Any additional kwargs (key-value mapping) will be modified if possible (e.g. aux_id).
        Regressive updates (see Notes) will be ignored.

        Raises
        ------
        OrderTrackingException: if order_id not in order manager
        OrderUpdateException: if kwargs could not be updated order

        Notes
        -----
        `created_at` and `signature` can only be modified if not already set, else an OrderUpdateException is raised.
        `status` and `size_matched` can only be updated if they do not regress (e.g., cannot update from
        INSERT_STATUS.MATCHED to INSERT_STATUS.LIVE, cannot update size_matched smaller than current size_matched),
        else they will be ignored.
        """
        ...

    def modify(self, order_id: str, **kwargs) -> None:
        """Update an order within the Order Manager. `order_id`is necessary to identify the order within
        the Order Manager. Any kwargs (key-value mapping) will be modified if possible.

        In contrast to :py:meth:`OrderManagerProtocol.update`, modify will try to modify any order attributes and
        does not check for regression.
        Because regressive modifications are allowed, use this method with care.

        Notes
        -----
        Does not auto-cast any values, unless corresponding order object has on_setattr defined to auto-cast values.
        """

    def market_order(
        self,
        amount: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias | None,
        book: OrderBook | None,
        max_size: NumericAlias | None,
        neg_risk: bool | None = None,
        **kwargs,
    ) -> tuple[FrozenOrder, PostOrderResponse]:
        """Creates a market order, posts it to the exchange and adds it to the Order Manager to be tracked.

        See :py:meth:`OrderManagerProtocol.post` for details (i.e. in case of an exception during posting the order).

        Parameters
        ----------
        amount
        token_id
        side
        tick_size: float | NumericAlias | None
            if None: first tries to infer tick_size from `book`, and if no `book` specified, then performs REST call
        book
        max_size: NumericAlias | None
            only necessary if side=SIDE.SELL. Maximal number of shares to sell (should not be greater than current
            position size)
        neg_risk

        Notes
        -----
        If amount is pre-round accordingly (precision), then tick_size can be set to any sufficiently small
        min tick_size (i.e. 0.001 or any smaller), and does not affect order creation anymore (as long as
        tick_size is sufficiently small).
        """
        ...

    def limit_order(
        self,
        price: NumericAlias,
        size: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias | None,
        tif: TIME_IN_FORCE,
        expiration: int | None,
        neg_risk: bool | None = None,
        **kwargs,
    ) -> tuple[FrozenOrder, PostOrderResponse]:
        """Creates a limit order, posts it to the exchange and adds it to the Order Manager to be tracked.

        See :py:meth:`OrderManagerProtocol.post` for details (i.e. in case of an exception during posting the order).

        Notes
        -----
        If price is pre-round accordingly (precision), then tick_size can be set to any sufficiently small
        min tick_size (i.e. 0.001 or any smaller), and does not affect order creation anymore (as long as
        tick_size is sufficiently small).
        """
        ...

    def post(self, order: AnyOrderProtocol) -> tuple[FrozenOrder, PostOrderResponse]:
        """Post an order to the exchange. Order will be added to the Order Manager automatically.

        If posting an order raises an exception, i.e. status UNMATCHED, the order will only be tracked
        (added to the Order Manager) if we received an OrderId from the exchange (regardless of the order status).
        If no OrderID was received from the exchange, the order will NOT be tracked!
        See :py:func:`polypy.rest.api.post_order` which exception may be raised.

        Notes
        -----
        Recipe: in case of an exception that still is able to assign an OrderID (e.g. status UNMATCHED),
        use unique value for aux_id field (if :py:class:`polypy.order.base.Order`) to retrieve order via
        :py:meth:`OrderManagerProtocol.retrieve` and check status.
        """
        ...

    def cancel(
        self,
        order: AnyOrderProtocol | str | list[AnyOrderProtocol | str],
        mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
    ) -> tuple[FrozenOrder | list[FrozenOrder], CancelOrdersResponse]:
        """Cancel order(s). If order id (str) is used, then order has to be already tracked by the Order Manager.
        Order/s is/are not untracked automatically.

        - If order objects are used, they will be added to the Order Manager automatically for tracking.
        - If any order is not tracked and cannot be tracked (e.g. wrong token_id), order cancellation will not be
          attempted at all (no cancellation attempt for any of the orders). Cancellation attempt will only be
          attempted if all orders are either already tracked or are trackable.
        - If FrozenWrapper (frozen order), order MUST already be tracked by the Order Manager (cannot add frozen order).
        - If exception is raised during cancellation (e.g. server-side error),
          at least successfully canceled orders will be updated accordingly.

        Notes
        -----
        Call self.sync() beforehand if already posted (submitted) orders were added to the Order Manager.
        Note, there is no implicit sync call inside the method.

        Raises
        ------
        OrderTrackingException: If orders cannot be tracked
        OrderUpdateException: If exception during cancellation
        """

    def cancel_all(
        self,
        mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
        statuses: INSERT_STATUS | list[INSERT_STATUS] = CANCELABLE_INSERT_STATI,
    ) -> tuple[list[FrozenOrder], CancelOrdersResponse]:
        """Cancel all orders tracked by the Order Manager. Orders are not automatically untracked.
        This only cancels all order of the Order Manager BUT NOT all orders of the associated account!

        If exception is raised, at least successfully canceled orders will be updated accordingly.

        Notes
        -----
        Call self.sync() beforehand if already posted (submitted) orders were added to the Order Manager or if
        Order Manager contains orders in status DEFINED in order to determine current order status.
        Note, there is no implicit sync call inside the method.
        """
        ...

    def sync(
        self,
        order: AnyOrderProtocol | str | list[AnyOrderProtocol | str] | None,
        mode_not_existent: Literal["except", "remove", "warn"],
    ) -> tuple[FrozenOrder | list[FrozenOrder], list[OrderProtocol]]:
        """Synchronize order(s) with information from exchange response.

        - If order object, tracks orders automatically, if trackable.
        - If any order is not tracked and cannot be tracked (e.g. wrong token_id), sync will not be
          attempted at all (no get_order/ REST call attempt for any of the orders). Sync attempt will only be
          attempted if all orders are either already tracked or are trackable.
        - If order_ids as strings, then orders have to be added to the Order Manager beforehand.
        - If None, sync all orders.
        - Any order not in status [DEFINED, LIVE, DELAYED] will be ignored (no exception raised)

        This is particularly useful, when Order Manager is not assigned to a stream and
        has to manage updates manually (not recommended though).

        Raises
        ------
        OrderTrackingException: If orders cannot be tracked
        OrderUpdateException: If exception during REST call

        Notes
        -----
        Call self.sync() if operating on already posted (submitted) orders. It is highly advised to
        use self.market_order or self.limit_order instead of creating orders manually.

        Only syncs orders in states DEFINED (if order id is known in advance), LIVE and DELAYED.
        """
        ...

    def clean(
        self,
        statuses: INSERT_STATUS | list[INSERT_STATUS] = TERMINAL_INSERT_STATI,
        expiration: int = -1,
    ) -> list[OrderProtocol]:
        """Untrack all orders with specified status OR expiration. This does not cancel any orders.

        Parameters
        ----------
        statuses: list[INSERT_STATUS] | None,
            if None, untrack MATCHED, UNMATCHED, CANCELED.
        expiration: int | IGNORE_EXPIRATION = IGNORE_EXPIRATION
            unix timestamp in millis

        Notes
        -----
        Users are advised, to regularly also clean DEFINED (but only if no stream updates are expected) to clear
        out any orders that failed to post.
        """
        ...


class MarketOrderFactory(Protocol):
    def __call__(
        self,
        amount: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias,
        neg_risk: bool,
        chain_id: CHAIN_ID,
        private_key: PrivateKey | str | PrivateKeyType,
        maker: str | None,
        signature_type: SIGNATURE_TYPE,
        book: OrderBook,
        max_size: NumericAlias | None,
        **kwargs,
    ) -> OrderProtocol:
        ...


class LimitOrderFactory(Protocol):
    def __call__(
        self,
        price: NumericAlias,
        size: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias,
        neg_risk: bool,
        chain_id: CHAIN_ID,
        private_key: PrivateKey | str | PrivateKeyType,
        maker: str | None,
        signature_type: SIGNATURE_TYPE,
        tif: TIME_IN_FORCE,
        expiration: int,
        **kwargs,
    ) -> OrderProtocol:
        ...


def _parse_to_list(x: Any | list[Any]) -> tuple[bool, list[Any]]:
    if isinstance(x, list):
        return False, x
    elif isinstance(x, (set, tuple)):
        return False, list(x)

    return True, [x]


def _filter_update_kwargs(kwargs: dict) -> dict:
    kwargs.pop("status", None)
    kwargs.pop("size_matched", None)
    kwargs.pop("strategy_id", None)
    kwargs.pop("created_at", None)
    return kwargs


def _update_order_kwargs(
    order: OrderProtocol, kwargs: dict
) -> tuple[dict, OrderProtocol]:
    missed_kwargs = {}

    for k, v in kwargs.items():
        try:
            order.__setattr__(k, v)
        except (AttributeError, OrderUpdateException):
            missed_kwargs[k] = v

    return missed_kwargs, order


def _cvt_order_numeric(order: OrderProtocol, val: Any | None) -> NumericAlias:
    return None if val is None else order.numeric_type(val)


_ERR_MSG_INVALID = "self.invalidate() was called. Order Manager is invalidated."


# todo move lock to custom dict factory, make all operations atomic
class OrderManager(OrderManagerProtocol):
    """Dict-based order manager with locking. Can be used with multithreading, but not multiprocessing."""

    def __init__(
        self,
        rest_endpoint: ENDPOINT | str,
        private_key: PrivateKey | str | PrivateKeyType,
        api_key: str,
        secret: str,
        passphrase: str,
        maker_funder: str | None,
        signature_type: SIGNATURE_TYPE | None,
        chain_id: CHAIN_ID,
        max_size: int | None = None,
        market_order_factory: MarketOrderFactory = create_market_order,
        limit_order_factory: LimitOrderFactory = create_limit_order,
    ):
        """Dict-based order manager.

        Parameters
        ----------
        rest_endpoint
        private_key
        api_key
        secret
        passphrase
        maker_funder: str | None
            wallet address corresponding to `signature_type`.
            If wallet address is the same as used on Polymarket UI, use SIGNATURE_TYPE.POLY_PROXY.
            If None, uses checksum_address of private key, which requires SIGNATURE_TYPE.EOA - which in most
            cases is not what you want.
        signature_type: SIGNATURE_TYPE | None
            signature type.
            If None, uses SIGNATURE_TYPE.EOA (which most probably is not what you want) - see `maker_funder`.
        chain_id
        max_size
        market_order_factory
        limit_order_factory
        """
        self.rest_endpoint = rest_endpoint

        self.private_key = parse_private_key(private_key)
        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase

        self.maker_funder = maker_funder
        self.signature_type = signature_type
        self.chain_id = chain_id

        self.max_size = math.inf if max_size is None else max_size

        self._create_market_order = market_order_factory
        self._create_limit_order = limit_order_factory

        self.order_dict: dict[str, OrderProtocol] = {}
        self.token_dict: dict[str, int] = {}
        self.lock = threading.RLock()

        self._invalid_token: bool = False
        self._invalid_reason: str = _ERR_MSG_INVALID

    def __contains__(self, order_id: str) -> bool:
        return order_id in self.order_dict

    @property
    def order_ids(self) -> KeysView[str]:
        return self.order_dict.keys()

    @property
    def token_ids(self) -> KeysView[str]:
        return self.token_dict.keys()

    @property
    def valid(self) -> bool:
        return not self._invalid_token

    @staticmethod
    def _is_trackable(order: OrderProtocol) -> None:
        if isinstance(order, FrozenOrder):
            raise OrderTrackingException(
                "Cannot track frozen order (which, e.g. is the case when already tracked by an Order Manager)."
            )

        if order.id is None or order.id == "":
            raise OrderTrackingException(
                "Can only track orders, which already have an ID assigned (=submitted)."
            )

    def invalidate(self, reason: str | None = None) -> None:
        self._invalid_token = True

        if reason is None:
            reason = (
                "E.g., Order Manager tried to create an order for an asset ID that is "
                "not tracked by the corresponding User Stream that it was assigned to."
            )
        self._invalid_reason = f"{_ERR_MSG_INVALID} Reason: {reason}"

    def _validate(self) -> None:
        if self._invalid_token:
            raise ManagerInvalidException(self._invalid_reason)

    def track(self, order: OrderProtocol, sync: bool) -> None:
        self._is_trackable(order)

        with self.lock:
            self._validate()
            if len(self.order_dict) >= self.max_size:
                raise OrderTrackingException(
                    f"Exceeding max_size={self.max_size}. Either set a higher max_size, or use"
                    f"OrderManager.clean() or OrderManager.untrack() to free up space."
                )

        # do not lock during sync, as sync may take a while
        if sync:
            try:
                # sync will also add the order to be tracked if successful
                self.sync(order, mode_not_existent="except")
            except Exception as e:
                # if we fail to sync, we have to remove the order again before re-raising
                self.untrack(order.id, sync=False)
                raise OrderUpdateException(f"Failed to sync order: {order}.") from e
        else:
            with self.lock:
                self._insert_order(order)

    def _inc_token_id_count(self, token_id: str) -> None:
        if token_id not in self.token_dict:
            self.token_dict[token_id] = 1
        else:
            self.token_dict[token_id] += 1

    def _insert_order(self, order: OrderProtocol) -> None:
        if order.id not in self.order_dict:
            self._inc_token_id_count(order.token_id)

        # we check, that for a given order_id the corresponding token_id hasn't changed:
        #   at the time when the order is added, we increase the token counter for token A
        #   if the token_id had changed at the time of popping the order to token B, we would
        #   decrease the token counter for token B instead of token A, which would invalidate our token counter
        # in use cases, where the order manager is solely used in tight conjunction with a UserStream,
        #   the above scenario is extreme unlikely - though for more manual use, we still perform this check
        #   to be safe
        if (
            order.id in self.order_dict
            and order.token_id != self.order_dict[order.id].token_id
        ):
            raise OrderTrackingException(
                f"Order ID already tracked, but changed token_id. Can only overwrite if token_ids match. "
                f"Previous order: {self.order_dict[order.id]}. New order: {order}."
            )

        self.order_dict[order.id] = order

        if len(self.token_dict) > len(self.order_dict):
            raise PolyPyException(
                "Internal exception: len(self.token_dict) > len(self.order_dict)"
            )

    def _dec_token_id_count(self, token_id: str) -> None:
        self.token_dict[token_id] -= 1

        if self.token_dict[token_id] == 0:
            del self.token_dict[token_id]

    def _pop_order(self, order_id: str) -> OrderProtocol | None:
        order = self.order_dict.pop(order_id, None)
        if order is not None:
            self._dec_token_id_count(order.token_id)

        if len(self.token_dict) > len(self.order_dict):
            raise PolyPyException(
                "Internal exception: len(self.token_dict) > len(self.order_dict)"
            )

        return order

    def untrack(self, order_id: str, sync: bool) -> OrderProtocol | None:
        with self.lock:
            self._validate()
            if sync:
                # sync first before popping from dict
                try:
                    self.sync(self.order_dict[order_id], mode_not_existent="warn")
                except KeyError:
                    # no such order in order manager, so return None
                    return None
                except Exception as e:
                    self._pop_order(order_id)
                    raise OrderUpdateException(
                        f"Order has been removed from the Order Manager but failed to sync order. "
                        f"Order ID: {order_id}."
                    ) from e

            return self._pop_order(order_id)

    # noinspection PyProtocol
    def get(self, **kwargs) -> list[FrozenOrder]:
        with self.lock:
            self._validate()
            try:
                return [frozen_order(self.order_dict[kwargs["id"]])]
            except KeyError:
                return [
                    frozen_order(order)
                    for order in self.order_dict.values()
                    if all(
                        hasattr(order, k) and getattr(order, k) == v
                        for k, v in kwargs.items()
                    )
                ]

    def get_by_id(self, order_id: str) -> FrozenOrder | None:
        with self.lock:
            self._validate()
            try:
                return frozen_order(self.order_dict[order_id])
            except KeyError:
                return None

    def _get_order(self, order_id: str) -> OrderProtocol | None:
        try:
            return self.order_dict[order_id]

        except KeyError as e:
            raise OrderGetException(
                f"{datetime.datetime.now()} | Order not found for id: {order_id}."
            ) from e

    # noinspection PyProtocol
    def update(
        self,
        order_id: str,
        status: INSERT_STATUS | None = None,
        size_matched: NumericAlias | str | None = None,
        strategy_id: str | None = None,
        created_at: int | None = None,
        **kwargs,
    ) -> None:
        with self.lock:
            self._validate()
            order = self._get_order(order_id)
            size_matched = _cvt_order_numeric(order, size_matched)

            order = update_order(
                order=order,
                status=status,
                size_matched=size_matched,
                strategy_id=strategy_id,
                created_at=created_at,
            )
            kwargs = _filter_update_kwargs(kwargs)
            missed_kwargs, order = _update_order_kwargs(order, kwargs)

        if len(missed_kwargs) > 0:
            raise OrderUpdateException(
                f"Failed to set the following attribute mappings: {missed_kwargs}. Order: {order}."
            )

    # noinspection PyProtocol
    def modify(self, order_id: str, **kwargs) -> None:
        with self.lock:
            self._validate()
            order = self._get_order(order_id)
            missed_kwargs, order = _update_order_kwargs(order, kwargs)

        if len(missed_kwargs) > 0:
            raise OrderUpdateException(
                f"Failed to set the following attribute mappings: {missed_kwargs}. Order: {order}."
            )

    # noinspection PyProtocol
    def market_order(
        self,
        amount: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias | None,
        book: OrderBook | None,
        max_size: NumericAlias | None,
        neg_risk: bool | None = None,
        **kwargs,
    ) -> tuple[FrozenOrder, PostOrderResponse]:
        """Creates a market order, posts it to the exchange and adds it to the Order Manager to be tracked.

        Notes
        -----
        Example for kwargs: aux_id='some additional info' in case of polypy.order.Order class.
        """
        if tick_size is None:
            if book is None:
                tick_size = get_tick_size(self.rest_endpoint, token_id)
            else:
                tick_size = book.tick_size

        if neg_risk is None:
            # get_neg_risk is lru cached
            neg_risk = get_neg_risk(self.rest_endpoint, token_id)

        order = self._create_market_order(
            amount=amount,
            token_id=token_id,
            side=side,
            tick_size=tick_size,
            neg_risk=neg_risk,
            chain_id=self.chain_id,
            private_key=self.private_key,
            maker=self.maker_funder,
            signature_type=self.signature_type,
            book=book,
            max_size=max_size,
            **kwargs,
        )

        return self.post(order)

    # noinspection PyProtocol
    def limit_order(
        self,
        price: NumericAlias,
        size: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias | None,
        tif: TIME_IN_FORCE,
        expiration: int | None,
        neg_risk: bool | None = None,
        **kwargs,
    ) -> tuple[FrozenOrder, PostOrderResponse]:
        """Creates a limit order, posts it to the exchange and adds it to the Order Manager to be tracked.

        Notes
        -----
        Example for kwargs: aux_id='some additional info' in case of polypy.order.Order class.
        """
        expiration = 0 if expiration is None else expiration
        if expiration < 0:
            raise OrderCreationException("`expiration` must be >= 0.")

        if tick_size is None:
            tick_size = get_tick_size(self.rest_endpoint, token_id)

        if neg_risk is None:
            # get_neg_risk is lru cached
            neg_risk = get_neg_risk(self.rest_endpoint, token_id)

        order = self._create_limit_order(
            price=price,
            size=size,
            token_id=token_id,
            side=side,
            tick_size=tick_size,
            neg_risk=neg_risk,
            chain_id=self.chain_id,
            private_key=self.private_key,
            maker=self.maker_funder,
            signature_type=self.signature_type,
            tif=tif,
            expiration=expiration,
            **kwargs,
        )

        return self.post(order)

    def post(self, order: AnyOrderProtocol) -> tuple[FrozenOrder, PostOrderResponse]:
        # posting an order might raise an exception, i.e. server side error (e.g. order unmatched)
        with self.lock:
            self._validate()
            try:
                order, response = post_order(
                    endpoint=self.rest_endpoint,
                    order=order,
                    private_key=self.private_key,
                    api_key=self.api_key,
                    secret=self.secret,
                    passphrase=self.passphrase,
                )
            finally:
                with contextlib.suppress(OrderTrackingException):
                    self.track(order, sync=False)

        return frozen_order(order), response

    # noinspection SpellCheckingInspection
    def _uptrack_orders(self, orders: list[AnyOrderProtocol]) -> None:
        untracked_orders = [
            order for order in orders if order.id not in self.order_dict.keys()
        ]

        # first check all orders are trackable before adding any of the orders
        for order in untracked_orders:
            self._is_trackable(order)

        for order in untracked_orders:
            self.track(order, sync=False)

    # noinspection SpellCheckingInspection
    def _uptrack_get_orders(
        self, orders: list[AnyOrderProtocol | str]
    ) -> list[OrderProtocol]:
        try:
            if not isinstance(orders[0], str):  # order objects
                # handle untracked orders
                self._uptrack_orders(orders)

                # get order ids, because in case of frozen order,
                #   we have to retrieve the original order object
                orders = [order.id for order in orders]

            orders = [self.order_dict[order_id] for order_id in orders]
        except KeyError as e:
            raise OrderGetException(
                "Not all order ids tracked by the Order Manager. "
                "Make sure to track orders first. "
                "Orders have not been processed (no single REST call submitted)!"
            ) from e
        except OrderTrackingException as e:
            raise OrderTrackingException(
                "Not all orders trackable by the Order Manager. "
                "Make sure that order objects are trackable or already tracked. "
                "Orders have not been processed (no single REST call submitted)!"
            ) from e

        return orders

    def cancel(
        self,
        orders: AnyOrderProtocol | str | list[AnyOrderProtocol | str],
        mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
    ) -> tuple[FrozenOrder | list[FrozenOrder], CancelOrdersResponse]:
        if isinstance(orders, (list, tuple, set)) and not orders:
            return [], CancelOrdersResponse(None, {})

        _single, orders = _parse_to_list(orders)

        with self.lock:
            self._validate()

            orders = self._uptrack_get_orders(orders)

            orders, response = cancel_orders(
                endpoint=self.rest_endpoint,
                orders=orders,
                private_key=self.private_key,
                api_key=self.api_key,
                secret=self.secret,
                passphrase=self.passphrase,
                mode_not_canceled=mode_not_canceled,
            )

        orders = [frozen_order(order) for order in orders]
        if _single:
            orders = orders[0]

        return orders, response

    def cancel_all(
        self,
        mode_not_canceled: Literal["except", "warn", "ignore"] = "except",
        statuses: INSERT_STATUS | list[INSERT_STATUS] = CANCELABLE_INSERT_STATI,
    ) -> tuple[list[FrozenOrder], CancelOrdersResponse]:
        _, statuses = _parse_to_list(statuses)

        with self.lock:
            self._validate()
            if orders := [
                order for order in self.order_dict.values() if order.status in statuses
            ]:
                return self.cancel(orders, mode_not_canceled)
            else:
                return [], CancelOrdersResponse([], {})

    def _assert_mode_not_existent(
        self,
        mode_not_existent: Literal["warn", "remove", "except"],
        missed_orders: list[OrderProtocol],
    ) -> None:
        if not missed_orders:
            return

        if mode_not_existent == "except":
            raise OrderUpdateException(
                f"Following orders could not be updated: {missed_orders}"
            )
        elif mode_not_existent == "warn":
            warnings.warn(f"Following orders could not be updated: {missed_orders}")
        elif mode_not_existent == "remove":
            warnings.warn(f"Following orders could not be updated: {missed_orders}")
            for x in missed_orders:
                self.untrack(x.id, False)
        else:
            raise PolyPyException(f"Unknown `mode_not_existent`={mode_not_existent}")

    def _parse_syncable_orders(
        self, order: AnyOrderProtocol | str | list[AnyOrderProtocol | str] | None
    ) -> tuple[bool, list[AnyOrderProtocol]]:
        if order is None:
            _single, orders = False, list(self.order_dict.values())
        else:
            _single, orders = _parse_to_list(order)
            orders = self._uptrack_get_orders(orders)

        # only sync DEFINED, LIVE, DELAYED
        orders = [
            x
            for x in orders
            if x.status
            in {INSERT_STATUS.DEFINED, INSERT_STATUS.LIVE, INSERT_STATUS.DELAYED}
        ]

        return _single, orders

    def sync(
        self,
        order: AnyOrderProtocol | str | list[AnyOrderProtocol | str] | None,
        mode_not_existent: Literal["except", "remove", "warn"],
    ) -> tuple[FrozenOrder | list[FrozenOrder], list[OrderProtocol]]:
        with self.lock:
            self._validate()
            _single, orders = self._parse_syncable_orders(order)

            orders, responses = get_order_updates(
                endpoint=self.rest_endpoint,
                orders=orders,
                private_key=self.private_key,
                api_key=self.api_key,
                secret=self.secret,
                passphrase=self.passphrase,
                mode_not_existent="ignore",
            )

        missed_orders = [x for x, r in zip(orders, responses) if r is None]
        self._assert_mode_not_existent(mode_not_existent, missed_orders)

        orders = [frozen_order(x) for x in orders]

        if _single:
            orders, responses = orders[0], responses[0]

        return orders, missed_orders

    def clean(
        self,
        statuses: INSERT_STATUS | list[INSERT_STATUS] = TERMINAL_INSERT_STATI,
        expiration: int = -1,
    ) -> list[OrderProtocol]:
        # filter out status and expiration

        if isinstance(statuses, (list, set, tuple)) and not statuses:
            return []

        _, statuses = _parse_to_list(statuses)

        rem_orders = []

        with self.lock:
            self._validate()
            orders = list(self.order_dict.values())

        for order in orders:
            has_valid_expiration = order.expiration is not None and order.expiration > 0

            if order.status in statuses or (
                has_valid_expiration and order.expiration <= expiration
            ):
                rem_orders.append(order)

        for order in rem_orders:
            self.untrack(order.id, False)

        return rem_orders
