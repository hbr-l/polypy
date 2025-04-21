import datetime
import logging
import math
import threading
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    KeysView,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    Union,
)

import msgspec.json
from hexbytes import HexBytes
from web3.types import TxReceipt

from polypy.book import OrderBook
from polypy.constants import ENDPOINT, N_DIGITS_SIZE, USDC
from polypy.ctf import MarketIdQuintet, MarketIdTriplet
from polypy.exceptions import (
    ManagerInvalidException,
    PolyPyException,
    PositionTrackingException,
    PositionTransactionException,
)
from polypy.manager.cache import AugmentedConversionCache, ConversionCacheProtocol
from polypy.manager.rpc_ops import (
    MTX,
    tx_batch_operate_positions,
    tx_convert_positions,
    tx_merge_positions,
    tx_redeem_positions,
    tx_split_position,
)
from polypy.manager.rpc_proc import RPCSettings, _act_conversion_cache_diff
from polypy.order.common import INSERT_STATUS, SIDE
from polypy.position import (
    ACT_SIDE,
    FrozenPosition,
    Position,
    PositionFactory,
    PositionProtocol,
    frozen_position,
)
from polypy.rest.api import get_markets_gamma_model, get_midpoints
from polypy.rounding import round_down, round_down_tenuis_up
from polypy.structs import RelayerResponse
from polypy.trade import TRADE_STATUS
from polypy.typing import NumericAlias, infer_numeric_type

if TYPE_CHECKING:
    from polypy.manager.order import OrderManagerProtocol


MarketTriplet: TypeAlias = MarketIdTriplet | tuple[str, str, str]
MarketQuintet: TypeAlias = MarketIdQuintet | tuple[str, str, str, str, str]


class PositionManagerProtocol(Protocol):
    """Manage positions. Interface layer for UserStream. Must implement locking if necessary (e.g. sharedMem).

    Position Manager is for managing currently active positions. It is not meant as database to track historic
    data, e.g. total_bought. For that purpose, implement custom class with appropriate logging or callbacks.
    Because an account can be used to drive multiple strategies, each with its own Position Manager, a sync()
    method is not possible (i.e., if we get data for a specific position of that account, it is not clear,
    how to distribute this position (size) amongst all its related Position Managers).
    Instead, use polypy.rest.api.get_positions() and handle sync of position data manually if necessary.
    """

    def __contains__(self, asset_id: str) -> bool:
        ...

    @property
    def gid(self) -> int:
        """Global ID for global ordering of locks to avoid deadlocking.
        Must be unique per object (e.g., unique random int, object id() of underlying shared mem container, etc. ...)
        """
        ...

    # noinspection PyTypeHints
    @property
    def asset_ids(self) -> list[str | Literal[USDC]] | KeysView[str | Literal[USDC]]:
        """List of asset IDs (including USDC)."""
        ...

    @property
    def balance(self) -> NumericAlias:
        """USDC balance, currently settled cash."""
        ...

    @property
    def balance_total(self) -> NumericAlias:
        """USDC balance including pending transactions."""
        ...

    @property
    def balance_available(self) -> NumericAlias:
        """USDC balance minus pending withdrawals due to trades."""
        ...

    @property
    def valid(self) -> bool:
        """True, if .invalidate() has not been called. False, if Position Manager has been invalidated."""
        ...

    def _get_or_create_position(
        self, asset_id: str, allow_create: bool
    ) -> PositionProtocol:
        """Get position or create a new one if `allow_create=True`, else raise PositionTrackingException.
        For compatibility with other modules only - should not be used by user!

        Notes
        -----
        Use __contains__ to check if positions exists (less overhead then get_by_id(...)).
        """

    def _locked_call(self, fn: Callable, *args, **kwargs) -> Any:
        """Call fn with lock acquired and release afterward.
        For compatibility with other modules only - should not be used by user!
        """

    def invalidate(self, reason: str | None = None) -> None:
        """Invalidate Position Manager s.t. any successive call will raise an exception.
        This method is mainly used for loose coupling of the Position Manager and the User Stream
        """

    def transact(
        self,
        asset_id: str,
        delta_size: NumericAlias,
        price: NumericAlias,
        trade_id: str,
        side: SIDE,
        trade_status: TRADE_STATUS,
        allow_create: bool,
    ) -> None:
        """Transaction to buy or sell asset_id. Must not be USDC."""
        ...

    def withdraw(self, amount: NumericAlias) -> None:
        """Withdraw USDC from Position Manager.
        This does NOT actually withdraw from the account, but only reduces USDC for this very Position Manager.
        """
        ...

    def deposit(self, amount: NumericAlias) -> None:
        """Deposit USDC from Position Manager.
        This does NOT actually deposit to the account, but only increases USDC for this very Position Manager.
        """
        ...

    def clean(self) -> list[PositionProtocol]:
        """Remove positions, that are empty/zero."""
        ...

    def get(self, **kwargs) -> list[FrozenPosition]:
        """Get by search criteria."""
        ...

    # noinspection PyTypeHints
    def get_by_id(self, asset_id: str | Literal[USDC]) -> FrozenPosition | None:
        """Get position by asset ID."""
        ...

    def total(
        self,
        midpoints: dict[str, NumericAlias | OrderBook | None] | None,
        tick_size: float | None,
    ) -> NumericAlias:
        # noinspection GrazieInspection
        """Get total amount (value) of Position Manager.

        Parameters
        ----------
        midpoints: dict[str, NumericAlias | OrderBook | None] | None
            key: asset_id, value: NumericAlias (supplied midpoint), order book or None (REST call)
        tick_size: float | None
            if float: same tick_size for all assets
            if None: use n_digits_size from USDC position (which defaults to 5)
        """
        ...

    def track(self, position: PositionProtocol) -> None:
        """Add position to Position Manager."""
        ...

    def untrack(self, asset_id: str) -> PositionProtocol | None:
        """Remove position from Position Manager"""
        ...

    def create_position(
        self,
        asset_id: str,
        size: NumericAlias,
        n_digits_size: int = N_DIGITS_SIZE,
        **kwargs,
    ) -> FrozenPosition:
        """Create and track new position"""
        ...

    def split_positions(
        self,
        market_triplet: MarketTriplet,
        amount: NumericAlias,
        rpc_settings: RPCSettings,
        neg_risk: bool | None = None,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        """Split USDC into YES and NO positions (assets/tokens) within a market.

        Notes
        -----
        If fallback to RPC: does not check whether POL balance is sufficient for covering gas!
        Use `get_balance_POL(...)` and check manually if necessary.
        """
        ...

    def merge_positions(
        self,
        market_triplet: MarketTriplet,
        size: NumericAlias | None,
        rpc_settings: RPCSettings,
        neg_risk: bool | None = None,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        """Merge YES and NO positions (assets/tokens) of a market into USDC.

        Notes
        -----
        If fallback to RPC: does not check whether POL balance is sufficient for covering gas!
        Use `get_balance_POL(...)` and check manually if necessary."""
        ...

    def redeem_positions(
        self,
        position_managers: list["PositionManagerProtocol"],
        market_triplet: MarketTriplet,
        rpc_settings: RPCSettings,
        outcome: Literal["YES", "NO"] | None = None,
        neg_risk: bool | None = None,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        """Redeem resolved positions (assets/tokens) of a market.

        1) Because redeem does not specify a size but redeems the entire position all at once
        (at least for non-negative markets) per wallet, we have to perform redeem on all Position Managers
        associated with that wallet that might have a position of the corresponding market_triplet
        - else sizes and balances will be erroneous!

        2) Because we have to lock every Position Manager, be aware that this has the potential for deadlocking!

        3) Redeems for YES and NO simultaneously (though one of them will be 0 USDC).

        4) If outcome is None, outcome will be queried via REST call. If outcome provided, be sure that token
        is actually redeemable, else position calculation will be off since this is not checked in this case.

        Notes
        -----
        If fallback to RPC: does not check whether POL balance is sufficient for covering gas!
        Use `get_balance_POL(...)` and check manually if necessary."""
        ...

    def convert_positions(
        self,
        cvt_market_quintets: MarketQuintet | list[MarketQuintet],
        all_market_quintets: list[MarketQuintet] | None,
        size: NumericAlias | None,
        bookkeep_deposit: bool,
        rpc_settings: RPCSettings,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        """Convert NO position/s to complementary YES positions and potentially USDC.

        Users should manually call `pull_augmented_conversions` periodically.

        Parameters
        ----------
        cvt_market_quintets
        all_market_quintets: list[MarketQuintet] | None,
            - if None: all markets will be fetched via Gamma API (recommended),
            - if not None: specify all markets including closed (!) markets.
              Consider using `get_neg_risk_markets` for this. If `pull_augmented_conversions` gets called,
              this might fix-up any missed markets, but users should not rely on this mechanism though.
        size
        bookkeep_deposit: bool
            bookkeeping when converting positions can be done in two ways:
            1) `bookkeep_deposit=True`: buy YESs with `price=0`, sell NOs with `price=0` and deposit `(NOs - 1) USDC`
            2) `bookkeep_deposit=False`:  buy YESs with `price=1 / (NOs + YESs)`, sell NOs with `price=1-price_yes`
                and no separate deposit.
            Approach 1) will be faster due to fewer rounding operations. Though, depending on the concrete
            `PositionProtocol` implementation, buying and selling at `price=0` may be unfavorable (e.g.,
            if Position class keeps track of an average entry price, etc. - though if using
            `polypy.Position`, this is not the case...), and modelling with a separate deposit operation might be
            unfavorable as well depending on the user's bookkeeping approach.
            Note, that in special cases, approach 2) might still use an additional `deposit` operation to compensate
            numeric instability during rounding operations to account for marginal amounts of USDC.
            Also see Notes regarding requirements for `PositionProtocol` (use `round_down_tenuis_up`
            whenever multiplication is in play when using custom `PositionProtocol` implementation!)
        rpc_settings

        Notes
        -----
        In order to correctly compute PositionManager.balance when using `bookkeep_deposit=False`,
        `PositionProtocol`'s specific implementation must (!) use `round_down_tenuis_up` instead of
        simple `round_down` when computing transaction volumes (i.e., size * price, or any multiplication in general)
        for USDC position!

        If fallback to RPC: does not check whether POL balance is sufficient for covering gas!
        Use `get_balance_POL(...)` and check manually if necessary."""
        ...

    def batch_operate_positions(
        self,
        batch_mtx: list[MTX],
        rpc_settings: RPCSettings,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        """Perform multiple transactions (split, merge, redeem, convert) as one batch.

        Redeeming is not batchable.
        `batch_operate_positions(...)` does NOT (!) check upfront whether the chain of operations is actually feasible
        in terms of bookkeeping etc. (e.g., merge_positions might reduce position size of token_id A s.t. following
        convert_positions might fail due to insufficient position size in token_id A).
        User has to be sure, that chaining operations is feasible and executable!
        Otherwise, might fail, and depending on the point of failure, PositionManager might be in a corrupted and
        irrecoverable state (internal cache, etc.).

        Notes
        -----
        To reduce REST request, consider specifying all arguments including optional arguments in MTX.
        To retrieve negative risk market `all_market_quintets`, use `get_neg_risk_markets`.

        If fallback to RPC: does not check whether POL balance is sufficient for covering gas!
        Use `get_balance_POL(...)` and check manually if necessary."""
        ...

    def pull_augmented_conversions(self) -> bool:
        """If a negative risk market adds a new outcome, in which we previously performed a NO-conversion before,
        then we have to update new complementary YES positions accordingly (increase).

        If conversion has been performed, then this function has to be called periodically
        and manually to ensure correct positions - no auto-calling.
        """
        ...

    def buying_power(
        self,
        order_managers: Union["OrderManagerProtocol", list["OrderManagerProtocol"]],
    ) -> NumericAlias:
        """Compute current buying power, which is available cash minus all open buy orders"""
        ...


def buying_power(
    position_manager: PositionManagerProtocol,
    order_managers: list["OrderManagerProtocol"],
    n_digits_amount: int,
    mode: Literal["available", "total", "settled"],
) -> NumericAlias:
    """Might suffer from float imprecision rounding error if not Decimal type used."""

    if mode == "available":
        cash = position_manager.balance_available
    elif mode == "total":
        cash = position_manager.balance_total
    else:
        cash = position_manager.balance

    # todo rounding correct?
    buy_amount = sum(
        round_down(
            sum(
                order.amount
                for order in order_manager.get(side=SIDE.BUY, status=INSERT_STATUS.LIVE)
            ),
            n_digits_amount,
        )
        for order_manager in order_managers
    )
    return round_down(cash - buy_amount, n_digits_amount)


def _is_trackable(position: PositionProtocol) -> None:
    if isinstance(position, FrozenPosition):
        raise PositionTrackingException(
            "Cannot track frozen position (which, e.g. is the case when already tracked by a Position Manager)."
        )

    if position.asset_id is None or position.asset_id == "":
        raise PositionTrackingException(
            "Can only track positions with valid asset_id assigned."
        )


def _is_position(x: Any) -> bool:
    return hasattr(x, "asset_id") and hasattr(x, "size") and hasattr(x, "act")


_ERR_MSG_INVALID = "self.invalidate() was called. Position Manager is invalidated."


# todo move lock to custom dict factory, make all operations atomic
class PositionManager(PositionManagerProtocol):
    # noinspection PyTypeHints
    def __init__(
        self,
        rest_endpoint: ENDPOINT | str | None,
        gamma_endpoint: str | ENDPOINT | None,
        usdc_position: PositionProtocol | NumericAlias,
        max_size: int | None = None,
        position_factory: type[PositionProtocol] | PositionFactory = Position,
        conversion_cache: ConversionCacheProtocol | None = None,
    ) -> None:
        """

        Parameters
        ----------
        rest_endpoint
        gamma_endpoint
        usdc_position
        max_size
        position_factory
        conversion_cache
        """
        self.position_factory = (
            position_factory.create
            if hasattr(position_factory, "create")
            else position_factory
        )

        self.rest_endpoint = rest_endpoint
        self.gamma_endpoint = gamma_endpoint

        self.max_size = math.inf if max_size is None else max_size
        self.position_dict: dict[str, PositionProtocol] = {
            USDC: usdc_position
            if _is_position(usdc_position)
            else self.position_factory(
                asset_id=USDC, size=usdc_position, n_digits_size=5
            )
        }

        self.lock = threading.RLock()

        self._invalid_token: bool = False
        self._invalid_reason: str = _ERR_MSG_INVALID

        self.conversion_cache = conversion_cache

    @classmethod
    def create(
        cls,
        usdc_position: PositionProtocol | NumericAlias,
        max_size: int | None = None,
    ) -> Self:
        return cls(
            ENDPOINT.REST,
            ENDPOINT.GAMMA,
            usdc_position,
            max_size,
            Position,
            AugmentedConversionCache(
                ENDPOINT.GAMMA, 2 * max_size if max_size is not None else None
            ),
        )

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self.position_dict

    def __str__(self):
        positions = {
            p.asset_id: (p.size, p.size_available, p.size_total)
            for p in self.position_dict.values()
        }
        return (
            f"Positions: {positions} "
            f"\nREST: {self.rest_endpoint} "
            f"\nGAMMA: {self.gamma_endpoint} "
            f"\nmax_size: {self.max_size}"
            f"\nConversion Cache: {self.conversion_cache}"
            f"\nInvalidation state: {(self._invalid_token, self._invalid_reason if self._invalid_token else None)}"
        )

    @property
    def gid(self) -> int:
        return id(self.position_dict)

    # noinspection PyTypeHints
    @property
    def asset_ids(self) -> KeysView[str | Literal[USDC]]:
        return self.position_dict.keys()

    @property
    def valid(self) -> bool:
        return not self._invalid_token

    def invalidate(self, reason: str | None = None) -> None:
        self._invalid_token = True

        self._invalid_reason = f"{self._invalid_reason} Reason: '{reason}'."

    def _validate(self) -> None:
        if self._invalid_token:
            raise ManagerInvalidException(self._invalid_reason)

    def _get_or_create_position(
        self, asset_id: str, allow_create: bool
    ) -> PositionProtocol:
        try:
            return self.position_dict[asset_id]
        except KeyError as e:
            if not allow_create:
                raise PositionTrackingException(
                    f"{datetime.datetime.now()} | Position not found for id: {asset_id}."
                ) from e

            numeric_type = infer_numeric_type(self.position_dict[USDC].size)
            self.create_position(asset_id, numeric_type(0))
            return self.position_dict[asset_id]

    def _locked_call(self, fn: Callable, *args, **kwargs) -> Any:
        with self.lock:
            self._validate()
            return fn(*args, **kwargs)

    def _act_position(
        self, asset_id: str, size: NumericAlias, allow_create: bool
    ) -> None:
        position = self._get_or_create_position(asset_id, allow_create)
        position.size = round_down(position.size + size, position.n_digits_size)

    def transact(
        self,
        asset_id: str,
        delta_size: NumericAlias,
        price: NumericAlias,
        trade_id: str,
        side: SIDE,
        trade_status: TRADE_STATUS,
        allow_create: bool,
    ) -> None:
        if asset_id == USDC:
            raise PositionTransactionException(
                f"Cannot transact on asset_id={asset_id}. "
                f"Use self.deposit() and self.withdraw() to adjust USDC balance."
            )

        with self.lock:
            self._validate()
            position = self._get_or_create_position(asset_id, allow_create)

            usdc = self._get_or_create_position(USDC, False)

            if side is SIDE.BUY:
                position_side = ACT_SIDE.TAKER
            elif side is SIDE.SELL:
                position_side = ACT_SIDE.MAKER
            else:
                raise PositionTransactionException(f"Unknown side: {side}.")

            position.act(
                delta_size=delta_size,
                trade_id=trade_id,
                act_side=position_side,
                trade_status=trade_status,
            )

            counter_party_side = (
                ACT_SIDE.TAKER if position_side is ACT_SIDE.MAKER else ACT_SIDE.MAKER
            )
            usdc.act(
                delta_size=round_down_tenuis_up(
                    delta_size * price, usdc.n_digits_size, 4
                ),
                trade_id=trade_id,
                act_side=counter_party_side,
                trade_status=trade_status,
            )

    @property
    def balance(self) -> NumericAlias:
        with self.lock:
            self._validate()
            return self.position_dict[USDC].size

    @property
    def balance_available(self) -> NumericAlias:
        with self.lock:
            self._validate()
            return self.position_dict[USDC].size_available

    @property
    def balance_total(self) -> NumericAlias:
        with self.lock:
            self._validate()
            return self.position_dict[USDC].size_total

    def withdraw(self, amount: NumericAlias) -> None:
        with self.lock:
            self._validate()
            self._act_position(USDC, -amount, False)

    def deposit(self, amount: NumericAlias) -> None:
        with self.lock:
            self._validate()
            self._act_position(USDC, amount, False)

    def clean(self) -> list[PositionProtocol]:
        with self.lock:
            self._validate()
            rem_pos = [
                position
                for asset_id, position in self.position_dict.items()
                if asset_id != USDC and position.empty
            ]

            for rp in rem_pos:
                self.position_dict.pop(rp.asset_id)

            return rem_pos

    # noinspection PyProtocol
    def get(self, **kwargs) -> list[FrozenPosition]:
        with self.lock:
            self._validate()
            try:
                return [frozen_position(self.position_dict[kwargs["asset_id"]])]
            except KeyError:
                return [
                    frozen_position(position)
                    for position in self.position_dict.values()
                    if all(
                        hasattr(position, k) and getattr(position, k) == v
                        for k, v in kwargs.items()
                    )
                ]

    # noinspection PyTypeHints
    def get_by_id(self, asset_id: str | Literal[USDC]) -> FrozenPosition | None:
        with self.lock:
            self._validate()
            try:
                return frozen_position(self.position_dict[asset_id])
            except KeyError:
                return None

    def _fill_midpoints(
        self, missed: set[str], midpoints: dict[str, NumericAlias | None]
    ) -> dict[str, NumericAlias]:
        if missed:
            numeric_type = infer_numeric_type(self.position_dict[USDC].size)

            rest_midpoints = get_midpoints(self.rest_endpoint, missed, numeric_type)

            midpoints |= {asset_id: rest_midpoints[asset_id] for asset_id in missed}

        return midpoints

    def _fill_no_orderbook_midpoints(
        self, missed: set[str], midpoints: dict[str, NumericAlias | None]
    ) -> dict[str, NumericAlias]:
        if missed:
            numeric_type = infer_numeric_type(self.position_dict[USDC].size)

            markets: list[dict] = get_markets_gamma_model(
                self.gamma_endpoint, token_ids=missed
            )

            for market in markets:
                # todo tight coupling: refactor to Struct instead of dict
                tokens = msgspec.json.decode(
                    market["clobTokenIds"], type=tuple[str, str]
                )
                prices = msgspec.json.decode(
                    market["outcomePrices"], type=tuple[str, str]
                )
                if tokens[0] in missed:
                    midpoints[tokens[0]] = numeric_type(prices[0])
                elif tokens[1] in missed:
                    midpoints[tokens[1]] = numeric_type(prices[1])
                else:
                    raise PolyPyException(f"{tokens} not in missed midpoints.")

        return midpoints

    def _fill_usdc_midpoint(
        self,
        midpoints: dict[str, NumericAlias | None],
    ) -> dict[str, NumericAlias | None]:
        numeric_type = infer_numeric_type(self.position_dict[USDC].size)

        midpoints[USDC] = numeric_type(1)

        return midpoints

    def _parse_midpoints(
        self, midpoints: dict[str, NumericAlias | OrderBook | None] | None
    ) -> dict[str, NumericAlias]:
        if midpoints is None:
            midpoints = {}

        # parse order book vs NumericAlias
        midpoints = {
            asset_id: val.midpoint_price if hasattr(val, "midpoint_price") else val
            for asset_id, val in midpoints.items()
        }

        # default fill USDC
        midpoints = self._fill_usdc_midpoint(midpoints)

        # get asset ids for which we have to manually fetch midpoint
        missed = (self.position_dict.keys() - midpoints.keys()) | {
            key for key, val in midpoints.items() if val is None
        }
        midpoints = self._fill_midpoints(missed, midpoints)

        # get asset ids of resolved markets/ no order book
        missed = (self.position_dict.keys() - midpoints.keys()) | {
            key for key, val in midpoints.items() if val is None
        }
        return self._fill_no_orderbook_midpoints(missed, midpoints)

    def total(
        self,
        midpoints: dict[str, NumericAlias | OrderBook | None] | None,
        tick_size: float | None,
    ) -> NumericAlias:
        with self.lock:
            self._validate()
            midpoints = self._parse_midpoints(midpoints)

            if tick_size is None:
                tick_size_digits = self.position_dict[USDC].n_digits_size
            else:
                tick_size_digits = -int(math.log10(tick_size))

            # todo round intermediate results as well?
            return round_down_tenuis_up(
                sum(
                    position.size_total * midpoints[asset_id]
                    for asset_id, position in self.position_dict.items()
                ),
                tick_size_digits,
                4,
            )

    def track(self, position: PositionProtocol) -> None:
        _is_trackable(position)

        with self.lock:
            self._validate()
            if len(self.position_dict) >= self.max_size:
                raise PositionTrackingException(
                    f"Exceeding max_size={self.max_size}.  Either set a higher max_size, or use"
                    f"PositionManager.clean() or PositionManager.untrack() to free up space."
                )

            self.position_dict[position.asset_id] = position

    def untrack(self, asset_id: str) -> PositionProtocol | None:
        if asset_id == USDC:
            raise PositionTrackingException(
                "Cannot remove USDC position. Use track() to override if necessary."
            )

        with self.lock:
            self._validate()
            return self.position_dict.pop(asset_id, None)

    # noinspection PyProtocol
    def create_position(
        self,
        asset_id: str,
        size: NumericAlias,
        n_digits_size: int = N_DIGITS_SIZE,
        **kwargs,
    ) -> FrozenPosition:
        size = float(size) if isinstance(size, int) else size
        position = self.position_factory(
            asset_id=asset_id, size=size, n_digits_size=n_digits_size, **kwargs
        )
        self.track(position)
        return frozen_position(position)

    def split_positions(
        self,
        market_triplet: MarketTriplet,
        amount: NumericAlias,
        rpc_settings: RPCSettings,
        neg_risk: bool | None = None,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        with self.lock:
            self._validate()

            return tx_split_position(
                rpc_settings=rpc_settings,
                position_manager=self,
                market_triplet=market_triplet,
                amount=amount,
                neg_risk=neg_risk,
                endpoint_rest=self.rest_endpoint,
            )

    def merge_positions(
        self,
        market_triplet: MarketTriplet,
        size: NumericAlias | None,
        rpc_settings: RPCSettings,
        neg_risk: bool | None = None,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        with self.lock:
            self._validate()

            return tx_merge_positions(
                rpc_settings=rpc_settings,
                position_manager=self,
                market_triplet=market_triplet,
                size=size,
                neg_risk=neg_risk,
                endpoint_rest=self.rest_endpoint,
            )

    def redeem_positions(
        self,
        position_managers: list["PositionManagerProtocol"],
        market_triplet: MarketTriplet,
        rpc_settings: RPCSettings,
        outcome: Literal["YES", "NO"] | None = None,
        neg_risk: bool | None = None,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        logging.info(
            "Please note the following points:\n "
            "1) `position_manager`: specify all (!) PositionManagers of same wallet address (!) which "
            "hold positions in `market_triplet`, else a PositionTransactionException will be raised "
            "since positions will be miscalculated else-wise (PositionManagers must not be of same type "
            "but must follow PositionManagerProtocol)\n "
            "2) During redeeming, all Position Manager will be locked, which has potential for deadlocking:\n"
            "   - If multiple PositionManagers redeem at the same time (unlikely due to lock ordering)\n"
            "   - If multiple PositionManagers share the same conversion cache (very likely to deadlock then)"
        )

        if outcome is not None:
            logging.info(
                "If outcome is not None, be sure that token is actually redeemable "
                "since in this case there are no further checks."
            )

        # sort to avoid deadlock when acquiring lock
        pms = list(set([self] + position_managers))
        pms = [pm for _, pm in sorted(zip([pm.gid for pm in pms], pms))]

        fn = partial(
            tx_redeem_positions,
            rpc_settings=rpc_settings,
            position_managers=pms,
            market_triplet=market_triplet,
            outcome=outcome,
            neg_risk=neg_risk,
            endpoint_rest=self.rest_endpoint,
        )

        for pm in reversed(pms):
            fn = partial(pm._locked_call, fn)

        return fn()

    def convert_positions(
        self,
        cvt_market_quintets: MarketQuintet | list[MarketQuintet],
        all_market_quintets: list[MarketQuintet] | None,
        size: NumericAlias | None,
        bookkeep_deposit: bool,
        rpc_settings: RPCSettings,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        # all_market_quintets not necessary for rpc but for proper bookkeeping in position manager
        if self.conversion_cache is None:
            raise PositionTrackingException(
                "No cache configured. Cannot convert negative risk market."
            )

        with self.lock:
            self._validate()

            return tx_convert_positions(
                rpc_settings=rpc_settings,
                position_manager=self,
                conversion_cache=self.conversion_cache,
                cvt_market_quintets=cvt_market_quintets,
                all_market_quintets=all_market_quintets,
                size=size,
                bookkeep_deposit=bookkeep_deposit,
                endpoint_gamma=self.gamma_endpoint,
            )

    def batch_operate_positions(
        self,
        batch_mtx: list[MTX],
        rpc_settings: RPCSettings,
    ) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
        if self.conversion_cache is None:
            raise PositionTrackingException(
                "No cache configured. Cannot convert negative risk market."
            )

        with self.lock:
            self._validate()

            return tx_batch_operate_positions(
                rpc_settings=rpc_settings,
                position_manager=self,
                conversion_cache=self.conversion_cache,
                batch_mtx=batch_mtx,
                endpoint_rest=self.rest_endpoint,
                endpoint_gamma=self.gamma_endpoint,
            )

    def pull_augmented_conversions(self) -> bool:
        if self.conversion_cache is None:
            raise PositionTrackingException(
                "No cache configured. Cannot convert negative risk market."
            )

        ret = False

        with self.lock:
            _, diffs = self.conversion_cache.pull()

            for re_size, yes_token_ids in diffs:
                has_acted = _act_conversion_cache_diff(
                    position_manager=self, re_size=re_size, token_1_ids=yes_token_ids
                )
                ret = ret or has_acted

        return ret

    def buying_power(
        self,
        order_managers: Union["OrderManagerProtocol", list["OrderManagerProtocol"]],
    ) -> NumericAlias:
        if not isinstance(order_managers, list):
            order_managers = [order_managers]

        with self.lock:
            self._validate()

        return buying_power(
            self, order_managers, self.position_dict[USDC].n_digits_size, "available"
        )
