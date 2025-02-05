import datetime
import math
import threading
from typing import TYPE_CHECKING, Any, KeysView, Literal, Protocol, Union

from polypy.book import OrderBook
from polypy.constants import ENDPOINT, SIG_DIGITS_SIZE
from polypy.exceptions import PositionTrackingException, PositionTransactionException
from polypy.order.common import INSERT_STATUS, SIDE
from polypy.position import (
    ACT_SIDE,
    USDC,
    FrozenPosition,
    Position,
    PositionProtocol,
    frozen_position,
    PositionFactory,
)
from polypy.rest.api import get_midpoints
from polypy.rounding import round_down, round_down_tenuis_up
from polypy.trade import TRADE_STATUS
from polypy.typing import NumericAlias

if TYPE_CHECKING:
    from polypy.manager.order import OrderManagerProtocol


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
            if None: use size_sig_digits from USDC position (which defaults to 5)
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
        size_sig_digits: int = SIG_DIGITS_SIZE,
        **kwargs,
    ) -> FrozenPosition:
        """Create and track new position"""
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
    amount_sig_digits: int,
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
            amount_sig_digits,
        )
        for order_manager in order_managers
    )
    return round_down(cash - buy_amount, amount_sig_digits)


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


# todo move lock to custom dict factory, make all operations atomic
class PositionManager(PositionManagerProtocol):
    def __init__(
        self,
        rest_endpoint: ENDPOINT | str | None,
        usdc_position: PositionProtocol | NumericAlias,
        max_size: int | None = None,
        position_factory: type[PositionProtocol] | PositionFactory = Position,
    ):
        """

        Parameters
        ----------
        rest_endpoint
        usdc_position
        max_size
        position_factory
        """
        self.position_factory = (
            position_factory.create
            if hasattr(position_factory, "create")
            else position_factory
        )

        self.rest_endpoint = rest_endpoint

        self.max_size = math.inf if max_size is None else max_size
        self.position_dict: dict[str, PositionProtocol] = {
            USDC: usdc_position
            if _is_position(usdc_position)
            else self.position_factory(
                asset_id=USDC, size=usdc_position, size_sig_digits=5
            )
        }

        self.lock = threading.RLock()

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self.position_dict

    # noinspection PyTypeHints
    @property
    def asset_ids(self) -> KeysView[str | Literal[USDC]]:
        return self.position_dict.keys()

    def _get_position(self, asset_id: str) -> PositionProtocol | None:
        try:
            return self.position_dict[asset_id]

        except KeyError as e:
            raise PositionTrackingException(
                f"{datetime.datetime.now()} | Position not found for id: {asset_id}."
            ) from e

    def _get_or_create_position(
        self, asset_id: str, allow_create: bool
    ) -> PositionProtocol | None:
        try:
            return self.position_dict[asset_id]
        except KeyError as e:
            if not allow_create:
                raise PositionTrackingException(
                    f"{datetime.datetime.now()} | Position not found for id: {asset_id}."
                ) from e

            numeric_type = type(self.position_dict[USDC].size)
            numeric_type = numeric_type if numeric_type is not int else float
            self.create_position(asset_id, numeric_type(0))
            return self.position_dict[asset_id]

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
            position = self._get_or_create_position(asset_id, allow_create)

            usdc = self._get_position(USDC)

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
                    delta_size * price, usdc.size_sig_digits, 4
                ),
                trade_id=trade_id,
                act_side=counter_party_side,
                trade_status=trade_status,
            )

    @property
    def balance(self) -> NumericAlias:
        with self.lock:
            return self.position_dict[USDC].size

    @property
    def balance_available(self) -> NumericAlias:
        with self.lock:
            return self.position_dict[USDC].size_available

    @property
    def balance_total(self) -> NumericAlias:
        with self.lock:
            return self.position_dict[USDC].size_total

    def withdraw(self, amount: NumericAlias) -> None:
        with self.lock:
            self.position_dict[USDC].size = round_down(
                self.position_dict[USDC].size - amount,
                self.position_dict[USDC].size_sig_digits,
            )

    def deposit(self, amount: NumericAlias) -> None:
        with self.lock:
            self.position_dict[USDC].size = round_down(
                self.position_dict[USDC].size + amount,
                self.position_dict[USDC].size_sig_digits,
            )

    def clean(self) -> list[PositionProtocol]:
        with self.lock:
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
            try:
                return frozen_position(self.position_dict[asset_id])
            except KeyError:
                return None

    def _fill_midpoints(
        self, missed: set[str], midpoints: dict[str, NumericAlias | None]
    ) -> dict[str, NumericAlias]:
        if missed:
            numeric_type = type(self.position_dict[USDC].size)
            numeric_type = numeric_type if numeric_type is not int else float

            rest_midpoints = get_midpoints(self.rest_endpoint, missed, numeric_type)

            midpoints |= {asset_id: rest_midpoints[asset_id] for asset_id in missed}

        return midpoints

    def _fill_usdc_midpoint(
        self,
        midpoints: dict[str, NumericAlias | None],
    ) -> dict[str, NumericAlias | None]:
        numeric_type = type(self.position_dict[USDC].size)
        numeric_type = numeric_type if numeric_type is not int else float

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

        return self._fill_midpoints(missed, midpoints)

    def total(
        self,
        midpoints: dict[str, NumericAlias | OrderBook | None] | None,
        tick_size: float | None,
    ) -> NumericAlias:
        with self.lock:
            midpoints = self._parse_midpoints(midpoints)

            if tick_size is None:
                tick_size_digits = self.position_dict[USDC].size_sig_digits
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
            return self.position_dict.pop(asset_id, None)

    # noinspection PyProtocol
    def create_position(
        self,
        asset_id: str,
        size: NumericAlias,
        size_sig_digits: int = SIG_DIGITS_SIZE,
        **kwargs,
    ) -> FrozenPosition:
        size = float(size) if isinstance(size, int) else size
        position = self.position_factory(
            asset_id=asset_id, size=size, size_sig_digits=size_sig_digits, **kwargs
        )
        self.track(position)
        return frozen_position(position)

    def buying_power(
        self,
        order_managers: Union["OrderManagerProtocol", list["OrderManagerProtocol"]],
    ) -> NumericAlias:
        if not isinstance(order_managers, list):
            order_managers = [order_managers]

        return buying_power(
            self, order_managers, self.position_dict[USDC].size_sig_digits, "available"
        )
