import warnings
from enum import StrEnum
from typing import Any, Literal, NoReturn, Protocol, Self

import attrs

from polypy.constants import SIG_DIGITS_SIZE
from polypy.exceptions import (
    PositionException,
    PositionNegativeException,
    PositionTransactionException,
)
from polypy.rounding import round_down
from polypy.trade import TRADE_STATUS
from polypy.typing import NumericAlias

USDC = "usdc"


# noinspection PyPep8Naming
class ACT_SIDE(StrEnum):
    MAKER = "MAKER"  # outflow/ spend
    TAKER = "TAKER"  # inflow/ receive


class PositionProtocol(Protocol):
    # noinspection PyTypeHints
    asset_id: str | Literal[USDC]
    size: NumericAlias
    size_sig_digits: int
    """int: should default to polypy.constant.SIG_DIGITS_SIZE."""

    @classmethod
    def create(
        cls,
        asset_id: str,
        size: NumericAlias,
        size_sig_digits: int = SIG_DIGITS_SIZE,
        **kwargs,
    ) -> Self:
        ...

    @property
    def size_available(self) -> NumericAlias:
        ...

    @property
    def size_total(self) -> NumericAlias:
        ...

    @property
    def empty(self) -> bool:
        ...

    def act(
        self,
        delta_size: NumericAlias,
        trade_id: str,
        act_side: ACT_SIDE,
        trade_status: TRADE_STATUS,
    ) -> None:
        ...


class FrozenPosition:
    def __init__(self, position: PositionProtocol) -> None:
        object.__setattr__(self, "_wrapped_position", position)

    def __getattr__(self, name) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped_position"), name)

    def __setattr__(self, name, value) -> NoReturn:
        raise PositionException(f"{self.__class__.__name__} is read-only.")

    def __delattr__(self, name) -> NoReturn:
        raise PositionException(f"{self.__class__.__name__} is read-only.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"_wrapped_position={object.__getattribute__(self, '_wrapped_position')}"
            f")"
        )


def frozen_position(position: PositionProtocol) -> FrozenPosition:
    return FrozenPosition(position)


def _validate_allow_neg(inst: "Position", _, val):
    if inst.allow_neg is False and val < 0:
        raise PositionNegativeException(f"No negative position allowed for {inst}.")

    return val


def _frozen(_, attr, ___) -> NoReturn:
    raise PositionException(f"Frozen attribute: {attr.name}.")


# todo sharedMem version
@attrs.define
class Position:
    """Simple Position implementation.

    Transactions (settlement) will be immediately performed on TRADE_STATUS.MATCHED.
    Transactions will be reverted on TRADE_STATUS.FAILED.
    All other TRADE_STATUS will be ignored.

    This might lead to incorrect results, when:
    1) Any duplicate transaction performed will lead to incorrect results (double transaction).
    2) If TRADE_STATUS.MATCHED missed, no transaction will be calculated at all.
    3) If TRADE_STATUS.FAILED hit, but missed TRADE_STATUS.MATCHED, transaction will still be reverted
       (no matter if transaction related to TRADE_STATUS.MATCHED was conducted or not),
       which will lead to incorrect results

    If a more robust and somehow idempotent implementation is required, use `polypy.position.CSMPosition`.
    """

    # noinspection PyTypeHints
    asset_id: str | Literal[USDC] = attrs.field(on_setattr=_frozen, converter=str)
    """Asset ID or USDC."""

    size: NumericAlias = attrs.field(on_setattr=_validate_allow_neg)
    """Confirmed size."""

    allow_neg: bool = attrs.field(on_setattr=_frozen)
    """Allow negative size."""

    size_sig_digits: int = attrs.field()
    """Number of decimal places for size."""

    # noinspection PyTypeHints
    @classmethod
    def create(
        cls,
        asset_id: str,
        size: NumericAlias,
        size_sig_digits: int = SIG_DIGITS_SIZE,
        allow_neg: bool = False,
        **_,
    ) -> Self:
        numeric_type = type(size) if type(size) is not int else float
        return cls(
            asset_id=asset_id,
            size=numeric_type(size),
            allow_neg=allow_neg,
            size_sig_digits=size_sig_digits,
        )

    @property
    def size_available(self) -> NumericAlias:
        """Available and settled size to perform trades on (buying power)."""
        return self.size

    @property
    def size_total(self) -> NumericAlias:
        """Total position size including pending size."""
        return self.size

    @property
    def empty(self) -> bool:
        try:
            return self.size == 0
        except TypeError:
            return self.size == type(self.size)(0)

    def _act_taker(
        self,
        delta_size: NumericAlias,
        trade_id: str,
        trade_status: TRADE_STATUS,
    ) -> None:
        # we only consider .MATCHED and .FAILED
        if trade_status is TRADE_STATUS.MATCHED:
            self.size = round_down(self.size + delta_size, self.size_sig_digits)
        elif trade_status is TRADE_STATUS.FAILED:
            # todo assumption: FAILED is always preceded by MATCHED
            # revert trade
            self.size = round_down(self.size - delta_size, self.size_sig_digits)
        # all other states will be ignored

    def _act_maker(
        self,
        delta_size: NumericAlias,
        trade_id: str,
        trade_status: TRADE_STATUS,
    ) -> None:
        # we only consider .MATCHED and .FAILED
        if trade_status is TRADE_STATUS.MATCHED:
            self.size = round_down(self.size - delta_size, self.size_sig_digits)
        elif trade_status is TRADE_STATUS.FAILED:
            # todo assumption: FAILED is always preceded by MATCHED
            self.size = round_down(self.size + delta_size, self.size_sig_digits)

    def act(
        self,
        delta_size: NumericAlias,
        trade_id: str,
        act_side: ACT_SIDE,
        trade_status: TRADE_STATUS,
    ) -> None:
        if trade_status is TRADE_STATUS.FAILED:
            warnings.warn(
                f"Trade status FAILED for asset_id={self.asset_id}, delta_size={delta_size}, "
                f"trade_id={trade_id}, trade_side={act_side}, trade_status={trade_status}."
            )

        delta_size = round_down(delta_size, self.size_sig_digits)

        if act_side is ACT_SIDE.TAKER:
            self._act_taker(
                delta_size=delta_size,
                trade_id=trade_id,
                trade_status=trade_status,
            )
        elif act_side is ACT_SIDE.MAKER:
            self._act_maker(
                delta_size=delta_size,
                trade_id=trade_id,
                trade_status=trade_status,
            )
        else:
            raise PositionException(f"Unknown trader_side: {act_side}")


def _validate_pending_maker(_, __, val: NumericAlias) -> NumericAlias:
    if val > 0:
        raise PositionException(f"Value not le0: {val}.")

    # todo comment in?
    # if not inst.allow_neg and -val > inst.size + inst.pending_taker:
    #     raise PositionException(f"Overspending for :{inst}.")

    return val


def _validate_pending_taker(_, __, val: NumericAlias) -> NumericAlias:
    if val < 0:
        raise PositionException(f"Value not ge0: {val}.")
    return val


def _validate_settlement_status(_, __, val: TRADE_STATUS) -> TRADE_STATUS:
    if val in [TRADE_STATUS.MATCHED, TRADE_STATUS.FAILED, TRADE_STATUS.RETRYING]:
        raise PositionException(
            "TRADE_STATUS.MATCHED, .FAILED and .RETRYING not allowed as `settlement_status`."
        )
    return val


@attrs.define
class CSMPosition(Position):
    """Clearing-Settlement-Mechanism Position,
    which tracks the entire lifecycle of the trade during transaction.

    This implementation mitigate false double transactions if a TRADE_STATE is repeated by accident and
    is robust to missing at least preceding states up to settlement.

    Nonetheless, potential pitfalls are:
    1) If settlement state is set to TRADE_STATUS.MINED, but missed all (MATCHED AND! MINED) states except
        TRADE_STATUS.CONFIRMED (received message), then transaction will be missed.
    2) If settlement state is set to TRADE_STATUS.MINED and trade has reached settlement, TRADE_STATUS.FAILED
        will have no effect, which results in not reverting the trade.
    """

    pending_maker: NumericAlias = attrs.field(
        validator=_validate_pending_maker, on_setattr=_validate_pending_maker
    )
    """Outflow, that is pending but not yet activated/confirmed."""

    pending_taker: NumericAlias = attrs.field(
        validator=_validate_pending_taker, on_setattr=_validate_pending_taker
    )
    """Inflow, that is pending but not yet activated/confirmed."""

    pending_trade_ids: dict[str, NumericAlias] = attrs.field(
        factory=dict, on_setattr=_frozen
    )
    """Dict of trade IDs and sizes corresponding to pending transactions. 
    Avoids double spending/receiving on `pending_maker` and `pending_taker`."""

    max_size_trade_ids: int = attrs.field(
        default=1e6, on_setattr=_frozen, converter=int
    )
    """Max size `pending_trade_ids` can take to avoid memory leaks."""

    settlement_status: TRADE_STATUS = attrs.field(
        default=TRADE_STATUS.CONFIRMED,
        converter=TRADE_STATUS,
        validator=_validate_settlement_status,
    )
    """Trade status at which a trade can be assumed to be settled. 
    Must not be MATCHED, RETRYING or FAILED.
    If settlement_status=MINED, but not state hit other than CONFIRMED (e.g. missed all previous
    states), this will result in a missing transaction (though this should not be the case, else
    there is likely a server-side error)."""

    # noinspection PyTypeHints
    @classmethod
    def create(
        cls,
        asset_id: str,
        size: NumericAlias,
        size_sig_digits: int = SIG_DIGITS_SIZE,
        pending_maker: NumericAlias = 0,
        pending_taker: NumericAlias = 0,
        settlement_status: TRADE_STATUS = TRADE_STATUS.CONFIRMED,
        allow_neg: bool = False,
        pending_trade_ids: dict[str, NumericAlias] | None = None,
        max_size_trade_ids: int = 1_000_000,
        **_,
    ) -> Self:
        numeric_type = type(size) if type(size) is not int else float
        pending_trade_ids = {} if pending_trade_ids is None else pending_trade_ids
        return cls(
            asset_id=asset_id,
            size=numeric_type(size),
            pending_maker=numeric_type(pending_maker),
            pending_taker=numeric_type(pending_taker),
            settlement_status=settlement_status,
            allow_neg=allow_neg,
            pending_trade_ids=pending_trade_ids,
            max_size_trade_ids=max_size_trade_ids,
            size_sig_digits=size_sig_digits,
        )

    @property
    def size_available(self) -> NumericAlias:
        """Available and settled size to perform trades on (buying power)."""
        return round_down(self.size + self.pending_maker, self.size_sig_digits)

    @property
    def size_total(self) -> NumericAlias:
        """Total position size including pending size."""
        return round_down(
            self.size + self.pending_maker + self.pending_taker, self.size_sig_digits
        )

    @property
    def empty(self) -> bool:
        try:
            return (
                self.size
                == self.pending_taker
                == self.pending_maker
                == len(self.pending_trade_ids)
                == 0
            )
        except TypeError:
            return (
                self.size
                == self.pending_taker
                == self.pending_maker
                == len(self.pending_trade_ids)
                == type(self.size)(0)
            )

    def _add_trade_id(self, trade_id: str, size: NumericAlias) -> None:
        # todo compress string?
        if len(self.pending_trade_ids) >= self.max_size_trade_ids:
            raise PositionTransactionException(
                f"Exceeding max_size_trade_ids={self.max_size_trade_ids}. Position: {self}."
            )

        if trade_id in self.pending_trade_ids:
            raise PositionTransactionException(
                f"trade_id={trade_id} already in pending_trade_ids. "
                f"Transaction was aborted (no changes)."
                f"Position: {self}."
            )

        self.pending_trade_ids[trade_id] = size

    def _remove_trade_id(self, trade_id: str, size: NumericAlias) -> bool:
        if trade_id in self.pending_trade_ids:
            if self.pending_trade_ids[trade_id] != size:
                raise PositionTransactionException(
                    f"Sizes do not match for trade_id={trade_id}. "
                    f"New size={size}, previous size={self.pending_trade_ids[trade_id]}."
                )

            del self.pending_trade_ids[trade_id]
            return True

        return False

    def _init_act_taker(self, delta_size: NumericAlias, trade_id: str) -> None:
        self._add_trade_id(trade_id, delta_size)
        self.pending_taker += delta_size

    def _finalize_act_taker(self, delta_size: NumericAlias, trade_id: str) -> None:
        if self._remove_trade_id(trade_id, delta_size):
            self.pending_taker -= delta_size

        self.size += delta_size
        self.size = round_down(self.size, self.size_sig_digits)

    def _act_taker(
        self,
        delta_size: NumericAlias,
        trade_id: str,
        trade_status: TRADE_STATUS,
    ) -> None:
        # inflow/ receive
        if trade_status is TRADE_STATUS.MATCHED:
            self._init_act_taker(delta_size, trade_id)
        elif trade_status is self.settlement_status:
            self._finalize_act_taker(delta_size, trade_id)
        elif trade_status is TRADE_STATUS.FAILED:
            if self._remove_trade_id(trade_id, delta_size):
                self.pending_taker -= delta_size
        elif trade_status is TRADE_STATUS.RETRYING:
            if trade_id not in self.pending_trade_ids:
                # in this case, we missed MATCHED
                self._init_act_taker(delta_size, trade_id)
        elif trade_status is TRADE_STATUS.MINED:
            # this can only be the case, if settlement_status is CONFIRMED
            if trade_id not in self.pending_trade_ids:
                # in this case, we missed MATCHED
                self._init_act_taker(delta_size, trade_id)
            # else: trade has already been registered
        elif trade_status is TRADE_STATUS.CONFIRMED:
            # this can only be the case, if settlement_status is MINED
            if trade_id in self.pending_trade_ids:
                # in this case, we missed MINED
                self._finalize_act_taker(delta_size, trade_id)
            # else: trade id is not in pending_trade_ids, so we have no clue, if we missed it all along,
            # or if trade has already been finalized and therefore removed from pending_trade_ids
        else:
            raise PositionTransactionException(
                f"Unknown parameter set: "
                f"delta_size={delta_size}, trade_id={trade_id}, trade_status={trade_status}."
            )

        self.pending_taker = round_down(self.pending_taker, self.size_sig_digits)

    def _init_act_maker(self, delta_size: NumericAlias, trade_id: str) -> None:
        self._add_trade_id(trade_id, -delta_size)
        self.pending_maker -= delta_size

    def _finalize_act_maker(self, delta_size: NumericAlias, trade_id: str) -> None:
        if self._remove_trade_id(trade_id, -delta_size):
            self.pending_maker += delta_size

        self.size -= delta_size
        self.size = round_down(self.size, self.size_sig_digits)

    def _act_maker(
        self,
        delta_size: NumericAlias,
        trade_id: str,
        trade_status: TRADE_STATUS,
    ) -> None:
        # outflow/ spend
        if trade_status is TRADE_STATUS.MATCHED:
            self._init_act_maker(delta_size, trade_id)
        elif trade_status is self.settlement_status:
            self._finalize_act_maker(delta_size, trade_id)
        elif trade_status is TRADE_STATUS.FAILED:
            if self._remove_trade_id(trade_id, -delta_size):
                self.pending_maker += delta_size
        elif trade_status is TRADE_STATUS.RETRYING:
            if trade_id not in self.pending_trade_ids:
                # in this case we missed MATCHED
                self._init_act_maker(delta_size, trade_id)
        elif trade_status is TRADE_STATUS.MINED:
            # settlement_status must be CONFIRMED
            if trade_id not in self.pending_trade_ids:
                # we missed MATCHED
                self._init_act_maker(delta_size, trade_id)
            # else: trade already registered
        elif trade_status is TRADE_STATUS.CONFIRMED:
            # settlement_status must be MINED
            if trade_id in self.pending_trade_ids:
                # we missed MINED
                self._finalize_act_maker(delta_size, trade_id)
            # else: we cannot tell, if already settled, or we missed all previous states
        else:
            raise PositionTransactionException(
                f"Unknown parameter set: "
                f"delta_size={delta_size}, trade_id={trade_id}, trade_status={trade_status}."
            )

        self.pending_maker = round_down(self.pending_maker, self.size_sig_digits)


class PositionFactory(Protocol):
    def __call__(
        self,
        asset_id: str,
        size: NumericAlias,
        size_sig_digits: int = SIG_DIGITS_SIZE,
        **kwargs,
    ) -> PositionProtocol:
        ...
