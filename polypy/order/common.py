import warnings
from enum import StrEnum
from typing import Any, Callable, NoReturn, Protocol

from polypy.exceptions import OrderUpdateException
from polypy.signing import SIGNATURE_TYPE
from polypy.typing import NumericAlias


# noinspection PyPep8Naming
class INSERT_STATUS(StrEnum):
    # Polymarket status
    LIVE = "LIVE"
    MATCHED = "MATCHED"
    DELAYED = "DELAYED"
    UNMATCHED = "UNMATCHED"

    # PolyPy status
    DEFINED = "DEFINED"
    CANCELED = "CANCELED"


# noinspection PyPep8Naming
class TIME_IN_FORCE(StrEnum):
    FOK = "FOK"
    """Fill-Or-Kill"""
    GTC = "GTC"
    """Good-Till-Cancel"""
    GTD = "GTD"
    """Good-Till-Day"""


class SIDE(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


TERMINAL_INSERT_STATI = (
    INSERT_STATUS.CANCELED,
    INSERT_STATUS.MATCHED,
    INSERT_STATUS.UNMATCHED,
)


CANCELABLE_INSERT_STATI = (
    INSERT_STATUS.DEFINED,  # todo necessary?
    INSERT_STATUS.LIVE,
    INSERT_STATUS.DELAYED,
)


class OrderProtocol(Protocol):
    """Protocol interface for generic order object.

    Notes
    -----
    We __discard__ the following attributes from the Polymarket API:
    - outcome: this is redundant to market info (given that token id is known)
    - order_owner: redundant to owner
    - owner: same as api_key
    - associate_trades: only available for limit orders in update message;
        but is also retrievable via trades information (redundant)
    - matched_amount: only available for MakerOrder of matched trade, can not be computed from size_matched (e.g.
        order might be filled by different counter-orders with different (best) prices);
        but is also retrievable via trades information (redundant)
    - transactions_hashes: only available as response after submitting order and trade messages, but not order
        update messages;
        but is also retrievable via trades information (redundant)
    """

    id: str | None
    tif: TIME_IN_FORCE
    status: INSERT_STATUS
    signature: str | None

    numeric_type: type[NumericAlias] | Callable[[float | str], NumericAlias]

    price: NumericAlias  # property derived from taker/maker amount
    """NumericAlias: Only meaningful in case of a Limit Order."""
    size: NumericAlias  # property derived from taker/maker amount
    """NumericAlias: Only meaningful in case of a Limit Order."""
    amount: NumericAlias  # property derived from taker/maker amount
    """NumericAlias: In case of a Limit order, amount might deviate from amount actually matched/filled. 
    Only relay on amount in case of a Market Order."""

    size_matched: NumericAlias
    """NumericAlias: In case of a Market Order, size_matched will not be consistent with the original size defined.
    Only rely on size_matched in case of a Limit Order."""
    # we do not define amount_matched, because order might be filled by multiple counter-orders
    #   at different (better) prices, which we first would need to retrieve from trades record
    #   (better include this in TradeManager)

    strategy_id: str | None
    created_at: int | None
    """Unix timestamp when order was created (get_order REST) or placement ws message received"""
    defined_at: int

    @property
    def token_id(self) -> str:
        ...

    @property
    def taker_amount(self) -> int:
        ...

    @property
    def maker_amount(self) -> int:
        ...

    @property
    def expiration(self) -> int:
        ...

    @property
    def side(self) -> SIDE:
        ...

    @property
    def signature_type(self) -> SIGNATURE_TYPE:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...

    def to_payload(self, api_key: str) -> dict[str, dict | str]:
        ...


class FrozenOrder:
    def __init__(self, order: OrderProtocol) -> None:
        object.__setattr__(self, "_wrapped_order", order)

    def __getattr__(self, name) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped_order"), name)

    def __setattr__(self, name, value) -> NoReturn:
        raise OrderUpdateException(
            f"{self.__class__.__name__} is read-only. Cannot set {name}={value}."
        )

    def __delattr__(self, name) -> NoReturn:
        raise OrderUpdateException(
            f"{self.__class__.__name__} is read-only. Cannot delete {name}."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(_wrapped_order={object.__getattribute__(self, '_wrapped_order')})"


def frozen_order(order: OrderProtocol) -> FrozenOrder:
    return FrozenOrder(order)


def update_order(
    order: OrderProtocol,
    status: INSERT_STATUS | None = None,
    size_matched: NumericAlias | None = None,
    strategy_id: str | None = None,
    created_at: int | None = None,
) -> OrderProtocol:
    if strategy_id is not None:
        order.strategy_id = strategy_id

    if status is not None:
        if order.status not in TERMINAL_INSERT_STATI:
            order.status = status
        elif order.status is not status:
            warnings.warn(
                f"Ignoring `status={status}`, order is in terminal state: {order.status}"
            )
        # else: order_status = status = TERMINAL_INSERT_STATI -> ignore
    # else: status = None -> ignore

    if size_matched is not None and size_matched >= order.size_matched:
        # todo use order.numeric_type(size_matched)? in all cases, size_matched is pre-casted outside plus
        #   standard Order class has on_setattr which auto-casts
        order.size_matched = size_matched
    elif size_matched is not None:
        warnings.warn(
            f"Ignoring `size_matched={size_matched}`, current size_matched is greater: {order.size_matched}"
        )
    # else: size_matched = None ignore

    if created_at is not None:
        if order.created_at is None or order.created_at <= 0:
            order.created_at = created_at
        elif order.created_at != created_at:
            warnings.warn(
                f"Ignoring `created_at={created_at}`, order.created_at is already set: {order.created_at}"
            )
        # else: order.created_at == created_at -> ignore
    # else: created_at = None -> ignore

    return order


def infer_numeric_type(x: Any) -> type:
    t = type(x)
    return float if t is int else t
