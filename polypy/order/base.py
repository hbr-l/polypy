import datetime
import math
import warnings
from typing import Any, Callable, NoReturn, Self

import attrs
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from poly_eip712_structs import EIP712Struct

from polypy.constants import ZERO_ADDRESS
from polypy.exceptions import OrderCreationException, OrderUpdateException
from polypy.order.common import INSERT_STATUS, SIDE, TIME_IN_FORCE
from polypy.order.eip712 import SIDE_INDEX, EIP712Order, order_id_hash, order_signature
from polypy.rounding import round_half_even
from polypy.signing import SIGNATURE_TYPE, parse_private_key
from polypy.typing import NumericAlias

# in Polymarket documentation:
# - Order: REST -> base order
# - MakerOrder: REST -> only limit orders can (but not have to) be maker order
# - OpenOrder: REST -> only limit orders can be open

# - MakerOrder: WS -> only limit orders can (but not have to) be maker order
# - Order: WS -> limit order (placement, update, cancellation), maker only will be matched immediately without placement


def compute_expiration_timestamp(
    tif: TIME_IN_FORCE,
    expire: int | datetime.timedelta | datetime.datetime,
    security_s: int = 60,
) -> int:
    """Compute expiration timestamp.

    Parses `expire` into a timestamp and adds `security_s` on top (required by Polymarket).

    Parameters
    ----------
    tif: TIME_IN_FORCE,
        if not GTD, then always returns 0
    expire: int | datetime.timedelta | datetime.datetime,
        - int: duration in seconds until order expires
        - datetime.timedelta: duration until order expires
        - datetime.datetime: order expires at specified datetime
    security_s: int, default = 60

    Returns
    -------
    int: timestamp of expiration, ready to be sent via REST call
    """
    # in second timestamp
    if tif is not TIME_IN_FORCE.GTD:
        return 0

    if isinstance(expire, int):
        return int(datetime.datetime.now().timestamp()) + expire + security_s
    elif isinstance(expire, datetime.timedelta):
        return int((datetime.datetime.now() + expire).timestamp()) + security_s
    elif isinstance(expire, datetime.datetime):
        if expire <= datetime.datetime.now():
            raise OrderCreationException("Expiration date is in the past.")
        return int(expire.timestamp()) + security_s
    else:
        raise OrderCreationException(
            f"Unknown type for `delta`: {type(expire)}. Input: {expire}."
        )


def check_valid_price(price: NumericAlias, tick_size: NumericAlias) -> None:
    if not (tick_size <= price <= 1 - tick_size):
        raise OrderCreationException(
            f"Invalid price. Must be 'tick_size <= price <= 1 - tick_size'. "
            f"Got: price={type(price)}({price}) and tick_size={type(tick_size)}({tick_size})."
        )


def tick_size_digits(tick_size: NumericAlias) -> int:
    if tick_size <= 0 or tick_size >= 1:
        raise OrderCreationException("`tick_size` has to be in (0, 1)")

    exp = math.log10(tick_size)
    if exp != int(exp):
        raise OrderCreationException("`tick_size` has to be base10.")

    return -int(exp)


def cvt_tick_size(
    tick_size: NumericAlias, numeric_type: type | Callable[[Any], NumericAlias]
) -> tuple[NumericAlias, int]:
    nb_digits = tick_size_digits(tick_size)

    if not isinstance(tick_size, numeric_type):
        try:
            tick_size = numeric_type(str(tick_size))
        except ValueError:
            tick_size = round_half_even(numeric_type(tick_size), nb_digits)

    return tick_size, nb_digits


def _cvt_numeric_type(val: Any, inst: "Order") -> NumericAlias:
    return inst.numeric_type(val)


def _validate_numeric_type(inst: "Order", attr, val: NumericAlias) -> None:
    if not isinstance(val, inst.numeric_type):
        raise OrderCreationException(
            f"{attr.name} must be of same type as `numeric_type`={inst.numeric_type}."
        )


def _validate_not_none(_, attr, val: str | None) -> None:
    if not val:
        raise OrderCreationException(f"{attr.name} must not be None or empty string.")


def _optional_frozen(inst: "Order", attr, val: Any) -> Any:
    if getattr(inst, attr.name) is not None:
        raise OrderUpdateException("Can only be set if None. Read-only if once set.")
    return val


def _frozen(_, attr, ___) -> NoReturn:
    raise OrderUpdateException(f"Frozen attribute: {attr.name}.")


def _parse_optional_defined_at(defined_at: int | None) -> int:
    if defined_at is None:
        defined_at = int(1_000 * datetime.datetime.now().timestamp())
    return defined_at


def _parse_optional_order_id(
    order_id: str | None, eip712order: EIP712Order, domain: EIP712Struct
) -> str:
    if order_id is None:
        order_id = order_id_hash(eip712order, domain)
    else:
        warnings.warn(
            "Manually specified `order_id` will not be validated separately "
            "(test injection purpose only). "
            "Use `Order.validate_id()` if necessary."
        )
    return order_id


def _parse_optional_signature(
    signature: str | None,
    order_id: str,
    eip712order: EIP712Order,
    private_key: PrivateKey | str | PrivateKeyType,
) -> str:
    if signature is None:
        signature = order_signature(eip712order, order_id, private_key)
    else:
        warnings.warn(
            "Manually specified `signature` will not be validated separately "
            "(test injection purpose only). "
            "Use `Order.validate_signature()` if necessary."
        )
    return signature


def _parse_optional_size_matched(
    size_matched: NumericAlias | None,
    numeric_type: type[NumericAlias] | Callable[[int], NumericAlias],
) -> NumericAlias:
    if size_matched is None:
        size_matched = numeric_type(0)
    elif type(size_matched) != numeric_type:
        raise OrderCreationException(
            f"Type of size_matched={type(size_matched)} does not match numeric_type={numeric_type}."
        )

    return size_matched


# todo shared memory version
@attrs.define
class Order:
    eip712order: EIP712Order = attrs.field(on_setattr=_frozen)

    id: str = attrs.field(
        converter=attrs.converters.optional(str),
        on_setattr=_frozen,
        validator=_validate_not_none,
    )
    signature: str = attrs.field(
        converter=attrs.converters.optional(str),
        on_setattr=_frozen,
        validator=_validate_not_none,
    )

    tif: TIME_IN_FORCE = attrs.field(converter=TIME_IN_FORCE, on_setattr=_frozen)

    strategy_id: str | None = attrs.field(converter=attrs.converters.optional(str))
    aux_id: str | None = attrs.field(converter=attrs.converters.optional(str))

    status: INSERT_STATUS = attrs.field(converter=INSERT_STATUS)
    created_at: int | None = attrs.field(
        converter=attrs.converters.optional(int),
        on_setattr=[attrs.setters.convert, _optional_frozen],
    )
    """In seconds"""
    defined_at: int = attrs.field(converter=int, on_setattr=_frozen)
    """In millis"""

    numeric_type: type[NumericAlias] | Callable[
        [float | str], NumericAlias
    ] = attrs.field(on_setattr=_frozen)

    # noinspection PyTypeChecker
    size_matched: NumericAlias = attrs.field(
        validator=_validate_numeric_type,
        converter=attrs.Converter(_cvt_numeric_type, takes_self=True),
    )

    @classmethod
    def create(
        cls,
        token_id: str,
        side: SIDE,
        taker_amount: int,
        maker_amount: int,
        private_key: PrivateKey | str | PrivateKeyType,
        maker: str | None,
        signature_type: SIGNATURE_TYPE,
        domain: EIP712Struct,
        tif: TIME_IN_FORCE = TIME_IN_FORCE.GTC,
        salt: int | None = None,
        order_id: str | None = None,
        signature: str | None = None,
        size_matched: NumericAlias | None = None,
        strategy_id: str | None = None,
        aux_id: str | None = None,
        status: INSERT_STATUS = INSERT_STATUS.DEFINED,
        created_at: int | None = None,
        defined_at: int | None = None,
        expiration: int = 0,
        nonce: int = 0,
        fee_rate_bps: int = 0,
        taker: str = ZERO_ADDRESS,
        signer: str | None = None,
        numeric_type: type[NumericAlias] | Callable[[int], NumericAlias] = float,
    ) -> Self:
        private_key = parse_private_key(private_key)

        # noinspection PyTypeChecker
        eip712order = EIP712Order.create(
            salt=salt,
            maker=maker,
            signer=signer,
            taker=taker,
            token_id=int(token_id),
            maker_amount=maker_amount,
            taker_amount=taker_amount,
            nonce=nonce,
            fee_rate_bps=fee_rate_bps,
            side=SIDE_INDEX[side],
            signature_type=signature_type,
            expiration=expiration,
            private_key=private_key,
        )

        defined_at = _parse_optional_defined_at(defined_at)
        order_id = _parse_optional_order_id(order_id, eip712order, domain)
        signature = _parse_optional_signature(
            signature, order_id, eip712order, private_key
        )
        size_matched = _parse_optional_size_matched(size_matched, numeric_type)

        return cls(
            eip712order=eip712order,
            id=order_id,
            signature=signature,
            tif=tif,
            strategy_id=strategy_id,
            aux_id=aux_id,
            status=status,
            created_at=created_at,
            defined_at=defined_at,
            size_matched=size_matched,
            numeric_type=numeric_type,
        )

    def validate_id(self, domain: EIP712Struct) -> bool:
        order_id = order_id_hash(self.eip712order, domain)
        return self.id == order_id

    def validate_signature(
        self, private_key: PrivateKey | str | PrivateKeyType
    ) -> bool:
        signature = order_signature(self.eip712order, self.id, private_key)
        return self.signature == signature

    def to_dict(self) -> dict[str, Any]:
        eip712_dict = self.eip712order.to_dict()
        eip712_dict["signature"] = self.signature
        eip712_dict["side"] = str(self.side)
        eip712_dict["expiration"] = str(eip712_dict["expiration"])
        eip712_dict["nonce"] = str(eip712_dict["nonce"])
        eip712_dict["feeRateBps"] = str(eip712_dict["feeRateBps"])
        eip712_dict["makerAmount"] = str(eip712_dict["makerAmount"])
        eip712_dict["takerAmount"] = str(eip712_dict["takerAmount"])
        eip712_dict["tokenId"] = str(eip712_dict["tokenId"])
        eip712_dict["signatureType"] = int(eip712_dict["signatureType"])
        return eip712_dict

    def to_payload(self, api_key: str) -> dict[str, dict | str]:
        return {"order": self.to_dict(), "owner": api_key, "orderType": self.tif.value}

    @property
    def price(self) -> NumericAlias:
        # price = amount / size

        # SELL:
        #   maker_amt -> size
        #   taker_amt -> amount
        # BUY:
        #   taker_amt -> size
        #   maker_amt -> amount

        taker_amt = self.numeric_type(self.taker_amount)
        maker_amt = self.numeric_type(self.maker_amount)

        if self.side is SIDE.SELL:
            return taker_amt / maker_amt
        else:
            return maker_amt / taker_amt

    @property
    def size(self) -> NumericAlias:
        if self.side is SIDE.SELL:
            return self.numeric_type(self.maker_amount) / 1_000_000
        else:
            return self.numeric_type(self.taker_amount) / 1_000_000

    @property
    def size_open(self) -> NumericAlias:
        return self.size - self.size_matched

    @property
    def amount(self) -> NumericAlias:
        if self.side is SIDE.SELL:
            return self.numeric_type(self.taker_amount) / 1_000_000
        else:
            return self.numeric_type(self.maker_amount) / 1_000_000

    @property
    def token_id(self) -> str:
        return str(self.eip712order.tokenId)

    @property
    def asset_id(self) -> str:
        """Alias to `token_id`."""
        return str(self.eip712order.tokenId)

    @property
    def expiration(self) -> int:
        """In seconds"""
        return self.eip712order.expiration

    @property
    def side(self) -> SIDE:
        return SIDE(self.eip712order.side.name)

    @property
    def signature_type(self) -> SIGNATURE_TYPE:
        return self.eip712order.signatureType

    @property
    def fee_rate_bps(self) -> int:
        return self.eip712order.feeRateBps

    @property
    def taker_amount(self) -> int:
        return self.eip712order.takerAmount

    @property
    def maker_amount(self) -> int:
        return self.eip712order.makerAmount

    @property
    def salt(self) -> int:
        return self.eip712order.salt

    @property
    def maker(self) -> str:
        return self.eip712order.maker

    @property
    def taker(self) -> str:
        return self.eip712order.taker

    @property
    def signer(self) -> str:
        return self.eip712order.signer

    @property
    def nonce(self) -> int:
        return self.eip712order.nonce
