"""
Terminology
-----------
- Order
- Trade/ Fill
- Position
"""
import datetime
import math
from enum import StrEnum
from typing import Any, Self

import attrs
from eth_keys.datatypes import PrivateKey
from poly_eip712_structs import EIP712Struct

from polypy.constants import ZERO_ADDRESS
from polypy.exceptions import OrderCreationException
from polypy.order.eip712 import (
    SIDE_INDEX,
    SIGNATURE_TYPE,
    EIP712Order,
    order_signature,
    parse_private_key,
)
from polypy.typing import NumericAlias

# in Polymarket documentation:
# - Order: REST
# - MakerOrder: REST
# - OpenOrder: REST
# - MakerOrder: WS
# - Order: WS


# noinspection PyPep8Naming
class PLACEMENT_STATUS(StrEnum):
    INSITU = "insitu"
    LIVE = "live"
    MATCHED = "matched"
    DELAYED = "delayed"
    UNMATCHED = "unmatched"


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


# todo move?
def millis_now() -> int:
    return int(1_000 * datetime.datetime.now().timestamp())


def is_valid_price(price: NumericAlias, tick_size: NumericAlias) -> bool:
    return tick_size <= price <= 1 - tick_size


def tick_size_digits(
    tick_size: NumericAlias, exc_type: type[Exception] = ValueError
) -> int:
    if tick_size <= 0 or tick_size >= 1:
        raise exc_type("`tick_size` has to be in (0, 1)")

    exp = math.log10(tick_size)
    if exp != int(exp):
        raise exc_type("`tick_size` has to be base10.")

    return -int(exp)


@attrs.define
class BaseOrder:
    eip712order: EIP712Order = attrs.field(on_setattr=attrs.setters.frozen)

    id: str | None = attrs.field(converter=attrs.converters.optional(str))
    """Will be created after posting the order."""
    signature: str | None = attrs.field(converter=attrs.converters.optional(str))

    tif: TIME_IN_FORCE = attrs.field(
        converter=TIME_IN_FORCE, on_setattr=attrs.setters.frozen
    )
    strategy_id: str | None = attrs.field(converter=attrs.converters.optional(str))
    aux_id: str | None = attrs.field(converter=attrs.converters.optional(str))

    placement_status: PLACEMENT_STATUS = attrs.field(converter=PLACEMENT_STATUS)
    transaction_hashes: list[str] = attrs.field(
        validator=attrs.validators.instance_of(list)
    )
    created_at: int | None = attrs.field(converter=attrs.converters.optional(int))

    defined_at: int = attrs.field(converter=int, on_setattr=attrs.setters.frozen)

    @classmethod
    def _parse_create_args(
        cls,
        token_id: str,
        side: SIDE,
        taker_amount: int,
        maker_amount: int,
        private_key: PrivateKey,
        domain: EIP712Struct,
        signature: str | None,
        transaction_hashes: list[str] | None,
        defined_at: int | None,
        expiration: int,
        nonce: int,
        fee_rate_bps: int,
        taker: str,
        maker: str | None,
        signer: str | None,
        signature_type: SIGNATURE_TYPE,
    ):
        private_key = parse_private_key(private_key)
        # noinspection PyTypeChecker
        side_idx: SIDE_INDEX = SIDE_INDEX[side]

        eip712order = EIP712Order.create(
            maker=maker,
            signer=signer,
            taker=taker,
            token_id=int(token_id),
            maker_amount=maker_amount,
            taker_amount=taker_amount,
            nonce=nonce,
            fee_rate_bps=fee_rate_bps,
            side=side_idx,
            signature_type=signature_type,
            expiration=expiration,
            private_key=private_key,
        )

        if defined_at is None:
            defined_at = millis_now()

        if transaction_hashes is None:
            transaction_hashes = []

        if signature is None:
            signature = order_signature(eip712order, domain, private_key)

        return (
            eip712order,
            signature,
            transaction_hashes,
            defined_at,
        )

    @classmethod
    def create(
        cls,
        token_id: str,
        side: SIDE,
        taker_amount: int,
        maker_amount: int,
        private_key: PrivateKey,
        domain: EIP712Struct,
        tif: TIME_IN_FORCE = TIME_IN_FORCE.GTC,
        signature: str | None = None,
        strategy_id: str | None = None,
        aux_id: str | None = None,
        idx: str | None = None,
        placement_status: PLACEMENT_STATUS = PLACEMENT_STATUS.INSITU,
        transaction_hashes: list[str] | None = None,
        created_at: int | None = None,
        defined_at: int | None = None,
        expiration: int = 0,
        nonce: int = 0,
        fee_rate_bps: int = 0,  # todo really int?
        taker: str = ZERO_ADDRESS,
        maker: str | None = None,
        signer: str | None = None,
        signature_type: SIGNATURE_TYPE = SIGNATURE_TYPE.EOA,
    ) -> Self:
        # noinspection DuplicatedCode
        (
            eip712order,
            signature,
            transaction_hashes,
            defined_at,
        ) = cls._parse_create_args(
            token_id=token_id,
            side=side,
            taker_amount=taker_amount,
            maker_amount=maker_amount,
            private_key=private_key,
            domain=domain,
            signature=signature,
            transaction_hashes=transaction_hashes,
            defined_at=defined_at,
            expiration=expiration,
            nonce=nonce,
            fee_rate_bps=fee_rate_bps,
            taker=taker,
            maker=maker,
            signer=signer,
            signature_type=signature_type,
        )

        return cls(
            eip712order=eip712order,
            id=idx,
            signature=signature,
            tif=tif,
            strategy_id=strategy_id,
            aux_id=aux_id,
            placement_status=placement_status,
            transaction_hashes=transaction_hashes,
            created_at=created_at,
            defined_at=defined_at,
        )

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

    @property
    def token_id(self) -> str:
        return str(self.eip712order.tokenId)

    @property
    def expiration(self) -> int:
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
