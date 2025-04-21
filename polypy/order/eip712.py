from enum import IntEnum
from typing import Any, Self

import attrs
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from eth_utils.crypto import keccak
from poly_eip712_structs import Address, EIP712Struct, Uint

from polypy.constants import ZERO_ADDRESS
from polypy.exceptions import OrderCreationException
from polypy.signing import (
    SIGNATURE_TYPE,
    _eth_sign_hash,
    generate_seed,
    parse_private_key,
    prepend_zx,
    private_key_checksum_address,
)


# noinspection PyPep8Naming
class SIDE_INDEX(IntEnum):
    BUY = 0
    SELL = 1


def _cvt_expiration(x: Any | None) -> int:
    return 0 if x is None else int(x)


@attrs.define
class EIP712Order:
    """Order object from REST API"""

    salt: int = attrs.field(converter=int)
    maker: str = attrs.field(converter=str)
    signer: str | None = attrs.field(converter=attrs.converters.optional(str))
    taker: str = attrs.field(converter=str)
    tokenId: int = attrs.field(converter=int)
    makerAmount: int = attrs.field(converter=int)
    takerAmount: int = attrs.field(converter=int)
    nonce: int = attrs.field(converter=int, validator=attrs.validators.ge(0))
    feeRateBps: int = attrs.field(converter=int, validator=attrs.validators.ge(0))
    side: SIDE_INDEX = attrs.field(converter=SIDE_INDEX)
    signatureType: SIGNATURE_TYPE = attrs.field(
        validator=attrs.validators.instance_of(SIGNATURE_TYPE),
    )
    expiration: int = attrs.field(
        default=0, converter=_cvt_expiration, validator=attrs.validators.ge(0)
    )

    def __attrs_post_init__(self):
        if self.signer is None:
            self.signer = self.maker

    @classmethod
    def create(
        cls,
        token_id: int,
        side: SIDE_INDEX,
        taker_amount: int,
        maker_amount: int,
        private_key: PrivateKey,
        maker: str | None,
        signature_type: SIGNATURE_TYPE | None,
        expiration: int = 0,
        nonce: int = 0,
        fee_rate_bps: int = 0,
        taker: str = ZERO_ADDRESS,
        signer: str | None = None,
        salt: int | None = None,
    ) -> Self:
        if maker is None:
            maker = private_key_checksum_address(private_key)

        if signer is None:
            signer = private_key_checksum_address(private_key)

        if signature_type is None:
            signature_type = SIGNATURE_TYPE.EOA

        if salt is None:
            salt = generate_seed()

        return cls(
            salt=salt,
            maker=maker,
            signer=signer,
            taker=taker,
            tokenId=token_id,
            makerAmount=maker_amount,
            takerAmount=taker_amount,
            nonce=nonce,
            feeRateBps=fee_rate_bps,
            side=side,
            signatureType=signature_type,
            expiration=expiration,
        )

    def to_dict(self):
        return {
            "salt": self.salt,
            "maker": self.maker,
            "signer": self.signer,
            "taker": self.taker,
            "tokenId": self.tokenId,
            "makerAmount": self.makerAmount,
            "takerAmount": self.takerAmount,
            "expiration": self.expiration,
            "nonce": self.nonce,
            "feeRateBps": self.feeRateBps,
            "side": self.side,
            "signatureType": self.signatureType,
        }


struct_sig = (
    "Order(uint256 salt,address maker,address signer,address taker,"
    "uint256 tokenId,uint256 makerAmount,uint256 takerAmount,uint256 expiration,"
    "uint256 nonce,uint256 feeRateBps,uint8 side,uint8 signatureType)"
)
order_members = [
    (Uint(256), "salt"),
    (Address(), "maker"),
    (Address(), "signer"),
    (Address(), "taker"),
    (Uint(256), "tokenId"),
    (Uint(256), "makerAmount"),
    (Uint(256), "takerAmount"),
    (Uint(256), "expiration"),
    (Uint(256), "nonce"),
    (Uint(256), "feeRateBps"),
    (Uint(8), "side"),
    (Uint(8), "signatureType"),
]


def order_encode_value(x: EIP712Order) -> bytes:
    encoded_values = [typ.encode_value(getattr(x, name)) for typ, name in order_members]
    return b"".join(encoded_values)


def order_hash_struct(x: EIP712Order) -> bytes:
    type_hash = keccak(text=struct_sig)
    return keccak(b"".join([type_hash, order_encode_value(x)]))


def order_signable_bytes(x: EIP712Order, domain: EIP712Struct) -> bytes:
    return b"\x19\x01" + domain.hash_struct() + order_hash_struct(x)


def order_id_hash(x: EIP712Order, domain: EIP712Struct) -> str:
    return prepend_zx(keccak(order_signable_bytes(x, domain)).hex())


def order_signature(
    x: EIP712Order, order_id: str, private_key: str | PrivateKey | PrivateKeyType
) -> str:
    key = parse_private_key(private_key)  # make sure we have PrivateKey obj
    if x.signer != private_key_checksum_address(key):
        raise OrderCreationException("Signer does not match")

    signed_hash_struct = _eth_sign_hash(order_id, key).signature.hex()
    return prepend_zx(signed_hash_struct)
