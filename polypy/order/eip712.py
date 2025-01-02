import traceback
from datetime import datetime, timezone
from enum import IntEnum
from functools import lru_cache
from random import random
from typing import Self

import attrs

# noinspection PyProtectedMember
from eth_account._utils.signing import sign_message_hash
from eth_account.datastructures import SignedMessage
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from eth_keys.exceptions import ValidationError
from eth_typing import Hash32
from eth_utils import to_checksum_address
from eth_utils.crypto import keccak
from hexbytes import HexBytes
from poly_eip712_structs import Address, EIP712Struct, Uint, make_domain

from polypy.constants import CHAIN_ID, ENDPOINT, EXCHANGE_ADDRESS, ZERO_ADDRESS
from polypy.exceptions import OrderCreationException, PolyPyException
from polypy.rest import get_neg_risk


def generate_seed() -> int:
    """
    Pseudo random seed
    """
    now = datetime.now().replace(tzinfo=timezone.utc).timestamp()
    return int(round(now * random()))


# noinspection PyPep8Naming
class SIDE_INDEX(IntEnum):
    BUY = 0
    SELL = 1


# noinspection PyPep8Naming
class SIGNATURE_TYPE(IntEnum):
    EOA = 0
    POLY_PROXY = 1
    POLY_GNOSIS_SAFE = 2


@attrs.define
class EIP712Order:
    """Order object from REST API"""

    salt: int = attrs.field(init=False, factory=generate_seed)
    maker: str = attrs.field(converter=to_checksum_address)
    signer: str | None = attrs.field(
        converter=attrs.converters.optional(to_checksum_address)
    )
    taker: str = attrs.field(converter=to_checksum_address)
    tokenId: int = attrs.field(converter=int)
    makerAmount: int = attrs.field(converter=int)
    takerAmount: int = attrs.field(converter=int)
    nonce: int = attrs.field(converter=int, validator=attrs.validators.ge(0))
    feeRateBps: int = attrs.field(converter=int, validator=attrs.validators.ge(0))
    side: SIDE_INDEX = attrs.field(converter=SIDE_INDEX)
    signatureType: SIGNATURE_TYPE = attrs.field(
        default=SIGNATURE_TYPE.EOA,
        validator=attrs.validators.instance_of(SIGNATURE_TYPE),
    )
    expiration: int = attrs.field(
        default=0, converter=int, validator=attrs.validators.ge(0)
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
        expiration: int = 0,
        nonce: int = 0,
        fee_rate_bps: int = 0,
        taker: str = ZERO_ADDRESS,
        maker: str | None = None,
        signer: str | None = None,
        signature_type: SIGNATURE_TYPE = SIGNATURE_TYPE.EOA,
    ) -> Self:
        private_key = parse_private_key(private_key)

        if maker is None:
            maker = private_key.public_key.to_checksum_address()

        if signer is None:
            signer = private_key.public_key.to_checksum_address()

        return cls(
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


def prepend_zx(in_str: str) -> str:
    """
    Prepend 0x to the input string if it is missing
    """
    s = in_str
    if len(s) > 2 and s[:2] != "0x":
        s = f"0x{s}"
    return s


def order_encode_value(x: EIP712Order) -> bytes:
    encoded_values = [typ.encode_value(getattr(x, name)) for typ, name in order_members]
    return b"".join(encoded_values)


def order_hash_struct(x: EIP712Order) -> bytes:
    type_hash = keccak(text=struct_sig)
    return keccak(b"".join([type_hash, order_encode_value(x)]))


def order_signable_bytes(x: EIP712Order, domain: EIP712Struct) -> bytes:
    return b"\x19\x01" + domain.hash_struct() + order_hash_struct(x)


@lru_cache(maxsize=16)
def parse_private_key(
    key: PrivateKeyType | str | PrivateKey,
) -> PrivateKey:
    if isinstance(key, PrivateKey):
        return key

    hb_key = HexBytes(key)

    try:
        return PrivateKey(hb_key)
    except ValidationError as original_exception:
        raise ValueError(
            "Private key must be exactly 32 bytes long, but got "
            f"{len(hb_key)} bytes."
        ) from original_exception


def _eth_sign_hash(
    message_hash: Hash32 | str, private_key: PrivateKey
) -> SignedMessage:
    msg_hash_bytes = HexBytes(message_hash)
    if len(msg_hash_bytes) != 32:
        raise ValueError("The message hash must be exactly 32-bytes")

    (v, r, s, eth_signature_bytes) = sign_message_hash(private_key, msg_hash_bytes)
    return SignedMessage(
        message_hash=msg_hash_bytes,
        r=r,
        s=s,
        v=v,
        signature=HexBytes(eth_signature_bytes),
    )


def order_signature(
    x: EIP712Order, domain: EIP712Struct, private_key: str | PrivateKey | PrivateKeyType
) -> str:
    hash_struct = prepend_zx(keccak(order_signable_bytes(x, domain)).hex())

    key = parse_private_key(private_key)  # make sure we have PrivateKey obj
    if x.signer != key.public_key.to_checksum_address():
        raise OrderCreationException("Signer does not match")

    signed_hash_struct = _eth_sign_hash(hash_struct, key).signature.hex()
    return prepend_zx(signed_hash_struct)


@lru_cache(maxsize=4)
def get_domain(chain_id: CHAIN_ID, exchange_address: EXCHANGE_ADDRESS) -> EIP712Struct:
    return make_domain(
        name="Polymarket CTF Exchange",
        version="1",
        chainId=str(chain_id),
        verifyingContract=to_checksum_address(exchange_address),
    )


def polymarket_domain(
    chain_id: CHAIN_ID,
    neg_risk: bool | None,
    token_id: str | None = None,
    rest_endpoint: str | ENDPOINT | None = None,
) -> EIP712Struct:
    if neg_risk is None:
        try:
            neg_risk = get_neg_risk(rest_endpoint, token_id)
        except Exception as e:
            raise PolyPyException(
                f"Cannot get neg_risk via REST. Did you configure 'rest_endpoint' correctly?. "
                f"Full traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}."
            ) from e

    if neg_risk:
        # noinspection PyTypeChecker
        exchange_addr: EXCHANGE_ADDRESS = EXCHANGE_ADDRESS[f"{chain_id.name}_NEG_RISK"]
    else:
        # noinspection PyTypeChecker
        exchange_addr: EXCHANGE_ADDRESS = EXCHANGE_ADDRESS[chain_id.name]

    return get_domain(chain_id, exchange_addr)
