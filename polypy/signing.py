import base64
import hashlib
import hmac
from datetime import datetime, timezone
from enum import IntEnum
from functools import lru_cache
from random import random
from typing import Any

# noinspection PyProtectedMember
from eth_account._utils.signing import sign_message_hash
from eth_account.datastructures import SignedMessage
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from eth_typing import Hash32
from eth_utils import ValidationError, to_checksum_address
from hexbytes import HexBytes
from poly_eip712_structs import EIP712Struct, make_domain

from polypy.constants import CHAIN_ID, EXCHANGE_ADDRESS
from polypy.exceptions import PolyPyException


# noinspection PyPep8Naming
class SIGNATURE_TYPE(IntEnum):
    EOA = 0
    POLY_PROXY = 1
    POLY_GNOSIS_SAFE = 2


def generate_seed() -> int:
    """
    Pseudo random seed
    """
    now = datetime.now().replace(tzinfo=timezone.utc).timestamp()
    return int(round(now * random()))


def prepend_zx(in_str: str) -> str:
    """
    Prepend 0x to the input string if it is missing
    """
    s = in_str
    if len(s) > 2 and s[:2] != "0x":
        s = f"0x{s}"
    return s


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


@lru_cache(maxsize=4)
def private_key_checksum_address(private_key: str | PrivateKey | PrivateKeyType) -> str:
    private_key = parse_private_key(private_key)
    return private_key.public_key.to_checksum_address()


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


def get_domain(chain_id: CHAIN_ID, exchange_address: EXCHANGE_ADDRESS) -> EIP712Struct:
    return make_domain(
        name="Polymarket CTF Exchange",
        version="1",
        chainId=str(chain_id),
        verifyingContract=to_checksum_address(exchange_address),
    )


@lru_cache(maxsize=4)
def polymarket_domain(chain_id: CHAIN_ID, neg_risk: bool) -> EIP712Struct:
    # sourcery skip: remove-redundant-if, simplify-boolean-comparison

    if neg_risk is True:
        # noinspection PyTypeChecker
        exchange_addr: EXCHANGE_ADDRESS = EXCHANGE_ADDRESS[f"{chain_id.name}_NEG_RISK"]
    elif neg_risk is False:
        # noinspection PyTypeChecker
        exchange_addr: EXCHANGE_ADDRESS = EXCHANGE_ADDRESS[chain_id.name]
    else:
        raise PolyPyException(f"neg_risk must be bool. Got: {neg_risk}.")

    return get_domain(chain_id, exchange_addr)


def build_hmac_signature(
    secret: str,
    timestamp: int,
    method: str,
    request_path: str,
    body: Any | None,
) -> str:
    """
    Creates an HMAC signature by signing a payload with the secret

    Notes
    -----
    Based on: https://github.com/Polymarket/py-clob-client/blob/main/py_clob_client/signing/hmac.py#L6
    """

    # replace single quotes with double quotes to generate the same hmac message as go and typescript
    body = str(body).replace("'", '"') if body else ""
    # todo json dumps instead of str? -> special cases... e.g. if body is str already

    message = f"{timestamp}{method}{request_path}{body}"

    key = base64.urlsafe_b64decode(secret)
    hmac_hash = hmac.new(key, message.encode("utf-8"), hashlib.sha256)

    return base64.urlsafe_b64encode(hmac_hash.digest()).decode("utf-8")
