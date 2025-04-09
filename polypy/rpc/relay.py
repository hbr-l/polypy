from typing import Literal

import msgspec.json
import requests
from eth_account.datastructures import SignedMessage
from eth_account.messages import encode_defunct
from eth_utils.crypto import keccak
from web3.eth import Eth

from polypy.constants import ENDPOINT, PROXY_WALLET_FACTORY, RELAY_HUB
from polypy.exceptions import RelayerException

# noinspection PyProtectedMember
from polypy.rest.api import _request
from polypy.signing import int_to_padded_hex, parse_private_key, prepend_zx, remove_zx
from polypy.structs import RelayerResponse


# noinspection PyTypeHints
def _encode_msg(
    from_addr: str,
    proxy_wallet_factory_addr: str | Literal[PROXY_WALLET_FACTORY],
    data: str,
    gas_limit: int,  # units of gas
    nonce: int,
    relay_hub_addr: str | Literal[RELAY_HUB],
    relay_addr: str,
    relayer_fee: int = 0,  # todo unknown unit
    gas_price_wei: int = 0,  # todo guess wei
):
    data_hex = remove_zx(data)
    data_pad = len(data_hex) + (len(data_hex) % 2)

    components = [
        "726c783a",
        remove_zx(from_addr),
        remove_zx(proxy_wallet_factory_addr),
        data_hex.zfill(data_pad),  # ensure even length
        int_to_padded_hex(relayer_fee),
        int_to_padded_hex(gas_price_wei),
        int_to_padded_hex(gas_limit),
        int_to_padded_hex(nonce),
        remove_zx(relay_hub_addr),
        remove_zx(relay_addr),
    ]

    return "".join(components)


def _hash_message(enc_msg: str) -> str:
    message = keccak(hexstr=prepend_zx(enc_msg)).hex()
    return prepend_zx(message)


def _generate_signature(hash_msg: str, private_key: str) -> SignedMessage:
    return Eth.account.sign_message(
        encode_defunct(hexstr=hash_msg),
        parse_private_key(private_key),
    )


# noinspection PyTypeHints
def generate_payload(
    addr: str,
    private_key: str,
    funder_address: str,
    proxy_wallet_factory_addr: str | Literal[PROXY_WALLET_FACTORY],
    data: str,
    gas_limit: int,  # units of gas
    nonce: int,
    relay_hub_addr: str | Literal[RELAY_HUB],
    relay_addr: str,
    relayer_fee: int = 0,  # todo unknown unit
    gas_price_wei: int = 0,  # todo guess wei
) -> dict:
    signature = _generate_signature(
        _hash_message(
            _encode_msg(
                from_addr=addr,
                proxy_wallet_factory_addr=proxy_wallet_factory_addr,
                data=data,
                relayer_fee=relayer_fee,
                gas_price_wei=gas_price_wei,
                gas_limit=gas_limit,
                nonce=nonce,
                relay_hub_addr=relay_hub_addr,
                relay_addr=relay_addr,
            )
        ),
        private_key,
    )

    return {
        "from": addr,
        "to": proxy_wallet_factory_addr,
        "proxyWallet": funder_address,
        "data": data,
        "nonce": str(nonce),
        "signature": prepend_zx(signature.signature.hex()),
        "signatureParams": {
            "gasPrice": str(gas_price_wei),
            "gasLimit": str(gas_limit),
            "relayerFee": str(relayer_fee),
            "relayHub": relay_hub_addr,
            "relay": relay_addr,
        },
        "type": "PROXY",
    }


def get_relay_info(
    endpoint: str | ENDPOINT,
    addr: str,
    cookies: dict[str, str],
) -> tuple[str, int]:
    if any(x is None for x in cookies.values()):
        raise RelayerException(f"'cookies' must not contain None. Got: {cookies}")

    resp = requests.get(
        f"{endpoint}/relay-payload",
        params={"address": addr, "type": "PROXY"},
        cookies=cookies,
    )

    try:
        resp.raise_for_status()
    except Exception as e:
        e.add_note(resp.text)
        raise e

    resp = resp.json()
    return resp["address"], int(resp["nonce"])


def submit(
    endpoint: str | ENDPOINT, payload: dict, cookies: dict[str, str]
) -> RelayerResponse:
    # resp = requests.post(f"{endpoint}/submit", params=payload, cookies=cookies)
    # try:
    #     resp.raise_for_status()
    # except Exception as e:
    #     e.add_note(resp.text)
    #     raise e
    resp = _request(f"{endpoint}/submit", "POST", None, payload, cookies=cookies)
    return msgspec.json.decode(resp.text, type=RelayerResponse)
