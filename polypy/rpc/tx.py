from functools import lru_cache
from typing import Literal, Self

import attrs
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from hexbytes import HexBytes
from web3 import HTTPProvider, Web3
from web3.eth import Contract
from web3.middleware import ExtraDataToPOAMiddleware
from web3.types import TxParams, TxReceipt, Wei

from polypy.constants import (
    CHAIN_ID,
    COLLATERAL,
    CONDITIONAL,
    ENDPOINT,
    NEGATIVE_RISK_ADAPTER,
    PROXY_WALLET_FACTORY,
)
from polypy.rounding import round_down
from polypy.rpc.abi import ERC20_ABI, ERC1155_ABI, PROXY_FACTORY_ABI
from polypy.signing import private_key_checksum_address
from polypy.typing import NumericAlias


class W3POA(Web3):
    def __init__(
        self,
        rpc_url: str | ENDPOINT,
        private_key: str | PrivateKeyType | PrivateKey,
        maker_funder: str | None,
    ) -> None:
        super().__init__(HTTPProvider(rpc_url))

        self.wallet_addr = private_key_checksum_address(private_key)
        self.maker_funder = (
            maker_funder if maker_funder is not None else self.wallet_addr
        )

        self.eth.default_account = self.wallet_addr
        self.middleware_onion.inject(
            ExtraDataToPOAMiddleware, name="geth_poa_middleware", layer=0
        )


@lru_cache(maxsize=2)
def _erc1155_contract(w3: W3POA, collateral: str | COLLATERAL) -> Contract:
    # we cannot define a global ERC20_CONTRACT because we need the rpc endpoint for calling
    #   'balanceOf' and since rpc endpoint is defined at PositionManager, we use the w3 instance
    return w3.eth.contract(address=collateral, abi=ERC1155_ABI)


@lru_cache(maxsize=8)  # todo up cache size?
def _erc20_contract(w3: W3POA, collateral: str | COLLATERAL) -> Contract:
    # we cannot define a global ERC20_CONTRACT because we need the rpc endpoint for calling
    #   'approve' and 'allowance' and since rpc endpoint is defined at PositionManager, we use
    #   the w3 instance
    return w3.eth.contract(address=collateral, abi=ERC20_ABI)


@lru_cache(maxsize=4)  # todo up cache size?
def _proxy_contract(w3: W3POA, proxy_wallet_factory_address: str) -> Contract:
    # we cannot define global PROXY_CONTRACT because we need the rpc endpoint for calling 'buildTransaction'
    #   and since rpc endpoint is defined at PositionManager, we use the w3 instance

    # noinspection PyTypeChecker
    return w3.eth.contract(address=proxy_wallet_factory_address, abi=PROXY_FACTORY_ABI)


def _get_gas_price_wei(
    w3: W3POA,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
) -> NumericAlias:
    gas_price_wei = w3.eth.gas_price * gas_factor
    gas_price_wei = int(round_down(gas_price_wei, 0))

    if max_gas_price is not None:
        gas_price_wei = min(gas_price_wei, max_gas_price)

    return gas_price_wei


@attrs.define
class ProxyDataFrame:
    encoded_data: str
    to: str | NEGATIVE_RISK_ADAPTER | CONDITIONAL | COLLATERAL

    @classmethod
    def from_erc20(cls, encoded_data: str, chain_id: CHAIN_ID) -> Self:
        # noinspection PyTypeChecker
        return cls(encoded_data, COLLATERAL[chain_id.name])

    @classmethod
    def from_erc1155(
        cls, encoded_data: str, neg_risk: bool, chain_id: CHAIN_ID
    ) -> Self:
        # noinspection PyTypeChecker
        return cls(
            encoded_data,
            NEGATIVE_RISK_ADAPTER[chain_id.name]
            if neg_risk
            else CONDITIONAL[chain_id.name],
        )

    def to_dict(self) -> dict:
        return {
            "to": self.to,
            "typeCode": 1,
            "data": self.encoded_data,
            "value": 0,
        }


# noinspection PyTypeHints
def generate_txn_params(
    w3: W3POA,
    proxy_frames: ProxyDataFrame | list[ProxyDataFrame],
    gas_price_wei: Wei,
    nonce: int,
    proxy_wallet_factory_addr: str | Literal[PROXY_WALLET_FACTORY],
) -> TxParams:
    factory = _proxy_contract(w3, proxy_wallet_factory_addr)

    if not isinstance(proxy_frames, list):
        proxy_frames = [proxy_frames]

    proxy_txns = [px.to_dict() for px in proxy_frames]

    # noinspection PyTypeChecker
    return factory.functions.proxy(proxy_txns).build_transaction(
        {
            "from": w3.wallet_addr,
            "gasPrice": gas_price_wei,
            "nonce": nonce,
        }
    )


def transact_txn(
    w3: W3POA,
    txn: TxParams,
    private_key: str | PrivateKeyType | PrivateKey,
    receipt_timeout: float | None,
) -> tuple[HexBytes, TxReceipt | None]:
    signed_txn = w3.eth.account.sign_transaction(txn, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

    if receipt_timeout is not None:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, receipt_timeout)
    else:
        receipt = None

    return tx_hash, receipt
