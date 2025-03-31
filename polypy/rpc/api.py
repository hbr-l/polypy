import warnings
from collections import namedtuple
from typing import Literal

from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from hexbytes import HexBytes
from web3.types import TxReceipt

from polypy.constants import (
    COLLATERAL,
    CONDITIONAL_TOKENS,
    ENDPOINT,
    NEGATIVE_RISK_ADAPTER,
    PROXY_WALLET_FACTORY,
    RELAY,
    RELAY_HUB,
)
from polypy.rpc.encode import (
    encode_convert,
    encode_merge,
    encode_redeem,
    encode_redeem_neg_risk,
    encode_split,
)
from polypy.rpc.relay import generate_payload, submit
from polypy.rpc.tx import (
    W3POA,
    approve_erc20,
    estimate_gas,
    estimate_gas_price_wei,
    generate_transaction_hash,
    transact_txn,
)
from polypy.signing import private_key_checksum_address
from polypy.structs import RelayerResponse
from polypy.typing import NumericAlias

# todo integrate into position manager
# todo neg_risk: bool -> get_neg_risk in pos_manager


# todo test
# todo transaction kwargs namedtuple?, approval kwargs


_approval_excl_args = namedtuple(
    "_approval_excl_args",
    ["approve_amount_usdc", "auto_approve_erc20", "approve_collateral"],
)
_relayer_excl_args = namedtuple(
    "_relayer_excl_args",
    [
        "maker_funder",
        "max_gas_limit",
        "relay_hub",
        "relay",
    ],
)


# noinspection PyTypeHints
def _prepare_relayer_payload(
    data: str,
    w3: W3POA,
    neg_risk: bool,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_price_wei: NumericAlias,
    max_gas_limit: NumericAlias | None,
    relay_hub: str | RELAY_HUB,
    relay: str | RELAY,
    conditional: str | CONDITIONAL_TOKENS,
    proxy_wallet_factory: str | Literal[PROXY_WALLET_FACTORY],
    neg_risk_adapter: str | NEGATIVE_RISK_ADAPTER,
) -> dict:
    txn = generate_transaction_hash(
        w3,
        data,
        neg_risk,
        gas_price_wei,
        private_key,
        proxy_wallet_factory,
        neg_risk_adapter,
        conditional,
    )

    gas_limit = estimate_gas(w3, txn)
    if max_gas_limit is not None:
        gas_limit = min(gas_limit, max_gas_limit)

    return generate_payload(
        private_key,
        maker_funder,
        proxy_wallet_factory,
        data,
        gas_limit,
        w3.eth.get_transaction_count(private_key_checksum_address(private_key)),
        relay_hub,
        relay,
        0,
        0,
    )


def _auto_approve_erc20(
    w3: W3POA,
    private_key: PrivateKey | str | PrivateKeyType,
    conditional: str | CONDITIONAL_TOKENS,
    receipt_timeout: float | None,
    approval_args: _approval_excl_args,
) -> tuple[HexBytes | None, TxReceipt | None,]:
    if approval_args.auto_approve_erc20:
        _, tx_hash_erc20, receipt_erc20 = approve_erc20(
            w3,
            approval_args.approve_amount_usdc,
            private_key,
            conditional,
            approval_args.approve_collateral,
            receipt_timeout,
        )
    else:
        tx_hash_erc20, receipt_erc20 = None, None

    return tx_hash_erc20, receipt_erc20


# noinspection PyTypeHints
def _transact(
    data: str,
    w3: W3POA,
    neg_risk: bool,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    allow_fallback_unrelayed: bool,
    max_gas_price: NumericAlias | None,
    max_gas_limit: NumericAlias | None,
    endpoint_relayer: str | ENDPOINT | None,
    relay_hub: str | RELAY_HUB,
    relay: str | RELAY,
    conditional: str | CONDITIONAL_TOKENS,
    proxy_wallet_factory: str | Literal[PROXY_WALLET_FACTORY],
    neg_risk_adapter: str | NEGATIVE_RISK_ADAPTER,
    receipt_timeout: float | None,
    approval_args: _approval_excl_args,
) -> tuple[
    RelayerResponse | None,
    HexBytes | None,
    TxReceipt | None,
    HexBytes | None,
    TxReceipt | None,
]:
    gas_price_wei = estimate_gas_price_wei(w3) * gas_factor

    if max_gas_price is not None:
        gas_price_wei = min(gas_price_wei, max_gas_price)

    if endpoint_relayer is not None:
        payload = _prepare_relayer_payload(
            data,
            w3,
            neg_risk,
            private_key,
            maker_funder,
            gas_price_wei,
            max_gas_limit,
            relay_hub,
            relay,
            conditional,
            proxy_wallet_factory,
            neg_risk_adapter,
        )

        try:
            resp = submit(endpoint_relayer, payload)
            return resp, None, None, None, None
        except Exception as e:
            if not allow_fallback_unrelayed:
                raise e
            else:
                warnings.warn(
                    f"Relayer failed for transaction: {payload}. "
                    f"Falling back to direct on-chain transaction including gas costs. "
                    f"Traceback: {str(e)}"
                )

    tx_hash_erc20, receipt_erc20 = _auto_approve_erc20(
        w3, private_key, conditional, receipt_timeout, approval_args
    )

    # we don't know if the relayer increases the nonce, so for safety’s sake,
    # we recompute the transaction hash (though, erc20 approval will for sure increase nonce)
    txn = generate_transaction_hash(
        w3,
        data,
        neg_risk,
        gas_price_wei,
        private_key,
        proxy_wallet_factory,
        neg_risk_adapter,
        conditional,
    )

    # no relayer or fallback
    tx_hash, tx_receipt = transact_txn(w3, txn, private_key, receipt_timeout)

    return None, tx_hash, tx_receipt, tx_hash_erc20, receipt_erc20


# noinspection PyTypeHints
def split_position(
    w3: W3POA,
    condition_id: str,
    amount_usdc: NumericAlias,
    neg_risk: bool,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    allow_fallback_unrelayed: bool,
    auto_approve_erc20: bool,
    max_gas_price: NumericAlias | None,
    max_gas_limit: NumericAlias | None,
    endpoint_relayer: str | ENDPOINT | None,
    relay_hub: str | RELAY_HUB,
    relay: str | RELAY,
    collateral: str | COLLATERAL,
    conditional: str | CONDITIONAL_TOKENS,
    proxy_wallet_factory: str | Literal[PROXY_WALLET_FACTORY],
    neg_risk_adapter: str | NEGATIVE_RISK_ADAPTER,
    receipt_timeout: float | None,
) -> tuple[
    RelayerResponse | None,
    HexBytes | None,
    TxReceipt | None,
    HexBytes | None,
    TxReceipt | None,
]:
    data = encode_split(condition_id, amount_usdc, collateral)
    approval_args = _approval_excl_args(amount_usdc, auto_approve_erc20, collateral)

    return _transact(
        data,
        w3,
        neg_risk,
        private_key,
        maker_funder,
        gas_factor,
        allow_fallback_unrelayed,
        max_gas_price,
        max_gas_limit,
        endpoint_relayer,
        relay_hub,
        relay,
        conditional,
        proxy_wallet_factory,
        neg_risk_adapter,
        receipt_timeout,
        approval_args,
    )


# noinspection PyTypeHints
def merge_position(
    w3: W3POA,
    condition_id: str,
    size: NumericAlias,
    neg_risk: bool,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    allow_fallback_unrelayed: bool,
    max_gas_price: NumericAlias | None,
    max_gas_limit: NumericAlias | None,
    endpoint_relayer: str | ENDPOINT | None,
    relay_hub: str | RELAY_HUB,
    relay: str | RELAY,
    collateral: str | COLLATERAL,
    conditional: str | CONDITIONAL_TOKENS,
    proxy_wallet_factory: str | Literal[PROXY_WALLET_FACTORY],
    neg_risk_adapter: str | NEGATIVE_RISK_ADAPTER,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None, None, None,]:
    data = encode_merge(condition_id, size, collateral)
    approval_args = _approval_excl_args(0, False, None)

    return _transact(
        data,
        w3,
        neg_risk,
        private_key,
        maker_funder,
        gas_factor,
        allow_fallback_unrelayed,
        max_gas_price,
        max_gas_limit,
        endpoint_relayer,
        relay_hub,
        relay,
        conditional,
        proxy_wallet_factory,
        neg_risk_adapter,
        receipt_timeout,
        approval_args,
    )


# noinspection PyTypeHints
def redeem_position(
    w3: W3POA,
    condition_id: str,
    size_yes_neg_risk: NumericAlias | None,
    size_no_neg_risk: NumericAlias | None,
    neg_risk: bool,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    allow_fallback_unrelayed: bool,
    max_gas_price: NumericAlias | None,
    max_gas_limit: NumericAlias | None,
    endpoint_relayer: str | ENDPOINT | None,
    relay_hub: str | RELAY_HUB,
    relay: str | RELAY,
    collateral: str | COLLATERAL,
    conditional: str | CONDITIONAL_TOKENS,
    proxy_wallet_factory: str | Literal[PROXY_WALLET_FACTORY],
    neg_risk_adapter: str | NEGATIVE_RISK_ADAPTER,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None, None, None,]:
    if neg_risk:
        data = encode_redeem_neg_risk(condition_id, size_yes_neg_risk, size_no_neg_risk)
    else:
        data = encode_redeem(condition_id, collateral)

    approval_args = _approval_excl_args(0, False, None)

    return _transact(
        data,
        w3,
        neg_risk,
        private_key,
        maker_funder,
        gas_factor,
        allow_fallback_unrelayed,
        max_gas_price,
        max_gas_limit,
        endpoint_relayer,
        relay_hub,
        relay,
        conditional,
        proxy_wallet_factory,
        neg_risk_adapter,
        receipt_timeout,
        approval_args,
    )


# noinspection PyTypeHints
def convert_position(
    w3: W3POA,
    condition_id: str,
    size: NumericAlias,
    question_ids: list[str],
    neg_risk: bool,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    allow_fallback_unrelayed: bool,
    max_gas_price: NumericAlias | None,
    max_gas_limit: NumericAlias | None,
    endpoint_relayer: str | ENDPOINT | None,
    relay_hub: str | RELAY_HUB,
    relay: str | RELAY,
    conditional: str | CONDITIONAL_TOKENS,
    proxy_wallet_factory: str | Literal[PROXY_WALLET_FACTORY],
    neg_risk_adapter: str | NEGATIVE_RISK_ADAPTER,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None, None, None,]:
    data = encode_convert(condition_id, question_ids, size)
    approval_args = _approval_excl_args(0, False, None)

    return _transact(
        data,
        w3,
        neg_risk,
        private_key,
        maker_funder,
        gas_factor,
        allow_fallback_unrelayed,
        max_gas_price,
        max_gas_limit,
        endpoint_relayer,
        relay_hub,
        relay,
        conditional,
        proxy_wallet_factory,
        neg_risk_adapter,
        receipt_timeout,
        approval_args,
    )
