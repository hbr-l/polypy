import logging
import warnings
from decimal import Decimal

from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from eth_utils import from_wei
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract
from web3.types import TxParams, TxReceipt, Wei

from polypy.constants import (
    CHAIN_ID,
    COLLATERAL,
    CONDITIONAL,
    ENDPOINT,
    ERC20_WEI_UNIT,
    POL_WEI_UNIT,
    PROXY_WALLET_FACTORY,
    RELAY_HUB,
)
from polypy.rpc.encode import (
    _usdc_to_dec_plus_marginWei,
    _usdc_to_wei_plus_marginWei,
    encode_approve,
    encode_convert,
    encode_merge,
    encode_redeem,
    encode_redeem_neg_risk,
    encode_split,
)
from polypy.rpc.mtx_protocols import (
    ConvertTransaction,
    MergeTransaction,
    RedeemTransaction,
    SplitTransaction,
)
from polypy.rpc.relay import generate_payload, get_relay_info, submit
from polypy.rpc.tx import (
    W3POA,
    ProxyDataFrame,
    _erc20_contract,
    _erc1155_contract,
    _get_gas_price_wei,
    generate_txn_params,
    transact_txn,
)
from polypy.structs import RelayerResponse
from polypy.typing import NumericAlias


def _prepare_relayer_payload(
    txn_params: TxParams,
    w3: W3POA,
    endpoint_relayer: str | ENDPOINT,
    cookies: dict[str, str],
    private_key: str | PrivateKeyType | PrivateKey,
    maker_funder: str,
    max_gas_limit: int | None,
) -> dict:
    relay_address, relay_nonce = get_relay_info(
        endpoint=endpoint_relayer, addr=w3.wallet_addr, cookies=cookies
    )

    gas_limit = w3.eth.estimate_gas(txn_params)
    if max_gas_limit is not None:
        gas_limit = min(gas_limit, max_gas_limit)

    return generate_payload(
        addr=w3.wallet_addr,
        private_key=private_key,
        funder_address=maker_funder,
        proxy_wallet_factory_addr=PROXY_WALLET_FACTORY,
        data=txn_params["data"],
        gas_limit=gas_limit,
        nonce=relay_nonce,
        relay_hub_addr=RELAY_HUB,
        relay_addr=relay_address,
        relayer_fee=0,
        gas_price_wei=0,
    )


def _transact(
    proxy_data: ProxyDataFrame | list[ProxyDataFrame],
    approve_usdc: NumericAlias | None,
    w3: W3POA,
    chain_id: CHAIN_ID,
    private_key: str | PrivateKeyType | PrivateKey,
    maker_funder: str,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    allow_fallback_unrelayed: bool,
    endpoint_relayer: str | ENDPOINT | None,
    cookies: dict[str, str] | None,
    max_gas_limit_relayer: int | None,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None,]:
    gas_price_wei = _get_gas_price_wei(
        w3=w3, gas_factor=gas_factor, max_gas_price=max_gas_price
    )

    if not isinstance(proxy_data, list):
        proxy_data = [proxy_data]

    # noinspection PyTypeChecker
    txn = generate_txn_params(
        w3=w3,
        proxy_frames=proxy_data,
        nonce=get_nonce(w3),
        gas_price_wei=gas_price_wei,
        proxy_wallet_factory_addr=PROXY_WALLET_FACTORY,
    )

    if endpoint_relayer is not None:
        try:
            payload = _prepare_relayer_payload(
                txn_params=txn,
                w3=w3,
                endpoint_relayer=endpoint_relayer,
                cookies=cookies,
                private_key=private_key,
                maker_funder=maker_funder,
                max_gas_limit=max_gas_limit_relayer,
            )
            resp = submit(endpoint=endpoint_relayer, payload=payload, cookies=cookies)
            return resp, None, None
        except Exception as e:
            if not allow_fallback_unrelayed:
                raise e
            else:
                warnings.warn(
                    f"Relayer failed for transaction: {txn}. "
                    f"Falling back to on-chain RPC transaction inducing gas costs. "
                    f"Traceback: {str(e)}"
                )

    nonce = get_nonce(w3)
    # noinspection PyTypedDict
    txn["nonce"] = nonce

    # noinspection PyTypeChecker
    if approve_usdc is not None and not is_sufficient_approval_erc20(
        w3=w3, amount=approve_usdc, chain_id=chain_id
    ):
        # noinspection PyTypeChecker
        proxy_approval = ProxyDataFrame.from_erc20(
            encoded_data=encode_approve(CONDITIONAL[chain_id.name], approve_usdc),
            chain_id=chain_id,
        )

        # proxy_data: list[ProxyDataFrame]
        # [proxy_approval] + proxy_data -> [proxy_approval, *proxy_data]
        txn = generate_txn_params(
            w3=w3,
            proxy_frames=[proxy_approval] + proxy_data,
            nonce=nonce,
            gas_price_wei=gas_price_wei,
            proxy_wallet_factory_addr=PROXY_WALLET_FACTORY,
        )

    # no relayer or fallback: send transaction directly via rpc
    tx_hash, tx_receipt = transact_txn(w3, txn, private_key, receipt_timeout)

    return None, tx_hash, tx_receipt


def split_positions(
    w3: W3POA,
    condition_id: str,
    amount_usdc: NumericAlias,
    neg_risk: bool,
    chain_id: CHAIN_ID,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    allow_fallback_unrelayed: bool,
    endpoint_relayer: str | ENDPOINT | None,
    max_gas_limit_relayer: int | None,
    polymarketnonce: str | None,
    polymarketsession: str | None,
    polymarketauthtype: str | None,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    # noinspection PyTypeChecker
    data = encode_split(
        condition_id=condition_id,
        amount_usdc=amount_usdc,
        collateral=COLLATERAL[chain_id.name],
    )

    cookies = {
        "polymarketnonce": polymarketnonce,
        "polymarketsession": polymarketsession,
        "polymarketauthtype": polymarketauthtype,
    }

    return _transact(
        proxy_data=ProxyDataFrame.from_erc1155(
            encoded_data=data, neg_risk=neg_risk, chain_id=chain_id
        ),
        approve_usdc=amount_usdc,
        w3=w3,
        endpoint_relayer=endpoint_relayer,
        cookies=cookies,
        chain_id=chain_id,
        private_key=private_key,
        maker_funder=maker_funder,
        gas_factor=gas_factor,
        max_gas_price=max_gas_price,
        max_gas_limit_relayer=max_gas_limit_relayer,
        allow_fallback_unrelayed=allow_fallback_unrelayed,
        receipt_timeout=receipt_timeout,
    )


def merge_positions(
    w3: W3POA,
    condition_id: str,
    size: NumericAlias,
    neg_risk: bool,
    chain_id: CHAIN_ID,
    private_key: str | PrivateKey | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    allow_fallback_unrelayed: bool,
    endpoint_relayer: str | ENDPOINT | None,
    max_gas_limit_relayer: int | None,
    polymarketnonce: str | None,
    polymarketsession: str | None,
    polymarketauthtype: str | None,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    # noinspection PyTypeChecker
    data = encode_merge(
        condition_id=condition_id, size=size, collateral=COLLATERAL[chain_id.name]
    )

    # noinspection DuplicatedCode
    cookies = {
        "polymarketnonce": polymarketnonce,
        "polymarketsession": polymarketsession,
        "polymarketauthtype": polymarketauthtype,
    }

    return _transact(
        proxy_data=ProxyDataFrame.from_erc1155(
            encoded_data=data, neg_risk=neg_risk, chain_id=chain_id
        ),
        approve_usdc=None,
        w3=w3,
        endpoint_relayer=endpoint_relayer,
        cookies=cookies,
        chain_id=chain_id,
        private_key=private_key,
        maker_funder=maker_funder,
        gas_factor=gas_factor,
        max_gas_price=max_gas_price,
        max_gas_limit_relayer=max_gas_limit_relayer,
        allow_fallback_unrelayed=allow_fallback_unrelayed,
        receipt_timeout=receipt_timeout,
    )


def redeem_positions(
    w3: W3POA,
    condition_id: str,
    size_yes_neg_risk: NumericAlias | None,
    size_no_neg_risk: NumericAlias | None,
    neg_risk: bool,
    chain_id: CHAIN_ID,
    private_key: str | PrivateKey | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    allow_fallback_unrelayed: bool,
    endpoint_relayer: str | ENDPOINT | None,
    max_gas_limit_relayer: int | None,
    polymarketnonce: str | None,
    polymarketsession: str | None,
    polymarketauthtype: str | None,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    if neg_risk:
        data = encode_redeem_neg_risk(condition_id, size_yes_neg_risk, size_no_neg_risk)
    else:
        # noinspection PyTypeChecker
        data = encode_redeem(condition_id, COLLATERAL[chain_id.name])

    # noinspection DuplicatedCode
    cookies = {
        "polymarketnonce": polymarketnonce,
        "polymarketsession": polymarketsession,
        "polymarketauthtype": polymarketauthtype,
    }

    return _transact(
        proxy_data=ProxyDataFrame.from_erc1155(
            encoded_data=data, neg_risk=neg_risk, chain_id=chain_id
        ),
        approve_usdc=None,
        w3=w3,
        endpoint_relayer=endpoint_relayer,
        cookies=cookies,
        chain_id=chain_id,
        private_key=private_key,
        maker_funder=maker_funder,
        gas_factor=gas_factor,
        max_gas_price=max_gas_price,
        max_gas_limit_relayer=max_gas_limit_relayer,
        allow_fallback_unrelayed=allow_fallback_unrelayed,
        receipt_timeout=receipt_timeout,
    )


def convert_positions(
    w3: W3POA,
    neg_risk_market_id: str,
    size: NumericAlias,
    question_ids: list[str],
    chain_id: CHAIN_ID,
    private_key: str | PrivateKey | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    allow_fallback_unrelayed: bool,
    endpoint_relayer: str | ENDPOINT | None,
    max_gas_limit_relayer: int | None,
    polymarketnonce: str | None,
    polymarketsession: str | None,
    polymarketauthtype: str | None,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    data = encode_convert(neg_risk_market_id, question_ids, size)

    cookies = {
        "polymarketnonce": polymarketnonce,
        "polymarketsession": polymarketsession,
        "polymarketauthtype": polymarketauthtype,
    }

    return _transact(
        proxy_data=ProxyDataFrame.from_erc1155(
            encoded_data=data, neg_risk=True, chain_id=chain_id
        ),
        approve_usdc=None,
        w3=w3,
        endpoint_relayer=endpoint_relayer,
        cookies=cookies,
        chain_id=chain_id,
        private_key=private_key,
        maker_funder=maker_funder,
        gas_factor=gas_factor,
        max_gas_price=max_gas_price,
        max_gas_limit_relayer=max_gas_limit_relayer,
        allow_fallback_unrelayed=allow_fallback_unrelayed,
        receipt_timeout=receipt_timeout,
    )


def batch_operate_positions(
    w3: W3POA,
    batch_txn: list[
        SplitTransaction | MergeTransaction | RedeemTransaction | ConvertTransaction
    ],
    chain_id: CHAIN_ID,
    private_key: PrivateKey | str | PrivateKeyType,
    maker_funder: str,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    allow_fallback_unrelayed: bool,
    endpoint_relayer: str | ENDPOINT | None,
    max_gas_limit_relayer: int | None,
    polymarketnonce: str | None,
    polymarketsession: str | None,
    polymarketauthtype: str | None,
    receipt_timeout: float | None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    approve_usdc = sum(
        t.amount_usdc for t in batch_txn if isinstance(t, SplitTransaction)
    )

    if approve_usdc == 0:
        approve_usdc = None
    else:
        logging.info(
            "Approving cumulative USDC amount of all SplitTransactions at once. "
            "This might lead to over-approval (but is uncritical for executing transaction). "
            "You might want to reset approval afterwards manually via approve_erc20(...)."
        )

    cookies = {
        "polymarketnonce": polymarketnonce,
        "polymarketsession": polymarketsession,
        "polymarketauthtype": polymarketauthtype,
    }

    proxy_frames = [
        ProxyDataFrame.from_erc1155(
            encoded_data=t.encode(chain_id=chain_id),
            neg_risk=t.neg_risk,
            chain_id=chain_id,
        )
        for t in batch_txn
    ]

    return _transact(
        proxy_data=proxy_frames,
        approve_usdc=approve_usdc,
        w3=w3,
        endpoint_relayer=endpoint_relayer,
        cookies=cookies,
        chain_id=chain_id,
        private_key=private_key,
        maker_funder=maker_funder,
        gas_factor=gas_factor,
        max_gas_price=max_gas_price,
        max_gas_limit_relayer=max_gas_limit_relayer,
        allow_fallback_unrelayed=allow_fallback_unrelayed,
        receipt_timeout=receipt_timeout,
    )


# noinspection PyPep8Naming
def approve_USDC(
    w3: W3POA,
    amount: NumericAlias,
    private_key: str | PrivateKeyType | PrivateKey,
    chain_id: CHAIN_ID,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    receipt_timeout: float | None,
) -> tuple[HexBytes, TxReceipt]:
    # noinspection PyTypeChecker
    contract = _erc20_contract(w3, COLLATERAL[chain_id.name])

    gas_price_wei = _get_gas_price_wei(w3, gas_factor, max_gas_price)

    txn = contract.functions.approve(
        _spender=CONDITIONAL[chain_id.name], _value=_usdc_to_wei_plus_marginWei(amount)
    ).build_transaction(
        {"from": w3.wallet_addr, "gasPrice": gas_price_wei, "nonce": get_nonce(w3)}
    )

    return transact_txn(w3, txn, private_key, receipt_timeout)


# noinspection PyPep8Naming
def auto_approve_USDC(
    w3: W3POA,
    amount: NumericAlias,
    private_key: str | PrivateKeyType | PrivateKey,
    chain_id: CHAIN_ID,
    gas_factor: NumericAlias,
    max_gas_price: int | None,
    receipt_timeout: float | None,
) -> tuple[HexBytes | None, TxReceipt | None]:
    if not is_sufficient_approval_erc20(w3=w3, amount=amount, chain_id=chain_id):
        # not sufficient approval, so we reset approval
        return approve_USDC(
            w3=w3,
            amount=amount,
            private_key=private_key,
            chain_id=chain_id,
            gas_factor=gas_factor,
            max_gas_price=max_gas_price,
            receipt_timeout=receipt_timeout,
        )

    # else: sufficient approval, no need to do anything

    return None, None


def estimate_gas_price_wei(w3: Web3) -> Wei:
    # todo w3.eth.set_gas_price_strategy(strat) and w3.eth.generate_gas_price()
    #   i.e., with strat from web3.gas_strategies.time_based import medium_gas_price_strategy,
    #   but more sophisticated gas price strategies take quite some time for sampling last blocks,
    #   so might not be so beneficial after all

    return w3.eth.gas_price


# noinspection PyPep8Naming
def get_allowance_USDC(
    w3: W3POA,
    chain_id: CHAIN_ID,
    contract: Contract | None = None,
) -> Decimal:
    if contract is None:
        # noinspection PyTypeChecker
        contract = _erc20_contract(w3, COLLATERAL[chain_id.name])

    return from_wei(
        contract.functions.allowance(
            _owner=w3.wallet_addr, _spender=CONDITIONAL[chain_id.name]
        ).call(),
        ERC20_WEI_UNIT,
    )


def is_sufficient_approval_erc20(
    w3: W3POA,
    amount: NumericAlias,
    chain_id: CHAIN_ID,
) -> bool:
    # noinspection PyTypeChecker
    contract = _erc20_contract(w3, COLLATERAL[chain_id.name])

    # raw_amount = int(amount * (10**decimals_)) + 1    # decimals_ == 6, so just use "mwei"
    requested_amount: Decimal = _usdc_to_dec_plus_marginWei(amount)
    approved_amount: Decimal = get_allowance_USDC(
        w3=w3,
        chain_id=chain_id,
        contract=contract,
    )

    return approved_amount >= requested_amount


def get_nonce(w3: W3POA) -> int:
    return w3.eth.get_transaction_count(w3.wallet_addr)


# noinspection PyPep8Naming
def get_balance_POL(
    w3: W3POA,
    unit: str = POL_WEI_UNIT,
) -> Decimal:
    return from_wei(w3.eth.get_balance(w3.wallet_addr), unit)


def get_balance_token(
    w3: W3POA,
    token_id: str | int,
    chain_id: CHAIN_ID,
    unit: str = ERC20_WEI_UNIT,
) -> Decimal:
    # noinspection PyTypeChecker
    contract = _erc1155_contract(w3, CONDITIONAL[chain_id.name])
    balance = contract.functions.balanceOf(w3.maker_funder, int(token_id)).call()

    return from_wei(balance, unit)


# noinspection PyPep8Naming
def get_balance_USDC(
    w3: W3POA,
    chain_id: CHAIN_ID,
    unit: str = ERC20_WEI_UNIT,
) -> Decimal:
    # noinspection PyTypeChecker
    contract = _erc20_contract(w3, COLLATERAL[chain_id.name])
    balance = contract.functions.balanceOf(w3.maker_funder).call()

    return from_wei(balance, unit)
