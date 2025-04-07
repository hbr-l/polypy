from decimal import Decimal

from eth_utils import from_wei, to_bytes, to_checksum_address, to_wei
from web3 import Web3

from polypy.constants import (
    APPROVAL_MARGIN_WEI,
    COLLATERAL,
    CONDITIONAL,
    ERC20_WEI_UNIT,
    ZERO_HASH,
)
from polypy.rpc.abi import CTF_ABI, ERC20_ABI, NEGATIVE_RISK_ADAPTER_ABI
from polypy.typing import NumericAlias, dec

CTF_INTERFACE = Web3().eth.contract(abi=CTF_ABI)
NEG_RISK_INTERFACE = Web3().eth.contract(abi=NEGATIVE_RISK_ADAPTER_ABI)
ERC20_INTERFACE = Web3().eth.contract(abi=ERC20_ABI)


# noinspection PyPep8Naming
def _usdc_to_wei_plus_marginWei(amount_usdc: NumericAlias) -> int:
    return to_wei(amount_usdc, ERC20_WEI_UNIT) + APPROVAL_MARGIN_WEI


# noinspection PyPep8Naming
def _usdc_to_dec_plus_marginWei(amount_usdc: NumericAlias) -> Decimal:
    return dec(amount_usdc) + from_wei(APPROVAL_MARGIN_WEI, ERC20_WEI_UNIT)


def encode_approve(
    conditional_token: str | CONDITIONAL, amount_usdc: NumericAlias
) -> str:
    approve_fn = ERC20_INTERFACE.functions.approve

    # noinspection PyProtectedMember
    return approve_fn(
        conditional_token, _usdc_to_wei_plus_marginWei(amount_usdc)
    )._encode_transaction_data()


def encode_split(
    condition_id: str, amount_usdc: NumericAlias, collateral: str | COLLATERAL
) -> str:
    """

    Parameters
    ----------
    collateral: str,
        collateral token address, usually COLLATERAL.POLYGON
    condition_id: str,
        condition ID of market/ market ID
    amount_usdc: NumericAlias
        amount in USDC, will be parsed to mwei unit

    Returns
    -------

    """
    # encode the function call
    split_position_fn = CTF_INTERFACE.functions.splitPosition

    # noinspection PyProtectedMember
    return split_position_fn(
        to_checksum_address(collateral),
        to_bytes(hexstr=ZERO_HASH),  # HashZero equivalent
        condition_id,
        [1, 2],  # Standard partition for binary outcomes
        to_wei(amount_usdc, ERC20_WEI_UNIT),
    )._encode_transaction_data()


def encode_merge(
    condition_id: str, size: NumericAlias, collateral: str | COLLATERAL
) -> str:
    merge_position_fn = CTF_INTERFACE.functions.mergePositions

    # noinspection PyProtectedMember
    return merge_position_fn(
        to_checksum_address(collateral),
        to_bytes(hexstr=ZERO_HASH),  # HashZero equivalent
        condition_id,
        [1, 2],  # Standard partition for binary outcomes
        to_wei(size, ERC20_WEI_UNIT),
    )._encode_transaction_data()


def encode_redeem(condition_id: str, collateral: str | COLLATERAL) -> str:
    redeem_position_fn = CTF_INTERFACE.functions.redeemPositions

    # noinspection PyProtectedMember
    return redeem_position_fn(
        to_checksum_address(collateral),
        to_bytes(hexstr=ZERO_HASH),  # HashZero equivalent
        condition_id,
        [1, 2],  # Standard partition for binary outcomes
    )._encode_transaction_data()


def encode_redeem_neg_risk(
    condition_id: str, size_yes: NumericAlias, size_no: NumericAlias
) -> str:
    redeem_position_fn = NEG_RISK_INTERFACE.functions.redeemPositions

    # noinspection PyProtectedMember
    return redeem_position_fn(
        condition_id,
        # yes tokens | no tokens
        [to_wei(size_yes, ERC20_WEI_UNIT), to_wei(int(size_no), ERC20_WEI_UNIT)],
    )._encode_transaction_data()


def _get_index_set(question_ids: list[str]) -> int:
    indices = [int(idx[-2:], 16) for idx in question_ids]
    unique_indices = list(set(indices))
    return sum(1 << idx for idx in unique_indices)


def encode_convert(
    neg_risk_market_id: str, question_ids: list[str], size: NumericAlias
) -> str:
    index_set = _get_index_set(question_ids)

    convert_position_fn = NEG_RISK_INTERFACE.functions.convertPositions

    # noinspection PyProtectedMember
    return convert_position_fn(
        neg_risk_market_id,
        str(index_set),
        to_wei(size, ERC20_WEI_UNIT),
    )._encode_transaction_data()
