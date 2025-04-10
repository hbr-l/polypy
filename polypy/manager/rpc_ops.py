from typing import TYPE_CHECKING, Literal, Self

import attrs
from hexbytes import HexBytes
from web3.types import TxReceipt

from polypy.ctf import MarketIdQuintet, MarketIdTriplet
from polypy.exceptions import (
    PolyPyException,
    PositionTransactionException,
    RelayerException,
)
from polypy.manager.cache import ConversionCacheProtocol
from polypy.manager.rpc_proc import (
    RPCSettings,
    _assert_redeem_sizes_onchain,
    _parse_outcome,
    _tx_post_convert_positions,
    _tx_post_merge_positions,
    _tx_post_redeem_positions,
    _tx_post_split_positions,
    _tx_pre_convert_position,
    _tx_pre_merge_positions,
    _tx_pre_redeem_positions,
    _tx_pre_split_position,
)
from polypy.rest.api import get_neg_risk
from polypy.rpc.api import (
    batch_operate_positions,
    convert_positions,
    merge_positions,
    redeem_positions,
    split_positions,
)
from polypy.rpc.mtx_protocols import (
    MTX_TYPE,
    ConvertTransaction,
    MergeTransaction,
    SplitTransaction,
)
from polypy.structs import RelayerResponse
from polypy.typing import NumericAlias

if TYPE_CHECKING:
    from polypy.manager.position import PositionManagerProtocol


@attrs.define
class MTX:
    type: MTX_TYPE
    market_triplet: MarketIdTriplet | None = None
    amount: NumericAlias | None = None
    size: NumericAlias | None = None
    outcome: Literal["YES", "NO"] | None = None
    cvt_market_quintets: MarketIdQuintet | list[MarketIdQuintet] | None = None
    all_market_quintets: list[MarketIdQuintet] | None = None
    neg_risk: bool | None = None

    @classmethod
    def split(
        cls,
        market_triplet: MarketIdTriplet,
        amount: NumericAlias,
        neg_risk: bool | None = None,
    ) -> Self:
        return cls(
            type=MTX_TYPE.SPLIT,
            market_triplet=market_triplet,
            amount=amount,
            neg_risk=neg_risk,
        )

    @classmethod
    def merge(
        cls,
        market_triplet: MarketIdTriplet,
        size: NumericAlias | None,
        neg_risk: bool | None = None,
    ) -> Self:
        return cls(
            type=MTX_TYPE.MERGE,
            market_triplet=market_triplet,
            size=size,
            neg_risk=neg_risk,
        )

    @classmethod
    def convert(
        cls,
        cvt_market_quintets: MarketIdQuintet | list[MarketIdQuintet],
        all_market_quintets: list[MarketIdQuintet] | None,
        size: NumericAlias | None,
    ) -> Self:
        return cls(
            type=MTX_TYPE.CONVERT,
            cvt_market_quintets=cvt_market_quintets,
            all_market_quintets=all_market_quintets,
            size=size,
        )


def check_relay_modus(
    rpc_settings: RPCSettings,
    polymarketnonce: str | None,
    polymarketsession: str | None,
    polymarketauthtype: str | None,
) -> None:
    if (
        rpc_settings.endpoint_relayer is None
        and polymarketnonce is not None
        and polymarketsession is not None
        and polymarketauthtype is not None
    ):
        raise RelayerException(
            "'polymarketnonce', 'polymarketsession' and 'polymarketauthtype' defined "
            "but 'endpoint_relayer' is None. Cannot send transaction to relayer. "
            "Did not initialize any transaction."
        )


def _check_onchain_response(
    resp: RelayerResponse, txn_hash: HexBytes, txn_receipt: TxReceipt
) -> None:
    if resp is None and txn_receipt is None:
        raise PositionTransactionException(
            f"Failed to transact position! resp={resp}, txn_hash={txn_hash}, txn_receipt={txn_receipt}."
        )

    if txn_receipt is not None and txn_receipt["status"] != 1:
        raise PositionTransactionException(
            f"Failed to transact position! resp={resp}, txn_hash={txn_hash}, txn_receipt={txn_receipt}."
        )

    if resp is not None and resp.state != "STATE_NEW":
        raise PositionTransactionException(
            f"Failed to transact position! resp={resp}, txn_hash={txn_hash}, txn_receipt={txn_receipt}."
        )


def _parse_split_mtx(
    mtx: MTX, rpc_settings: RPCSettings, position_manager: "PositionManagerProtocol"
) -> SplitTransaction:
    condition_id, amount_usdc, neg_risk = _tx_pre_split_position(
        rpc_settings=rpc_settings,
        position_manager=position_manager,
        market_triplet=mtx.market_triplet,
        amount=mtx.amount,
        neg_risk=mtx.neg_risk,
    )
    return SplitTransaction(
        condition_id=condition_id,
        amount_usdc=amount_usdc,
        neg_risk=neg_risk,
    )


def _parse_merge_mtx(
    mtx: MTX, rpc_settings: RPCSettings, position_manager: "PositionManagerProtocol"
) -> MergeTransaction:
    condition_id, size, neg_risk = _tx_pre_merge_positions(
        rpc_settings=rpc_settings,
        position_manager=position_manager,
        market_triplet=mtx.market_triplet,
        size=mtx.size,
        neg_risk=mtx.neg_risk,
    )
    return MergeTransaction(condition_id=condition_id, size=size, neg_risk=neg_risk)


def _parse_convert_mtx(
    mtx: MTX, rpc_settings: RPCSettings, position_manager: "PositionManagerProtocol"
) -> ConvertTransaction:
    (
        neg_risk_market_id,
        size,
        question_ids,
        all_markets_quintets,
    ) = _tx_pre_convert_position(
        rpc_settings=rpc_settings,
        position_manager=position_manager,
        cvt_market_quintets=mtx.cvt_market_quintets,
        all_market_quintets=mtx.all_market_quintets,
        size=mtx.size,
    )

    # update mtx    todo test
    mtx.all_market_quintets = all_markets_quintets

    return ConvertTransaction(
        neg_risk_market_id=neg_risk_market_id,
        size=size,
        question_ids=question_ids,
    )


def _tx_pre_batch_operate_positions(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    batch_mtx: list[MTX],
) -> list[SplitTransaction | MergeTransaction | ConvertTransaction]:
    transactions = []

    for mtx in batch_mtx:
        if mtx.type is MTX_TYPE.SPLIT:
            transactions.append(
                _parse_split_mtx(
                    mtx=mtx,
                    rpc_settings=rpc_settings,
                    position_manager=position_manager,
                )
            )
        elif mtx.type is MTX_TYPE.MERGE:
            transactions.append(
                _parse_merge_mtx(
                    mtx=mtx,
                    rpc_settings=rpc_settings,
                    position_manager=position_manager,
                )
            )
        elif mtx.type is MTX_TYPE.REDEEM:
            raise PolyPyException("REDEEM not allowed in batch operation.")
        elif mtx.type is MTX_TYPE.CONVERT:
            transactions.append(
                _parse_convert_mtx(
                    mtx=mtx,
                    rpc_settings=rpc_settings,
                    position_manager=position_manager,
                )
            )
        else:
            raise PolyPyException(f"Unknown mtx type. Got: {mtx.type}")

    return transactions


def _tx_post_batch_operate_positions(
    position_manager: "PositionManagerProtocol",
    conversion_cache: ConversionCacheProtocol,
    batch_mtx: list[MTX],
    batch_txn: list[SplitTransaction | MergeTransaction | ConvertTransaction],
) -> None:
    for mtx, ta in zip(batch_mtx, batch_txn):
        if mtx.type is MTX_TYPE.SPLIT:
            _tx_post_split_positions(
                position_manager=position_manager,
                market_triplet=mtx.market_triplet,
                amount=ta.amount_usdc,
            )
        elif mtx.type is MTX_TYPE.MERGE:
            _tx_post_merge_positions(
                position_manager=position_manager,
                market_triplet=mtx.market_triplet,
                size=ta.size,
            )
        elif mtx.type is MTX_TYPE.REDEEM:
            raise PolyPyException("REDEEM not allowed in batch operation.")
        elif mtx.type is MTX_TYPE.CONVERT:
            _tx_post_convert_positions(
                position_manager=position_manager,
                conversion_cache=conversion_cache,
                cvt_market_quintets=mtx.cvt_market_quintets,
                all_market_quintets=mtx.all_market_quintets,
                size=ta.size,
            )
        else:
            raise PolyPyException(f"Unknown mtx type. Got: {mtx.type}")


def tx_split_position(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
    amount: NumericAlias,
    neg_risk: bool | None = None,
    polymarketnonce: str | None = None,
    polymarketsession: str | None = None,
    polymarketauthtype: str | None = None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    check_relay_modus(
        rpc_settings=rpc_settings,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
    )

    condition_id, amount, neg_risk = _tx_pre_split_position(
        rpc_settings=rpc_settings,
        position_manager=position_manager,
        market_triplet=market_triplet,
        amount=amount,
        neg_risk=neg_risk,
    )

    resp, txn_hash, txn_receipt = split_positions(
        w3=rpc_settings.w3,
        condition_id=condition_id,
        amount_usdc=amount,
        neg_risk=neg_risk,
        chain_id=rpc_settings.chain_id,
        private_key=rpc_settings.private_key,
        maker_funder=rpc_settings.maker,
        gas_factor=rpc_settings.gas_factor,
        max_gas_price=rpc_settings.max_gas_price,
        allow_fallback_unrelayed=rpc_settings.allow_fallback_unrelayed,
        endpoint_relayer=rpc_settings.endpoint_relayer,
        max_gas_limit_relayer=rpc_settings.max_gas_limit_relayer,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
        receipt_timeout=rpc_settings.receipt_timeout,
    )

    # check response
    _check_onchain_response(resp, txn_hash, txn_receipt)

    _tx_post_split_positions(
        position_manager=position_manager, market_triplet=market_triplet, amount=amount
    )

    return resp, txn_hash, txn_receipt


def tx_merge_positions(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
    size: NumericAlias | None,
    neg_risk: bool | None = None,
    polymarketnonce: str | None = None,
    polymarketsession: str | None = None,
    polymarketauthtype: str | None = None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    check_relay_modus(
        rpc_settings=rpc_settings,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
    )

    condition_id, size, neg_risk = _tx_pre_merge_positions(
        rpc_settings=rpc_settings,
        position_manager=position_manager,
        market_triplet=market_triplet,
        size=size,
        neg_risk=neg_risk,
    )

    resp, txn_hash, txn_receipt = merge_positions(
        w3=rpc_settings.w3,
        condition_id=condition_id,
        size=size,
        neg_risk=neg_risk,
        chain_id=rpc_settings.chain_id,
        private_key=rpc_settings.private_key,
        maker_funder=rpc_settings.maker,
        gas_factor=rpc_settings.gas_factor,
        max_gas_price=rpc_settings.max_gas_price,
        allow_fallback_unrelayed=rpc_settings.allow_fallback_unrelayed,
        endpoint_relayer=rpc_settings.endpoint_relayer,
        max_gas_limit_relayer=rpc_settings.max_gas_limit_relayer,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
        receipt_timeout=rpc_settings.receipt_timeout,
    )

    _check_onchain_response(resp, txn_hash, txn_receipt)
    _tx_post_merge_positions(
        position_manager=position_manager, market_triplet=market_triplet, size=size
    )
    return resp, txn_hash, txn_receipt


def tx_convert_positions(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    conversion_cache: ConversionCacheProtocol,
    cvt_market_quintets: MarketIdQuintet | list[MarketIdQuintet],
    all_market_quintets: list[MarketIdQuintet] | None,
    size: NumericAlias | None,
    polymarketnonce: str | None = None,
    polymarketsession: str | None = None,
    polymarketauthtype: str | None = None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    # all_market_quintets not needed for rpc but for proper bookkeeping in position manager
    check_relay_modus(
        rpc_settings=rpc_settings,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
    )

    (
        neg_risk_market_id,
        size,
        question_ids,
        all_market_quintets,
    ) = _tx_pre_convert_position(
        rpc_settings=rpc_settings,
        position_manager=position_manager,
        cvt_market_quintets=cvt_market_quintets,
        all_market_quintets=all_market_quintets,
        size=size,
    )

    resp, txn_hash, txn_receipt = convert_positions(
        w3=rpc_settings.w3,
        neg_risk_market_id=neg_risk_market_id,
        size=size,
        question_ids=question_ids,
        chain_id=rpc_settings.chain_id,
        private_key=rpc_settings.private_key,
        maker_funder=rpc_settings.maker,
        gas_factor=rpc_settings.gas_factor,
        max_gas_price=rpc_settings.max_gas_price,
        allow_fallback_unrelayed=rpc_settings.allow_fallback_unrelayed,
        endpoint_relayer=rpc_settings.endpoint_relayer,
        max_gas_limit_relayer=rpc_settings.max_gas_limit_relayer,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
        receipt_timeout=rpc_settings.receipt_timeout,
    )

    _check_onchain_response(resp, txn_hash, txn_receipt)
    _tx_post_convert_positions(
        position_manager=position_manager,
        conversion_cache=conversion_cache,
        cvt_market_quintets=cvt_market_quintets,
        all_market_quintets=all_market_quintets,
        size=size,
    )

    return resp, txn_hash, txn_receipt


def tx_batch_operate_positions(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    conversion_cache: ConversionCacheProtocol,
    batch_mtx: list[MTX],
    polymarketnonce: str | None = None,
    polymarketsession: str | None = None,
    polymarketauthtype: str | None = None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    check_relay_modus(
        rpc_settings=rpc_settings,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
    )

    batch_txn = _tx_pre_batch_operate_positions(
        rpc_settings, position_manager, batch_mtx
    )

    resp, txn_hash, txn_receipt = batch_operate_positions(
        w3=rpc_settings.w3,
        batch_txn=batch_txn,
        chain_id=rpc_settings.chain_id,
        private_key=rpc_settings.private_key,
        maker_funder=rpc_settings.maker,
        gas_factor=rpc_settings.gas_factor,
        max_gas_price=rpc_settings.max_gas_price,
        allow_fallback_unrelayed=rpc_settings.allow_fallback_unrelayed,
        endpoint_relayer=rpc_settings.endpoint_relayer,
        max_gas_limit_relayer=rpc_settings.max_gas_limit_relayer,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
        receipt_timeout=rpc_settings.receipt_timeout,
    )

    _check_onchain_response(resp, txn_hash, txn_receipt)
    _tx_post_batch_operate_positions(
        position_manager=position_manager,
        conversion_cache=conversion_cache,
        batch_mtx=batch_mtx,
        batch_txn=batch_txn,
    )

    return resp, txn_hash, txn_receipt


def tx_redeem_positions(
    rpc_settings: RPCSettings,
    position_managers: list["PositionManagerProtocol"],
    market_triplet: MarketIdTriplet,
    outcome: Literal["YES", "NO"] | None = None,
    neg_risk: bool | None = None,
    polymarketnonce: str | None = None,
    polymarketsession: str | None = None,
    polymarketauthtype: str | None = None,
) -> tuple[RelayerResponse | None, HexBytes | None, TxReceipt | None]:
    check_relay_modus(
        rpc_settings=rpc_settings,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
    )

    if neg_risk is None:
        neg_risk = get_neg_risk(rpc_settings.endpoint_rest, market_triplet[1])

    condition_id = market_triplet[0]
    size_tuples = [
        _tx_pre_redeem_positions(position_manager=pm, market_triplet=market_triplet)
        for pm in position_managers
    ]
    size_1 = sum(s[0] for s in size_tuples)
    size_2 = sum(s[1] for s in size_tuples)

    _assert_redeem_sizes_onchain(
        rpc_settings=rpc_settings,
        size_1=size_1,
        size_2=size_2,
        market_triplet=market_triplet,
    )

    resp, txn_hash, txn_receipt = redeem_positions(
        w3=rpc_settings.w3,
        condition_id=condition_id,
        size_yes_neg_risk=size_1,
        size_no_neg_risk=size_2,
        neg_risk=neg_risk,
        chain_id=rpc_settings.chain_id,
        private_key=rpc_settings.private_key,
        maker_funder=rpc_settings.maker,
        gas_factor=rpc_settings.gas_factor,
        max_gas_price=rpc_settings.max_gas_price,
        allow_fallback_unrelayed=rpc_settings.allow_fallback_unrelayed,
        endpoint_relayer=rpc_settings.endpoint_relayer,
        max_gas_limit_relayer=rpc_settings.max_gas_limit_relayer,
        polymarketnonce=polymarketnonce,
        polymarketsession=polymarketsession,
        polymarketauthtype=polymarketauthtype,
        receipt_timeout=rpc_settings.receipt_timeout,
    )
    _check_onchain_response(resp, txn_hash, txn_receipt)

    outcome = _parse_outcome(
        rpc_settings=rpc_settings, market_triplet=market_triplet, outcome=outcome
    )

    for (size_y, size_n), pm in zip(size_tuples, position_managers):
        _tx_post_redeem_positions(
            position_manager=pm,
            market_triplet=market_triplet,
            size_yes=size_y,
            size_no=size_n,
            outcome=outcome,
        )

    return resp, txn_hash, txn_receipt
