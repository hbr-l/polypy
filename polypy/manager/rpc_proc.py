import logging
from typing import TYPE_CHECKING, Literal

import attrs
from _decimal import Decimal
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey
from eth_utils import keccak

from polypy.constants import CHAIN_ID, ENDPOINT, ERC20_WEI_UNIT
from polypy.ctf import MarketIdQuintet, MarketIdTriplet
from polypy.exceptions import (
    PolyPyException,
    PositionTrackingException,
    PositionTransactionException,
)
from polypy.manager.cache import ConversionCacheProtocol, _check_conversion_all_quintets
from polypy.order.common import SIDE
from polypy.rest.api import get_market, get_neg_risk, get_neg_risk_market
from polypy.rpc.api import get_balance_token
from polypy.rpc.tx import W3POA
from polypy.signing import generate_seed
from polypy.trade import IDEAL_TRADE_LIFECYCLE
from polypy.typing import NumericAlias, dec, infer_numeric_type

if TYPE_CHECKING:
    from polypy.manager.position import PositionManagerProtocol


@attrs.define
class RPCSettings:
    endpoint_rest: str | ENDPOINT
    endpoint_gamma: str | ENDPOINT
    endpoint_rpc: str | ENDPOINT
    endpoint_relayer: str | ENDPOINT | None
    chain_id: CHAIN_ID
    private_key: str | PrivateKeyType | PrivateKey = attrs.field(repr=False)
    maker: str | None
    gas_factor: NumericAlias
    max_gas_price: int | None
    allow_fallback_unrelayed: bool
    max_gas_limit_relayer: int | None
    receipt_timeout: float | None

    w3: W3POA = attrs.field(default=None, init=False)

    def __attrs_post_init__(self):
        self.w3 = W3POA(self.endpoint_rpc, self.private_key, self.maker)


def _rand_trade_id(x: str | int, addendum: str) -> str:
    _hash = keccak(f"{x}_{generate_seed()}".encode()).hex()
    return f"{_hash}_{addendum}"


def _transact_position(
    position_manager: "PositionManagerProtocol",
    cvt_type: bool,
    asset_id: str,
    delta_size: NumericAlias,
    price: NumericAlias,
    trade_id: str,
    side: SIDE,
    allow_create: bool,
) -> None:
    if cvt_type:
        # noinspection PyProtectedMember
        position = position_manager._get_or_create_position(
            asset_id=asset_id, allow_create=allow_create
        )
        numeric_type = infer_numeric_type(position.size)
        delta_size = numeric_type(delta_size)
        price = numeric_type(price)

    # we pass through all trade statu of a trade lifecycle since
    # we do not know the specific implementation of position
    for status in IDEAL_TRADE_LIFECYCLE:
        position_manager.transact(
            asset_id=asset_id,
            delta_size=delta_size,
            price=price,
            trade_id=trade_id,
            side=side,
            trade_status=status,
            allow_create=allow_create,
        )


def _tx_pre_split_position(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
    amount: NumericAlias,
    neg_risk: bool | None,
) -> tuple[str, NumericAlias, bool]:
    if amount > position_manager.balance_available:
        raise PositionTransactionException(
            f"{amount} exceeds available balance {position_manager.balance_available}."
        )

    if neg_risk is None:
        neg_risk = get_neg_risk(rpc_settings.endpoint_rest, market_triplet[1])

    return market_triplet[0], amount, neg_risk


def _tx_post_split_positions(
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
    amount: NumericAlias,
) -> None:
    # when splitting, we give amount USDC to get amount positions: buy
    _transact_position(
        position_manager,
        True,
        market_triplet[1],
        amount,
        Decimal("0.5"),
        _rand_trade_id(market_triplet[1], "SPLIT"),
        SIDE.BUY,
        True,
    )
    _transact_position(
        position_manager,
        True,
        market_triplet[2],
        amount,
        Decimal("0.5"),
        _rand_trade_id(market_triplet[2], "SPLIT"),
        SIDE.BUY,
        True,
    )


def _tx_pre_merge_positions(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
    size: NumericAlias | None,
    neg_risk: bool | None,
) -> tuple[str, NumericAlias, bool]:
    # noinspection PyProtectedMember
    pos_1 = position_manager._get_or_create_position(market_triplet[1], False)
    # noinspection PyProtectedMember
    pos_2 = position_manager._get_or_create_position(market_triplet[2], False)

    size_1 = pos_1.size_available if pos_1 is not None else 0
    size_2 = pos_2.size_available if pos_2 is not None else 0
    size_min = min(size_1, size_2)

    if size is None:
        size = size_min

    if size > size_min:
        raise PositionTransactionException(
            f"{size} exceeds available positions. "
            f"PositionYes.size_available={size_1}, PositionNo.size_available={size_2}."
        )

    if size <= 0:
        raise PositionTransactionException(f"Got size={size} =< 0.")

    if neg_risk is None:
        neg_risk = get_neg_risk(rpc_settings.endpoint_rest, market_triplet[1])

    return market_triplet[0], size, neg_risk


def _tx_post_merge_positions(
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
    size: NumericAlias,
) -> None:
    # when merging, we receive amount USDC and give amount positions: sell
    _transact_position(
        position_manager,
        True,
        market_triplet[1],
        size,
        Decimal("0.5"),
        _rand_trade_id(market_triplet[1], "MERGE"),
        SIDE.SELL,
        True,
    )
    _transact_position(
        position_manager,
        True,
        market_triplet[2],
        size,
        Decimal("0.5"),
        _rand_trade_id(market_triplet[2], "MERGE"),
        SIDE.SELL,
        True,
    )


def _parse_all_market_quintets(
    rpc_settings: RPCSettings,
    cvt_market_quintets: list[MarketIdQuintet],
    all_market_quintets: list[MarketIdQuintet] | None,
) -> list[MarketIdQuintet]:
    if all_market_quintets is None:
        _, all_market_quintets = get_neg_risk_market(
            endpoint_gama=rpc_settings.endpoint_gamma,
            include_closed=True,
            condition_id=cvt_market_quintets[0][0],
            token_id=None,
            market_slug=None,
        )

    _check_conversion_all_quintets(all_market_quintets)
    # todo additional checks?

    return all_market_quintets


def _tx_pre_convert_position(
    rpc_settings: RPCSettings,
    position_manager: "PositionManagerProtocol",
    cvt_market_quintets: MarketIdQuintet | list[MarketIdQuintet],
    all_market_quintets: list[MarketIdQuintet] | None,
    size: NumericAlias | None,
) -> tuple[str, NumericAlias, list[str], list[MarketIdQuintet]]:
    if all_market_quintets is not None:
        logging.info(
            "'all_market_quintets' must contain all conditions associated with the NegRiskMarket,"
            "else converting positions might lead to erroneous position sizes and balance - "
            "user responsibility is advised."
        )
    logging.info(
        "Users have to periodically call `PositionManager.update_augmented_conversions(...)` "
        "in order to correctly account for new YES shares in case a new outcome/condition is added to "
        "the negative risk market (see Polymarket's Augmented Negative Risk Adapter). "
        "If the PositionManager is associated with an UserStream which automates `.update_augmented_conversion(...)`,"
        "be aware that this incurs two Gamma API REST calls per update."
    )

    if not isinstance(cvt_market_quintets, list):
        cvt_market_quintets = [cvt_market_quintets]

    all_market_quintets = _parse_all_market_quintets(
        rpc_settings=rpc_settings,
        cvt_market_quintets=cvt_market_quintets,
        all_market_quintets=all_market_quintets,
    )

    neg_risk_market_ids = {m[1] for m in all_market_quintets} | {
        m[1] for m in cvt_market_quintets
    }
    if "" in neg_risk_market_ids or None in neg_risk_market_ids:
        raise PositionTransactionException(
            "Not all 'market_quintets' are NegRiskMarket!"
        )
    if len(neg_risk_market_ids) > 1:
        raise PositionTransactionException(
            f"Not all 'market_quintets' have the same neg_risk_market_id. "
            f"'neg_risk_market_ids'={neg_risk_market_ids}."
        )

    # noinspection PyProtectedMember
    positions_no = [
        position_manager._get_or_create_position(m[4], False)
        for m in cvt_market_quintets
    ]
    sizes_no = min(p.size_available if p is not None else 0 for p in positions_no)

    if size is None:
        size = sizes_no

    if size < 0 or sizes_no < 0:
        raise PositionTransactionException(
            "Cannot convert size<0. "
            "Make sure 'cvt_market_quintets' only contains NO positions with size > 0."
        )

    if size > sizes_no:
        raise PositionTransactionException(
            f"'size'={size} exceeds the smallest NO-size={sizes_no} specified in 'cvt_market_quintets'."
        )

    return (
        cvt_market_quintets[0][1],
        size,
        [m[2] for m in cvt_market_quintets],
        all_market_quintets,
    )


def _act_conversion_cache_diff(
    position_manager: "PositionManagerProtocol",
    re_size: Decimal,
    token_1_ids: list[str],
) -> bool:
    if re_size <= 0:
        return False

    for asset_id in token_1_ids:
        if asset_id in position_manager:
            raise PositionTrackingException(
                f"{asset_id} already in PositionManager. Cannot process post-conversion event."
            )
        _transact_position(
            position_manager,
            True,  # re_size in Decimal, so we cvt
            asset_id,
            re_size,
            0,
            _rand_trade_id(asset_id, "AUG_CONVERT"),
            SIDE.BUY,
            True,
        )

    return True


def _tx_post_convert_positions(
    position_manager: "PositionManagerProtocol",
    conversion_cache: ConversionCacheProtocol,
    cvt_market_quintets: list[MarketIdQuintet],
    all_market_quintets: list[MarketIdQuintet],
    size: NumericAlias,
) -> None:
    no_token_ids = {m[4] for m in cvt_market_quintets}
    yes_token_ids = {m[3] for m in all_market_quintets if m[4] not in no_token_ids}

    # usdc = size * (len(no_token_ids) - 1) # according to neg risk conversion logic

    # decrease all NO positions: this is like selling the positions
    # if we have N NO positions and Y YES positions,
    #   we collect size * price_n * N in USDC by selling N positions,
    #   and we spend size * price_y * Y in USDC by buying Y positions,
    #   in which the difference/profit has to be equal to size * (N - 1) according
    #   to negative risk conversion logic
    # -> size * price_n * N - size * price_y * Y = size * (N - 1)
    # -> price_n * N - price_y * Y = N - 1          | price_n = 1 - price_y
    # -> (1 - price_y) * N - price_y * Y = N - 1
    # -> N - price_y * N - price_y * Y = N - 1      | -N
    # -> - price_y * N - price_y * Y = -1
    # -> price_y * N + price_y * Y = 1
    # -> price_y = 1 / (N + Y) = 1 / C
    # -> price_n = 1 - price_y = 1 - (1 / C)
    # todo this might induce numerical instability,
    #  alternative: set price=0 and use separate position_manager.deposit(size * (N - 1)) transaction,
    #  but this might mess with specific position implementation
    n, y = len(no_token_ids), len(yes_token_ids)
    price_yes = 1 / Decimal(n + y)
    price_no = 1 - price_yes
    for no_token in no_token_ids:
        # NO is assumed to exist, because we can only convert NO positions that we actually own
        _transact_position(
            position_manager,
            True,
            no_token,
            size,
            price_no,  # in Decimal, so we cvt
            _rand_trade_id(no_token, "CONVERT"),
            SIDE.SELL,
            False,
        )

    # increase complementary YES positions
    for yes_token in yes_token_ids:
        _transact_position(
            position_manager,
            True,
            yes_token,
            size,
            price_yes,
            _rand_trade_id(yes_token, "CONVERT"),
            SIDE.BUY,
            True,
        )

    # if everything went up to here, we can update the conversion cache
    re_size, token_1_ids = conversion_cache.update(
        size=size, all_market_quintets=all_market_quintets
    )  # returns diff and updates cache all at once
    # increment YES positions that have been added after previous conversion (not current conversion!)
    _act_conversion_cache_diff(
        position_manager=position_manager, re_size=re_size, token_1_ids=token_1_ids
    )


def _assert_redeem_sizes_onchain(
    rpc_settings: RPCSettings,
    size_1: NumericAlias,
    size_2: NumericAlias,
    market_triplet: MarketIdTriplet,
) -> None:
    token_size_1 = get_balance_token(
        rpc_settings.w3, market_triplet[1], rpc_settings.chain_id, ERC20_WEI_UNIT
    )
    token_size_2 = get_balance_token(
        rpc_settings.w3, market_triplet[2], rpc_settings.chain_id, ERC20_WEI_UNIT
    )

    # size > token_size: position managers account for more cumulative size than we actually have
    #   on-chain, which is incorrect anyway (we have more size on sheet than we actually own...)
    # size < token_size: some tokens are not covered by specified position managers, so there might be
    #   a position manager left on which the change of position sizes will not be correctly calculated
    #   since redeeming is only possible for all-or-none, which leaves uncovered position managers with
    #   incorrect position sizes, which is incorrect as well
    # size == token_size: constraint to check for

    if abs(dec(token_size_1) - size_1) > 0.05:
        # give some leeway, as token_size_1 might be converted from float and might incur float imprecision
        raise PositionTransactionException(
            f"Redeem size={size_1} != on-chain token size={token_size_1} for {market_triplet[1]}. "
            f"Make sure to provide all Position Managers holding redeemable tokens of the market_triplet, "
            f"else position sizes cannot be computed correctly for uncovered Position Managers. "
            f"No transactions have been executed."
        )
    if abs(dec(token_size_2) - size_2) > 0.05:
        raise PositionTransactionException(
            f"Redeem size={size_2} != on-chain token size={token_size_2} for {market_triplet[2]}. "
            f"Make sure to provide all Position Managers holding redeemable tokens of the market_triplet, "
            f"else position sizes cannot be computed correctly for uncovered Position Managers. "
            f"No transactions have been executed."
        )


def _tx_pre_redeem_positions(
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
) -> tuple[NumericAlias, NumericAlias]:
    if market_triplet[1] in position_manager:
        # noinspection PyProtectedMember
        pos_1 = position_manager._get_or_create_position(market_triplet[1], False)
        size_1 = pos_1.size_available
    else:
        size_1 = 0

    if market_triplet[2] in position_manager:
        # noinspection PyProtectedMember
        pos_2 = position_manager._get_or_create_position(market_triplet[2], False)
        size_2 = pos_2.size_available
    else:
        size_2 = 0

    if min(size_1, size_2) < 0:
        raise PositionTransactionException(
            f"Got negative size. "
            f"Cannot redeem {market_triplet[1]} (size={size_1}) "
            f"and/or {market_triplet[2]} (size={size_2})."
        )

    return size_1, size_2


def _parse_outcome(
    rpc_settings: RPCSettings,
    market_triplet: MarketIdTriplet,
    outcome: Literal["YES", "NO"] | None,
) -> Literal["YES", "NO"]:
    if outcome is None:
        market = get_market(rpc_settings.endpoint_rest, market_triplet[0])
        if market.tokens[0].winner is True:
            outcome = "YES"
        elif market.tokens[1].winner is True:
            outcome = "NO"
        else:
            raise PolyPyException(f"Undefined outcome for: {market_triplet[0]}")

    if outcome not in ["YES", "NO"]:
        raise PositionTransactionException(f"Unknown argument 'outcome'={outcome}.")

    # noinspection PyTypeChecker
    return outcome


def _tx_post_redeem_positions(
    position_manager: "PositionManagerProtocol",
    market_triplet: MarketIdTriplet,
    size_yes: NumericAlias,
    size_no: NumericAlias,
    outcome: Literal["YES", "NO"] | None,
) -> None:
    # only winner pays out 1 USDC * size
    # this is equivalent so selling, since we receive USDC and spend shares
    # price is either 1 or 0 depending on outcome

    if market_triplet[1] in position_manager:
        # we don't use size_yes > 0 as if-condition, because floating point comparison can be imprecise
        _transact_position(
            position_manager,
            False,
            market_triplet[1],
            size_yes,
            1 if outcome == "YES" else 0,
            _rand_trade_id(market_triplet[1], "REDEEM"),
            SIDE.SELL,
            False,
        )

    if market_triplet[2] in position_manager:
        # we don't use size_no > 0 as if-condition, because floating point comparison can be imprecise
        _transact_position(
            position_manager,
            False,
            market_triplet[2],
            size_no,
            1 if outcome == "NO" else 0,
            _rand_trade_id(market_triplet[2], "REDEEM"),
            SIDE.SELL,
            False,
        )

    # not every position manager will have each market_triplet, so we don't check if
    #   even at least one transaction has been executed (there's a possibility, that the position manager
    #   does not contain the market_triplet at all)
