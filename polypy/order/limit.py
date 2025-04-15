from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey

from polypy.constants import CHAIN_ID, SIG_DIGITS_SIZE, ZERO_ADDRESS
from polypy.exceptions import OrderCreationException
from polypy.order.base import Order, check_valid_price, cvt_tick_size
from polypy.order.common import INSERT_STATUS, SIDE, TIME_IN_FORCE
from polypy.rounding import (
    round_floor,
    round_floor_tenuis_ceil,
    round_half_even,
    scale_1e06,
)
from polypy.signing import SIGNATURE_TYPE, polymarket_domain
from polypy.typing import NumericAlias, infer_numeric_type


def limit_order_taker_maker_amount(
    side: SIDE,
    price: NumericAlias,
    size: NumericAlias,
    tick_size_sig_digits: int,
    order_size_sig_digits: int,
    extra_precision_buffer: int = 4,
) -> tuple[int, int]:
    # todo premature (double) rounding of price and size might lead to loss of precision -> only round end result, i.e.:
    #   1. round_price = round_half_even(...) -> prune
    #   2. round_floor_tenuis_ceil(round_x_amt, ..., ...) -> change to round_floor_tenuis_ceil(size * price, ..., ...)
    #   for now we keep it this way because tests (ported) from py-clob-client will fail otherwise
    #   -> though this might need further investigation in the future

    # todo whole rounding approach needs thorough revision

    amount_sig_digits = tick_size_sig_digits + order_size_sig_digits
    round_price = round_half_even(price, tick_size_sig_digits)
    round_size = round_floor(size, order_size_sig_digits)

    if side is SIDE.BUY:
        # round_floor: we do not want to buy more than specified size
        #   as this would mean spending more than planned (safe side: spend slightly less)
        round_taker_amt = round_size

        round_maker_amt = round_taker_amt * round_price
        round_maker_amt = round_floor_tenuis_ceil(
            round_maker_amt, amount_sig_digits, extra_precision_buffer
        )

    elif side is SIDE.SELL:
        # round_floor: we do not want to sell more than specified size
        #   as this would mean selling more than we have (safe side: sell slightly less than position)
        round_maker_amt = round_size

        round_taker_amt = round_maker_amt * round_price
        round_taker_amt = round_floor_tenuis_ceil(
            round_taker_amt, amount_sig_digits, extra_precision_buffer
        )

    else:
        raise OrderCreationException(f"Unknown side: {side}")

    return (
        scale_1e06(round_maker_amt),
        scale_1e06(round_taker_amt),
    )


def create_limit_order(
    price: NumericAlias,
    size: NumericAlias,
    token_id: str,
    side: SIDE,
    tick_size: float | NumericAlias,
    neg_risk: bool,
    chain_id: CHAIN_ID,
    private_key: PrivateKey | str | PrivateKeyType,
    maker: str | None,
    signature_type: SIGNATURE_TYPE,
    tif: TIME_IN_FORCE = TIME_IN_FORCE.GTC,
    signature: str | None = None,
    strategy_id: str | None = None,
    aux_id: str | None = None,
    idx: str | None = None,
    status: INSERT_STATUS = INSERT_STATUS.DEFINED,
    size_matched: NumericAlias | None = None,
    price_matched: NumericAlias | None = None,
    created_at: int | None = None,
    defined_at: int | None = None,
    expiration: int = 0,
    nonce: int = 0,
    fee_rate_bps: int = 0,
    taker: str = ZERO_ADDRESS,
    signer: str | None = None,
    sig_digits_order_size: int = SIG_DIGITS_SIZE,
    extra_precision_buffer: int = 4,
) -> Order:
    """Build Limit Order.

    Notes
    -----
    price and size will be round automatically according to tick_size and SIG_DIGITS_SIZE.
    If price is pre-round accordingly (precision), then tick_size can be set to any sufficiently small
    min tick_size (i.e. 0.001 or any smaller), and does not affect order creation anymore (as long as tick_size is
    sufficiently small).
    """
    domain = polymarket_domain(chain_id, neg_risk)
    numeric_type = infer_numeric_type(price)
    tick_size, nb_tick_digits = cvt_tick_size(tick_size, numeric_type)

    check_valid_price(price, tick_size)

    maker_amount, taker_amount = limit_order_taker_maker_amount(
        side,
        price,
        size,
        nb_tick_digits,
        sig_digits_order_size,
        extra_precision_buffer,
    )

    return Order.create(
        token_id=token_id,
        side=side,
        taker_amount=taker_amount,
        maker_amount=maker_amount,
        private_key=private_key,
        domain=domain,
        size_matched=size_matched,
        price_matched=price_matched,
        tif=tif,
        signature=signature,
        strategy_id=strategy_id,
        aux_id=aux_id,
        idx=idx,
        status=status,
        created_at=created_at,
        defined_at=defined_at,
        expiration=expiration,
        nonce=nonce,
        fee_rate_bps=fee_rate_bps,
        taker=taker,
        maker=maker,
        signer=signer,
        signature_type=signature_type,
        numeric_type=numeric_type,
    )
