from typing import TYPE_CHECKING, Protocol

import numpy as np
from eth_account.types import PrivateKeyType
from eth_keys.datatypes import PrivateKey

from polypy.constants import CHAIN_ID, N_DIGITS_SIZE, ZERO_ADDRESS
from polypy.exceptions import OrderCreationException, PolyPyException
from polypy.order.base import Order, check_valid_price, cvt_tick_size
from polypy.order.common import (
    INSERT_STATUS,
    SIDE,
    TIME_IN_FORCE,
    TIME_IN_FORCE_MARKET,
    OrderProtocol,
)
from polypy.rounding import (
    round_floor,
    round_floor_tenuis_ceil,
    round_half_even,
    scale_1e06,
)
from polypy.signing import SIGNATURE_TYPE, polymarket_domain
from polypy.typing import NumericAlias, infer_numeric_type

if TYPE_CHECKING:
    from polypy.book.order_book import OrderBookProtocol


def market_order_taker_maker_amount(
    side: SIDE,
    amount: NumericAlias,
    price: NumericAlias,
    n_digits_tick_size: int,
    n_digits_order_size: int,
    extra_precision_buffer: int = 4,
    max_size: NumericAlias | None = None,
) -> tuple[int, int]:
    # todo premature (double) rounding of price and size might lead to loss of precision -> only round end result, i.e.:
    #   1. round_price = round_half_even(...) -> prune
    #   2. round_floor_tenuis_ceil(round_x_amt, ..., ...) -> change to round_floor_tenuis_ceil(size * price, ..., ...)
    #   for now we keep it this way because tests (ported) from py-clob-client will fail otherwise
    #   -> though this might need further investigation in the future

    # todo whole rounding approach needs thorough revision

    n_digits_amount = n_digits_tick_size + n_digits_order_size
    round_price = round_half_even(price, n_digits_tick_size)
    round_amount = round_floor(amount, n_digits_order_size)

    if side is SIDE.BUY:
        # round_floor: we do not want to spend more cash than specified
        #   (safe side: spend slightly less)
        round_maker_amt = round_amount

        round_taker_amt = round_maker_amt / round_price
        round_taker_amt = round_floor_tenuis_ceil(
            round_taker_amt, n_digits_amount, extra_precision_buffer
        )
    elif side is SIDE.SELL:
        if max_size is None:
            raise OrderCreationException(
                "Market Sell Order needs to specify a max order size to prevent overspending."
            )

        # round_floor: to not sell more shares than we actually own (size = round_maker_amt),
        #   we want to approximate size a little bit less/smaller.
        #   Therefore, we either have to round down the numerator or round up the denominator.
        #   Denominator (price) has already been round half even earlier, so we round down numerator=amount.
        round_taker_amt = round_amount

        round_maker_amt = round_taker_amt / round_price  # size
        round_maker_amt = round_floor_tenuis_ceil(
            round_maker_amt, n_digits_amount, extra_precision_buffer
        )

        if round_maker_amt > max_size:
            raise OrderCreationException(
                f"Market Sell Order size (nb. shares: {round_maker_amt}) "
                f"exceeds specified max order size (`max_size`: {max_size})."
            )
    else:
        raise OrderCreationException(f"Unknown side: {side}")

    return (
        scale_1e06(round_maker_amt),
        scale_1e06(round_taker_amt),
    )


def terminal_price(side: SIDE, tick_size: NumericAlias) -> NumericAlias:
    if side is SIDE.BUY:
        return 1 - tick_size
    elif side is SIDE.SELL:
        return tick_size
    else:
        raise OrderCreationException(f"Unknown side: {side}.")


def is_marketable_amount(
    book: "OrderBookProtocol",
    side: SIDE,
    amount: NumericAlias,
) -> bool:
    if side is SIDE.BUY:
        prices, sizes = book.asks
    elif side is SIDE.SELL:
        prices, sizes = book.bids
    else:
        raise PolyPyException(f"Unknown side: {side}.")

    total_amount = np.sum(np.asarray(prices) * np.asarray(sizes))
    return amount <= total_amount


def create_market_order(
    amount: NumericAlias,
    token_id: str,
    side: SIDE,
    tick_size: float | NumericAlias,
    neg_risk: bool,
    chain_id: CHAIN_ID,
    private_key: PrivateKey | str | PrivateKeyType,
    maker: str | None,
    signature_type: SIGNATURE_TYPE,
    tif: TIME_IN_FORCE = TIME_IN_FORCE.FOK,
    book: "OrderBookProtocol | None" = None,
    max_size: NumericAlias | None = None,
    salt: int | None = None,
    order_id: str | None = None,
    signature: str | None = None,
    strategy_id: str | None = None,
    aux_id: str | None = None,
    status: INSERT_STATUS = INSERT_STATUS.DEFINED,
    created_at: int | None = None,
    defined_at: int | None = None,
    nonce: int = 0,
    fee_rate_bps: int = 0,
    taker: str = ZERO_ADDRESS,
    signer: str | None = None,
    n_digits_order_size: int = N_DIGITS_SIZE,
    extra_precision_buffer: int = 4,
    price: NumericAlias | None = None,
) -> Order:
    """Build market order defined via amount (instead of size).

    Parameters
    ----------
    amount: NumericAlias,
        will be rounded to `n_digits_order_size`, which defaults to 2 - even though usually amount
        has 4 to 5 decimal digits (depending on tick size). Though, this might change in the future to
        rounding to 4 to 5 decimal digits.
    token_id
    side
    tick_size
    private_key
    chain_id
    neg_risk
    tif: TIME_IN_FORCE, default=TIME_IN_FORCE.FOK
    book: "OrderBookProtocol" | None,
        if defined, checks the oder book if enough liquidity is available in the order book to fill the order
    max_size: NumericAlias | None,
        only required if SIDE.SELL
    salt
    order_id
    signature
    strategy_id
    aux_id
    status
    created_at
    defined_at
    nonce
    fee_rate_bps
    taker
    maker
    signer
    signature_type
    n_digits_order_size
    extra_precision_buffer
    price: NumericAlias | None
        If None, sets price to tick_size (SELL) or 1 - ticksize (BUY),
        which is equivalent to a deep-in-the-book limit order.
        Setting price is mostly an interface for testing purpose.

    Returns
    -------

    Notes
    -----
    amount (and price) will be rounded automatically according to tick_size and N_DIGITS_SIZE.
    If amount is pre-round accordingly (precision), then tick_size can be set to any sufficiently small
    min tick_size (i.e. 0.001 or any smaller), and does not affect order creation anymore (as long as tick_size is
    sufficiently small).
    """
    if not tif in TIME_IN_FORCE_MARKET:
        raise OrderCreationException(
            f"Market order: `tif` must be one of {TIME_IN_FORCE_MARKET}. Got tif={tif}."
        )

    domain = polymarket_domain(chain_id, neg_risk)
    numeric_type = infer_numeric_type(amount)
    tick_size, nb_tick_digits = cvt_tick_size(tick_size, numeric_type)

    if price is None:
        price = terminal_price(side, tick_size)
        # if amount is Decimal, this will be quantized in round_half_even in market_order_taker_maker_amount
    check_valid_price(price, tick_size)

    if book is not None:
        if book.token_id != token_id:
            raise OrderCreationException(
                f"book.token_id={book.token_id} and `token_id`={token_id} do not match."
            )
        if not is_marketable_amount(book, side, amount):
            raise OrderCreationException("Order amount exceeds market liquidity.")

    maker_amount, taker_amount = market_order_taker_maker_amount(
        side,
        amount,
        price,
        nb_tick_digits,
        n_digits_order_size,
        extra_precision_buffer,
        max_size,
    )

    return Order.create(
        token_id=token_id,
        side=side,
        taker_amount=taker_amount,
        maker_amount=maker_amount,
        private_key=private_key,
        domain=domain,
        size_matched=numeric_type(0),
        tif=tif,
        salt=salt,
        order_id=order_id,
        signature=signature,
        strategy_id=strategy_id,
        aux_id=aux_id,
        status=status,
        created_at=created_at,
        defined_at=defined_at,
        expiration=0,
        nonce=nonce,
        fee_rate_bps=fee_rate_bps,
        taker=taker,
        maker=maker,
        signer=signer,
        signature_type=signature_type,
        numeric_type=numeric_type,
    )


class MarketOrderFactory(Protocol):
    def __call__(
        self,
        amount: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias,
        neg_risk: bool,
        chain_id: CHAIN_ID,
        private_key: PrivateKey | str | PrivateKeyType,
        maker: str | None,
        signature_type: SIGNATURE_TYPE,
        tif: TIME_IN_FORCE,
        book: "OrderBookProtocol",
        max_size: NumericAlias | None,
        *args,
        **kwargs,
    ) -> OrderProtocol:
        ...
