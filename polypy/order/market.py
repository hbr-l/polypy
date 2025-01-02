import traceback
from typing import Self

import attrs
from eth_keys.datatypes import PrivateKey
from poly_eip712_structs import EIP712Struct

from polypy.constants import CHAIN_ID, ENDPOINT, SIG_DIGITS_ORDER_SIZE, ZERO_ADDRESS
from polypy.exceptions import (
    OrderBookException,
    OrderCreationException,
    PolyPyException,
)
from polypy.order.base import (
    PLACEMENT_STATUS,
    SIDE,
    TIME_IN_FORCE,
    BaseOrder,
    is_valid_price,
    tick_size_digits,
)
from polypy.order.eip712 import SIGNATURE_TYPE, parse_private_key, polymarket_domain
from polypy.orderbook import OrderBook, calculate_marketable_price
from polypy.rest import get_tick_size
from polypy.rounding import (
    round_floor,
    round_floor_tenuis_ceil,
    round_half_even,
    scale_1e06,
)
from polypy.typing import NumericAlias


@attrs.define
class MarketOrder(BaseOrder):
    amount: NumericAlias = attrs.field(on_setattr=attrs.setters.frozen)

    @classmethod
    def build(
        cls,
        amount: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias | None,
        private_key: PrivateKey,
        domain: EIP712Struct,
        tif: TIME_IN_FORCE = TIME_IN_FORCE.GTC,
        max_size: NumericAlias | None = None,
        signature: str | None = None,
        strategy_id: str | None = None,
        aux_id: str | None = None,
        idx: str | None = None,
        placement_status: PLACEMENT_STATUS = PLACEMENT_STATUS.INSITU,
        transaction_hashes: list[str] | None = None,
        created_at: int | None = None,
        defined_at: int | None = None,
        expiration: int = 0,
        nonce: int = 0,
        fee_rate_bps: int = 0,  # todo really int?
        taker: str = ZERO_ADDRESS,
        maker: str | None = None,
        signer: str | None = None,
        signature_type: SIGNATURE_TYPE = SIGNATURE_TYPE.EOA,
        rest_endpoint: ENDPOINT | str | None = None,
        sig_digits_order_size: int = SIG_DIGITS_ORDER_SIZE,
        extra_precision_buffer: int = 4,
        price: NumericAlias | None = None,
    ) -> Self:
        if tick_size is None:
            try:
                return get_tick_size(rest_endpoint, token_id)
            except Exception as e:
                raise OrderCreationException(
                    f"Cannot get tick_size via REST. Did you configure 'tick_size_endpoint' correctly?. "
                    f"Full traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}."
                ) from e

        if price is None:
            if side is SIDE.BUY:
                price = 1 - tick_size
            elif side is SIDE.SELL:
                price = tick_size
            else:
                raise OrderCreationException(f"Unknown side: {side}.")

        if not is_valid_price(price, tick_size):
            raise OrderCreationException(
                f"Invalid price. Must be 'tick_size <= price <= 1 - tick_size'. "
                f"Got: price={price} and tick_size={tick_size}"
            )

        maker_amount, taker_amount, amount, price = market_order_taker_maker_amount(
            side,
            amount,
            price,
            tick_size_digits(tick_size, OrderCreationException),
            sig_digits_order_size,
            extra_precision_buffer,
            max_size,
        )

        # noinspection DuplicatedCode
        (
            eip712order,
            signature,
            transaction_hashes,
            defined_at,
        ) = cls._parse_create_args(
            token_id=token_id,
            side=side,
            taker_amount=taker_amount,
            maker_amount=maker_amount,
            private_key=private_key,
            domain=domain,
            signature=signature,
            transaction_hashes=transaction_hashes,
            defined_at=defined_at,
            expiration=expiration,
            nonce=nonce,
            fee_rate_bps=fee_rate_bps,
            taker=taker,
            maker=maker,
            signer=signer,
            signature_type=signature_type,
        )

        return cls(
            eip712order=eip712order,
            id=idx,
            signature=signature,
            tif=tif,
            strategy_id=strategy_id,
            aux_id=aux_id,
            placement_status=placement_status,
            transaction_hashes=transaction_hashes,
            created_at=created_at,
            defined_at=defined_at,
            amount=amount,
        )

    def is_marketable_liquidity(self, book: OrderBook) -> bool:
        if self.side is SIDE.BUY:
            prices, sizes = book.asks
        elif self.side is SIDE.SELL:
            prices, sizes = book.bids
        else:
            raise PolyPyException(f"Unknown side: {self.side}")

        try:
            calculate_marketable_price(prices, sizes, self.amount)
        except OrderBookException:
            return False
        return True


def build_market_order(
    amount: NumericAlias,
    token_id: str,
    side: SIDE,
    tick_size: float | NumericAlias | None,
    private_key: str,
    chain_id: CHAIN_ID,
    neg_risk: bool | None,
    book: OrderBook | None = None,
    tif: TIME_IN_FORCE = TIME_IN_FORCE.GTC,
    max_size: NumericAlias | None = None,
    signature: str | None = None,
    strategy_id: str | None = None,
    aux_id: str | None = None,
    idx: str | None = None,
    placement_status: PLACEMENT_STATUS = PLACEMENT_STATUS.INSITU,
    transaction_hashes: list[str] | None = None,
    created_at: int | None = None,
    defined_at: int | None = None,
    expiration: int = 0,
    nonce: int = 0,
    fee_rate_bps: int = 0,  # todo really int?
    taker: str = ZERO_ADDRESS,
    maker: str | None = None,
    signer: str | None = None,
    signature_type: SIGNATURE_TYPE = SIGNATURE_TYPE.EOA,
    rest_endpoint: ENDPOINT | str | None = None,
    sig_digits_order_size: int = SIG_DIGITS_ORDER_SIZE,
    extra_precision_buffer: int = 4,
    price: NumericAlias | None = None,
) -> MarketOrder:
    """

    Parameters
    ----------
    amount
    token_id
    side
    tick_size
    private_key
    chain_id
    neg_risk
    book: OrderBook | None,
        if defined, checks the oder book if enough liquidity is available in the order book to fill the order
    tif
    max_size: NumericAlias | None,
        only required if SIDE.SELL
    signature
    strategy_id
    aux_id
    idx
    placement_status
    transaction_hashes
    created_at
    defined_at
    expiration
    nonce
    fee_rate_bps
    taker
    maker
    signer
    signature_type
    rest_endpoint
    sig_digits_order_size
    extra_precision_buffer
    price: NumericAlias | None
        If None, sets price to tick_size (SELL) or 1 - ticksize (BUY),
        which is equivalent to a deep-in-the-book limit order.
        Setting price is mostly an interface for testing purpose.

    Returns
    -------

    Notes
    -----
    amount (and price) will be rounded automatically according to tick_size and SIG_DIGITS_ORDER_SIZE.
    """
    private_key = parse_private_key(private_key)
    domain = polymarket_domain(chain_id, neg_risk, token_id, rest_endpoint)
    order = MarketOrder.build(
        amount,
        token_id,
        side,
        tick_size,
        private_key,
        domain,
        tif,
        max_size,
        signature,
        strategy_id,
        aux_id,
        idx,
        placement_status,
        transaction_hashes,
        created_at,
        defined_at,
        expiration,
        nonce,
        fee_rate_bps,
        taker,
        maker,
        signer,
        signature_type,
        rest_endpoint,
        sig_digits_order_size,
        extra_precision_buffer,
        price,
    )

    if book is not None and order.is_marketable_liquidity(book) is False:
        raise OrderCreationException("Order amount exceeds market liquidity.")

    return order


def market_order_taker_maker_amount(
    side: SIDE,
    amount: NumericAlias,
    price: NumericAlias,
    tick_size_sig_digits: int,
    order_size_sig_digits: int,
    extra_precision_buffer: int = 4,
    max_size: NumericAlias | None = None,
) -> tuple[int, int, float, float]:
    # todo premature (double) rounding of price and size might lead to loss of precision -> only round end result, i.e.:
    #   1. round_price = round_half_even(...) -> prune
    #   2. round_floor_tenuis_ceil(round_x_amt, ..., ...) -> change to round_floor_tenuis_ceil(size * price, ..., ...)
    #   for now we keep it this way because tests (ported) from py-clob-client will fail otherwise
    #   -> though this might need further investigation in the future

    # todo whole rounding approach needs thorough revision

    amount_sig_digits = tick_size_sig_digits + order_size_sig_digits
    round_price = round_half_even(price, tick_size_sig_digits)
    round_amount = round_floor(amount, order_size_sig_digits)

    if side is SIDE.BUY:
        # round_floor: we do not want to spend more cash than specified
        #   (safe side: spend slightly less)
        round_maker_amt = round_amount

        round_taker_amt = round_maker_amt / round_price
        round_taker_amt = round_floor_tenuis_ceil(
            round_taker_amt, amount_sig_digits, extra_precision_buffer
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
            round_maker_amt, amount_sig_digits, extra_precision_buffer
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
        round_amount,
        round_price,
    )
