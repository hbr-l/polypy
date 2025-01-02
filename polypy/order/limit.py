import traceback
from typing import Self

import attrs
from eth_keys.datatypes import PrivateKey
from poly_eip712_structs import EIP712Struct

from polypy.constants import CHAIN_ID, ENDPOINT, SIG_DIGITS_ORDER_SIZE, ZERO_ADDRESS
from polypy.exceptions import OrderCreationException
from polypy.order.base import (
    PLACEMENT_STATUS,
    SIDE,
    TIME_IN_FORCE,
    BaseOrder,
    is_valid_price,
    tick_size_digits,
)
from polypy.order.eip712 import SIGNATURE_TYPE, parse_private_key, polymarket_domain
from polypy.rest import get_tick_size
from polypy.rounding import (
    round_floor,
    round_floor_tenuis_ceil,
    round_half_even,
    scale_1e06,
)
from polypy.typing import NumericAlias


def limit_order_taker_maker_amount(
    side: SIDE,
    price: NumericAlias,
    size: NumericAlias,
    tick_size_sig_digits: int,
    order_size_sig_digits: int,
    extra_precision_buffer: int = 4,
) -> tuple[int, int, NumericAlias, NumericAlias]:
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
        round_price,
        round_size,
    )


@attrs.define
class LimitOrder(BaseOrder):
    price: NumericAlias = attrs.field(on_setattr=attrs.setters.frozen)
    size: NumericAlias = attrs.field(on_setattr=attrs.setters.frozen)
    size_matched: NumericAlias | None = attrs.field(default=None)

    def __attrs_post_init__(self):
        # ensure same numeric type
        if self.size_matched is None:
            self.size_matched = type(self.size)(0)
        else:
            self.size_matched = type(self.size)(self.size_matched)

    @classmethod
    def build(
        cls,
        price: NumericAlias,
        size: NumericAlias,
        token_id: str,
        side: SIDE,
        tick_size: float | NumericAlias | None,
        private_key: PrivateKey,
        domain: EIP712Struct,
        tif: TIME_IN_FORCE = TIME_IN_FORCE.GTC,
        signature: str | None = None,
        strategy_id: str | None = None,
        aux_id: str | None = None,
        idx: str | None = None,
        placement_status: PLACEMENT_STATUS = PLACEMENT_STATUS.INSITU,
        size_matched: NumericAlias | None = None,
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
    ) -> Self:
        if tick_size is None:
            try:
                return get_tick_size(rest_endpoint, token_id)
            except Exception as e:
                raise OrderCreationException(
                    f"Cannot get tick_size via REST. Did you configure 'rest_endpoint' correctly?. "
                    f"Full traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}."
                ) from e

        if not is_valid_price(price, tick_size):
            raise OrderCreationException(
                f"Invalid price. Must be 'tick_size <= price <= 1 - tick_size'. "
                f"Got: price={price} and tick_size={tick_size}"
            )

        maker_amount, taker_amount, price, size = limit_order_taker_maker_amount(
            side,
            price,
            size,
            tick_size_digits(tick_size, OrderCreationException),
            sig_digits_order_size,
            extra_precision_buffer,
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
            price=price,
            size=size,
            size_matched=size_matched,
        )

    @property
    def size_open(self):
        return self.size - self.size_matched


def build_limit_order(
    price: NumericAlias,
    size: NumericAlias,
    token_id: str,
    side: SIDE,
    tick_size: float | NumericAlias | None,
    private_key: str,
    chain_id: CHAIN_ID,
    neg_risk: bool | None,
    tif: TIME_IN_FORCE = TIME_IN_FORCE.GTC,
    signature: str | None = None,
    strategy_id: str | None = None,
    aux_id: str | None = None,
    idx: str | None = None,
    placement_status: PLACEMENT_STATUS = PLACEMENT_STATUS.INSITU,
    size_matched: NumericAlias | None = None,
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
) -> LimitOrder:
    """Build Limit Order.

    Notes
    -----
    price and size will be round automatically according to tick_size and SIG_DIGITS_ORDER_SIZE
    """
    private_key = parse_private_key(private_key)
    domain = polymarket_domain(chain_id, neg_risk, token_id, rest_endpoint)
    return LimitOrder.build(
        price,
        size,
        token_id,
        side,
        tick_size,
        private_key,
        domain,
        tif,
        signature,
        strategy_id,
        aux_id,
        idx,
        placement_status,
        size_matched,
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
    )
