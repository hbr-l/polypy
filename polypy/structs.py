import datetime
from typing import Literal, Optional

import msgspec

from polypy.order.common import INSERT_STATUS, SIDE, TIME_IN_FORCE
from polypy.trade import TRADE_STATUS, TRADER_SIDE


class PostOrderResponse(msgspec.Struct, forbid_unknown_fields=True):
    success: bool
    errorMsg: str
    orderID: str
    transactionsHashes: tuple[str] | None
    status: str
    takingAmount: str
    makingAmount: str


class CancelOrdersResponse(msgspec.Struct, forbid_unknown_fields=True):
    canceled: list[str] | None
    """list[str]: list of canceled orders"""
    not_canceled: dict[str, str]
    """dict[str, str]: {order_id: reason} dictionary"""


class OpenOrderInfo(msgspec.Struct, forbid_unknown_fields=True):
    associate_trades: list[str]
    id: str
    status: INSERT_STATUS
    market: str
    original_size: str
    outcome: Literal["Yes", "No"]
    maker_address: str
    owner: str
    price: str
    side: SIDE
    size_matched: str
    asset_id: str
    expiration: int
    order_type: TIME_IN_FORCE
    created_at: int


class OpenOrderResponse(msgspec.Struct, forbid_unknown_fields=True):
    # {"data":[],"next_cursor":"LTE=","limit":500,"count":0}
    data: list[OpenOrderInfo]
    next_cursor: str
    limit: int
    count: int


class TokenInfo(msgspec.Struct, forbid_unknown_fields=True):
    token_id: str
    outcome: Literal["Yes", "No"]
    price: str
    winner: bool


class RewardRate(msgspec.Struct, forbid_unknown_fields=True):
    asset_address: str
    rewards_daily_rate: str


class Rewards(msgspec.Struct, forbid_unknown_fields=True):
    rates: RewardRate
    min_size: str
    max_spread: str


class MarketInfo(msgspec.Struct, forbid_unknown_fields=True):
    enable_order_book: bool
    active: bool
    closed: bool
    archived: bool
    accepting_orders: bool
    accepting_order_timestamp: datetime.datetime
    minimum_order_size: str
    minimum_tick_size: str
    condition_id: str
    question_id: str
    question: str
    description: str
    market_slug: str
    end_date_iso: datetime.datetime
    game_start_time: Optional[datetime.datetime]
    seconds_delay: str
    fpmm: str
    maker_base_fee: str
    taker_base_fee: str
    notifications_enabled: bool
    neg_risk: bool
    neg_risk_market_id: str
    neg_risk_request_id: str
    icon: str
    image: str
    rewards: Rewards
    is_50_50_outcome: bool
    tokens: tuple[TokenInfo]
    tags: tuple[str]


class MarketsResponse(msgspec.Struct):
    limit: int
    count: int
    next_cursor: str
    data: tuple[MarketInfo]


class MakerOrder(
    msgspec.Struct, forbid_unknown_fields=True, frozen=True, cache_hash=True
):
    asset_id: str
    fee_rate_bps: int
    maker_address: str
    matched_amount: str
    order_id: str
    outcome: Literal["Yes", "No"]
    owner: str
    price: str


class TradeWSInfo(
    msgspec.Struct,
    forbid_unknown_fields=True,
    tag_field="event_type",
    tag="trade",
    frozen=True,
    cache_hash=True,
):
    asset_id: str
    bucket_index: int
    fee_rate_bps: int
    id: str
    last_update: int
    maker_address: str
    maker_orders: tuple[MakerOrder] | None
    market: str
    match_time: int
    outcome: Literal["Yes", "No"]
    owner: str
    price: str
    side: SIDE
    size: str
    status: TRADE_STATUS
    taker_order_id: str
    timestamp: int
    trade_owner: str
    trader_side: TRADER_SIDE
    transaction_hash: str
    type: Literal["TRADE"]

    @property
    def event_type(self) -> str:
        return "trade"


class OrderWSInfo(
    msgspec.Struct,
    forbid_unknown_fields=True,
    tag_field="event_type",
    tag="order",
    frozen=True,
    cache_hash=True,
):
    asset_id: str
    associate_trades: tuple[str] | None
    created_at: int
    expiration: int
    id: str
    maker_address: str
    market: str
    order_owner: str
    order_type: TIME_IN_FORCE
    original_size: str
    outcome: Literal["Yes", "No"]
    owner: str
    price: str
    side: SIDE
    size_matched: str
    status: INSERT_STATUS
    timestamp: int
    type: Literal["PLACEMENT", "UPDATE", "CANCELLATION"]

    @property
    def event_type(self) -> str:
        return "order"
