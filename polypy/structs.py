import datetime
from typing import Literal, Union

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
    outcome: str
    price: float
    winner: bool


class RewardRate(msgspec.Struct, forbid_unknown_fields=True):
    asset_address: str
    rewards_daily_rate: float


class Rewards(msgspec.Struct, forbid_unknown_fields=True):
    rates: list[RewardRate] | None
    min_size: float | None
    max_spread: float | None


class MarketInfo(msgspec.Struct, forbid_unknown_fields=True):
    enable_order_book: bool
    active: bool
    closed: bool
    archived: bool
    accepting_orders: bool
    accepting_order_timestamp: datetime.datetime | None
    minimum_order_size: float
    minimum_tick_size: float
    condition_id: str
    question_id: str
    question: str
    description: str
    market_slug: str
    end_date_iso: datetime.datetime | None
    game_start_time: datetime.datetime | None
    seconds_delay: float
    fpmm: str
    maker_base_fee: float
    taker_base_fee: float
    notifications_enabled: bool
    neg_risk: bool
    neg_risk_market_id: str
    neg_risk_request_id: str
    icon: str
    image: str
    rewards: Rewards | None
    is_50_50_outcome: bool
    tokens: list[TokenInfo]
    tags: list[str] | None


class MarketsResponse(msgspec.Struct, forbid_unknown_fields=True):
    limit: int
    count: int
    next_cursor: str
    data: list[MarketInfo]


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
    side: SIDE
    outcome_index: int


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
    maker_orders: list[MakerOrder] | None
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


class RelayerResponse(msgspec.Struct, forbid_unknown_fields=True):
    transactionID: str
    transactionHash: str
    state: Literal[
        "STATE_NEW",
        "STATE_EXECUTED",
        "STATE_MINED",
        "STATE_INVALID",
        "STATE_CONFIRMED",
        "STATE_FAILED",
    ]


class OrderSummary(msgspec.Struct, forbid_unknown_fields=True):
    price: str
    size: str


class BookSummary(msgspec.Struct, forbid_unknown_fields=True):
    market: str
    asset_id: str
    timestamp: int
    hash: str
    bids: list[OrderSummary]
    asks: list[OrderSummary]

    min_order_size: str
    tick_size: str
    neg_risk: bool

    @property
    def event_type(self) -> str:
        return "summary"


class PriceChangeSummary(msgspec.Struct, forbid_unknown_fields=True):
    asset_id: str
    price: str
    size: str
    side: SIDE
    hash: str
    best_bid: str
    best_ask: str


class BookEvent(
    msgspec.Struct,
    forbid_unknown_fields=True,
    tag_field="event_type",
    tag="book",
):
    market: str
    asset_id: str
    timestamp: int
    hash: str
    bids: list[OrderSummary]
    asks: list[OrderSummary]

    @property
    def event_type(self) -> str:
        return "book"


class PriceChangeEvent(
    msgspec.Struct,
    forbid_unknown_fields=True,
    tag_field="event_type",
    tag="price_change",
):
    market: str
    price_changes: list[PriceChangeSummary]
    timestamp: int

    @property
    def event_type(self) -> str:
        return "price_change"


class TickSizeEvent(
    msgspec.Struct,
    forbid_unknown_fields=True,
    tag_field="event_type",
    tag="tick_size_change",
):
    market: str
    asset_id: str
    old_tick_size: str
    new_tick_size: str
    timestamp: int

    @property
    def event_type(self) -> str:
        return "tick_size_change"


class LastTradePriceEvent(
    msgspec.Struct,
    forbid_unknown_fields=True,
    tag_field="event_type",
    tag="last_trade_price",
):
    market: str
    asset_id: str
    fee_rate_bps: str
    price: str
    side: SIDE
    size: str
    timestamp: int

    @property
    def event_type(self) -> str:
        return "last_trade_price"


MarketEvent = Union[BookEvent, PriceChangeEvent, TickSizeEvent, LastTradePriceEvent]
