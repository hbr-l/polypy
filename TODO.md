# TODO

__bold__ -> scheduled next

## Features 
_(new/add)_
- [x] Implement rpc methods: redeem(), merge() and split() and add methods to PositionManager
  - (split and) merge:
    - https://github.com/Polymarket/examples/blob/main/examples/safeWallet/merge.ts
    - https://github.com/Polymarket/conditional-token-examples-py/blob/main/ctf_examples/merge.py
    - https://github.com/Polymarket/py-merge-split-positions/blob/main/merge-split.py
    - https://github.com/PolyTrader/polymarket-trading/tree/main/polymarket
  - redeem:
    - https://github.com/Polymarket/examples/blob/main/examples/safeWallet/redeem.ts
    - https://github.com/Polymarket/conditional-token-examples-py/blob/main/ctf_examples/redeem.py
- [x] SharedMemory implementation of `polypy.book.OrderBook` (enables MarketStream to run in separate process instead of only thread) -> SharedOrderBook
- [ ] __Better cache size handling__ (@lru_cache)
- [ ] __Integrate rate limiting function__ (global var which holds callback function)
- [ ] __Typing SharedOrderBook__ (currently not recognized as OrderBookProtocol)
- [ ] Implement function complement_book() in `polypy.book` to invert order book (complementary outcome)
- [ ] Add callback at on_setattr of order.status, which is called everytime INSERT_STATUS changes (more sophisticated order management)
- [ ] Implement auto-redeem function (currently not implemented since we need conditionId and tokenId but position managers only hold tokenIds)
- [ ] Add/implement more REST methods to polypy.rest.api.py
- [ ] historical redeem data for wallet: http GET https://data-api.polymarket.com/activity?type=REDEEM&user=<wallet-id> -> see [Polymarket Data API](https://polymarket.notion.site/Polymarket-Data-API-Docs-15fd316c50d58062bf8ee1b4bcf3d461)
- [ ] Fixed point numeric type (for the moment use `polypy.typing.dec()`)
- [ ] Compute order id upfront: makes buffering in UserStream obsolete (though, no docs on how to compute order id, so probably not possible...)
- [ ] Implement Trade class (analogously to Position and Order -> debatable: use `callback_msg` and `TradeWSInfo` in UserStream instead)
- [ ] Implement TradeManager analogously to OrderManager and PositionManager (additionally, merge trades into top-level trade 
if trade was split into multiple separate transactions -> debatable: use `callback_msg` and `TradeWSInfo` in UserStream instead)
- [ ] Implement Client class, that combines
  - Order Manager
  - Position Manager
  - Trade Manager
  - Order books
  - Market stream
  - User stream
  - Subscribing and removing of markets
  - Auto-setting `strategy_id`
- [ ] (Virtual) Stop-Loss orders (separate thread/process to check trigger conditions)
- [ ] SharedMemory implementation of `polypy.order.base.Order` (enables UserStream to run in separate process instead of only thread) 
- [ ] SharedMemory implementation of OrderManager and PositionManager:
  - [ ] SharedMemory implementation e.g., ManagedDict 
  - [ ] move `lock`/mutex to a _(sharedMem) dict factory_ and make all operations atomic (enable OrderManager and PositionManager across processes instead of only threads)
- [ ] Check `position_manager.buying_power` before submitting order (still skeptical, this should rather be managed by the user himself to keep overhead as low as possible)
- [ ] Implement `polypy.rewards` for basic rewards calculation

## Refactoring
_(backward-incompatible changes, changes in signatures)_
- [ ] __price_change handling chaotic and inefficient (price change summaries): on_msg of MarketStream__
- [ ] __replace `_get_or_create_position` with `get_by_id` in `rpc_proc.py` and remove `_get_or_create_position` from `PositionManagerProtocol`__
- [ ] __remove buffer from `UserStream` and adapt test_userstream (no buffer tests anymore)__
- [ ] __replace np.ndarray with NDArray for better type annotations__
- [ ] Standardize naming of `asset_id` vs `token_id`, `market_id` vs `condition_id`
- [ ] More specialized exception classes (esp. in OrderManager, PositionManager, UserStream) of `PolyPyException`
- [ ] Better substitute for `isinstance(...)` where possible
- [ ] Specialized msgspec.Struct instead of dicts for get_markets_gamma_model and get_events_gamma_model to resolve tight coupling in 
  - PositionManager._fill_no_orderbook_midpoints(...)
  - rest.api.get_neg_risk_market(...)
- [ ] More msgspec.Structs for decoding JSON
- [ ] Use more TypeVar for better typing (e.g., in `create_limit_order`, `create_market_order`)
- [ ] Remove INSERT_STATUS.DEFINED from `polypy.order.common.CANCELABLE_INSERT_STATI` (debatable...?, tend to keep)

## Fix
_(backward-compatible changes, no changes in signatures)_
- [x] `_tx_post_convert_positions` might induce numerical instability, alternative: set `price=0` and use separate `position_manager.deposit(size * (N - 1))`,
but this might mess with specific `PositionProtocol` implementation (resolved: let user choose bookkeeping method via args)
- [x] Rewrite `MarketStream`: 
  - [x] use AbstractStreamer
  - [x] if specified, merge complement asset_id (invert before) into order book as well (discarded)
  - [x] factor out _BookHashChecker_ for managing hash checks -> not necessary
  - [x] MarketStreamer._process_raw_message: really discard if locked? (original intention: minimize backpressure)
- [x] Rounding `amount` in market order to `amount_digits` (4 to 5 decimal places) instead of currently to `order_size_digits` (2 decimal places) (py_clob_client behavior=order_size_digits for now) -> this actually seems to be intended by Polymarket...
- [ ] __test `ShareLock` and `SharedRLock` on Posix__
- [ ] __`untrack_order_by_trade_terminal` in `UserStream` might fail in case of `CONFIRMED` status of `TradeWSInfo`-message was missed__
  (documentation: to be on the safe side, untrack/clean orders manually from `OrderManager` by `order.id` of return value when using
  `OrderManager.limit_order`, `OrderManager.market_order` or `OrderManager.post_order` if and only if (!) full taker order, 
  or even better by manually filtering for orders with order.status=MATCHED and 
  order.created_order < now-threshold (which also works for maker orders or partial taker orders !) - though this rarely should be the case because 
  this will only be necessary if `CONFIRMED` was missed)
- [ ] __Port `polypy.stream.common.AbstractStreamer` to async (performance and buffering)__
- [ ] Optimize `_coerce_inbound_idx()` (though not urgent)
- [ ] Parse `amount` in `_raise_post_order_exception` instead of defaulting to OrderPlacementFailure for better order update in case of an exception
- [ ] Additional checks for `all_market_quintets` in `_check_conversion_all_quintets(...)` (used in PositionManager.convert_positions and its conversion cache) (necessary?, probably not and current implementation is sufficient)
- [ ] Profile and optimize PositionManager rpc methods
- [ ] Better cache size handling instead of hard coding
- [ ] Fix typing of `FrozenOrder` and `FrozenPosition` (currently, autocomplete does not work for class attributes and methods)
- [ ] Rounding routines cost a lot of compute time
- [ ] Premature double rounding in `polypy.order.limit.limit_order_taker_maker_amount` and `polypy.order.market.market_order_taker_maker_amount`
- [ ] Lack of idempotency in `Position` and `CSMPosition` (see class docstring e.g., if duplicate transaction with identical trade_id is performed twice), 
though miscalculation will only happen when server-side error.
Additionally, both classes build upon the assumption, that FAILED is always preceded by MATCHED (which is quite sensible nonetheless)
- [ ] Overload function annotation and type hints in `polypy.manager.order_manager` and `polypy.rest.api` for better typing ([resource](https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/))
- [ ] Raise warning if `price` is rounded due to `tick_size` in `create_limit_order` and analogously in `create_market_order`
- [ ] Check if rounding is correct in `polypy.manager.position.buying_power` (round_down all summands?)
- [ ] Check and revisit `polypy.rounding.max_allowed_decimals`: alternative implementation in comments
- [ ] Re-implement overspending protection for `CSMPosition` in `polypy.position._validate_pending_maker()` (see comments L:231)? (not really needed since Polymarket does not wait till trade_status CONFIRMED but 
enables trading positions as soon as they hit MATCHED - at least given limited and manual testing so far: repeat experiments) -> evaluate need

## Open Unknowns
- [x] GTC security threshold for expiration? c.f. `polypy.order.base.compute_expiration_timestamp` and change if necessary -> always add 1 minute
- [x] Computation of order id? (instead of waiting until REST response is received -> would make buffering in UserStream obsolete) -> done, use order hash
- [ ] __Computation of token_id based on condition_id?__
  - https://github.com/Polymarket/ctf-utils/blob/main/src/utils.ts
  - https://github.com/Polymarket/ctf-utils/blob/main/test/util.test.ts
  - makes supplying token_ids to UserStream obsolete (more convenient if only market id needs to be supplied)
- [ ] When `price_change` market stream message is received, the `timestamp` field does not conform with the `hash`. This 
leads to iterating over timestamps in order to find the correct hash if and only if the local book is still in sync. Assumption:
`timestamp` is the _time of message emission_ and not the _time of book generation_.

## Repository and Git
- [ ] Documentation
- [ ] Examples
- [ ] Setup GitHub actions
- [ ] Setup versioning
- [ ] Rework profiling
- [ ] Rework tests
