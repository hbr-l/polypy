# TODO

## Features 
_(new/add)_
- [ ] Add/implement more REST methods to polypy.rest.api.py
- [ ] Implement rpc methods: redeem(), merge() and split() and add methods to PositionManager
  - (split and) merge:
    - https://github.com/Polymarket/examples/blob/main/examples/safeWallet/merge.ts
    - https://github.com/Polymarket/conditional-token-examples-py/blob/main/ctf_examples/merge.py
    - https://github.com/Polymarket/py-merge-split-positions/blob/main/merge-split.py
    - https://github.com/PolyTrader/polymarket-trading/tree/main/polymarket
  - redeem:
    - https://github.com/Polymarket/examples/blob/main/examples/safeWallet/redeem.ts
    - https://github.com/Polymarket/conditional-token-examples-py/blob/main/ctf_examples/redeem.py
    - historical redeem data for wallet: http GET https://data-api.polymarket.com/activity?type=REDEEM&user=<wallet-id> -> see [Polymarket Data API](https://polymarket.notion.site/Polymarket-Data-API-Docs-15fd316c50d58062bf8ee1b4bcf3d461)
- [ ] Implement function complement_book() in `polypy.book` to invert order book (complementary outcome)
- [ ] Implement TradeManager analogously to OrderManager and PositionManager (additionally, merge trades into top-level trade if trade was split into multiple separate transactions)
- [ ] Fixed point numeric type (for the moment use `polypy.typing.dec()`)
- [ ] Add callback at on_setattr of order.status, which is called everytime INSERT_STATUS changes (more sophisticated order management)
- [ ] Compute order id upfront: makes buffering in UserStream obsolete (though, no docs on how to compute order id, so probably not possible...)
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
- [ ] SharedMemory implementation of `polypy.book.OrderBook` (enables MarketStream to run in separate process instead of only thread) 
- [ ] SharedMemory implementation of OrderManager and PositionManager:
  - [ ] SharedMemory implementation e.g., ManagedDict 
  - [ ] move `lock`/mutex to a _(sharedMem) dict factory_ and make all operations atomic (enable OrderManager and PositionManager across processes instead of only threads)
- [ ] Check `position_manager.buying_power` before submitting order (still skeptical, this should rather be managed by the user himself to keep overhead as low as possible)
- [ ] Implement `polypy.rewards` for basic rewards calculation

## Refactoring
_(backward-incompatible changes, changes in signatures)_
- [ ] Standardize naming of `asset_id` vs `token_id`, `market_id` vs `condition_id`
- [ ] More specialized exception classes (esp. in OrderManager, PositionManager, UserStream)
- [ ] More msgspec.Structs for decoding JSON
- [ ] Use more TypeVar for better typing (e.g., `create_limit_order`, `create_market_order`)

## Fix
_(backward-compatible changes, no changes in signatures)_
- [ ] Rewrite `MarketStream`: 
  - [ ] use AbstractStreamer
  - [ ] if specified, merge complement asset_id (invert before) into order book as well
  - [ ] factor out _BookHashChecker_ for managing hash checks
  - [ ] MarketStreamer._process_raw_message: really discard if locked? (original intention: minimize backpressure)
- [ ] Fix typing of `FrozenOrder` and `FrozenPosition` (currently, autocomplete does not work for class attributes and methods)
- [ ] Rounding `amount` in market order to `amount_digits` (4 to 5 decimal places) instead of currently to `order_size_digits` (2 decimal places) (py_clob_client behavior=order_size_digits for now)
- [ ] Rounding routines cost a lot of compute time
- [ ] Premature double rounding in `polypy.order.limit.limit_order_taker_maker_amount` and `polypy.order.market.market_order_taker_maker_amount`
- [ ] Lack of idempotency in `Position` and `CSMPosition` (see class docstring e.g., if duplicate transaction with identical trade_id is performed twice), 
though miscalculation will only happen when server-side error.
Additionally, both classes build upon the assumption, that FAILED is always preceded by MATCHED (which is quite sensible nonetheless)
- [ ] Port `polypy.stream.common.AbstractStreamer` to async (performance and buffering)
- [ ] Overload function annotation and type hints in `polypy.manager.order_manager` and `polypy.rest.api` for better typing ([resource](https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/))
- [ ] Replace usage of `PolyPyException` with more specific (base) exceptions
- [ ] Raise warning if `price` is rounded due to `tick_size` in `create_limit_order` and analogously in `create_market_order`
- [ ] Check rounding correct in `polypy.manager.position.buying_power` (round_down all summands?)
- [ ] Check and revisit `polypy.rounding.max_allowed_decimals`: alternative implementation in comments
- [ ] Re-implement overspending protection for `CSMPosition` in `polypy.position._validate_pending_maker()` (see comments L:231)? (not really needed since Polymarket does not wait till trade_status CONFIRMED but 
enables trading positions as soon as they hit MATCHED - at least given limited and manual testing so far: repeat experiments) -> evaluate need

## Open Unknowns
- [ ] Computation of order id? (instead of waiting until REST response is received)
- [ ] Computation of token_id based on condition_id?
  - https://github.com/Polymarket/ctf-utils/blob/main/src/utils.ts
  - https://github.com/Polymarket/ctf-utils/blob/main/test/util.test.ts
  - https://chatgpt.com/c/678d8f6d-c324-8001-b919-d9b35a3772d1
- [ ] GTC security threshold for expiration? c.f. `polypy.order.base.compute_expiration_timestamp` and change if necessary
- [ ] When `price_change` market stream message is received, the `timestamp` field does not conform with the `hash`. This 
leads to iterating over timestamps in order to find the correct hash if and only if the local book is still in sync. Assumption:
`timestamp` is the _time of message emission_ and not the _time of book generation_. 

## Repository and Git
- [ ] Documentation
- [ ] Examples
- [ ] Setup Github actions
- [ ] Setup versioning
- [ ] Rework profiling
- [ ] Rework tests
