# PolyPy

> [Polymarket](https://www.polymarket.com) Python Wrapper - PolyPy


Goals
-----
`PolyPy` is a Python wrapper around [Polymarket's API](https://docs.polymarket.com/) and aims to implement a 
set of components for facilitating trading on Polymarket, whilst focusing on decent runtime performance.
`PolyPy`'s implementation is opinionated and biased regarding architecture and design choices, and therefore tailored
towards personal projects/ use in trading bots (e.g. running different strategies on the same account).
  
Although, [Polymarket's Python Client](https://github.com/Polymarket/py-clob-client/tree/main) is quite convenient, 
it is also comparably slow, i.e. it uses JSON-parsing from stdlib instead of much faster 
[msgspec](https://jcristharif.com/msgspec/), native dataclasses instead of 
[attrs](https://www.attrs.org/en/stable/init.html), etc., and has some disadvantageous implementation details.
  
> POLYPY IS NOT AN OFFICIAL POLYMARKET IMPLEMENTATION OR AFFILIATED WITH POLYMARKET IN ANY WAY!
> Please mind, that PolyPy is just a hobby project, might be subject to bugs, and is no professional software!

Quickstart
----------

````python
import time
import polypy as plp

# create an order book, which keeps tracks of resting limit orders
# due to the complementary character of the unified order book for YES and NO tokens,
# we only need to track one them
book = plp.OrderBook(token_id="...", tick_size=0.01)

# assign order book to a market stream, which updates the order book automatically in a separate thread
# this way, the order book is always up-to-date (including tick size)
market_stream = plp.MarketStream(
    ws_endpoint=plp.ENDPOINT.WS, books=book, check_hash_params=None, rest_endpoint=plp.ENDPOINT.REST
)
market_stream.start()

# create an order manager, which is used to create and manipulate orders, store and manage them
order_manager = plp.OrderManager(
    rest_endpoint=plp.ENDPOINT.REST,
    private_key="...",
    api_key="...",
    secret="...",
    passphrase="...",
    maker_funder="your_wallet_addr",
    signature_type=plp.SIGNATURE_TYPE.POLY_PROXY,
    chain_id=plp.CHAIN_ID.POLYGON
)

# create a position_manager with an initial bankroll (100 USDC), which stores and manages current positions (holdings)
position_manager = plp.PositionManager(rest_endpoint=plp.ENDPOINT.REST, gamma_endpoint=plp.ENDPOINT.GAMMA, usdc_position=100)

# assign order_manager and position_manager to a user stream
# this way, orders in order_manager and positions in position_manager will be updated automatically (e.g. if an 
# order is executed, the corresponding position will be created and/or updated in the position manager, and the
# order will be updated in the order_manager)
user_stream = plp.UserStream(
    ws_endpoint=plp.ENDPOINT.WS,
    tuple_manager=(order_manager, position_manager),
    market_triplets=("market_id", "yes_token_id", "no_token_id"),
    api_key="...",
    secret="...",
    passphrase="..."
)
user_stream.start()

# post an order 1 tick size above current best bid
best_bid = book.best_bid_price
order, response = order_manager.limit_order(
    price=best_bid + book.tick_size,
    size=10,
    token_id="yes_token_id",
    side=plp.SIDE.BUY,
    tick_size=book.tick_size,
    tif=plp.TIME_IN_FORCE.GTC,
    expiration=None,
    neg_risk=None
)

# check after 10 seconds
time.sleep(10)

# check order status
status = order_manager.get_by_id(order.id).status

# check cash position
balance = position_manager.balance

# check buying power (to submit next buy order)
position_manager.buying_power(order_manager)
````


Documentation
-------------
See [documentation](docs/guide.md) and [examples](examples).

Disclaimer
----------
_PolyPy_ is written for educational purpose only. The author of this software and accompanying
materials makes no representation or warranties with respect to the accuracy, applicability, fitness, or completeness of
the contents. Therefore, if you wish to apply this software or ideas contained in this software, you are taking full
responsibility for your action. All opinions stated in this software should not be taken as financial advice. Neither
as an investment recommendation. This software is provided "as is", use on your own risk and responsibility.  
  
PolyPy is licensed under GNU GPLv3.  
> __USE AT YOUR OWN RISK!__


Improved Implementation over `py_clob_client`
---------------------------------------------
- 3x faster order creation (no unnecessary REST calls)
- Market Sell orders, which are not available `py_clob_client`
- Local order book implementation
- Position implementation
- Market channel (order book updates) and user channel (order and position updates) stream implementation
- Fixed erroneous rounding routines in `py_clob_client`, i.e. `round_floor(4.6, 2) == 4.59 != 4.6`
- Fixed erroneous calculation of marketable price compared to `py_clob_client`
- Fixed bug in Orderbook hash compared to `py_clob_client` (especially, after receiving a "price_change" websocket message)
- Fixed bug in get_tick_size compared to `py_clob_client` which incorrectly caches tick sizes locally

Project Status
--------------
This project is under active development and its function and class signatures, as well as the repository structure 
will be subject to future changes.  
Quite a few of the REST calls are not yet implemented (e.g., _get_spread()_), as development currently focuses on features 
needed for personal projects/trading bots - though implementing REST calls by oneself should be relatively easy.  
Large parts of the code base are not yet fully tested, though most common classes and functions are covered 
fairly thoroughly.  
  
Especially `MarketStream` and `UserStream` will be subject to future changes (design pattern).

Known Issues
------------
- When defining a Market Order, amount will be rounded to the same number of decimal places as orders, which 
defaults to 2 - even though amount usually is round to 4 to 5 decimal places (depending on tick size). This is done in
`py_clob_client` as well. To maintain functional compatibility, `polypy` adopts this logic, though this might change 
in the future.
- Floats incur so-called "floating point imprecision", e.g. `0.1 + 0.2 = 30000000000000004`. `polypy` tries to
counteract this by appropriate rounding. Though, if precision is important, use Python's `decimal.Decimal` type.
Future versions might implement fixed point types instead of floats. Due to rounding implementation, this also
should be noticeable faster.

Todo and Planned Features
-------------------------
See [TODO.md](TODO.md) for a complete list of planned features and open fixes.

Development
-----------
### Developer Tooling
- black
- isort
- [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/): ```pytest --cov=polypy tests/``` or ```pytest --cov-report term-missing --cov=polypy tests/``` 
- pytest
- [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html)

### Requirements
- __python >= 3.11__ 
- see [requirements.txt](requirements.txt).

### Change Log
#### 2025/06/03
- In `OrderBook`:
  - Change `zeros_factory` to `zeros_factory_bid` and `zeros_factory_ask`
  - Fix minor bug in `from_dict` regarding tick size inference
  - Add `strict_type_check` argument to control checking routine w.r.t. of dtypes
#### 2025/05/28
- Add argument `untrack_order_by_trade_terminal` to `UserStream` to automatically untrack/clean orders in `OrderManager` by `TradeWSInfo` (websocket message) if `Order.status` in `TERMINAL_INSERT_STATI` and `TradeWSInfo.status` in `TERMINAL_TRADE_STATI` (though this will fail if `CONFIRMED` status of `TradeWSInfo`-message in websocket was missed)
- Add argument `cover_mode` to `UserStream` controlling the behavior when checking if all `token_id` of `OrderManager`s are contained in `market_triplets` (which, i.e. might not be the case if an `OrderManager` is assigned to multiple `UserStream`s)
- Fix bug in `UserStream` where positions in default `PositionManagers` where not automatically untracked based on `untrack_trade_status`
- Fix bug in `UserStream` when untracking a trade message, whose position is not in `PositionManager` any more
- Fix minor bug in `OrderManager.clean`, which now cleans per order status OR order expiration. Expiration-based cleaning now works also even if no order status is specified (which previously only worked if an order status was set as well)
#### 2025/05/25
- Simpler re-implementation of `MarketIdTriplet` and `MarketIdQuintet`
#### 2025/05/24
- Implement `__getstate__` and `__setstate__` for `OrderManager`, `PositionManager` and `RPCSettings`
#### 2025/05/20
- `FAK` (alia `IOC`) as additional `TIME_IN_FORCE` option
#### 2025/04/29
- GAMMA API can only retrieve up to 20 items at once (`get_markets_gamma_model`, `get_events_gamma_model`)
- Implement `__getnewargs__` for `MarketIdQuintet` and `MarketIdTriplet` to make them pickable
- Implement `coerce_inbound_prices` argument in `OrderBook`:
  - Occasionally, price quotes in the order book are outside \[0, 1\] (for unknown reasons, this seems to be an issue on Polymarket's side though)
  - This results in Out-Of-Index exceptions
  - If `coerce_inbound_prices=True`, invalid prices outside \[0, 1\] will be accumulated into prices `0` and `1` - though current implementation is not yet the most efficient
#### 2025/04/20
- Pre-compute `order_id`
  - OrderPlacementException (e.g., OrderPlacementFailure, OrderPlacementDelayed, etc.) returns order in its `.order` attribute (append order object to exception)
  - in `OrderManager`, order can now be tracked before being submitted to Polymarket via REST
  - no need to wait on REST response anymore to get oder_id (no need for buffering in UserStream anymore, though this fix is only scheduled in the future and not yet done...)
- Rename `MTX_TYPE` to `RPC_TYPE`
#### 2025/04/15
- In UserStream, when parsing TradeWSInfo in case of a taker order, use all maker orders to transact positions, which
is more accurate than taking the taker price information
- Renaming of 'sig_digits' to 'n_digits', and 'update_augmented_conversions' to 'pull_augmented_conversions'
#### 2025/03/31
- Implement monitoring ex-post added outcomes/conditions in augmented negative risk markets in UserStream
- Implement Gamma API (`get_markets_gamma_model`, `get_events_gamma_model`, `get_neg_risk_market`)
- Implement fetching "midpoints" for resolved markets in PositionManager (via Gamma API)
- Add `gamma_endpoint` as argument for PositionManager
- Implement on-chain actions: split, merge, redeem, convert
- Implement namedtuple 'MarketIdTriple' (e.g., used in UserStream) and 'MarketIdQuintet' (used in PositionManager)
