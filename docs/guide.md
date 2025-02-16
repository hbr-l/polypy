# User Guide
>work in progress

## Introduction
`polypy` is a wrapper around Polymarket's REST and websocket interface.
`polypy`'s implementation is biased and opinionated, and aims at delivering at least some level of comfort for interacting 
with Polymarket's interfaces, e.g., keeping track of orders and positions locally, maintaining a local oder book, etc.
Therefore, `polypy` offers custom (and to some extent customizable) implementations beyond Polymarket's very basic 
interfaces, though the very implementation might be considered opinionated towards an OOP design architecture (a more
functional package interface might be extended in the future).

### Installation
At the moment, `Polpy` has to be installed manually:
````
>> git clone https://github.com/hbr-l/polypy.git
>> cd polypy
>> pip install .
````
Instead of `pip install .`, you can also use:
````
>> pip install .[examples]  (includes dependencies for examples)
or
>> pip install -e .[dev]    (editable with development dependencies e.g., for testing)
````
Note, that depending on your terminal/shell, you might need to run `pip install '.[examples]'` (quotation marks) instead.

### Allowances and Authentication
1) In order to be able to trade on Polymarket, the correct token allowances must be set before orders can be placed via the API.
2) Certain functions and methods need an authentication via the account's api key, private key (for signing, will not 
actually be sent in any payload), secret, passphrase and wallet address (__NEVER SHARE OR PUBLISH YOUR PRIVATE KEY__).
  
_`polypy` does neither support setting allowances, nor creating API keys, as this is outside the scope of this package._
  
Please refer to [py-clob-client](https://github.com/Polymarket/py-clob-client/tree/main), the official 
[Polymarket documentation ](https://docs.polymarket.com/#introduction) or Polymarket's Discord for allowances and authentication.  
If you are using Mail/Magic-Login, please refer to [Magic-Login](#magic-login), which simplifies allowances and authentication
quite a lot (i.e., actually not necessary when using Magic-Login).

#### Magic-Login
If you use Mail/Magic-Login (btw. __NEVER TYPE IN A NUMBER CODE INTO MAGIC'S LOGIN PAD THAT YOU HAVEN'T ACTIVELY 
REQUESTED YOURSELF__), you can obtain the necessary credentials by:
1) Open DevTools of your browser and search for a websocket named "user". The first websocket message sent, contains
your api key, secret and passphrase
2) Your wallet address (usually used in `polypy` for `maker_funder`/`maker` argument) is displayed on Polymarket's website 
beneath your profile image (starts with "0x...."). Be aware, that is not the same address as the address where you deposit to!
3) Your private key (__NEVER SHARE YOUR PRIVATE KEY__) can be obtained via Polymarket's website: Go to your profile,
click "Edit profile" and then "Export Private Key".

### Floating Point Imprecision, Rounding and Decimal Type
In general, not all real numbers can be represented by floating point numbers. This stems from storing number in a 
binary number system and is independent of the programming language used:
````python
0.1 * 3
>> 0.30000000000000004  # instead of 0.3
````
These small errors might accumulate over the course of multiple arithmetic operations and might lead to unexpected 
results - especially when handling currencies and monetary units.  
  
As already stated, this really is independent of the very programming language in use.
`polypy` tries to minimize errors due to floating point imprecision by implementing dedicated rounding approaches in the 
`polypy.rounding` module (e.g., `polypy.rounding.round_half_even`), but actually calling these explicitly is usually not 
necessary.
Nonetheless, error might accumulate, and it is recommended to use Python's `decimal.Decimal` type instead (which is also faster, 
because of dedicated rounding methods for Decimal types).
  
For Decimal types as well as zeros array (analogously to `np.zeros`):
````python
import polypy as plp

x = plp.dec(0.1)
>> Decimal("0.1")

zeros = plp.zeros_dec(3)
>> np.array([Decimal("0"), Decimal("0"), Decimal("0")], dtype=object)
````

### Rate Limits, Trading Limits and Subscription Limits
_Note: the following limits are imposed by Polymarket and are independent of `polypy`s implementation._  
  
As of the time of writing this documentation, following limits apply:
1) __Rate limits__ for interacting with Polymarket REST:
   - `polypy.ENDPOINT.REST`: 125 requests/10 seconds
   - `polypy.ENDPOINT.DATA`: 100 requests/10 seconds
2) __Trading limits__ for trading on Polymarket:
   - min. limit order size: 5 shares
   - min. market order amount: 1 USDC
   - 10 orders/second - 100 burst
   - 2 orders/second per book (4/s by market) - 20 burst
   - 1 second minimal interval between order submission and cancellation
3) __Subscription limits__ for websockets:
   - `polypy.ENDPOINT.WS`: max. 500 subscriptions (e.g., market or asset IDs)

> Above limits might or might not be outdated. Users are advised to confirm limits beforehand. Unfortunately, there is
> no official documentation, but only Discord conversations regarding rate limits - take it with a grain of salt!

## Central Limit Order Book (CLOB)
### What is a CLOB?
Polymarket provides a L2 (central) limit order book (aka. 'CLOB').
A L2 limit order book shows the aggregated/summed volume of all resting limit orders at each price level.
  
Given the nature of prediction markets, each market (aka. bet) consists of either Yes or No shares. Each of these shares 
is implemented as a separate token, s.t. each market consists of two (or in case of multiple outcomes, multiple) tokens.
  
Each token has its own limit order book, though they are directly coupled via a complementary relationship:
Buying 1 Yes share is equivalent to selling 1 No share, and vice versa.  
Therefore, the bid of a Yes share (buy order) is directly related to the ask of a No share (sell order):  
$$p_{bid}^{yes} = 1 - p_{ask}^{no}$$  
and quantities:  
$$q_{bid}^{yes} = q_{ask}^{no}$$  
  
Because of this complement, both tokens in essence point to the same unified limit order book.  
By this, it is sufficient to only track the order book for one token, as the complement can easily be computed with above 
formulas.  
Also note, that best bid and best ask do not necessarily need to sum up to 1 (can be greater or less), as prices are 
solely set by market participants and their resting limit orders in the CLOB.  
For further reading about how the unified complementary CLOB works, please refer to: [An In-Depth Overview of Polymarket's Market Making Rewards Program](https://mirror.xyz/polymarket.eth/TOHA3ir5R76bO1vjTrKQclS9k8Dygma53OIzHztJSjk)

### Tick sizes
In general, each share price can only take on values in the interval `(0, 1)`.  
If a bet is won, the shareholder will be paid back 1 USDC, and if the bet is lost, the initial principal is lost to the 
counterparty.

Furthermore, share prices can only take on discrete levels, which are a multiple of the so called 'tick size' (just as stock
prices), e.g. '0.50', '0.51', '0.52', ... for a tick size of 0.01 USDC - meaning share prices can only move/jump by the 'tick size'.
Markets usually start at a tick size of 0.01 USDC (1 cent), but might change to 0.001 USDC if $p > 0.96$ (above) or $0.04 > p$ (below). 
Prices must conform to $ticksize < p <= 1 - ticksize$.

You can get the tick size by:
````python
import polypy as plp

token_id = "..."    # ID of token
tick_size = plp.get_tick_size(plp.ENDPOINT.REST, token_id)
````
or as we see later on:

````python
import time
import polypy as plp

token_id = "..."  # needs to be specified by user
tick_size = plp.get_tick_size(plp.ENDPOINT.REST, token_id)
# or set manually if you're certain where the current price is: 
# tick_size = 0.01 or tick_size = 0.001

book = plp.OrderBook(token_id, tick_size)
stream = plp.MarketStream(plp.ENDPOINT.WS, book, None, plp.ENDPOINT.REST)
stream.start()

while True:
   tick_size = book.tick_size   # up-to-date tick size
   ...
````
The `stream` listens to a websocket and continuously updates the order book, including the tick size 
(note: no stream = no live updates = no up-to-date tick size). See later paragraphs for more explanations.

### CLOB in `polypy`: `polypy.OrderBook`
````python
import polypy as plp

token_id = "..."    # needs to be specified by user
tick_size = 0.01

book = plp.OrderBook(token_id, tick_size)
````

The `OrderBook` class stores price and quantities (=size) for bids and asks at each level, as well as the tick size, as 
properties:
````python
import polypy as plp
book: plp.OrderBook

# book keeps track of (excerpt):
best_ask_p = book.best_ask_price    # numeric value, same type as book.dtype (see next paragraphs)
>> np.float64(0.254)

best_ask_q = book.best_ask_size     # numeric value, same type as book.dtype
>> np.float64(150.0)

mid = book.midpoint_price       # numeric value, same type as book.dtype
>> np.float64(0.235)

tick_size = book.tick_size      # numeric value, same type as book.dtype
>> 0.001

arr_ask_prices = book.ask_prices    # array of ask prices: contains only prices with non-zero sizes 
>> array([0.254, 0.37 , 0.4  , 0.999])

arr_ask_sizes = book.ask_sizes      # array of non-zero ask sizes, same type as book.dtype
>> array([150, 200, 125, 150000])

arr_raw_prices, arr_raw_sizes = book.asks
>> (array([0., 0.001, 0.002, ..., 0.998,   0.999, 1.], shape=(1001,)),
    array([0.,    0.,    0., ...,    0., 150000., 0.], shape=(1001,)))
# raw arrays: contains zero-size elements as well and has therefore length (1/tick_size)+1

bid_q = book.bid_size(0.2)          # get bid size at bid price = 0.2, same type as book.dtype (not dependent on input type)
>> np.float64(200.0)

bids_summary = book.bids_summary    # list of {"price": str(...), "size": str(...)} dicts per level
>> [{'price': '0.001', 'size': '211.27'},
    {'price': '0.002', 'size': '222'},
     ...,
    {'price': '0.216', 'size': '20'}]
````
_*Note: return type depends on which `zeros_factory` was defined at \_\_init\_\_ (see [Numeric Type](#numeric-type-float-vs-decimal)) 
and, in case of np.float64 vs np.float32, on your system._

- Asks are sorted in ascending order: best ask (lowest) at index 0.  
- Bids are sorted in descending order: best bid (highes) at index 0.  

Thereby, if you want to plot an order book depth chart, you can easily and directly use `np.cumsum` without the 
need to re-sort any arrays.
See [examples/order_book_streaming.py](../examples/order_book_streaming.py):    
<img src="order_book_depth_chart.png" alt="depth_chart" width="1000"/>

#### Marketable Price
The `OrderBook` can also be used to calculate the marketable price, which is the price at which an order would be filled 
in its entirety (immediate fill by crossing the spread):
````python
import polypy as plp

book = plp.OrderBook(...)
market_price = book.marketable_price(plp.SIDE.BUY, 150)
>> (np.float64(0.4), np.float64(162.1))
# marketable price, cumulative size, i.e.: 
#   there are currently 162.1 shares at price 0.4, which would match an order of size 150 immediately
````
Above example calculates the price at which a buy (marketable/aggressive limit) order of 150 USDC volume would fill immediately.
If no marketable price can be calculated, e.g., the amount exceeds the liquidity in the book, an `OrderBookException` will be raised. 
  
#### Numeric Type: Float vs. Decimal
The numeric type (cf. [Floating Point Imprecision](#floating-point-imprecision-rounding-and-decimal-type)) of the `OrderBook` 
can be controlled via the `zeros_factory` argument in `__init__`.  
By default, this is a zeros-array of type np.float64, though it is recommended to use the Decimal type via `plp.zeros_dec` 
as argument for `zeros_factory`:
````python
import polypy as plp

book = plp.OrderBook("[token_id]", plp.dec(0.01), zeros_factory=plp.zeros_dec)
book.dtype
>> Decimal
````

#### Manipulating the OrderBook
Whilst `OrderBook` exposes some methods to manipulate the order book directly (e.g., writing bids and asks into), usually 
there is absolutely no need to do so manually.
````python
# methods to manipulate book manually (excerpt):
book.set_asks(list_ask_prices, list_ask_sizes)      # override just new values
book.reset_asks(list_ask_prices, list_ask_sizes)    # override with new values and set all other values to zero
````
Instead, the `OrderBook` should rather be assigned to a [MarketStream](#market-stream), which updates the order book 
in real time within a separate thread, s.t. the user does not have to care for updating or manipulating the order book manually.
The `OrderBook` is designed s.t. only its properties and attributes should be accessed but not set (unless for a good reason).
  
#### Manually Updating the OrderBook
If you do not want to assign the order book to a `MarketStream`, then you still can update manually via:
````python
import polypy as plp

book = plp.OrderBook(...)

book.sync(plp.ENDPOINT.REST)    # update bids and asks (fetches data)
book.update_tick_size(plp.ENDPOINT.REST)    # update tick size (fetches data)
````

#### Advanced: Order Book Hash, Multiprocessing
The current state of the order book (bids and asks) can be captured via hashing:
````python
import polypy as plp

book = plp.OrderBook(...)

market_id = "..."   # market ID to which the token belongs to (do not confuse it with the token ID!)
timestamp = ...     # int: time of order book generation in millis
book.hash(market_id, timestamp)
````
The timestamp denotes the time of order book generation, which is when the order book last changed.
The hash can be used, to prove the conformity of a locally maintained order book (or can be used as e.g., a key for a dict).
Usually, the user does not need to care about the hash as `polypy` will check hash conformity with incoming websocket 
messages automatically.
  
The default implementation is not suitable for multiprocessing. If multiprocessing is necessary, the `zeros_factory` must 
return an array of zeros which is capable of being used within multiprocessing - typically one would choose a shared memory 
implementation for this purpose (_multiprocessing.SharedMemory_). Furthermore, the zeros array returned by `zeros_factory` 
must implement adequate locking/mutex as `OrderBook` does not use (and does not need) any sophisticated locking.

## Orders
In general, to perform a trade, an order has to be submitted to the order book, which then either 1) gets (partially) matched 
directly to a resting limit order in the order book, 2) will be posted to the order book as a resting limit order until it 
gets (partially) matched by a taker order, or 3) will be canceled if not filled or expired - the concrete behavior will 
depend on the very order type.
  
Thereby, an order is just a quote to buy or sell a token/asset, and only results in a real monetary transaction (aka position) 
if a counterparty is matched at the quoted price.
  
In the following section, we first define some nomenclature regarding different order types and their behavior, and then 
showcase how to place and manage orders. The section following this section will then elaborate on positions and how to 
manage those.

### Order Types
On Polymarket, order types differ in the following:
1) Maker vs Taker: whether an order does not match and will be posted on the order book ('makes' liquidity on the book), 
or order matches immediately ('takes' liquidity out of the book)
2) Limit vs Market: whether a number of shares ('size') at a corresponding price or just an amount in USDC ('amount') 
is specified in the order (note: Polymarket's definition of limit and market order differs from the conventional definition)
3) Time-in-Force/ expiration: When the order expires

#### Maker Order vs Taker Order
If you post an order and there is no counterparty willing to trade at your specified price and size, then no monetary
transaction will take place and the order will be resting on the order book. Therefore, you provide (aka 'make') liquidity 
in the order book, until a counterparty comes along willing to trade at your specified price and size. This 
is called a __maker order__.
  
If you instead post an order, which can be matched to a resting order on the order book, the transaction will take place 
and therefore, you take liquidity from the order book. Apart from the clearing and settlement mechanism (i.e. mining and 
confirming the trade on the blockchain), the transaction takes place immediately. This is called a __taker order__.
  
Note, that maker as well as taker order can be matched partially, e.g. if your order specifies 10 shares, it might be 
the case, that a counterparty is only willing to trade 5 shares, such that you remain with an open order of 5 shares. 
If you want to avoid partial fills, you can set the "Time-in-Force" to FOK - "Fill-Or-Kill", which is always a 
taker order and either will be matched in its entirety or will be canceled immediately (though, it is not possible to 
define FOK for resting limit orders - those can always incur partial fills).

#### Limit Order vs Market Order
A limit order defines a specific price and number of shares at which you want to buy or sell. If the order can be 
matched against resting orders on the order book sufficient to your specified price or better, then the order will 
be executed immediately (meaning transaction of shares and currency with one or more counterparties). In this 
case the limit order is a _taker order_. Note, that your order can be matched against multiple other orders at 
different price levels (as long as they are equal or better to your limit price e.g., if you want to buy for 0.5 USDC,
every price level <= 0.5 USDC), as well as being matched only partially if not enough size is available, which means 
the remaining order size will sit as a resting limit order on the order book.
  
If a limit order cannot be matched directly (e.g., bid (buy order) is lower than current best ask (resting sell order), 
or vice versa), it will be resting on the order book. In this case, the limit order is a _maker order_, and will only 
be matched against an arriving taker order.  
  
A market order is essentially just a limit order with a marketable price - which means it can be matched directly as a 
_taker order_. Therefore, market orders are always _taker orders_.  
  
Conversely, to common definition in e.g., stock exchanges, Polymarket makes the following distinction between both:
1) __Limit order__: defines limit price and size (number of shares)
2) __Market order__: defines an amount in USDC (and will execute as many shares for the given USDC amount as possible at 
the current market price)

|              | taker order     | maker order         | definition             |
|--------------|-----------------|---------------------|------------------------|
| market order | always          | never               | specify amount         |
| limit        | if marketable   | if not marketable   | specify price and size |
| definition   | immediate match | resting on the book |                        |

#### Time-in-Force
_Time in Force_ (TiF) defines the order behavior regarding matching and expiration.
On Polymarket, there are three _TiF_ options available:
1) __GTC__ "good till cancel": the order remains open/live/resting until canceled or fully matched (a partially matched 
order remains open for its remaining size to be matched).
2) __GTD__ "good til day": the order remains open/live/resting until a pre-set expiration date, canceled or fully matched.
3) __FOK__ "fill or kill": the order gets filled immediately or otherwise will be automatically canceled. This option is 
only valid for taker orders! (because non-marketable limit orders will be automatically canceled as they cannot be 
matched immediately) 

#### Notes on MakingAmount, TakingAmount and Precision
The size argument of an order has up to 2 decimal places of precision. Everything beyond that will be rounded to 2 decimal 
places (floor).  
The price argument of an order has either 2 or 3 decimal places of precision depending on the current [tick size](#tick-sizes) 
of the market. If the current market price is within \[0.4, 0.96], then `tick_size = 0.01`, which results in a precision 
of 2 decimal places. If the market price is in (0, 0.4) or (0.96, 1), then `tick_size = 0.001`, which results in 
3 decimal places (sub-cent) of precision.  
Prices will be round to the corresponding decimal places via half even.
As the amount is the product of _size * price_, it has a precision of 4 or 5 decimal places, depending on the current 
tick size.  
All inputs will be round to the corresponding decimal places automatically when an order is created in `polypy`, so the 
user does not have to worry about decimal places and precision when specifying an order.
  
_MakingAmount_ and _TakingAmount_ are terms, that should not be confused with _Taker Order_ and _Maker Order_, as those 
terms have NOTHING in common.  
_MakingAmount_ and _TakingAmount_ are terms often used in digital order protocols (e.g., 0x) and have the following meaning:
1) __MakingAmount__: The amount of USDC (in case of buy order) or the size/number of shares (in case of sell order) that you 
transfer to the counterparty of a trade ('making' the amount/size to the counterparty)
2) __TakingAmount__: The amount of USDC (in case of a sell order) or the size/number of shares (in case of buy order) that you
receive from the counterpart of a trade ('taking' the amount/size from the counterparty)

Thereby, these terms specify the flows that occur at a trade:

|            | MakingAmount | TakingAmount    |
|------------|--------------|-----------------|
| Buy order  | USDC spent   | shares received |
| Sell order | shares spent | USDC received   |

_MakingAmount_ and _TakingAmount_ are usually scaled by 1e06 and are integers without decimal places.  
Usually, the user of `polypy` will never have to (and should never directly) interact with the terms _MakingAmount_ and 
_TakingAmount_ directly, as everything is handled automatically by `polypy` under the hood.   
Nonetheless, it is good to be familiar with these terms.

### Order Implementation
`polpy`s standard order implementation class is located under `polypy.order.base.Order`.
Users can (if they really want to...) implement their own order implementation by following `polypy.order.commom.OrderProtocol`, 
which defines a standard interface such that the implementation can be used trough out all `polpy` functions and methods.
    
It is highly recommended, __not__ to initiate or manipulate `polypy.order.base.Order` directly (which is why it is not 
directly importable from package namespace), but instead either use a factory (see [Creating Orders](#creating-orders)), 
or even better use an `plp.OrderManager` (see [Order Manager](#order-manager)).  
  
The rest of this documentation refers to the standard order class implementation of `polypy`.

#### Attributes
An order object has the following __read-only__ attributes:
- `tif: plp.TIME_IN_FORCE`: see [Time-in-Force](#time-in-force)
- `defined_at: int`: time in _millis_ at which order was initialized. This is only kept as reference for the user and is neither 
used anywhere else in `polypy`, nor sent to the Polymarket API.
- `numeric_type: type[plp.NumericAlias]`: Python type which is used for decimal numbers e.g., `float` or `Decimal`
- `price: plp.NumericAlias`: specified price for order, same type as `numeric_type`, cf. below [notes on market vs limit order](#notes-on-price-size-and-amount---market-vs-limit-order)
- `size: plp.NumericAlias`: specified size for order, same type as `numeric_type`, cf. below [notes on market vs limit order](#notes-on-price-size-and-amount---market-vs-limit-order)
- `size_open: plp.NumericAlias`: order size which is not yet matched and therefore open/remaining (`size == size_open + size_matched`), 
same type as `numeric_type`, cf. below [notes on market vs limit order](#notes-on-price-size-and-amount---market-vs-limit-order)
- `amount: plp.NumericAlias`: specified amount for order, same type as `numeric_type`, cf. below [notes on market vs limit order](#notes-on-price-size-and-amount---market-vs-limit-order)
- `token_id: str`: token id
- `asset_id: str`: token id (alias for `token_id`)
- `expiration: int`: expiration time in _seconds_, if not `plp.TIME_IN_FORCE.GTD` then `=0`
- `side: plp.SIDE`: buy or sell order
- `eip712order: plp.order.eip712.EIP712Order`: special class implementing signing etc., usually irrelevant for the user
- `signature_type: plp.SIGNATURE_TYPE`: signature type which was used to sign the order
- `fee_rate_bps: int`: fee rate in basis points (usually =0)
- `taker_amount: int`: [TakerAmount](#notes-on-makingamount-takingamount-and-precision)
- `maker_amount: int`: [MakerAmount](#notes-on-makingamount-takingamount-and-precision)
- `salt: int`: salt which was used to sign the order
- `maker: str`: maker (funder) address (if [Magic Login](#magic-login) is used, then this is the user's wallet address displayed on Polymarket)
- `taker: str`: taker address, usually zero-address (all zeros)
- `signer: str`: signer address, usually address associated with private key
- `nonce: int`: Nonce which was used to sign the order

An order object has the following __write-once__ attributes (which are frozen once they are set):
- `id: str | None`: will be assigned by the exchange/Polymarket and returned within the response message when submitting an order
- `signature: str | None`: will be computed automatically if factory or Order Manager is used, else user has to compute and provide signature when initializing directly
- `created_at: int | None`: time in _seconds_ at which order is created on the exchange/Polymarket and will be returned within the response message when submitting an order

An order object has the following __read-and-write__ attributes:
- `status`
- `size_matched`
- `strategy_id`
- `aux_id`

#### Methods

#### Notes on `price`, `size` and `amount` - market vs limit order
(defined only once and then set forever)
(market vs limit order)
#### Notes on Multithreading and Multiprocessing

### Creating Orders
### Computing Expiration Timestamp
(expiration: compute_expiration_timestamp)
### Order Manager

## Positions
### Position Implementations
### Creating Positions
### Position Manager

## Trades
_tbd_

## Timestamps and Units
Polymarket uses different time scales for different timestamps. This is also reflected in `polypy`.
  
In __seconds__ unit:
- `Order.created_at`: timestamp at which the order is submitted to the exchange. This timestamp is generated and returned by Polymarket.
- `Order.expiration`: timestamp at which a `GTD` order will expire (minus security threshold of 60 seconds). This timestamp is computed by the user and sent to Polymarket.
`polypy.compute_expiration_timestamp(...)` (see [Creating Orders](#creating-orders)) can be used as a convenience function therefore.

In __millis__ unit:
- `timestamp` in `market` websocket message: timestamp of order book generation (aka when order book changed). This timestamp is generated and returned by Polymarket.
If an [order book](#clob-in-polypy-polypyorderbook) is assigned to a [market stream](#market-stream), then this will be handled automatically.
- `Order.defined_at`: timestamp at which the order object was initialized. This is only kept for reference and is not further used in `polypy`, not sent to Polymarket.

## Streams
### Market Stream
(order book only at one market stream)
### User Stream

## REST API functions

## Full Example