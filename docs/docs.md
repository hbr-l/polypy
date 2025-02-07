# Documentation
>work in progress

## Introduction
`PolyPy` is a wrapper around Polymarket's REST and websocket interface.
`PolyPy`'s implementation is biased and opinionated, and aims at delivering at least some level of comfort for interacting 
with Polymarket's interfaces, e.g., keeping track of orders and positions locally, maintaining a local oder book, etc.
Therefore, `PolyPy` offers custom (and to some extent customizable) implementations beyond Polymarket's very basic 
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
  
_`Polypy` does neither support setting allowances, nor creating API keys, as this is outside the scope of this package._
  
Please refer to [py-clob-client](https://github.com/Polymarket/py-clob-client/tree/main), the official 
[Polymarket documentation ](https://docs.polymarket.com/#introduction) or Polymarket's Discord for allowances and authentication.  
If you are using Mail/Magic-Login, please refer to [Magic-Login](#magic-login), which simplifies allowances and authentication
quite a lot (i.e., actually not necessary when using Magic-Login).

#### Magic-Login
If you use Mail/Magic-Login (btw. __NEVER TYPE IN A NUMBER CODE FROM ANY MAIL INTO MAGIC'S LOGIN PAD THAT YOU HAVEN'T 
REQUESTED YOURSELF OF FROM ANY THIRD PARTY__), you can obtain the necessary credentials by:
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
`polypy` tries to minimize errors due to floating point imprecision by implementing dedicated rounding approaches.  
Nonetheless, error might accumulate, and it is advised to use Python's `decimal.Decimal` type instead (which is also faster, 
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

## Orders
### Order Types
### Creating Orders
### Order Manager

## Positions
### Position Types
### Creating Positions
### Position Manager

## Trades
_tbd_

## Streams
### Market Stream
### User Stream

## REST API functions