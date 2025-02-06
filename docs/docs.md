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

### Authentication and Allowances
### Floating Point Imprecision, Rounding and Decimal Type
### Rate and Trading Limits

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