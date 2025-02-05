# PolyPy

> [Polymarket](https://www.polymarket.com) Python Wrapper - PolyPy


Goals
-----
`PolyPy` is a Python wrapper around [Polymarket's API](https://docs.polymarket.com/) and aims to implement a minimal 
set of components for facilitating trading on Polymarket, whilst focusing on acceptable (but not high-frequency) 
runtime performance.
`PolyPy`'s implementation is opinionated and biased with respect to architecture and design choices.
  
Although, [Polymarket's Python Client](https://github.com/Polymarket/py-clob-client/tree/main) is quite convenient, 
it is also comparably slow, i.e. it uses JSON-parsing from stdlib instead of much faster 
[msgspec](https://jcristharif.com/msgspec/), native dataclasses instead of 
[attrs](https://www.attrs.org/en/stable/init.html), etc., and has some disadvantageous implementation details.
  
> Please mind, that PolyPy is just a hobby project!


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

Documentation
-------------
See [documentation](docs) and [examples](examples). (work in progress)

Install
-------
Currently, `PolyPy` can only be installed locally:
1) git clone
2) cd polypy
3) pip install -e . (locally)

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
