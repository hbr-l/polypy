# PolyPy

> [Polymarket](https://www.polymarket.com) Python Wrapper - PolyPy


Goals
----------------------
PolyPy is a Python wrapper around [Polymarket's API](https://docs.polymarket.com/) and aims to implement a minimal 
set of components needed for building trading systems, whilst focusing on acceptable runtime performance.
PolyPy's implementation is opinionated and biased towards a minimal implementation, which for real systems needs to 
be extended by additional custom components such as logging, account balance management, etc.
  
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
__USE AT YOUR OWN RISK!__


Improved Implementation over `py_clob_client`
---------------------------------------------
- 3x faster order creation
- Market Sell orders, which are not available `py_clob_client`
- Market channel and user channel (WIP) stream implementation
- Positions implementation (WIP)
- Fixed erroneous rounding routines in `py_clob_client`, i.e. `round_floor(4.6, 2) == 4.59 != 4.6`
- Fixed erroneous calculation of marketable price
- Fixed bug in Orderbook hash in `py_clob_client` (especially, after receiving a "price_change" websocket message)

Project Status
--------------
This project is under active development and its function and class signatures, as well as the repository structure 
might be subject to future changes.

Install
-------
1) git clone
2) pip install -e . (locally)

Todo
----
- [ ] Proper Git Tooling and Actions
- [ ] Restructure repository
- [ ] Documentation
- [ ] Additional components
  - [ ] Rate Limiting
  - [ ] Positions
  - [ ] etc.

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
