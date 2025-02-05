import math
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_UP, Decimal

from polypy.typing import NumericAlias

# todo if custom ScaledInt type -> adapt implementations


DEC10 = Decimal(10)


def round_half_even(x: NumericAlias, sig_digits: int) -> NumericAlias:
    # built-in round also rounds half even but suffers from floating point error

    if isinstance(x, Decimal):
        return x.quantize(DEC10**-sig_digits)

    # NOTE: without this line, we would introduce float imprecision,
    #   i.e. round_floor(4.6, 2) = 4.59 instead of = 4.6
    # NOTE: this erroneous in py_clob_client!
    if max_allowed_decimals(x, sig_digits):
        return x

    scaler = 10**sig_digits
    return round(x * scaler) / scaler


def round_ceil(x: NumericAlias, sig_digits: int) -> NumericAlias:
    """Round ceil towards +infinity."""
    if isinstance(x, Decimal):
        return x.quantize(DEC10**-sig_digits, rounding=ROUND_CEILING)

    # NOTE: without this line, we would introduce float imprecision,
    #   i.e. round_floor(4.6, 2) = 4.59 instead of = 4.6
    # NOTE: this erroneous in py_clob_client!
    if max_allowed_decimals(x, sig_digits):
        return x

    scaler = 10**sig_digits
    return math.ceil(x * scaler) / scaler


def round_up(x: NumericAlias, sig_digits: int) -> NumericAlias:
    """Round up away from 0"""
    if isinstance(x, Decimal):
        return x.quantize(DEC10**-sig_digits, rounding=ROUND_UP)

    if max_allowed_decimals(x, sig_digits):
        return x

    scaler = 10**sig_digits
    if x > 0:
        return math.ceil(x * scaler) / scaler
    else:
        return math.floor(x * scaler) / scaler


def round_floor(x: NumericAlias, sig_digits: int) -> NumericAlias:
    """Round floor towards -infinity.

    Notes
    -----
    If argument was involved in multiplication, use `round_floor_tenuis_ceil` to account for precision increase
    during multiplication.
    """
    if isinstance(x, Decimal):
        return x.quantize(DEC10**-sig_digits, rounding=ROUND_FLOOR)

    # NOTE: without this line, we would introduce float imprecision,
    #   i.e. round_floor(4.6, 2) = 4.59 instead of = 4.6
    # NOTE: this is erroneous in py_clob_client!
    if max_allowed_decimals(x, sig_digits):
        return x

    scaler = 10**sig_digits
    return math.floor(x * scaler) / scaler


def round_down(x: NumericAlias, sig_digits: int) -> NumericAlias:
    """Round towards zero.

    Notes
    -----
    If argument was involved in multiplication, use `round_down_tenuis_up` to account for precision increase
    during multiplication.
    """
    if isinstance(x, Decimal):
        return x.quantize(DEC10**-sig_digits, rounding=ROUND_DOWN)

    if max_allowed_decimals(x, sig_digits):
        return x

    scaler = 10**sig_digits
    if x > 0:
        return math.floor(x * scaler) / scaler
    else:
        return math.ceil(x * scaler) / scaler


def scale_1e06(x: NumericAlias) -> int:
    scaler = DEC10**6 if isinstance(x, Decimal) else 1e06
    return int(round_half_even(x * scaler, 0))


def max_allowed_decimals(
    x: NumericAlias, max_ndecimals: int, eps: float = 1e-21
) -> bool:
    """Checks if the number of decimal places in 'x' is less than or equal to 'max_ndecimals'.

    This function checks if the number of decimal places in a given number
    (`x`) is within the specified limit (`max_ndecimals`).

    Parameters
    ----------
    x : NumericAlias
        The input number.
    max_ndecimals : int
        The maximum allowed number of decimal places.
    eps : float, optional
        Tolerance for floating-point comparisons. Defaults to 1e-21.

    Returns
    -------
    bool
        True if the number of decimal places in 'x' is less than or equal
        to 'max_ndecimals', False otherwise.
    """
    if isinstance(x, Decimal):
        # noinspection PyTypeChecker
        return abs(x.as_tuple().exponent) <= max_ndecimals

    # return x * 10**max_ndecimals % 1
    # suffers from floating point imprecision: better solution see below

    # todo: alternative implementation (more accurate but slower by 10x -> 1e-6 vs 1e-7 seconds):
    # x_round = x * 10**ndecimal_digits
    # ndigits = max(0, 15 - abs(int(math.log10(abs(x_round)))) - 1)
    # x_round = round(x_round, ndigits)
    # return False if x_round == 0 else x_round % 1 == 0

    scaled_x = round(x * (10**max_ndecimals))
    reconstructed_x = scaled_x / (10**max_ndecimals)
    return abs(x - reconstructed_x) < eps


def round_floor_tenuis_ceil(
    x: NumericAlias, sig_digits: int, extra_tenuis_digits: int
) -> NumericAlias:  # sourcery skip: assign-if-exp, reintroduce-else
    """Rounds a number to the specified number of significant digits. This function should be applied
    whenever multiplication (increase in precision) occurs (instead of simpler round_floor).

    This function primarily performs floor rounding.
    However, for values close to a rounding boundary,
    where the digits beyond `sig_digits` are "almost all 9s"
    within the precision of `extra_tenuis_digits`,
    the function may round up to ensure accurate results.


    Parameters
    ----------
    x : NumericAlias
        The input number.
    sig_digits : int
        The desired number of significant digits.
    extra_tenuis_digits : int
        The number of extra digits used for intermediate calculations
        to improve accuracy near rounding boundaries.

    Returns
    -------
    NumericAlias
        The rounded number.

    Notes
    -----
    Useful when rounding volume/amount in quote currency (e.g. USDC). Tenuis, Latin for thin slice.

    The rounding-up condition is expressed as:
        round_ceil(x, sig_digits) - x < 10**-(sig_digits + extra_tenuis_digits)
    In simple words:
        1) Take all digits beyond `sig_digits`
        2) Place new decimal point at `extra_tenuis_digits`
        3) If ceil changes the new number, we round up to `sig_digits`, else we round down to `sig_digits`
    Equivalently: ceil(fractional(x * 10 ** sig_digits) * 10 ** extra_tenuis_digits)

    Examples
    --------
    >>> round_floor_tenuis_ceil(0.1299999, 2, 1)
    0.13
    >>> round_floor_tenuis_ceil(0.1299999, 2, 4)
    0.13
    >>> round_floor_tenuis_ceil(0.1299999, 2, 5)
    0.12
    >>> round_floor_tenuis_ceil(0.1299999, 2, 6)
    0.12
    """
    if max_allowed_decimals(x, sig_digits):
        return x

    x = round_ceil(x, sig_digits + extra_tenuis_digits)

    if max_allowed_decimals(x, sig_digits):
        return x

    return round_floor(x, sig_digits)


def round_down_tenuis_up(
    x: NumericAlias, sig_digits: int, extra_tenuis_digits: int
) -> NumericAlias:  # sourcery skip: assign-if-exp, reintroduce-else
    if max_allowed_decimals(x, sig_digits):
        return x

    x = round_up(x, sig_digits + extra_tenuis_digits)

    if max_allowed_decimals(x, sig_digits):
        return x

    return round_down(x, sig_digits)
