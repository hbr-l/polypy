from collections.abc import Iterable
from decimal import Decimal
from typing import Any, Optional, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

NumericAlias = int | float | Decimal


def dec(x: NumericAlias | str) -> Decimal:
    return Decimal(str(x))


def zeros_dec(x: int, *_) -> np.ndarray:
    return np.array([Decimal(0)] * x, dtype=object)


def is_iter(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


def is_all_none(*args) -> bool:
    return all(x is None for x in args)


class ArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> dict:
        return ...


ArrayCoercible: TypeAlias = ArrayInterface | ArrayLike | np.ndarray


class ZerosProtocol(ArrayInterface):
    # noinspection PyUnusedLocal
    def __init__(self, shape: int | tuple[int, ...], dtype: Optional[type | np.dtype]):
        ...

    def __getitem__(self, item):
        ...

    def __setitem__(self, key, value):
        ...

    def __len__(self):
        ...

    def __iter__(self):
        ...


class ZerosFactoryFunc(Protocol):
    def __call__(
        self, shape: int | tuple[int, ...], dtype: Optional[type | np.dtype]
    ) -> ArrayCoercible:
        ...
