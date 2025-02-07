from decimal import Decimal
from typing import Optional, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

NumericAlias = int | float | Decimal


def dec(x: NumericAlias | str) -> Decimal:
    return Decimal(str(x))


def zeros_dec(x: int, *_) -> np.ndarray:
    return np.array([Decimal(0)] * x, dtype=object)


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
