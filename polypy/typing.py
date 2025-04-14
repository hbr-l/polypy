from decimal import Decimal
from typing import Any, Iterable, Optional, Protocol, Sequence, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike

NumericAlias = int | float | Decimal

T = TypeVar("T")


def dec(x: NumericAlias | str) -> Decimal:
    return Decimal(str(x))


def zeros_dec(x: int, *_) -> np.ndarray:
    return np.array([Decimal(0)] * x, dtype=object)


def is_iter(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


def is_all_none(*args) -> bool:
    return all(x is None for x in args)


def infer_numeric_type(x: Any) -> type:
    t = type(x)
    return float if t is int else t


class ArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> dict:
        return ...

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        ...


ArrayCoercible: TypeAlias = ArrayInterface | ArrayLike | np.ndarray

AdvancedIndex: TypeAlias = int | slice | Sequence[int] | Sequence[bool]


class ZerosProtocol(ArrayInterface):
    @property
    def dtype(self) -> Any:
        return ...

    # noinspection PyUnusedLocal
    def __init__(self, shape: int | tuple[int, ...], dtype: Optional[type | np.dtype]):
        ...

    def __getitem__(self, key: AdvancedIndex):
        ...

    def __setitem__(self, key: AdvancedIndex, value):
        ...

    def __len__(self):
        ...

    def __iter__(self):
        ...


class ZerosFactoryFunc(Protocol):
    def __call__(
        self, shape: int | tuple[int, ...], dtype: Optional[type | np.dtype]
    ) -> ZerosProtocol | np.ndarray:
        ...


def first_iterable_element(s: set[T] | Iterable[T]) -> T:
    for e in s:
        return e
