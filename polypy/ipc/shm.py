import os
from collections.abc import Iterable, Sequence
from decimal import Decimal
from multiprocessing import shared_memory
from multiprocessing.util import Finalize
from typing import Any, Callable, Self

import numpy as np
from numpy.typing import NDArray

from polypy.typing import AdvancedIndex, NumericAlias


class FinalizedSharedMemory(shared_memory.SharedMemory):
    def __init__(self, name: str, create: bool, size: int) -> None:
        super().__init__(name, create, size)

        self.owner = create
        self.finalizer = Finalize(self, self.cleanup, exitpriority=10)

    def __getstate__(self) -> dict:
        return {
            "name": self.name,
            "pid": os.getpid(),
        }

    def __setstate__(self, state: dict) -> None:
        super().__init__(name=state["name"], create=False)

        self.owner = os.getpid() == state["pid"]
        self.finalizer = Finalize(self, self.cleanup, exitpriority=10)

    def __del__(self):
        self.close()

    def cleanup(self) -> None:
        self.close()

        if self.owner:
            self.unlink()


def _nbytes(shape: int | Sequence[int], dtype: type | np.dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


class SharedArray:
    def __init__(
        self,
        shape: int | Sequence[int],
        shm_name: str | None,
        create: bool,
        dtype: type | np.dtype | str,
        fill_val: Any,
    ) -> None:
        self._dtype = dtype

        size = _nbytes(shape, self._dtype) if create else 0
        self.shm = FinalizedSharedMemory(create=create, name=shm_name, size=size)

        self._arr = np.ndarray(shape, dtype=self._dtype, buffer=self.shm.buf)

        if create:
            self._arr[:] = fill_val

    @classmethod
    def factory(
        cls, shm_name: str, create: bool, fill_val: Any
    ) -> Callable[[int | Sequence[int], type | np.dtype | str], Self]:
        def closure(shape: int | Sequence[int], dtype: type | np.dtype | str):
            return SharedArray(shape, shm_name, create, dtype, fill_val)

        return closure

    def __getstate__(self) -> dict:
        return {
            "shape": self._arr.shape,
            "shm": self.shm.__getstate__(),
            "dtype": self._dtype,
        }

    def __setstate__(self, state: dict) -> None:
        shm = FinalizedSharedMemory.__new__(FinalizedSharedMemory)
        shm.__setstate__(state["shm"])
        self.shm = shm

        self._dtype = state["dtype"]
        self._arr = np.ndarray(state["shape"], dtype=self._dtype, buffer=self.shm.buf)

    def __getitem__(self, item: AdvancedIndex) -> Any | NDArray[Any]:
        return self._arr[item]

    def __setitem__(self, key: AdvancedIndex, value: Any | Iterable[Any]) -> None:
        self._arr[key] = value

    def __array__(self) -> NDArray[Any]:
        return self[:]

    @property
    def array(self) -> np.ndarray:
        return self._arr

    @property
    def dtype(self) -> type:
        return self._arr.dtype

    @property
    def shape(self):
        return self._arr.shape

    # noinspection PyTypeChecker
    def __eq__(self, other: Any) -> NDArray[bool]:
        return self._arr == other

    def __ne__(self, other: Any) -> NDArray[bool]:
        return ~self.__eq__(other)

    def __len__(self) -> int:
        return len(self._arr)

    def __iter__(self):
        return iter(self[:])

    def __del__(self):
        self.close()

    def cleanup(self):
        """Calls self.close() and if owner also self.unlink()."""
        self.shm.cleanup()

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()


class SharedDecimalArray(SharedArray):
    """SharedMemory implementation of an array with Decimal type.

    Although this implementation takes measures to close and unlink the shared memory block by itself via finalizer,
    it is best to also manually call SharedZerosDec.close() in each child process and to call
    .close() AND .unlink() from the parent process. SharedZerosDec.unlink() should only be called once!
    Note, orphaned shared memory blocks might be possible in case of exceptions (especially on Windows).
    Please refer to Python's SharedMemory documentation.

    Notes
    -----
    This implementation assumes all __setitem__ values are of type Decimal and does no further checks!
    This implementation does not use locks, the user must take care of proper locking!
    """

    def __init__(
        self,
        shape: int | Sequence[int],
        shm_name: str | None,
        create: bool,
        n_decimals: int,
        fill_value: NumericAlias = 0,
    ) -> None:
        self.scale = 10**n_decimals

        super().__init__(shape, shm_name, create, np.int64, fill_value)

    def __getstate__(self) -> dict:
        return super().__getstate__() | {"scale": self.scale}

    def __setstate__(self, state: dict) -> None:
        self.scale = state["scale"]
        super().__setstate__(state)

    def __getitem__(self, item: AdvancedIndex) -> Decimal | NDArray[Decimal]:
        x = self._arr[item]

        if np.isscalar(x):
            # noinspection PyTypeChecker
            return Decimal(int(x)) / self.scale

        return np.fromiter(
            (Decimal(int(v)) / self.scale for v in x.ravel()), dtype=object
        ).reshape(x.shape)

    def __setitem__(
        self, key: AdvancedIndex, value: NumericAlias | Iterable[NumericAlias]
    ) -> None:
        # convert values to scaled int before placing them into self._arr
        if np.isscalar(value):
            value = int(value * self.scale)
        else:
            value = np.fromiter((int(v * self.scale) for v in value), dtype=self._dtype)

        self._arr[key] = value

    @property
    def dtype(self) -> type:
        return type(Decimal)

    # noinspection PyTypeChecker
    def __eq__(self, other: Decimal | int) -> NDArray[bool]:
        other = int(other * self.scale)
        return self._arr == other

    def __ne__(self, other: Decimal | int) -> NDArray[bool]:
        return ~self.__eq__(other)
