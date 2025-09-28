import math
import warnings
from typing import Any, Protocol, TypeVar

from _decimal import Decimal

from polypy.exceptions import OrderBookException
from polypy.ipc.shm import SharedZerosDec
from polypy.typing import NumericAlias


def _validate_base10(x: NumericAlias) -> None:
    exponent = math.log10(x)
    if exponent != int(exponent):
        raise OrderBookException(
            f"`tick_size` has to be base 10. Got: {x} with exponent {exponent}."
        )


def _validate_tick_size(val: Any, min_tick_size: NumericAlias) -> None:
    _validate_base10(val)
    if val < min_tick_size:
        raise OrderBookException(
            f"`tick_size`={val} less than `min_tick_size`={min_tick_size}."
        )


T = TypeVar("T")


class TickSizeProtocol(Protocol[T]):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        tick_size: NumericAlias | str,
        dtype: type[T],
        min_tick_size: float = 0.001,
    ) -> None:
        ...

    @property
    def min_tick_size(self) -> float:
        ...

    def get(self) -> T:
        ...

    def set(self, tick_size: NumericAlias | str) -> None:
        ...


class TickSizeFactory(Protocol):
    def __call__(
        self,
        tick_size: NumericAlias | str,
        dtype: type[T],
        min_tick_size: float = 0.001,
    ) -> TickSizeProtocol[T]:
        ...


class TickSize:
    def __init__(
        self,
        tick_size: NumericAlias | str,
        dtype: type[T],
        min_tick_size: float = 0.001,
    ) -> None:
        _validate_base10(min_tick_size)

        self.dtype = dtype
        self._min_tick_size = min_tick_size

        self.tick_size = None
        self.set(tick_size)

    @property
    def min_tick_size(self) -> float:
        return self._min_tick_size

    def get(self) -> T:
        return self.tick_size

    def set(self, tick_size: NumericAlias | str) -> None:
        if not isinstance(tick_size, self.dtype):
            tick_size = self.dtype(str(tick_size))
        _validate_tick_size(tick_size, self.dtype(str(self._min_tick_size)))
        self.tick_size = tick_size


class SharedTickSize:
    def __init__(
        self,
        tick_size: NumericAlias | str | None,
        *_,
        min_tick_size: float | None = 0.001,
        shm_name: str | None = None,
        create: bool = True,
        **__,
    ) -> None:
        if create:
            if tick_size is None or min_tick_size is None:
                raise OrderBookException(
                    "`tick_size` and `min_tick_size` must not be None if `create=True`."
                )

            _validate_base10(min_tick_size)

            self.tick_size = SharedZerosDec(2, shm_name, True)

            self.tick_size[1] = min_tick_size
            self.set(tick_size)
        else:
            self.tick_size = SharedZerosDec(2, shm_name, False)

            if tick_size is not None or min_tick_size is not None:
                warnings.warn(
                    f"`tick_size={tick_size}` and `min_tick_size={min_tick_size}` "
                    f"ignored for SharedTickSize if `create=False`. "
                    f"`shm_name={shm_name}`"
                )

    def __del__(self):
        self.tick_size.__del__()

    @property
    def min_tick_size(self) -> float:
        return float(self.tick_size[1])

    def get(self) -> Decimal:
        return self.tick_size[0]

    def set(self, tick_size: NumericAlias | str) -> None:
        tick_size = Decimal(str(tick_size))
        _validate_tick_size(tick_size, self.tick_size[1])
        self.tick_size[0] = tick_size

    def cleanup(self) -> None:
        self.tick_size.cleanup()

    def close(self) -> None:
        self.tick_size.close()

    def unlink(self) -> None:
        self.tick_size.unlink()
