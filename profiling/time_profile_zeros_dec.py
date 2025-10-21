import time
import timeit
from decimal import Decimal

import numpy as np


def zeros_dec_full(x: int, *_) -> np.ndarray:
    return np.full(x, Decimal(0), dtype=object)


_zeros_cache = Decimal(0)


def zeros_dec_full_cache(x: int, *_) -> np.ndarray:
    return np.full(x, _zeros_cache, dtype=object)


def zeros_dec_arr(x: int, *_) -> np.ndarray:
    return np.array([Decimal(0)] * x, dtype=object)


def main():
    n = 100
    s = 10_000

    time.sleep(1)

    start_t = timeit.default_timer()
    for _ in range(n):
        zeros_dec_arr(s)
    end_t = timeit.default_timer()
    print(f"zeros_dec_arr: {((end_t - start_t) / n)*1_000} millis")

    time.sleep(1)

    start_t = timeit.default_timer()
    for _ in range(n):
        zeros_dec_full(s)
    end_t = timeit.default_timer()
    print(f"zeros_dec_full: {((end_t - start_t) / n) * 1_000} millis")

    time.sleep(1)

    start_t = timeit.default_timer()
    for _ in range(n):
        zeros_dec_full_cache(s)
    end_t = timeit.default_timer()
    print(f"zeros_dec_full_cache: {((end_t - start_t) / n) * 1_000} millis")


if __name__ == "__main__":
    main()
