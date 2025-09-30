import multiprocessing
import time
from decimal import Decimal

import numpy as np

from polypy.book.tick import SharedTickSize
from polypy.ipc.shm import SharedArray, SharedDecimalArray


def test_shared_array_base():
    arr = SharedArray((2, 5), None, True, "U32", "")
    assert arr.shape == (2, 5)
    assert arr[0, 1] == ""

    arr[0, 0] = "test"
    assert arr[0, 0] == "test"

    arr[arr != ""] = "some"
    assert arr[0, 0] == "some"

    c_arr = np.array([["some", "", "", "", ""], ["", "", "", "", ""]])
    assert np.array_equal(arr[:], c_arr)

    arr[:] = "thing"
    assert all(j == "thing" for i in arr for j in i)

    arr.close()
    arr.unlink()


def test_shared_zeros_dec_base():
    arr = SharedDecimalArray((2, 5), None, True, 3)
    assert arr.shape == (2, 5)
    assert arr[0, 1] == Decimal(0)

    arr[0, 0] = Decimal("1.4")
    assert arr[0, 0] == Decimal("1.4")

    arr[arr != Decimal(0)] = Decimal(2)
    assert arr[0, 0] == Decimal(2)

    arr[0, 0] += Decimal(2)
    assert arr[0, 0] == Decimal(4)

    c_arr = np.array(
        [
            [Decimal(4), Decimal(0), Decimal(0), Decimal(0), Decimal(0)],
            [Decimal(0), Decimal(0), Decimal(0), Decimal(0), Decimal(0)],
        ]
    )
    assert np.array_equal(arr[:], c_arr)
    assert np.sum(arr[:] - c_arr) == Decimal(0)
    assert np.sum(arr - c_arr) == Decimal(0)

    arr[:] = Decimal(0)
    assert np.sum(arr) == 0

    arr.close()
    arr.unlink()


def worker_arr(arr: SharedDecimalArray):
    arr[0] = "test"
    time.sleep(0.2)


def test_shared_zeros_dec_multiprocessing():
    array = SharedArray((3,), None, True, "U32", "")
    assert all(i == "" for i in array)

    process = multiprocessing.Process(target=worker_arr, args=(array,))
    process.start()
    time.sleep(3)
    process.join(1)

    assert array[0] == "test"


def worker_arr_gen():
    arr = SharedArray((3,), "test_arr", False, "U32", None)
    arr[0] = "test"
    time.sleep(0.2)


def test_shared_zeros_dec_gen():
    array = SharedArray((3,), "test_arr", True, "U32", "1")
    assert all(i == "1" for i in array)

    process = multiprocessing.Process(target=worker_arr_gen)
    process.start()
    time.sleep(3)
    process.join(1)

    assert array[0] == "test"


def worker(arr: SharedDecimalArray):
    arr[0] = Decimal(1)
    time.sleep(0.2)


# noinspection PyRedeclaration
def test_shared_zeros_dec_multiprocessing():
    array = SharedDecimalArray((3,), None, True, 3)
    assert np.sum(array) == Decimal(0)

    process = multiprocessing.Process(target=worker, args=(array,))
    process.start()
    time.sleep(3)
    process.join(1)

    assert array[0] == Decimal(1)


def worker_gen():
    arr = SharedDecimalArray((3,), "test", create=False, n_decimals=3)
    arr[0] = Decimal(1)
    time.sleep(0.2)


# noinspection PyRedeclaration
def test_shared_zeros_dec_gen():
    array = SharedDecimalArray((3,), "test", True, 3)
    assert np.sum(array) == Decimal(0)

    process = multiprocessing.Process(target=worker_gen)
    process.start()
    time.sleep(3)
    process.join(1)

    assert array[0] == Decimal(1)


def worker_tick_size():
    tick = SharedTickSize(None, shm_name="tick", create=False)
    tick.set(0.001)
    time.sleep(0.2)


def test_shared_tick_size():
    tick = SharedTickSize(0.01, shm_name="tick", create=True)
    assert tick.get() == Decimal("0.01")

    process = multiprocessing.Process(target=worker_tick_size)
    process.start()
    time.sleep(3)
    process.join(1)

    assert tick.get() == Decimal("0.001")
