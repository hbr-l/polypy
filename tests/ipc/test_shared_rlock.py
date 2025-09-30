import multiprocessing
import time

import pytest

from polypy.ipc.lock import SharedProcRLock


def test_rlock_acquire_release():
    rlock = SharedProcRLock("test", True)
    assert rlock.acquire()
    assert rlock.release() is None


def test_rlock_reentrance():
    rlock = SharedProcRLock("test", True)
    assert rlock.acquire()
    assert rlock.acquire()
    assert rlock.acquire()
    assert rlock.release() is None
    assert rlock.release() is None
    assert rlock.release() is None


def worker():
    rlock = SharedProcRLock("test", False)
    rlock.acquire()
    time.sleep(5)
    rlock.release()


def test_rlock_blocks():
    rlock = SharedProcRLock("test", True)

    proc = multiprocessing.Process(target=worker)
    proc.start()
    time.sleep(2)

    assert rlock.acquire(False) is False
    time.sleep(3)
    proc.join(5)

    assert rlock.acquire()
    assert rlock.release() is None


def worker_context_reentrance():
    rlock = SharedProcRLock("test", False)
    with rlock:
        time.sleep(3)
        rlock.acquire()
        time.sleep(3)
        rlock.release()


def test_rlock_blocks_context_with_reentrance():
    rlock = SharedProcRLock("test", True)

    proc = multiprocessing.Process(target=worker_context_reentrance)
    proc.start()
    time.sleep(2)

    assert rlock.acquire(True, 2) is False
    time.sleep(4)
    proc.join(6)

    assert rlock.acquire()
    assert rlock.release() is None


def test_rlock_release_without_acquire():
    rlock = SharedProcRLock("test", True)

    with pytest.raises(RuntimeError):
        rlock.release()


def test_rlock_release_wrong_process():
    rlock = SharedProcRLock("test", True)

    proc = multiprocessing.Process(target=worker)
    proc.start()
    time.sleep(2)

    with pytest.raises(RuntimeError):
        rlock.release()
