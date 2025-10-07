import multiprocessing
import time

import pytest

from polypy.ipc.lock import SharedProcRLock


def test_rlock_acquire_release():
    rlock = SharedProcRLock("test_rlock_acquire_release", True)
    assert rlock.acquire()
    assert rlock.release() is None


def test_rlock_reentrance():
    rlock = SharedProcRLock("test_rlock_reentrance", True)
    assert rlock.acquire()
    assert rlock.acquire()
    assert rlock.acquire()
    assert rlock.release() is None
    assert rlock.release() is None
    assert rlock.release() is None


def worker(name: str):
    rlock = SharedProcRLock(name, False)
    rlock.acquire()
    time.sleep(8)
    rlock.release()


def test_rlock_blocks():
    rlock = SharedProcRLock("test_rlock_blocks", True)

    proc = multiprocessing.Process(target=worker, args=("test_rlock_blocks",))
    proc.start()
    time.sleep(3)

    assert rlock.acquire(False) is False
    time.sleep(3)
    proc.join(8)

    assert rlock.acquire()
    assert rlock.release() is None


def worker_context_reentrance(name: str):
    rlock = SharedProcRLock(name, False)
    with rlock:
        time.sleep(3)
        rlock.acquire()
        time.sleep(3)
        rlock.release()


def test_rlock_blocks_context_with_reentrance():
    rlock = SharedProcRLock("test_rlock_blocks_context_with_reentrance", True)

    proc = multiprocessing.Process(
        target=worker_context_reentrance,
        args=("test_rlock_blocks_context_with_reentrance",),
    )
    proc.start()
    time.sleep(3)

    assert rlock.acquire(True, 2) is False
    time.sleep(4)
    proc.join(8)

    assert rlock.acquire()
    assert rlock.release() is None


def test_rlock_release_without_acquire():
    rlock = SharedProcRLock("test_rlock_release_without_acquire", True)

    with pytest.raises(RuntimeError):
        rlock.release()


def test_rlock_release_wrong_process():
    rlock = SharedProcRLock("test_rlock_release_wrong_process", True)

    proc = multiprocessing.Process(
        target=worker, args=("test_rlock_release_wrong_process",)
    )
    proc.start()
    time.sleep(3)

    with pytest.raises(RuntimeError):
        rlock.release()
