import multiprocessing
import platform
import time

import pytest

if platform.system() != "Windows":
    from polypy.ipc.lock import PosixLock


def worker():
    lock = PosixLock("123")
    lock.acquire()
    time.sleep(5)
    lock.release()


@pytest.mark.skipif(platform.system() == "Windows", reason="PosixLock only on Posix")
def test_lock_status():
    lock = PosixLock("123")

    assert lock.acquire() is True
    lock.release()
    time.sleep(1)

    proc = multiprocessing.Process(target=worker)
    proc.start()

    time.sleep(2)
    assert lock.acquire(blocking=False) is False
    time.sleep(4)

    proc.join()

    assert lock.acquire(blocking=False) is True
    lock.release()
    lock.close()


def worker_context():
    with PosixLock("123"):
        time.sleep(5)


@pytest.mark.skipif(platform.system() == "Windows", reason="PosixLock only on Posix")
def test_lock_status_context():
    lock = PosixLock("123")

    proc = multiprocessing.Process(target=worker_context)
    proc.start()

    time.sleep(2)
    assert lock.acquire(blocking=False) is False
    time.sleep(4)

    proc.join()

    assert lock.acquire(blocking=False) is True
    lock.release()
    lock.close()
