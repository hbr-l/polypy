import multiprocessing
import platform
import time

import pytest

if platform.system() == "Windows":
    from polypy.ipc.lock import WinLock


def worker():
    lock = WinLock("123")
    lock.acquire()
    time.sleep(5)
    lock.release()


@pytest.mark.skipif(platform.system() != "Windows", reason="WinLock only on Windows")
def test_lock_status():
    lock = WinLock("123")

    assert lock.acquire() is True
    lock.release()
    time.sleep(1)

    proc = multiprocessing.Process(target=worker)
    proc.start()

    time.sleep(3)
    assert lock.acquire(blocking=False) is False
    time.sleep(4)

    proc.join()

    assert lock.acquire(blocking=False) is True
    lock.release()
    lock.close()


def worker_context():
    with WinLock("123"):
        time.sleep(5)


@pytest.mark.skipif(platform.system() != "Windows", reason="WinLock only on Windows")
def test_lock_status_context():
    lock = WinLock("123")

    proc = multiprocessing.Process(target=worker_context)
    proc.start()

    time.sleep(3)
    assert lock.acquire(blocking=False) is False
    time.sleep(4)

    proc.join()

    assert lock.acquire(blocking=False) is True
    lock.release()
    lock.close()
