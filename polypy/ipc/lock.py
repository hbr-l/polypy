import contextlib
import platform
import struct
import threading
import warnings
from multiprocessing.util import Finalize
from typing import Self

from polypy.ipc.shm import FinalizedSharedMemory

if platform.system() == "Windows":
    import win32api
    import win32event
else:
    import posix_ipc


class MixinLockContext:
    def acquire(self, blocking: bool = True, timeout=None) -> bool:
        raise NotImplementedError()

    def release(self) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        """Close handle"""
        raise NotImplementedError()

    def unlink(self) -> None:
        """Unlink handle. self.close() should be called before calling self.unlink()
        and should only be called once."""
        raise NotImplementedError()

    def cleanup(self) -> None:
        """Close and only if owner, then unlink (elif not owner, then do not unlink)."""
        raise NotImplementedError

    def __del__(self) -> None:
        self.cleanup()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WinLock(MixinLockContext):
    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = win32event.CreateMutex(None, False, f"Global\\{self.name}")

        if not self._lock:
            raise RuntimeError(f"Failed to create Windows mutex. name={self.name}")

        self.finalizer = Finalize(self, self.cleanup, exitpriority=10)

    def acquire(self, blocking: bool = True, timeout=None) -> bool:
        if not blocking:
            timeout_ms = 0
        elif timeout is None:
            timeout_ms = win32event.INFINITE
        else:
            timeout_ms = int(timeout * 1_000)

        return (
            win32event.WaitForSingleObject(self._lock, timeout_ms)
            == win32event.WAIT_OBJECT_0
        )

    def release(self) -> None:
        win32event.ReleaseMutex(self._lock)

    def close(self) -> None:
        # noinspection PyUnresolvedReferences
        win32api.CloseHandle(self._lock)

    def unlink(self) -> None:
        # Windows mutex does not need to unlink
        return

    def cleanup(self) -> None:
        self.close()
        # Windows mutex does not need to unlink


class PosixLock(MixinLockContext):
    def __init__(self, name: str) -> None:
        warnings.warn("Posix-based lock has not been tested!")

        self.name = name
        self._sem_name = f"/{self.name}"

        try:
            self._lock = posix_ipc.Semaphore(
                self._sem_name, posix_ipc.O_CREAT | posix_ipc.O_EXCL
            )
            self._created = True
        except posix_ipc.ExistentialError:
            self._lock = posix_ipc.Semaphore(self._sem_name)
            self._created = False

        self.finalizer = Finalize(self, self.cleanup, exitpriority=10)

    def acquire(self, blocking=True, timeout=None) -> bool:
        if not blocking:
            try:
                self._lock.acquire(timeout=0)
                return True
            except posix_ipc.BusyError:
                return False
        else:
            return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()

    def close(self) -> None:
        self._lock.close()

    def unlink(self) -> None:
        posix_ipc.unlink_semaphore(self._sem_name)

    def cleanup(self) -> None:
        self.close()
        if self._created:
            with contextlib.suppress(posix_ipc.ExistentialError):
                self.unlink()


class SharedProcLock(MixinLockContext):
    """Shareable lock across processes"""

    def __init__(self, name: str) -> None:
        if platform.system() == "Windows":
            self.impl = WinLock(name=name)
        else:
            self.impl = PosixLock(name=name)

    def acquire(self, blocking: bool = True, timeout=None) -> bool:
        return self.impl.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        self.impl.release()

    def close(self) -> None:
        self.impl.close()

    def unlink(self) -> None:
        self.impl.unlink()

    def cleanup(self) -> None:
        self.cleanup()


class SharedProcRLock(MixinLockContext):
    """Shareable RLock across processes"""

    def __init__(self, name: str, create: bool) -> None:
        self.global_lock_name = f"{name}_global_lock"
        self.state_lock_name = f"{name}_state_lock"
        self.state_memory_name = f"{name}_state_memory"

        if platform.system() == "Windows":
            self._lock = WinLock(self.global_lock_name)
            self._state_lock = WinLock(self.state_lock_name)
        else:
            self._lock = PosixLock(name=self.global_lock_name)
            self._state_lock = PosixLock(name=self.state_lock_name)

        # [0:4]: count
        # [4:8]: thread id
        self._state = FinalizedSharedMemory(
            self.state_memory_name, create=create, size=8
        )

    def acquire(self, blocking: bool = True, timeout=None) -> bool:
        curr_thr_id = threading.get_ident()

        self._state_lock.acquire(True, None)
        try:
            owner_thr_id = struct.unpack("<I", self._state.buf[4:8])[0]

            if owner_thr_id == curr_thr_id:
                # reentrant case
                self._state.buf[0:4] = struct.pack(
                    "<I", struct.unpack("<I", self._state.buf[0:4])[0] + 1
                )
                return True
        finally:
            self._state_lock.release()

        # not owned by us, acquire the main lock
        acquired = self._lock.acquire(blocking, timeout)

        if acquired:
            # update
            self._state_lock.acquire(True, None)
            try:
                self._state.buf[0:4] = struct.pack("<I", 1)
                self._state.buf[4:8] = struct.pack("<I", curr_thr_id)
            finally:
                self._state_lock.release()

        return acquired

    def release(self) -> None:
        curr_thr_id = threading.get_ident()

        self._state_lock.acquire(True, None)
        try:
            if struct.unpack("<I", self._state.buf[4:8])[0] != curr_thr_id:
                raise RuntimeError("Cannot release un-acquired lock")

            count = struct.unpack("<I", self._state.buf[0:4])[0] - 1
            if count == 0:
                # last release
                self._state.buf[0:8] = bytes(8)  # Clear both count and owner
                self._lock.release()
            else:
                # still held
                self._state.buf[0:4] = struct.pack("<I", count)
        finally:
            self._state_lock.release()

    def close(self) -> None:
        self._lock.close()
        self._state_lock.close()
        self._state.close()

    def unlink(self) -> None:
        self._lock.unlink()
        self._state_lock.unlink()
        self._state.unlink()

    def cleanup(self) -> None:
        self._lock.cleanup()
        self._state_lock.cleanup()
        self._state.cleanup()
