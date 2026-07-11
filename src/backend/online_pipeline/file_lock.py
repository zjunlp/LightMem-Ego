from __future__ import annotations

import os
import time
from pathlib import Path


try:
    from filelock import FileLock as FileLock
except Exception:

    class FileLock:
        def __init__(self, lock_file: str | Path, timeout: float = 30.0, poll_interval: float = 0.1) -> None:
            self.lock_file = Path(lock_file)
            self.timeout = float(timeout)
            self.poll_interval = float(poll_interval)
            self._fd: int | None = None

        def acquire(self) -> None:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            deadline = time.time() + max(0.0, self.timeout)
            while True:
                try:
                    self._fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(self._fd, str(os.getpid()).encode("utf-8"))
                    return
                except FileExistsError:
                    if time.time() >= deadline:
                        raise TimeoutError(f"timed out waiting for lock: {self.lock_file}")
                    time.sleep(self.poll_interval)

        def release(self) -> None:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None
            try:
                self.lock_file.unlink()
            except FileNotFoundError:
                pass

        def __enter__(self) -> "FileLock":
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.release()
