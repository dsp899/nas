from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


@contextmanager
def exclusive_file_lock(lock_path: str, *, poll_interval_s: float = 0.2, stale_after_s: float | None = 1800.0) -> Iterator[None]:
    _ensure_parent(lock_path)
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode('utf-8'))
            break
        except FileExistsError:
            if stale_after_s is not None:
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                    if age > stale_after_s:
                        os.remove(lock_path)
                        continue
                except FileNotFoundError:
                    continue
            time.sleep(poll_interval_s)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
