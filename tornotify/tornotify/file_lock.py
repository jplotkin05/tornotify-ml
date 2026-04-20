"""
Small cross-process file lock helper for CSV append logs.

The distributed scanner can run many Python worker processes on one machine or
across machines sharing a mounted output directory. Thread locks are not enough
in that mode, so writers also take a sidecar .lock file.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


@contextmanager
def locked_file(path: Path) -> Iterator[None]:
    """Hold an exclusive lock for writes related to path."""
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
