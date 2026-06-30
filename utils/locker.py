"""
Lightweight file-based directory lock.

Guarantees that only one atomcode task operates on a given workspace
directory at any moment, preventing concurrent git / file-write corruption.

Thread-safe + process-safe (cross-process via filesystem).
"""

import errno
import logging
import os
import time

try:
    from atomcode2api.config import settings
except ModuleNotFoundError:
    from config import settings

logger = logging.getLogger("atomcode2api.locker")


class DirectoryLockError(Exception):
    """Raised when the lock cannot be acquired."""


class DirectoryLockedError(DirectoryLockError):
    """Raised when the directory is already locked by another task."""


class DirectoryLock:
    """
    Context manager / decorator-style lock backed by a temporary file.

    Usage::

        with DirectoryLock("/path/to/workspace") as lock:
            # exclusive access
            ...
    """

    def __init__(self, path: str, task_id: str, timeout: float = 0) -> None:
        self._path = os.path.abspath(path)
        self._lock_file = os.path.join(self._path, settings.lock_filename)
        self._task_id = task_id
        self._timeout = timeout
        self._acquired = False

    # ── public API ─────────────────────────────────────────────────────

    @property
    def is_locked(self) -> bool:
        """Return *True* when the lock file currently exists."""
        return os.path.isfile(self._lock_file)

    def acquire(self) -> None:
        """
        Try to acquire the lock.

        Raises *DirectoryLockedError* when the lock is already held and
        *timeout* is 0 (the default), or when *timeout* seconds have
        elapsed without success.
        """
        deadline = time.monotonic() + self._timeout if self._timeout >= 0 else None

        while True:
            try:
                self._try_create_lock()
                self._acquired = True
                logger.info(
                    "Lock acquired for %s  task=%s", self._path, self._task_id
                )
                return
            except DirectoryLockedError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)

    def release(self) -> None:
        """Release the lock (no-op if not held)."""
        if not self._acquired:
            return
        try:
            os.remove(self._lock_file)
        except OSError:
            pass
        self._acquired = False
        logger.info(
            "Lock released for %s  task=%s", self._path, self._task_id
        )

    # ── context manager ────────────────────────────────────────────────

    def __enter__(self) -> "DirectoryLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ── internals ──────────────────────────────────────────────────────

    def _try_create_lock(self) -> None:
        # If the lock file already exists, check if its owner task is still alive.
        if os.path.isfile(self._lock_file):
            self._maybe_stale_lock()
            raise DirectoryLockedError(
                f"Workspace '{self._path}' is locked by another task "
                f"(lock file: {self._lock_file})"
            )

        try:
            with open(self._lock_file, "x", encoding="utf-8") as f:
                f.write(f"{self._task_id}\n")
        except FileExistsError:
            raise DirectoryLockedError(
                f"Workspace '{self._path}' was locked concurrently"
            )

    def _maybe_stale_lock(self) -> None:
        """
        Remove lock file if the process that held it no longer exists.

        Heuristic: if the lock file contains only an empty line or a PID
        that is not running, consider it stale.  This is best-effort.
        """
        try:
            with open(self._lock_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                os.remove(self._lock_file)
                logger.warning("Removed stale lock file (empty) in %s", self._path)
                return
            # The content is a UUID or PID – we cannot reliably check a
            # remote process via filesystem alone.  Only remove truly empty files.
        except OSError:
            pass


def check_locked(path: str) -> bool:
    """Convenience function: return *True* if *path* is currently locked."""
    return os.path.isfile(os.path.join(os.path.abspath(path), settings.lock_filename))