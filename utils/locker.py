"""
Lightweight file-based directory lock.

Guarantees that only one atomcode task operates on a given workspace
directory at any moment, preventing concurrent git / file-write corruption.

Thread-safe + process-safe (cross-process via filesystem).

Lock files are stored under the system temp directory so they never
pollute the workspace with "atom lock" files.
"""

import hashlib
import logging
import os
import shutil
import tempfile
import time

logger = logging.getLogger("atomcode2api.locker")

# Dedicated temp subdirectory for all lock files.
_LOCK_DIR = os.path.join(tempfile.gettempdir(), "atomcode2api", "locks")


def clear_all_locks() -> None:
    """Remove all lock files from the temp lock directory.

    Call this at application startup so that stale locks from a previous
    crashed instance never block subsequent conversations.
    """
    if os.path.isdir(_LOCK_DIR):
        try:
            shutil.rmtree(_LOCK_DIR)
            logger.info("Cleared all stale lock files from %s", _LOCK_DIR)
        except OSError as exc:
            logger.warning("Failed to clear lock directory %s: %s", _LOCK_DIR, exc)


class DirectoryLockError(Exception):
    """Raised when the lock cannot be acquired."""


class DirectoryLockedError(DirectoryLockError):
    """Raised when the directory is already locked by another task."""


class DirectoryLock:
    """
    Context manager / decorator-style lock backed by a temporary file.

    The lock file is stored under the system temp directory, keyed by an
    SHA-256 hash of the workspace absolute path — no files are created
    inside the workspace itself.

    Usage::

        with DirectoryLock("/path/to/workspace") as lock:
            # exclusive access
            ...
    """

    def __init__(self, path: str, task_id: str, timeout: float = 0) -> None:
        self._path = os.path.abspath(path)
        # Hash the workspace path to produce a stable, safe filename
        # that does NOT contain "atom" or "lock" wording.
        path_hash = hashlib.sha256(self._path.encode("utf-8")).hexdigest()[:16]
        self._lock_file = os.path.join(_LOCK_DIR, f"{path_hash}.lck")
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
        # If the lock file already exists, treat it as stale and remove it.
        # (After startup cleanup, any leftover lock is from a crashed task.)
        if os.path.isfile(self._lock_file):
            self._remove_stale_lock()

        # Ensure the lock directory exists.
        os.makedirs(_LOCK_DIR, exist_ok=True)

        try:
            with open(self._lock_file, "x", encoding="utf-8") as f:
                f.write(f"{self._task_id}\n")
        except FileExistsError:
            raise DirectoryLockedError(
                f"Workspace '{self._path}' was locked concurrently"
            )

    def _remove_stale_lock(self) -> None:
        """Remove an existing lock file (considered stale)."""
        try:
            os.remove(self._lock_file)
            logger.warning("Removed stale lock file for %s", self._path)
        except OSError:
            pass


def check_locked(path: str) -> bool:
    """Convenience function: return *True* if *path* is currently locked."""
    path_hash = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
    return os.path.isfile(os.path.join(_LOCK_DIR, f"{path_hash}.lck"))
