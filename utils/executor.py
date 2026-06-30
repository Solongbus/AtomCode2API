"""
Async subprocess runner for atomcode CLI.

Spools real-time stdout/stderr into a thread-safe in-memory store
so that the status endpoint can serve progressively-updated logs.

Design decisions
----------------
- A plain ``dict`` guarded by a ``threading.Lock`` is used instead of
  Redis for simplicity.  Swap it out when horizontal scaling is needed.
- The subprocess is started with ``creationflags=CREATE_NO_WINDOW`` on
  Windows so that no console window pops up.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    from atomcode2api.config import settings
except ModuleNotFoundError:
    from config import settings

try:
    from atomcode2api.utils.locker import DirectoryLock, DirectoryLockedError
except ModuleNotFoundError:
    from utils.locker import DirectoryLock, DirectoryLockedError

logger = logging.getLogger("atomcode2api.executor")

# ── Task status constants ──────────────────────────────────────────────
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# ── In-memory task store (thread-safe) ─────────────────────────────────
import threading  # noqa: E402 (import after constants)

_store_lock = threading.Lock()
_tasks: Dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _filter_user_visible_output_line(text: str) -> str:
    """Filter out daemon/engine diagnostic lines from user-visible output.

    Lines starting with ``[engine`` or ``[headless`` are daemon-internal
    diagnostics (model load, permission mode, etc.).  They still appear in
    the server log (via ``logger.info``) for debugging, but are excluded
    from the SSE stream sent to the IDE so the user only sees the AI's
    actual thinking and execution output.
    """
    stripped = text.strip()
    if stripped.startswith("[engine") or stripped.startswith("[headless"):
        return ""
    return text


# ── Public helpers ─────────────────────────────────────────────────────


def init_task(task_id: str, prompt: str, project_path: str) -> dict:
    """Create a new task record and store it."""
    record = {
        "task_id": task_id,
        "prompt": prompt,
        "project_path": project_path,
        "status": STATUS_QUEUED,
        "logs": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    with _store_lock:
        _tasks[task_id] = record
    return record


def get_task(task_id: str) -> Optional[dict]:
    """Return a *copy* of the task record, or *None*."""
    with _store_lock:
        raw = _tasks.get(task_id)
        if raw is None:
            return None
        return dict(raw)  # shallow copy


def _update_logs(task_id: str, chunk: str):
    """Append log chunk and set updated_at under lock."""
    with _store_lock:
        record = _tasks.get(task_id)
        if record is None:
            return
        record["logs"] += chunk
        # Keep only the last N lines to avoid unbounded memory growth.
        lines = record["logs"].splitlines(keepends=True)
        if len(lines) > settings.max_log_lines:
            lines = lines[-settings.max_log_lines :]
            lines.insert(0, "[truncated: earlier lines omitted]\n")
        record["logs"] = "".join(lines)
        record["updated_at"] = _now_iso()


def _set_status(task_id: str, status: str):
    with _store_lock:
        record = _tasks.get(task_id)
        if record is not None:
            record["status"] = status
            record["updated_at"] = _now_iso()


# ── Background coroutine ───────────────────────────────────────────────


async def run_atomcode(task_id: str, prompt: str, project_path: str):
    """
    Long-running background coroutine that:

    1. Acquires a directory lock on *project_path*.
    2. Spawns ``atomcode --execute <prompt> --yes`` as a subprocess.
    3. Streams stdout/stderr into the task record.
    4. Releases the lock on completion (success or failure).
    """
    lock = DirectoryLock(project_path, task_id)

    # ── 1. Lock ────────────────────────────────────────────────────────
    try:
        lock.acquire()
    except DirectoryLockedError as exc:
        _set_status(task_id, STATUS_FAILED)
        _update_logs(task_id, f"[lock-error] {exc}\n")
        logger.error("Task %s failed to acquire lock: %s", task_id, exc)
        return
    except Exception as exc:
        _set_status(task_id, STATUS_FAILED)
        _update_logs(task_id, f"[lock-error] Unexpected: {exc}\n")
        logger.exception("Task %s unexpected lock error", task_id)
        return

    # ── 2. Build command ───────────────────────────────────────────────
    prompt_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as tmp:
            tmp.write(prompt)
            prompt_file_path = tmp.name
    except Exception as exc:
        _set_status(task_id, STATUS_FAILED)
        _update_logs(task_id, f"[start-error] failed to create prompt file: {exc}\n")
        logger.exception("Task %s failed to create prompt file", task_id)
        lock.release()
        return

    cmd = [settings.atomcode_bin, "--prompt-file", prompt_file_path]
    cmd.extend(settings.atomcode_extra_args)

    binary_path = settings.atomcode_bin
    binary_exists = os.path.isfile(binary_path) if os.path.isabs(binary_path) else shutil.which(binary_path) is not None
    logger.info(
        "Task %s starting: cmd=%s cwd=%s binary_exists=%s",
        task_id,
        " ".join(cmd),
        project_path,
        binary_exists,
    )

    if os.path.isabs(binary_path) and not os.path.isfile(binary_path):
        msg = f"[start-error] Binary path does not exist: {binary_path}\n"
        _set_status(task_id, STATUS_FAILED)
        _update_logs(task_id, msg)
        logger.error("Task %s: %s", task_id, msg.strip())
        lock.release()
        return

    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=project_path,
            shell=False,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ},
            creationflags=creation_flags,
        )
    except FileNotFoundError as exc:
        msg = (
            f"[start-error] Failed to launch binary '{settings.atomcode_bin}': {exc}. "
            "This may indicate missing executable, missing runtime dependency, or environment restrictions.\n"
        )
        _set_status(task_id, STATUS_FAILED)
        _update_logs(task_id, msg)
        logger.error("Task %s: %s", task_id, msg.strip())
        lock.release()
        return
    except Exception as exc:
        _set_status(task_id, STATUS_FAILED)
        _update_logs(task_id, f"[start-error] {exc}\n")
        logger.exception("Task %s failed to start subprocess", task_id)
        lock.release()
        return

    # ── 3. Stream output ───────────────────────────────────────────────
    _set_status(task_id, STATUS_RUNNING)
    logger.info("Task %s started (pid=%d, cwd=%s)", task_id, proc.pid, project_path)

    async def _read_stream():
        """Read stdout line-by-one and update logs + server live log.

        On Windows, atomcode.exe uses a launcher/engine split architecture:
        the launcher process exits quickly and closes its stdout pipe handle,
        but the real AI engine may continue running and producing output
        through an inherited pipe handle.

        Strategy:
        1. First EOF: enter grace period (readline() with polling).
        2. After grace: check if the process (or any child) is still alive
           by testing the handle.  If so, keep polling — the engine may
           produce more output.
        3. Only declare the stream truly ended when the pipe is closed AND
           the subprocess has exited, to ensure no late data is missed.
        """
        assert proc.stdout is not None
        line_count = 0
        eof_count = 0          # consecutive EOF polls
        MAX_EOF_POLLS = 50     # ~15 s of grace (@ 0.3 s sleep)
        _line_remainder = b""  # local buffer for partial-line coalescing

        while True:
            # ── Try to read a chunk (non-line-buffered) ──────────────
            try:
                chunk = await asyncio.wait_for(
                    proc.stdout.read(4096), timeout=0.3
                )
            except asyncio.TimeoutError:
                chunk = b""

            if chunk:
                eof_count = 0
                _line_remainder += chunk
                # Emit complete lines, keep the partial tail.
                *complete, tail = _line_remainder.split(b"\n")
                _line_remainder = tail
                for raw in complete:
                    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    filtered_text = _filter_user_visible_output_line(text)
                    if filtered_text:
                        _update_logs(task_id, filtered_text + "\n")
                    logger.info("[task %s] %s", task_id, text)
                    line_count += 1
            else:
                # ── No data this poll ────────────────────────────────
                eof_count += 1
                # If nothing in the partial buffer, check EOF conditions.
                if not _line_remainder:
                    # NOTE: On Windows, atomcode.exe uses a launcher/engine split
                    # architecture. The launcher process (proc) exits quickly after
                    # spawning the real AI engine, so `proc.returncode is not None`
                    # does NOT mean output is done. We must NOT bail out early based
                    # on launcher exit — instead rely solely on MAX_EOF_POLLS to
                    # detect genuine pipe closure / engine termination.
                    if eof_count > MAX_EOF_POLLS:
                        logger.warning(
                            "Task %s read stream timeout after %d polls (~%.0fs)",
                            task_id,
                            eof_count,
                            eof_count * 0.3,
                        )
                        break
                    await asyncio.sleep(0.3)
                    continue

        # Flush the final partial line (no trailing newline).
        if _line_remainder:
            text = _line_remainder.decode("utf-8", errors="replace").rstrip("\r\n")
            if text:
                _update_logs(task_id, text + "\n")
                logger.info("[task %s] %s", task_id, text)
                line_count += 1

        logger.info("Task %s stream ended (%d lines)", task_id, line_count)

    try:
        await asyncio.wait_for(
            asyncio.gather(_read_stream(), proc.wait()),
            timeout=settings.task_timeout_seconds,
        )
    except asyncio.TimeoutError:
        proc.terminate()
        # Give it a few seconds to shut down gracefully, then force-kill.
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        _set_status(task_id, STATUS_FAILED)
        _update_logs(
            task_id,
            f"\n[timeout] Task exceeded {settings.task_timeout_seconds}s and was killed.\n",
        )
        logger.warning("Task %s timed out after %ds", task_id, settings.task_timeout_seconds)
    except Exception as exc:
        _set_status(task_id, STATUS_FAILED)
        _update_logs(task_id, f"\n[error] {exc}\n")
        logger.exception("Task %s encountered an unexpected error", task_id)
    else:
        # Normal completion – check return code.
        returncode = proc.returncode
        if returncode == 0:
            _set_status(task_id, STATUS_COMPLETED)
            logger.info("Task %s completed successfully", task_id)
        else:
            _set_status(task_id, STATUS_FAILED)
            _update_logs(
                task_id,
                f"\n[exit] Process exited with code {returncode}\n",
            )
            logger.warning("Task %s failed with exit code %d", task_id, returncode)
    finally:
        if prompt_file_path:
            try:
                os.unlink(prompt_file_path)
            except OSError:
                pass
        # ── 4. Always release the lock ─────────────────────────────────
        lock.release()