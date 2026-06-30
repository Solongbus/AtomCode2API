"""
atomcode-daemon process lifecycle manager.

Mirrors the logic from the VS Code extension's ``daemon/process.js``:

- Locates the daemon binary (bundled in vsix, on PATH, or user-specified).
- Starts the daemon as a subprocess.
- Performs health checks and version validation.
- Shuts down the daemon on exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from atomcode2api.daemon_client import DaemonClient
except ModuleNotFoundError:
    from daemon_client import DaemonClient

logger = logging.getLogger("atomcode2api.daemon_manager")

DAEMON_VERSION_FILE = "daemon-version.txt"

# Relative path inside the vsix bundle:
# extension/resources/bin/<platform>/atomcode-daemon(.exe)
BUNDLED_PLATFORM_DIRS = {
    "win32": "win32-x64",
    "darwin": {
        "arm64": "darwin-arm64",
        "x86_64": "darwin-x64",
    },
    "linux": {
        "aarch64": "linux-arm64",
        "x86_64": "linux-x64",
    },
}


def _platform_dir() -> Optional[str]:
    """Return the platform-specific subdirectory name for the bundled daemon."""
    sys_platform = sys.platform
    machine = platform.machine().lower()

    if sys_platform == "win32":
        return "win32-x64"
    elif sys_platform == "darwin":
        if machine in ("arm64", "aarch64"):
            return "darwin-arm64"
        else:
            return "darwin-x64"
    elif sys_platform == "linux":
        if machine in ("arm64", "aarch64"):
            return "linux-arm64"
        else:
            return "linux-x64"
    return None


def _daemon_binary_name() -> str:
    return "atomcode-daemon.exe" if os.name == "nt" else "atomcode-daemon"


class DaemonManager:
    """
    Manages the lifecycle of the atomcode-daemon process.

    Usage::

        mgr = DaemonManager(port=13456)
        await mgr.ensure_running()
        # ... use the daemon ...
        mgr.dispose()
    """

    def __init__(
        self,
        client: DaemonClient,
        binary_path: str = "",
        auto_start: bool = True,
        daemon_port: int = 13456,
    ) -> None:
        self.client = client
        self.binary_path = binary_path
        self.auto_start = auto_start
        self.daemon_port = daemon_port
        self._process: Optional[asyncio.subprocess.Process] = None
        self._ensure_promise: Optional[asyncio.Future] = None

    # ── Public API ───────────────────────────────────────────────────────

    async def ensure_running(self) -> bool:
        """
        Ensure the daemon is running and healthy.

        Returns True if the daemon is ready, False otherwise.
        """
        if self._ensure_promise is not None:
            return await self._ensure_promise

        self._ensure_promise = asyncio.ensure_future(self._ensure_impl())
        try:
            return await self._ensure_promise
        finally:
            self._ensure_promise = None

    async def shutdown(self) -> bool:
        """
        Gracefully shut down the daemon.

        Returns True if shutdown succeeded, False otherwise.
        """
        try:
            await self.client.shutdown()
            logger.info("Daemon shutdown signal sent")
            return True
        except Exception as exc:
            logger.warning("Failed to send shutdown signal: %s", exc)
            return False

    def dispose(self) -> None:
        """Clean up resources. Call on application shutdown."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            logger.info("Daemon process terminated")

    # ── Internal ─────────────────────────────────────────────────────────

    async def _ensure_impl(self) -> bool:
        # 1. Check if already running
        health = await self._try_get_health()
        if health:
            expected = self._get_expected_version()
            if not expected or health.get("version") == expected:
                logger.info("Daemon already running (version=%s)", health.get("version"))
                return True
            # Version mismatch — restart
            logger.warning(
                "Daemon version mismatch: running=%s, expected=%s. Restarting...",
                health.get("version"), expected,
            )
            shutdown_ok = await self.shutdown()
            if not shutdown_ok:
                logger.error("Could not stop old daemon; version mismatch remains")
                return False
            # Fall through to start()

        # 2. Not running — start if auto_start is enabled
        if not self.auto_start:
            logger.info("Daemon not running and auto_start is disabled")
            return False

        return await self._start()

    async def _try_get_health(self) -> Optional[dict]:
        try:
            return await self.client.health()
        except Exception:
            return None

    def _get_expected_version(self) -> Optional[str]:
        """Read the expected daemon version from the bundled version file."""
        # Search relative to this file's location (project root / ref / vsix_extracted)
        search_paths = [
            Path(__file__).resolve().parent.parent,  # atomcode2api/
            Path(__file__).resolve().parent.parent.parent,  # project root
            Path(__file__).resolve().parent.parent.parent / "ref" / "vsix_extracted" / "extension" / "resources" / "bin",
        ]
        for base in search_paths:
            vf = base / "resources" / "bin" / DAEMON_VERSION_FILE
            if vf.exists():
                return vf.read_text(encoding="utf-8").strip()
            # Also check extraction path directly
            vf2 = base / DAEMON_VERSION_FILE
            if vf2.exists():
                return vf2.read_text(encoding="utf-8").strip()
        return None

    async def _start(self) -> bool:
        """Start the daemon process and wait for it to become healthy."""
        binary = await self._find_binary()
        if not binary:
            logger.error("Could not find atomcode-daemon binary")
            return False

        logger.info("Starting daemon: %s --port %d", binary, self.daemon_port)

        try:
            self._process = await asyncio.create_subprocess_exec(
                str(binary),
                "--port", str(self.daemon_port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("Daemon binary not found: %s", binary)
            return False
        except Exception as exc:
            logger.error("Failed to start daemon: %s", exc)
            return False

        # Wait for the daemon to become healthy (poll up to 5 seconds)
        for attempt in range(10):
            await asyncio.sleep(0.5)
            if self._process.returncode is not None:
                logger.error("Daemon exited prematurely with code %d", self._process.returncode)
                return False
            health = await self._try_get_health()
            if health:
                logger.info("Daemon started successfully (version=%s)", health.get("version"))
                return True

        logger.error("Daemon started but did not become healthy within 5 seconds")
        self._process.terminate()
        return False

    async def _find_binary(self) -> Optional[Path]:
        """Locate the daemon binary: explicit path → bundled → PATH."""
        # 1. User-specified path
        if self.binary_path:
            p = Path(self.binary_path)
            if p.is_file():
                return p
            logger.warning("Specified binary path not found: %s", self.binary_path)

        # 2. Bundled in vsix extraction (relative to project root)
        platform_subdir = _platform_dir()
        if platform_subdir:
            binary_name = _daemon_binary_name()
            search_roots = [
                Path(__file__).resolve().parent.parent.parent
                / "ref" / "vsix_extracted" / "extension" / "resources" / "bin"
                / platform_subdir,
                Path(__file__).resolve().parent.parent.parent
                / "ref",
            ]
            for root in search_roots:
                candidate = root / binary_name
                if candidate.exists():
                    # Ensure executable
                    if os.name != "nt":
                        candidate.chmod(candidate.stat().st_mode | 0o111)
                    return candidate

        # 3. On PATH (e.g. `atomcode daemon` or standalone `atomcode-daemon`)
        binary_name = _daemon_binary_name()
        try:
            proc = await asyncio.create_subprocess_exec(
                "where" if os.name == "nt" else "which",
                binary_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout:
                path = stdout.decode("utf-8").strip().splitlines()[0]
                return Path(path)
        except Exception:
            pass

        # 4. Try `atomcode daemon` subcommand
        try:
            proc = await asyncio.create_subprocess_exec(
                "atomcode", "daemon", "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if proc.returncode == 0:
                # `atomcode daemon` is available — but we need the actual binary
                # The VS Code extension bundles it separately; fall back to PATH search.
                pass
        except Exception:
            pass

        logger.error(
            "atomcode-daemon not found. Install atomcode or place the binary on PATH."
        )
        return None
