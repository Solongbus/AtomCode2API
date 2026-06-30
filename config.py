"""
Application configuration schema.

All tunable settings are centralized here so that CLI flags, sandbox paths,
and timeouts can be adjusted without touching business logic.

Environment variables
---------------------
Every setting can be overridden via ``ATOMCODE_<UPPER_FIELD_NAME>`` env vars:

  - ``ATOMCODE_API_KEY``       → *api_key*
  - ``ATOMCODE_PORT``          → *port*
  - ``ATOMCODE_ATOMCODE_BIN``  → *atomcode_bin*
  - ``ATOMCODE_DEBUG``         → *debug*  (accepts true/1/yes)
  - etc.

Boolean fields accept ``true`` / ``1`` / ``yes`` (case-insensitive).
List fields accept comma-separated values.
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import List


# ── Helpers ─────────────────────────────────────────────────────────────

def _default_atomcode_bin() -> str:
    """Return a portable default path for the atomcode CLI.

    On Windows the binary lives under %LOCALAPPDATA%\AtomCode\atomcode.exe;
    on Linux/macOS it is expected to be on PATH as ``atomcode``.
    """
    if os.name == "nt":
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            candidate = os.path.join(localappdata, "AtomCode", "atomcode.exe")
            if os.path.isfile(candidate):
                return candidate
        # fallback: try PATH lookup
        which = shutil.which("atomcode")
        if which:
            return which
        # last resort — keep the well-known location even if absent
        return os.path.join(localappdata or r"C:\Users\%USERNAME%\AppData\Local",
                            "AtomCode", "atomcode.exe")
    return "atomcode"


@dataclass
class Settings:
    # ── FastAPI / Uvicorn ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8123
    debug: bool = False

    # ── API Key ────────────────────────────────────────────────────────
    # If non-empty, all agent/OpenAI endpoints require
    #   Authorization: Bearer <api_key>
    api_key: str = ""

    # ── Default workspace (for OpenAI-compatible endpoint) ─────────────
    # If empty, the OpenAI endpoint requires an explicit project_path
    # in the request body or X-Project-Path header.
    default_workspace: str = ""

    # ── Mode: "daemon" | "cli" ─────────────────────────────────────────
    # "daemon" — connects to (or auto-starts) atomcode-daemon HTTP API.
    #            Enables rich SSE events, session management, tool calls.
    # "cli"    — spawns `atomcode --execute <prompt>` as a subprocess.
    #            Simpler but loses structured streaming & multi-turn context.
    mode: str = "cli"

    # ── Daemon settings (used only when mode="daemon") ─────────────────
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 13456
    daemon_auto_start: bool = True
    # Path to atomcode-daemon binary (auto-detected if empty).
    daemon_binary_path: str = ""

    # ── CLI settings (used only when mode="cli") ───────────────────────
    # Name or absolute path of the atomcode CLI executable.
    # Default expands %LOCALAPPDATA%\AtomCode\atomcode.exe on Windows
    # so it works across different user accounts.
    atomcode_bin: str = field(default_factory=_default_atomcode_bin)

    # Extra flags appended to every invocation.
    # `-y` skips all permission prompts (non-interactive).
    atomcode_extra_args: List[str] = field(default_factory=lambda: ["-y"])

    # Shell to use on Windows; on Linux/macOS the default shell is used.
    shell: str = "powershell" if os.name == "nt" else ""

    # ── Timeouts ───────────────────────────────────────────────────────
    # Maximum wall-clock seconds a task is allowed to run.
    task_timeout_seconds: int = 600  # 10 minutes

    # ── Lock file ──────────────────────────────────────────────────────
    # Name of the temporary file created inside the workspace directory
    # to signal that a task is currently modifying it.
    lock_filename: str = ".atomcode.lock"

    # ── Task status backend ────────────────────────────────────────────
    # How many recent log lines to keep per task in memory.
    max_log_lines: int = 5000

    # ── Path validation ────────────────────────────────────────────────
    # When set, only projects under these parent directories are allowed.
    # An empty list means *any* absolute path is accepted.
    allowed_project_roots: List[str] = field(default_factory=list)


# ── Build singleton from env vars ──────────────────────────────────────

def _load_settings_from_env() -> dict:
    """Read env vars and map them to Settings field names.

    Primary rule:
      ATOMCODE_<UPPER_FIELD_NAME>  -> <field_name>

    Compatibility aliases:
      ATOMCODE_BIN                 -> atomcode_bin
    """
    overrides: dict = {}
    prefix = "ATOMCODE_"
    aliases = {
        "BIN": "atomcode_bin",
    }
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        raw_name = key[len(prefix) :]
        field_name = aliases.get(raw_name, raw_name.lower())
        overrides[field_name] = val
    return overrides


_settings_kwargs = _load_settings_from_env()

if _settings_kwargs:
    _defaults = Settings()
    cleaned = {}
    for fld_name, raw in _settings_kwargs.items():
        if not hasattr(_defaults, fld_name):
            continue
        default_val = getattr(_defaults, fld_name)
        if isinstance(default_val, bool):
            cleaned[fld_name] = raw.lower() in ("true", "1", "yes")
        elif isinstance(default_val, int):
            cleaned[fld_name] = int(raw)
        elif isinstance(default_val, list):
            cleaned[fld_name] = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            cleaned[fld_name] = raw
    _settings_kwargs = cleaned

settings = Settings(**_settings_kwargs)