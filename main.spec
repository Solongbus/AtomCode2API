# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for atomcode2api.exe

Usage:
    pyinstaller main.spec

Notes:
    - Uses ``onedir`` (not ``onefile``) because PySide6 is ~200 MB
      and Qt plugin paths break under ``onefile`` extraction.
    - Console is kept (console=True) so startup errors are visible.
      Remove ``--windowed`` from _buildEXE.bat once debugged.
    - All dynamic / conditional imports are listed in ``hiddenimports``
      because PyInstaller's static analyser cannot follow try/except
      import patterns.
"""

import os
import sys
from pathlib import Path

# ── Collect all PySide6 sub-modules ──────────────────────────────────────
# We list PySide6 itself in hiddenimports so that PyInstaller's built-in
# hooks (hook-PySide6.py etc.) fire automatically and collect Qt DLLs,
# plugins, translations, and data files — no manual collect_all needed.
from PyInstaller.utils.hooks import collect_submodules

pyside_imports = collect_submodules("PySide6")


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # PySide6 + all sub-modules (triggers built-in PySide6 hooks)
        "PySide6",
        # ── Dynamic / conditional imports from main.py ───────────────
        # ``from config import settings`` fallback
        "config",
        # ``from atomcode2api.config import settings``  (the non-fallback)
        "atomcode2api.config",
        # ``from utils.executor import ...`` fallback
        "utils.executor",
        "utils.locker",
        # ``from atomcode2api.utils.executor import ...`` (non-fallback)
        "atomcode2api.utils.executor",
        "atomcode2api.utils.locker",
        # daemon mode imports (inside  if settings.mode == "daemon")
        "daemon_client",
        "daemon_manager",
        "atomcode2api.daemon_client",
        "atomcode2api.daemon_manager",
        # GUI import (inside  if __name__ == "__main__")
        "gui",
        # Python < 3.10 compat
        "typing_inspection.introspection",
    ]
    + pyside_imports,  # auto-collected PySide6 sub-modules
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Unnecessary — reduce exe size
        "tkinter",
        "matplotlib",
        "numpy",
        "pandas",
        "PIL",
        "cv2",
        "scipy",
        "sympy",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="atomcode2api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── One-directory distribution ────────────────────────────────────────
# Much more reliable than --onefile for PySide6.  Outputs:
#   dist/atomcode2api/
#       atomcode2api.exe    <-- the launcher
#       _internal/          <-- all Python libs + PySide6 DLLs
#
# Double-click atomcode2api.exe to run (requires the _internal/ folder
# to be present beside it).
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="atomcode2api",
)
