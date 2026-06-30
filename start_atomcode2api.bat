@echo off
chcp 65001 >nul 2>&1
title atomcode2api

set ATOMCODE_API_KEY=atomcode2api-local-dev-key
set ATOMCODE_MODE=cli

rem Prefer absolute atomcode path to avoid PATH-dependent failures
set "ATOMCODE_BIN=%LOCALAPPDATA%\AtomCode\atomcode.exe"
if not exist "%ATOMCODE_BIN%" (
    for /f "delims=" %%I in ('where.exe atomcode 2^>nul') do (
        set "ATOMCODE_BIN=%%I"
        goto :atomcode_found
    )
)
:atomcode_found
if exist "%ATOMCODE_BIN%" (
    echo [OK] atomcode = %ATOMCODE_BIN%
) else (
    echo [ERROR] atomcode.exe not found.
    echo         Expected: %%LOCALAPPDATA%%\AtomCode\atomcode.exe
    echo         Please install AtomCode or update ATOMCODE_BIN.
    pause
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set ATOMCODE_DEFAULT_WORKSPACE=%SCRIPT_DIR%
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

echo ============================================
echo  atomcode2api - Starting Server (CLI mode)
echo ============================================
echo  API  : http://127.0.0.1:8123
echo  Docs : http://127.0.0.1:8123/docs
echo  Mode : %ATOMCODE_MODE%
echo ============================================

rem Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [INFO] No virtual environment found, using system Python.
)

rem Check and install dependencies
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r "%SCRIPT_DIR%requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
)

rem Check PySide6 (GUI dependency)
pip show PySide6 >nul 2>&1
if errorlevel 1 (
    echo [INFO] PySide6 not found -- installing...
    pip install PySide6
    if errorlevel 1 (
        echo [WARN] PySide6 installation failed; GUI will be unavailable.
        echo [WARN] The server will start in headless mode instead.
    ) else (
        echo [OK] PySide6 installed.
    )
)

rem Check port 8123 without killing anything automatically
echo [OK] Checking port 8123...
for /f "tokens=5" %%a in ('
    netstat -ano ^| findstr /c:":8123 " ^| findstr /i "LISTENING"
') do (
    echo [WARN] Port 8123 is already in use by PID %%a
    echo [WARN] Please stop the existing process first if startup fails.
    goto :port_checked
)
:port_checked
timeout /t 1 /nobreak >nul

echo.
echo [OK] Starting server at http://127.0.0.1:8123
echo [OK] API docs at http://127.0.0.1:8123/docs
echo [OK] CWD      = %cd%
echo [OK] APP DIR  = %SCRIPT_DIR%
echo.

python main.py

if errorlevel 1 (
    echo [ERROR] Server exited with code %errorlevel%
    pause
)
