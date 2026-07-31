@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
set "EXE=dist\BurnerEyeReport.exe"

if not exist "%PYTHON%" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python launcher "py" was not found.
        echo Install Python 3.10 or newer and run this file again.
        pause
        exit /b 1
    )

    echo [INFO] Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo [INFO] Checking Python version...
"%PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    pause
    exit /b 1
)

echo [INFO] Checking Tcl/Tk GUI runtime...
"%PYTHON%" -c "import tkinter as tk; interpreter = tk.Tcl(); print('[OK] Tcl/Tk ' + interpreter.eval('info patchlevel'))"
if errorlevel 1 (
    echo [ERROR] Tcl/Tk is unavailable. Repair Python and enable the "tcl/tk and IDLE" component.
    pause
    exit /b 1
)

echo [INFO] Checking build dependencies...
"%PYTHON%" -m pip install --disable-pip-version-check -r requirements-build.txt
if errorlevel 1 (
    echo [ERROR] Failed to install build dependencies.
    pause
    exit /b 1
)

echo [INFO] Building %EXE%...
"%PYTHON%" -m PyInstaller --noconfirm --clean BurnerEyeReport.spec
if errorlevel 1 (
    echo [ERROR] EXE build failed.
    pause
    exit /b 1
)

if not exist "%EXE%" (
    echo [ERROR] Build completed without creating %EXE%.
    pause
    exit /b 1
)

echo [OK] EXE created: %CD%\%EXE%
exit /b 0
