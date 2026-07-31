@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "tools\check_exe_status.ps1"
set "STATUS=%ERRORLEVEL%"

if "%STATUS%"=="0" (
    echo [OK] No EXE update is required.
    exit /b 0
)
if not "%STATUS%"=="10" if not "%STATUS%"=="11" (
    echo [ERROR] EXE status check failed with code %STATUS%.
    pause
    exit /b %STATUS%
)

echo [INFO] Updating EXE...
call build_exe.bat
if errorlevel 1 (
    echo [ERROR] Could not update the application.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "tools\check_exe_status.ps1"
if errorlevel 1 (
    echo [ERROR] The rebuilt EXE is still outdated.
    pause
    exit /b 1
)

echo [OK] EXE update completed.
exit /b 0
