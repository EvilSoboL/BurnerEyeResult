@echo off
setlocal
cd /d "%~dp0"

set "EXE=dist\BurnerEyeReport.exe"

call update_exe.bat
if errorlevel 1 (
    echo [ERROR] Could not update the application.
    pause
    exit /b 1
)

echo [INFO] Starting %EXE%...
start "" /wait "%EXE%" %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo [ERROR] Application exited with code %APP_EXIT_CODE%.
    pause
)
exit /b %APP_EXIT_CODE%
