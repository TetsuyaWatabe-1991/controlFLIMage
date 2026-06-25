@echo off
REM Launcher: prefers FLIMageProfileManager.exe (PS2EXE), else runs .ps1
cd /d "%~dp0"
set "FLIM_PROFILE_DIR=%~dp0"

if exist "%~dp0FLIMageProfileManager.exe" (
    start "" "%~dp0FLIMageProfileManager.exe"
    exit /b 0
)

if exist "%~dp0FLIMageProfileManager.ps1" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0FLIMageProfileManager.ps1"
    if errorlevel 1 (
        echo Failed to start FLIMage Profile Manager.
        pause
    )
    exit /b %errorlevel%
)

echo FLIMage Profile Manager not found. Build with Build-FLIMageProfileManager.ps1
pause
exit /b 1
