@echo off
REM Launches the PowerShell UI. The .exe in this folder is an older build.
cd /d "%~dp0"

if exist "%~dp0FLIMageProfileManager.ps1" (
    powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0FLIMageProfileManager.ps1"
    if errorlevel 1 (
        echo Failed to start FLIMage Profile Manager.
        pause
    )
    exit /b %errorlevel%
)

echo FLIMageProfileManager.ps1 was not found.
pause
exit /b 1
