@echo off
setlocal
REM Use Windows PowerShell 5.1 (System.IO.Ports is built-in). pwsh may need: Install-Package System.IO.Ports
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0olympus_turret_pipe_gui.ps1"
if errorlevel 1 pause
