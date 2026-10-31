@echo off
cd /d "%~dp0"
if not exist "%~dp0setup.done" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
)
title NpuStudio desktop worker
"%~dp0venv\Scripts\python.exe" "%~dp0worker_server.py"
pause
