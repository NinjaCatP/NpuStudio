@echo off
cd /d "%~dp0"
if not exist "%~dp0setup.done" (
  echo First run: setting up NpuStudio. This downloads Python and the AI model and takes a few minutes...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if not exist "%~dp0setup.done" (
    echo Setup failed. See setup.log in this folder.
    pause
    exit /b 1
  )
)
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0npustudio.py"
