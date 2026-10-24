@echo off
setlocal

where wsl >nul 2>nul
if errorlevel 1 (
  echo WSL is not installed or is not available in PATH.
  echo Install Ubuntu/WSL, then run this file again.
  pause
  exit /b 1
)

for /f "usebackq delims=" %%i in (`wsl wslpath -a "%~dp0"`) do set "WSL_PROJECT_DIR=%%i"

wsl bash -lc "cd '%WSL_PROJECT_DIR%' && bash scripts/start_app_from_archive_wsl.sh"

pause
