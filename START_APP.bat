@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

if exist "%~dp0.venv-app\Scripts\python.exe" (
  set "APP_PYTHON=%~dp0.venv-app\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
  echo The application is not installed yet.
  echo Run INSTALL_WINDOWS.bat first.
  pause
  exit /b 1
)

"%APP_PYTHON%" -m app.launcher
if errorlevel 1 pause
