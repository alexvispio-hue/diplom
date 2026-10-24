@echo off
setlocal
cd /d "%~dp0"
set "INSTALL_PROFILE=%~1"
if "%INSTALL_PROFILE%"=="" set "INSTALL_PROFILE=Lite"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_windows.ps1" -Profile "%INSTALL_PROFILE%"
if errorlevel 1 (
  echo.
  echo Installation failed. Review the message above.
  pause
  exit /b 1
)
pause
