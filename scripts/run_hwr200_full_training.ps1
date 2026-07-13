param(
    [switch]$PreflightOnly,
    [switch]$Resume,
    [switch]$FromBaseModel
)

$ErrorActionPreference = "Stop"
$Launcher = Join-Path $PSScriptRoot "launch_qwen_training.ps1"

$Arguments = @()
if ($PreflightOnly) { $Arguments += "-PreflightOnly" }
if ($Resume) { $Arguments += "-Resume" }
if ($FromBaseModel) { $Arguments += "-FromBaseModel" }

& powershell -NoProfile -ExecutionPolicy Bypass -File $Launcher @Arguments
exit $LASTEXITCODE
