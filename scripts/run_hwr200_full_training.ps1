param(
    [switch]$PreflightOnly,
    [switch]$Resume,
    [switch]$FromBaseModel,
    [string]$InitialAdapter = "models\qwen-vl-school-pages-lora",
    [string]$OutputDirectory = "models\qwen-vl-school-hwr200-full-pages-lora",
    [string]$RunName = "qwen3-vl-2b-hwr200-full-v1"
)

$ErrorActionPreference = "Stop"
$Launcher = Join-Path $PSScriptRoot "launch_qwen_training.ps1"

$Arguments = @()
if ($PreflightOnly) { $Arguments += "-PreflightOnly" }
if ($Resume) { $Arguments += "-Resume" }
if ($FromBaseModel) { $Arguments += "-FromBaseModel" }
$Arguments += @("-InitialAdapter", $InitialAdapter)
$Arguments += @("-OutputDirectory", $OutputDirectory)
$Arguments += @("-RunName", $RunName)

& powershell -NoProfile -ExecutionPolicy Bypass -File $Launcher @Arguments
exit $LASTEXITCODE
