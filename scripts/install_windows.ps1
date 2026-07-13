param(
    [ValidateSet("Lite", "Full", "Developer")]
    [string]$Profile = "Lite",
    [string]$DataRoot = "",
    [switch]$NoShortcut,
    [switch]$Start
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv-app"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

if (-not $DataRoot) {
    $DataRoot = Join-Path $env:LOCALAPPDATA "HandwritingOCR"
}
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$ModelsRoot = Join-Path $DataRoot "models"
$RuntimeRoot = Join-Path $DataRoot "data\runtime"

function Resolve-PythonCommand {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if ($Python) {
        try {
            & $Python.Source -c "import sys; assert sys.version_info >= (3, 10)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = $Python.Source; Prefix = @() }
            }
        } catch {
        }
    }

    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        try {
            & $Launcher.Source -3.13 -c "import sys; assert sys.version_info >= (3, 10)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = $Launcher.Source; Prefix = @("-3.13") }
            }
        } catch {
        }
    }

    $Winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "Python не найден, а Windows Package Manager недоступен. Установите Python 3.13 x64 и повторите запуск."
    }
    Write-Host "Installing the managed Python runtime..."
    & $Winget.Source install --id Python.Python.3.13 --exact --scope user --silent `
        --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Windows Package Manager could not install Python 3.13."
    }
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $Launcher) {
        throw "Python was installed but the py launcher is not available in this process. Restart Windows and run setup again."
    }
    return @{ Command = $Launcher.Source; Prefix = @("-3.13") }
}

Set-Location $ProjectRoot
$PythonCommand = Resolve-PythonCommand
if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "Creating the application runtime..."
    & $PythonCommand.Command @($PythonCommand.Prefix) -m venv $VenvPath
}

Write-Host "Installing the $Profile profile..."
& $PythonPath -m pip install --upgrade pip
& $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-base.txt")
if ($Profile -eq "Lite") {
    & $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-trocr.txt")
} elseif ($Profile -eq "Full") {
    & $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-qwen.txt")
} else {
    & $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-ml.txt")
    & $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")
}

@(
    $DataRoot,
    $ModelsRoot,
    (Join-Path $DataRoot "data\uploads"),
    (Join-Path $DataRoot "data\processed"),
    (Join-Path $DataRoot "data\runtime\jobs"),
    (Join-Path $DataRoot "data\runtime\exports"),
    (Join-Path $DataRoot "data\runtime\logs"),
    (Join-Path $DataRoot "data\training\user_feedback")
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

$ForwardDataRoot = $DataRoot.Replace("\", "/")
$ForwardModelsRoot = $ModelsRoot.Replace("\", "/")
$DatabasePath = (Join-Path $DataRoot "data\ocr_history.db").Replace("\", "/")
$EnvironmentPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path -LiteralPath $EnvironmentPath)) {
    @"
APP_NAME=Handwriting OCR
APP_VERSION=0.4.0-dev
INSTALL_PROFILE=$Profile
UPLOAD_DIR=$ForwardDataRoot/data/uploads
PROCESSED_DIR=$ForwardDataRoot/data/processed
FEEDBACK_DIR=$ForwardDataRoot/data/training/user_feedback
JOBS_DIR=$ForwardDataRoot/data/runtime/jobs
EXPORTS_DIR=$ForwardDataRoot/data/runtime/exports
LOGS_DIR=$ForwardDataRoot/data/runtime/logs
USER_DICTIONARY_PATH=$ForwardDataRoot/data/runtime/user_dictionary.json
MODELS_DIR=$ForwardModelsRoot
MODEL_REGISTRY_PATH=$ForwardDataRoot/data/runtime/model_registry.json
MODEL_DOWNLOAD_STATE_PATH=$ForwardDataRoot/data/runtime/model_downloads.json
HF_HOME=$ForwardDataRoot/cache/huggingface
DATABASE_URL=sqlite:///$DatabasePath
RECOGNIZER_BACKEND=trocr
QWEN_MODEL_PRESET=light
QWEN_DEVICE_MAP=auto
QWEN_TORCH_DTYPE=auto
"@ | Set-Content -LiteralPath $EnvironmentPath -Encoding UTF8
}

if (-not $NoShortcut) {
    $Desktop = [Environment]::GetFolderPath("Desktop")
    $ShortcutPath = Join-Path $Desktop "Handwriting OCR.lnk"
    $Shell = New-Object -ComObject WScript.Shell
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = Join-Path $ProjectRoot "START_APP.bat"
    $Shortcut.WorkingDirectory = $ProjectRoot
    $Shortcut.Description = "Local handwriting recognition"
    $Shortcut.Save()
}

& $PythonPath (Join-Path $ProjectRoot "scripts\system_diagnostics.py") `
    --output (Join-Path $RuntimeRoot "installation_diagnostics.json")

Write-Host "Installation completed: $Profile"
Write-Host "User data: $DataRoot"
Write-Host "Models are installed separately from the System page or scripts\manage_models.py."

if ($Start) {
    & (Join-Path $ProjectRoot "START_APP.bat")
}
