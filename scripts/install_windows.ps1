param(
    [switch]$NoShortcut,
    [switch]$Start
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv-app"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10 or newer is required. Install Python from python.org and run the installer again."
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "Creating local Python environment..."
    python -m venv $VenvPath
}

Write-Host "Installing application dependencies..."
& $PythonPath -m pip install --upgrade pip
& $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-base.txt")
& $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements-inference.txt")

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot ".env.example") -Destination (Join-Path $ProjectRoot ".env")
}

@("uploads", "processed", "runtime", "training\user_feedback") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "data\$_") | Out-Null
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

Write-Host "Installation completed. Run START_APP.bat to open the application."
Write-Host "The first recognition may download model weights to the local Hugging Face cache."

if ($Start) {
    & (Join-Path $ProjectRoot "START_APP.bat")
}
