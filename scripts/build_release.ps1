param(
    [string]$Version = "0.2.0",
    [string]$OutputDirectory = "dist"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputDirectory))
$StagingRoot = [System.IO.Path]::GetFullPath((Join-Path $OutputRoot "handwriting-ocr-$Version"))
$ArchivePath = [System.IO.Path]::GetFullPath((Join-Path $OutputRoot "handwriting-ocr-$Version-windows.zip"))

if (-not $OutputRoot.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output directory must be inside the project workspace."
}
if (-not $StagingRoot.StartsWith($OutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Invalid staging directory."
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if (Test-Path -LiteralPath $StagingRoot) {
    Remove-Item -LiteralPath $StagingRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
New-Item -ItemType Directory -Force -Path $StagingRoot | Out-Null

@("app", "ui", ".streamlit") | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $_) -Destination $StagingRoot -Recurse
}

@(
    ".env.example",
    "README.md",
    "pyproject.toml",
    "requirements-base.txt",
    "requirements-inference.txt",
    "INSTALL_WINDOWS.bat",
    "START_APP.bat",
    "Dockerfile",
    "docker-compose.yml"
) | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $_) -Destination $StagingRoot
}

New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "scripts") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\install_windows.ps1") -Destination (Join-Path $StagingRoot "scripts")
New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "docs") | Out-Null
@("installation.md", "product_roadmap.md") | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\$_") -Destination (Join-Path $StagingRoot "docs")
}
New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "models") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "models\README.md") -Destination (Join-Path $StagingRoot "models")

@("uploads", "processed", "runtime", "training\user_feedback") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "data\$_") | Out-Null
}

Get-ChildItem -LiteralPath $StagingRoot -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}
Get-ChildItem -LiteralPath $StagingRoot -Recurse -File | Where-Object {
    $_.Extension -in ".pyc", ".pyo"
} | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
}

$ReleaseFiles = @(Get-ChildItem -LiteralPath $StagingRoot -Recurse -File)
if ($ReleaseFiles.Count -lt 20) {
    throw "Release staging is incomplete: only $($ReleaseFiles.Count) files were collected."
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $StagingRoot,
    $ArchivePath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)
Remove-Item -LiteralPath $StagingRoot -Recurse -Force

Write-Host "Release archive created: $ArchivePath"
