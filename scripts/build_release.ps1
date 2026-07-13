param(
    [string]$Version = "0.4.0-dev",
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

@("app", "ui", ".streamlit", "installer") | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $_) -Destination $StagingRoot -Recurse
}

@(
    ".env.example",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "requirements-base.txt",
    "requirements-trocr.txt",
    "requirements-qwen.txt",
    "requirements-inference.txt",
    "requirements-ml.txt",
    "requirements-dev.txt",
    "constraints-base.txt",
    "INSTALL_WINDOWS.bat",
    "START_APP.bat",
    "Dockerfile",
    "docker-compose.yml"
) | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $_) -Destination $StagingRoot
}

New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "scripts") | Out-Null
@("install_windows.ps1", "system_diagnostics.py", "manage_models.py", "audit_acceptance.py") | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "scripts\$_") -Destination (Join-Path $StagingRoot "scripts")
}
New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "docs") | Out-Null
@("installation.md", "product_roadmap.md", "product_technical_specification.md", "architecture.md", "page_recognition.md") | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\$_") -Destination (Join-Path $StagingRoot "docs")
}
New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "models") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "models\README.md") -Destination (Join-Path $StagingRoot "models")
New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "manifests") | Out-Null
@("datasets", "benchmarks") | ForEach-Object {
    $ManifestSource = Join-Path $ProjectRoot "manifests\$_"
    if (Test-Path -LiteralPath $ManifestSource) {
        Copy-Item -LiteralPath $ManifestSource -Destination (Join-Path $StagingRoot "manifests") -Recurse
    }
}

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

$ChecksumLines = Get-ChildItem -LiteralPath $StagingRoot -Recurse -File | ForEach-Object {
    $RelativePath = $_.FullName.Substring($StagingRoot.Length + 1).Replace("\", "/")
    $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $RelativePath"
}
$ChecksumLines | Set-Content -LiteralPath (Join-Path $StagingRoot "SHA256SUMS.txt") -Encoding UTF8

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $StagingRoot,
    $ArchivePath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)
Remove-Item -LiteralPath $StagingRoot -Recurse -Force

$ArchiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
"$ArchiveHash  $([System.IO.Path]::GetFileName($ArchivePath))" | Set-Content `
    -LiteralPath "$ArchivePath.sha256" -Encoding ASCII

Write-Host "Release archive created: $ArchivePath"
