param(
    [Parameter(Mandatory = $true)]
    [string]$Benchmark,
    [string]$OutputDirectory = "data\evaluation\gold-v1",
    [string]$ApiUrl = "http://127.0.0.1:8000/api",
    [string[]]$Modes = @("trocr_lines", "qwen_lines", "qwen_page", "hybrid"),
    [string]$Model
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$Modes | ForEach-Object {
    $Mode = $_
    $Arguments = @(
        "scripts\run_ocr_benchmark.py",
        "--benchmark", $Benchmark,
        "--mode", $Mode,
        "--api-url", $ApiUrl,
        "--output", (Join-Path $OutputDirectory "$Mode.jsonl")
    )
    if ($Model) {
        $Arguments += @("--model", $Model)
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline failed: $Mode"
    }
}
