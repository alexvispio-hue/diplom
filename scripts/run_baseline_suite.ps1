param(
    [Parameter(Mandatory = $true)]
    [string]$Benchmark,
    [string]$OutputDirectory = "data\evaluation\gold-v1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

@("trocr_lines", "qwen_page", "hybrid") | ForEach-Object {
    $Mode = $_
    & $Python "scripts\run_ocr_benchmark.py" `
        --benchmark $Benchmark `
        --mode $Mode `
        --output (Join-Path $OutputDirectory "$Mode.jsonl")
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline failed: $Mode"
    }
}
