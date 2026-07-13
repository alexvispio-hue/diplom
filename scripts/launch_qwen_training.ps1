param(
    [string]$Model = "Qwen/Qwen3-VL-2B-Instruct",
    [string]$InitialAdapter = "models\qwen-vl-school-hwr200-pages-lora",
    [string]$OutputDirectory = "models\qwen-vl-school-hwr200-full-pages-lora",
    [string]$RunName = "qwen3-vl-2b-hwr200-full-v1",
    [int]$Epochs = 1,
    [int]$GradientAccumulationSteps = 8,
    [int]$MaxImageSide = 768,
    [switch]$FromBaseModel,
    [switch]$Resume,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $ProjectRoot ".venv-app\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment was not found. Install the Developer profile first."
}

Set-Location $ProjectRoot
$TrainDataset = Join-Path $ProjectRoot "data\training\hwr200_qwen_pages_full\train.jsonl"
$EvalDataset = Join-Path $ProjectRoot "data\training\hwr200_qwen_pages_full\val.jsonl"
$ReviewMetadata = Join-Path $ProjectRoot "data\training\hwr200_qwen_pages_full\page_metadata.jsonl"
$DatasetManifest = Join-Path $ProjectRoot "manifests\datasets\hwr200-pages-v1.json"
$OutputPath = Join-Path $ProjectRoot $OutputDirectory
$PreflightOutput = Join-Path $ProjectRoot "data\runtime\training_logs\qwen_training_preflight.json"
$RunManifest = Join-Path $OutputPath "run_manifest.json"
$PipelineStatus = Join-Path $ProjectRoot "data\runtime\training_logs\hwr200_full_pipeline_status.json"
New-Item -ItemType Directory -Force -Path (Split-Path $PipelineStatus) | Out-Null

if ($Resume) {
    if (-not (Test-Path -LiteralPath $RunManifest)) {
        throw "Cannot resume: run manifest was not found at $RunManifest"
    }
} elseif (Test-Path -LiteralPath $OutputPath) {
    $existingFiles = @(Get-ChildItem -LiteralPath $OutputPath -Force)
    if ($existingFiles.Count -gt 0) {
        throw "Output directory is not empty. Use -Resume for an interrupted run or choose -OutputDirectory."
    }
}

& $Python "scripts\validate_hwr200_full_dataset.py"
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed; training was not started."
}

$PreflightArguments = @(
    "scripts\preflight_qwen_training.py",
    "--train-dataset", "data/training/hwr200_qwen_pages_full/train.jsonl",
    "--eval-dataset", "data/training/hwr200_qwen_pages_full/val.jsonl",
    "--review-metadata", "data/training/hwr200_qwen_pages_full/page_metadata.jsonl",
    "--dataset-manifest", "manifests/datasets/hwr200-pages-v1.json",
    "--model", $Model,
    "--output", "data/runtime/training_logs/qwen_training_preflight.json"
)
if (-not $FromBaseModel) {
    $PreflightArguments += @("--initial-adapter", $InitialAdapter)
} else {
    $PreflightArguments += "--no-initial-adapter"
}
& $Python @PreflightArguments
if ($LASTEXITCODE -ne 0) {
    throw "Training preflight failed; training was not started."
}
if ($PreflightOnly) {
    Write-Host "Preflight completed. Training was not started."
    exit 0
}

@{
    phase = "train"
    status = "running"
    message = "Qwen training launcher passed dataset and runtime preflight."
    updated_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $PipelineStatus -Encoding UTF8

$TrainArguments = @(
    "scripts\train_qwen_vl_lora.py",
    "--train-dataset", "data\training\hwr200_qwen_pages_full\train.jsonl",
    "--eval-dataset", "data\training\hwr200_qwen_pages_full\val.jsonl",
    "--dataset-manifest", "manifests\datasets\hwr200-pages-v1.json",
    "--review-metadata", "data\training\hwr200_qwen_pages_full\page_metadata.jsonl",
    "--review-policy", "all",
    "--variant-policy", "all",
    "--model", $Model,
    "--output-dir", $OutputDirectory,
    "--run-name", $RunName,
    "--epochs", $Epochs,
    "--batch-size", "1",
    "--gradient-accumulation-steps", $GradientAccumulationSteps,
    "--learning-rate", "1e-5",
    "--max-image-side", $MaxImageSide,
    "--gradient-checkpointing",
    "--bf16",
    "--logging-steps", "25",
    "--save-steps", "250",
    "--eval-steps", "250",
    "--early-stopping-patience", "3",
    "--local-files-only"
)
if (-not $FromBaseModel) {
    $TrainArguments += @("--initial-adapter", $InitialAdapter)
}
if ($Resume) {
    $TrainArguments += @("--resume-from-checkpoint", "latest")
}

& $Python @TrainArguments
if ($LASTEXITCODE -ne 0) {
    @{
        phase = "train"
        status = "failed"
        message = "Qwen training process exited with code $LASTEXITCODE."
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $PipelineStatus -Encoding UTF8
    throw "Qwen training failed with exit code $LASTEXITCODE."
}

@{
    phase = "train"
    status = "passed"
    message = "Qwen training completed; evaluation remains a separate quality-gate step."
    run_manifest = $RunManifest
    updated_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $PipelineStatus -Encoding UTF8
Write-Host "Qwen training completed. Run manifest: $RunManifest"
