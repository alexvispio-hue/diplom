param(
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "data\runtime\training_logs"
$LogPath = Join-Path $LogDir "hwr200_full_pipeline.log"
$StatusPath = Join-Path $LogDir "hwr200_full_pipeline_status.json"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-PipelineStatus {
    param([string]$Phase, [string]$Status, [string]$Message)
    @{
        phase = $Phase
        status = $Status
        message = $Message
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

Start-Transcript -LiteralPath $LogPath -Append | Out-Null
try {
    Write-PipelineStatus -Phase "prepare" -Status "running" -Message "Building OCR drafts and page labels for all HWR200 documents."

    & $Python "scripts\prepare_hwr200_full_qwen.py" `
        --stage all `
        --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw "HWR200 preparation failed with exit code $LASTEXITCODE." }

    Write-PipelineStatus -Phase "validate" -Status "running" -Message "Validating all labels, images, variants, and author splits."
    & $Python "scripts\validate_hwr200_full_dataset.py"
    if ($LASTEXITCODE -ne 0) { throw "HWR200 validation failed; Qwen training was not started." }

    Write-PipelineStatus -Phase "train" -Status "running" -Message "Fine-tuning Qwen on the complete validated HWR200 page dataset."
    & $Python "scripts\train_qwen_vl_lora.py" `
        --train-dataset "data\training\hwr200_qwen_pages_full\train.jsonl" `
        --eval-dataset "data\training\hwr200_qwen_pages_full\val.jsonl" `
        --dataset-manifest "manifests\datasets\hwr200-pages-v1.json" `
        --review-metadata "data\training\hwr200_qwen_pages_full\page_metadata.jsonl" `
        --review-policy all `
        --variant-policy all `
        --initial-adapter "models\qwen-vl-school-hwr200-pages-lora" `
        --output-dir "models\qwen-vl-school-hwr200-full-pages-lora" `
        --run-name "qwen3-vl-2b-hwr200-full-v1" `
        --epochs 1 `
        --batch-size 1 `
        --gradient-accumulation-steps 8 `
        --learning-rate 1e-5 `
        --max-image-side 768 `
        --gradient-checkpointing `
        --bf16 `
        --logging-steps 25 `
        --save-steps 250 `
        --eval-steps 250 `
        --early-stopping-patience 3 `
        --local-files-only
    if ($LASTEXITCODE -ne 0) { throw "Full HWR200 Qwen training failed with exit code $LASTEXITCODE." }

    Write-PipelineStatus -Phase "complete" -Status "passed" -Message "Full HWR200 preparation, validation, and Qwen training completed."
}
catch {
    Write-PipelineStatus -Phase "failed" -Status "failed" -Message $_.Exception.Message
    throw
}
finally {
    Stop-Transcript | Out-Null
}
