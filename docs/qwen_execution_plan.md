# Qwen execution plan

- model: `Qwen/Qwen3-VL-2B-Instruct`
- install profile: `cuda-cu130`
- runtime status: `passed`
- missing packages: `none`
- missing inputs: `none`

## Steps

### install_ml

- title: Install ML dependencies
- requires network: `True`
- blocked by: `none`

```bash
C:\Users\Aleksandr\Desktop\diplom\.venv\Scripts\python.exe scripts/setup_ml_environment.py --profile cuda-cu130 --install
```

### qwen_smoke

- title: Run Qwen smoke inference
- requires network: `True`
- blocked by: `none`

```bash
C:\Users\Aleksandr\Desktop\diplom\.venv\Scripts\python.exe scripts/qwen_runtime_check.py --run-smoke --page --model Qwen/Qwen3-VL-2B-Instruct --json
```

### page_experiment

- title: Compare TrOCR and Qwen page modes
- requires network: `False`
- blocked by: `none`

```bash
C:\Users\Aleksandr\Desktop\diplom\.venv\Scripts\python.exe scripts/run_page_experiment.py --pages-csv C:\Users\Aleksandr\Desktop\diplom\data\training\page_lines_experiment_smoke\test_pages.csv --modes trocr_lines qwen_lines qwen_page hybrid --output-dir data/evaluation/page_experiment_qwen_light
```

### build_sft

- title: Build Qwen SFT JSONL
- requires network: `False`
- blocked by: `none`

```bash
C:\Users\Aleksandr\Desktop\diplom\.venv\Scripts\python.exe scripts/build_qwen_sft_dataset.py --lines-csv C:\Users\Aleksandr\Desktop\diplom\data\evaluation\layout_export_smoke\lines.csv --pages-csv C:\Users\Aleksandr\Desktop\diplom\data\training\page_lines_experiment_smoke\test_pages.csv --output data/training/qwen_sft_light/mixed.jsonl --include-layout
```

### lora_dry_run

- title: Validate Qwen SFT data before training
- requires network: `False`
- blocked by: `none`

```bash
C:\Users\Aleksandr\Desktop\diplom\.venv\Scripts\python.exe scripts/train_qwen_vl_lora.py --dataset C:\Users\Aleksandr\Desktop\diplom\data\training\qwen_sft_light\mixed.jsonl --model Qwen/Qwen3-VL-2B-Instruct --dry-run --max-samples 5
```

### lora_train

- title: Train Qwen LoRA adapter
- requires network: `True`
- blocked by: `none`

```bash
C:\Users\Aleksandr\Desktop\diplom\.venv\Scripts\python.exe scripts/train_qwen_vl_lora.py --dataset C:\Users\Aleksandr\Desktop\diplom\data\training\qwen_sft_light\mixed.jsonl --model Qwen/Qwen3-VL-2B-Instruct --output-dir models/qwen-vl-ocr-lora --epochs 1 --batch-size 1 --gradient-accumulation-steps 1 --max-image-side 512 --gradient-checkpointing --fp16
```
