# Page-level Qwen dataset for school notebooks

This dataset is prepared from the local `school_notebooks_RU` copy:

```text
data/training/schooldataset
```

The page model is trained on full notebook pages. The previous word and fragment recognizer can stay as a strong auxiliary model; this pipeline focuses on teaching Qwen to read the whole page and, in a separate task, return line-level layout.

## Conversion

Quick smoke conversion:

```bash
python scripts/convert_school_notebooks_qwen.py \
  --source-dir data/training/schooldataset \
  --output-dir data/training/school_qwen_pages_smoke \
  --splits train val test \
  --max-pages-per-split 5
```

Full conversion:

```bash
python scripts/convert_school_notebooks_qwen.py \
  --source-dir data/training/schooldataset \
  --output-dir data/training/school_qwen_pages \
  --splits train val test
```

The full converted dataset currently contains:

| Split | Pages | Lines | Qwen SFT examples |
| --- | ---: | ---: | ---: |
| train | 1556 | 63395 | 3112 |
| val | 150 | 6450 | 300 |
| test | 150 | 6254 | 300 |

No images were missing during conversion.

## Output layout

```text
data/training/school_qwen_pages/
  conversion_stats.json
  train_pages.csv
  val_pages.csv
  test_pages.csv
  train_lines.csv
  val_lines.csv
  test_lines.csv
  pages_text/
    train/
    val/
    test/
  qwen_sft/
    train.jsonl
    val.jsonl
    test.jsonl
```

Each page produces two Qwen SFT examples:

- full-page OCR: image to plain text with preserved line breaks;
- layout OCR: image to a JSON array of text lines with bounding boxes.

If line-level examples are also needed later, add:

```bash
--include-line-examples
```

For the current development step this is intentionally disabled, because the separate word/fragment model already works well and the priority is page recognition.

## Training start

Small validation pass without loading the model:

```bash
python scripts/train_qwen_vl_lora.py \
  --dataset data/training/school_qwen_pages/qwen_sft/train.jsonl \
  --max-samples 8 \
  --dry-run
```

Short LoRA training run:

```bash
python scripts/train_qwen_vl_lora.py \
  --dataset data/training/school_qwen_pages/qwen_sft/train.jsonl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --output-dir models/qwen-vl-school-pages-lora \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-image-side 768 \
  --gradient-checkpointing \
  --fp16
```

For a GPU with less memory, lower `--max-image-side` to `512`. For a stronger GPU, increase it to `1024` before increasing the batch size.

## Evaluation

After training, evaluate on validation pages first:

```bash
python scripts/run_page_experiment.py \
  --pages-csv data/training/school_qwen_pages/val_pages.csv \
  --output-dir data/evaluation/school_qwen_pages_val \
  --modes qwen_page qwen_layout hybrid \
  --limit 30
```

Then run the same command on the test split without `--limit`:

```bash
python scripts/run_page_experiment.py \
  --pages-csv data/training/school_qwen_pages/test_pages.csv \
  --output-dir data/evaluation/school_qwen_pages_test \
  --modes qwen_page qwen_layout hybrid
```

The main metrics remain CER and WER. For layout mode, additionally inspect whether line order and bounding boxes are stable enough for the hybrid pipeline.
