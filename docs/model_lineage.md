# Qwen Model Lineage

## Confirmed chain

```text
Qwen/Qwen3-VL-2B-Instruct
  -> models/qwen-vl-school-pages-lora
     schooldataset model; intended parent for the full HWR200 run
  -> models/qwen-vl-school-hwr200-pages-lora
     short 108-step HWR200 warm-start experiment
```

`models/qwen-vl-ocr-lora` is a separate smoke adapter trained on only three
examples. It is useful for checking the training pipeline, but it is not the
school-dataset model and must not be used as the production starting point.

## Main run

The default launcher continues `models/qwen-vl-school-pages-lora` and writes
the result to `models/qwen-vl-school-hwr200-full-pages-lora`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\launch_qwen_training.ps1
```

The full HWR200 dataset contains 33,030 image variants. The short HWR200
adapter remains available as an explicitly selected comparison branch, with a
different output directory and run name.
