import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "AntiplagiatCompany/HWR200"
DEFAULT_PATTERNS = [
    "README.md",
    ".gitattributes",
    "annotations.zip",
    "hwr200_0_19.zip",
    "hwr200_20_39.zip",
    "hwr200_40_59.zip",
    "hwr200_60_79.zip",
    "hwr200_80_99.zip",
    "hwr200_100_119.zip",
    "hwr200_120_139.zip",
    "hwr200_140_159.zip",
    "hwr200_160_179.zip",
    "hwr200_180_199.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the HWR200 dataset from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/hwr200_hf"))
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in DEFAULT_PATTERNS:
        print(f"downloading: {filename}", flush=True)
        downloaded_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            filename=filename,
            local_dir=args.output_dir,
        )
        print(f"downloaded: {downloaded_path}", flush=True)


if __name__ == "__main__":
    main()
