from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.dataset_manifest import build_dataset_manifest, write_dataset_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible JSON manifest for a page dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--license", dest="license_name", required=True)
    return parser.parse_args()


def current_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    manifest = build_dataset_manifest(
        args.dataset_dir,
        name=args.name,
        version=args.version,
        source=args.source,
        source_revision=args.source_revision,
        license_name=args.license_name,
        conversion_commit=current_commit(),
    )
    write_dataset_manifest(args.output, manifest)
    print(f"dataset manifest: {args.output}")


if __name__ == "__main__":
    main()
