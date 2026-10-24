import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements-ml.txt"
INSTALL_PROFILES = {
    "default": None,
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda-cu130": "https://download.pytorch.org/whl/cu130",
}


def build_pip_command(
    python_executable: str,
    requirements: Path,
    extra_index_url: str | None = None,
    profile: str = "default",
    upgrade: bool = False,
) -> list[str]:
    if profile not in INSTALL_PROFILES:
        choices = ", ".join(sorted(INSTALL_PROFILES))
        raise ValueError(f"Unknown ML install profile: {profile}. Available profiles: {choices}.")
    profile_index_url = INSTALL_PROFILES[profile]
    if extra_index_url and profile_index_url:
        raise ValueError("Use either --profile with a built-in index or --extra-index-url, not both.")

    command = [python_executable, "-m", "pip", "install", "-r", str(requirements)]
    if upgrade:
        command.append("--upgrade")
    index_url = extra_index_url or profile_index_url
    if index_url:
        command.extend(["--extra-index-url", index_url])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ML dependencies for TrOCR/Qwen experiments.")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument(
        "--profile",
        choices=sorted(INSTALL_PROFILES),
        default="default",
        help=(
            "Built-in pip index profile. Use cuda-cu130 for the recommended local NVIDIA GPU path, "
            "cpu for CPU-only wheels, or default for normal PyPI resolution."
        ),
    )
    parser.add_argument(
        "--extra-index-url",
        help=(
            "Optional extra pip index for torch wheels, for example a PyTorch CUDA index. "
            "Leave empty for the default PyPI/CPU-compatible install."
        ),
    )
    parser.add_argument("--upgrade", action="store_true", help="Pass --upgrade to pip install.")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Actually run pip install. Without this flag the script only prints the command.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_pip_command(
        python_executable=sys.executable,
        requirements=args.requirements,
        extra_index_url=args.extra_index_url,
        profile=args.profile,
        upgrade=args.upgrade,
    )
    print(" ".join(command))
    if not args.install:
        print("dry-run: add --install to run this command.")
        return
    subprocess.check_call(command)


if __name__ == "__main__":
    main()
