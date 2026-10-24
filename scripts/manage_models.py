import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_download_service import ModelDownloadService
from app.services.model_registry_service import ModelRegistryService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage local OCR models and adapters.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("discover")

    verify = subparsers.add_parser("verify")
    verify.add_argument("model_id")

    promote = subparsers.add_parser("promote")
    promote.add_argument("model_id")

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("backend", choices=["trocr", "qwen"])

    remove = subparsers.add_parser("remove")
    remove.add_argument("model_id")
    remove.add_argument("--delete-files", action="store_true")

    install = subparsers.add_parser("install")
    install.add_argument("model_id")
    install.add_argument("repo_id")
    install.add_argument("--backend", choices=["trocr", "qwen"], required=True)
    install.add_argument("--revision")
    install.add_argument("--expected-sha256")
    install.add_argument(
        "--stage",
        choices=["baseline", "experimental", "candidate", "rejected"],
        default="experimental",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = ModelRegistryService()
    result = None
    if args.command == "list":
        result = registry.list_models()
    elif args.command == "discover":
        registry.discover_local_adapters()
        result = registry.list_models()
    elif args.command == "verify":
        result = registry.verify(args.model_id)
    elif args.command == "promote":
        result = registry.promote(args.model_id)
    elif args.command == "rollback":
        result = registry.rollback(args.backend)
    elif args.command == "remove":
        registry.unregister(args.model_id, delete_files=args.delete_files)
        result = {"status": "removed", "model_id": args.model_id}
    elif args.command == "install":
        downloads = ModelDownloadService()
        item = downloads.create(
            model_id=args.model_id,
            repo_id=args.repo_id,
            backend=args.backend,
            revision=args.revision,
            expected_sha256=args.expected_sha256,
            stage=args.stage,
        )
        downloads.run(item["id"])
        result = downloads.get(item["id"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
