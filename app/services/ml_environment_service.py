import importlib.metadata
import importlib.util


REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "accelerate": "accelerate",
    "qwen-vl-utils": "qwen_vl_utils",
    "peft": "peft",
    "trl": "trl",
}
TROCR_RUNTIME_PACKAGES = {"torch", "transformers", "sentencepiece"}
QWEN_RUNTIME_PACKAGES = {"torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils"}


def package_status(distribution_name: str, import_name: str) -> dict:
    installed = importlib.util.find_spec(import_name) is not None
    version = None
    if installed:
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return {
        "package": distribution_name,
        "import_name": import_name,
        "installed": installed,
        "version": version,
    }


def collect_status() -> dict:
    packages = [
        package_status(distribution_name, import_name)
        for distribution_name, import_name in REQUIRED_PACKAGES.items()
    ]
    torch_cuda_available = None
    torch_device_count = None
    if next(item for item in packages if item["package"] == "torch")["installed"]:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
        torch_device_count = int(torch.cuda.device_count())

    return {
        "packages": packages,
        "ready_for_trocr": all(
            item["installed"] for item in packages if item["package"] in TROCR_RUNTIME_PACKAGES
        ),
        "ready_for_qwen_inference": all(
            item["installed"] for item in packages if item["package"] in QWEN_RUNTIME_PACKAGES
        ),
        "ready_for_qwen_lora": all(item["installed"] for item in packages),
        "torch_cuda_available": torch_cuda_available,
        "torch_device_count": torch_device_count,
    }


def missing_packages(status: dict, required_packages: set[str]) -> list[str]:
    return [
        item["package"]
        for item in status["packages"]
        if item["package"] in required_packages and not item["installed"]
    ]
