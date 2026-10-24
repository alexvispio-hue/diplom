from scripts.check_ml_environment import collect_status, package_status


def test_package_status_reports_missing_package() -> None:
    status = package_status("definitely-not-installed-package", "definitely_not_installed_package")

    assert status["package"] == "definitely-not-installed-package"
    assert status["installed"] is False
    assert status["version"] is None


def test_collect_status_has_expected_readiness_flags() -> None:
    status = collect_status()

    assert {package["package"] for package in status["packages"]} >= {"torch", "torchvision", "transformers", "peft"}
    assert isinstance(status["ready_for_trocr"], bool)
    assert isinstance(status["ready_for_qwen_inference"], bool)
    assert isinstance(status["ready_for_qwen_lora"], bool)
    assert "torch_cuda_available" in status
