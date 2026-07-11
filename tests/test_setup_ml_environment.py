from pathlib import Path

import pytest

from scripts.setup_ml_environment import build_pip_command


def test_build_pip_command_uses_requirements_file() -> None:
    command = build_pip_command(
        python_executable="python",
        requirements=Path("requirements-ml.txt"),
    )

    assert command == ["python", "-m", "pip", "install", "-r", "requirements-ml.txt"]


def test_build_pip_command_can_add_upgrade_and_extra_index() -> None:
    command = build_pip_command(
        python_executable="python",
        requirements=Path("requirements-ml.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu130",
        upgrade=True,
    )

    assert "--upgrade" in command
    assert command[-2:] == ["--extra-index-url", "https://download.pytorch.org/whl/cu130"]


def test_build_pip_command_can_use_cuda_profile() -> None:
    command = build_pip_command(
        python_executable="python",
        requirements=Path("requirements-ml.txt"),
        profile="cuda-cu130",
    )

    assert command[-2:] == ["--extra-index-url", "https://download.pytorch.org/whl/cu130"]


def test_build_pip_command_can_use_cpu_profile() -> None:
    command = build_pip_command(
        python_executable="python",
        requirements=Path("requirements-ml.txt"),
        profile="cpu",
    )

    assert command[-2:] == ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]


def test_build_pip_command_rejects_profile_and_manual_index_together() -> None:
    with pytest.raises(ValueError, match="either --profile"):
        build_pip_command(
            python_executable="python",
            requirements=Path("requirements-ml.txt"),
            profile="cuda-cu130",
            extra_index_url="https://example.test/simple",
        )
