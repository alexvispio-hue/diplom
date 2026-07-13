from __future__ import annotations

import platform
import os
import shutil
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import BASE_DIR, get_settings
from app.services.ml_environment_service import collect_status
from app.services.model_registry_service import ModelRegistryService
from app.storage.database import schema_version


class DiagnosticsService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def collect(self) -> dict[str, Any]:
        ml_status = collect_status()
        storage = self._storage_status()
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "application": {
                "name": self.settings.app_name,
                "version": self.settings.app_version,
                "project_root": str(BASE_DIR),
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "platform": platform.platform(),
                "database_schema": schema_version(),
                "process_id": os.getpid(),
                "process_memory_bytes": _process_memory_bytes(),
            },
            "runtime": ml_status,
            "storage": storage,
            "ports": {
                "api_8000": self._port_state(8000),
                "ui_8501": self._port_state(8501),
            },
            "models": ModelRegistryService().list_models(),
            "recent_errors": self._recent_errors(),
        }

    def _storage_status(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.settings.upload_dir.anchor or BASE_DIR)
        return {
            "database_url": self.settings.database_url,
            "uploads": str(self.settings.upload_dir),
            "processed": str(self.settings.processed_dir),
            "jobs": str(self.settings.jobs_dir),
            "exports": str(self.settings.exports_dir),
            "feedback": str(self.settings.feedback_dir),
            "models": str(self.settings.models_dir.resolve()),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
        }

    @staticmethod
    def _port_state(port: int) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.15)
            return "listening" if connection.connect_ex(("127.0.0.1", port)) == 0 else "available"

    def _recent_errors(self) -> list[str]:
        errors: list[str] = []
        for log_path in sorted(self.settings.logs_dir.glob("*.log"), key=_modified, reverse=True)[:5]:
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                lowered = line.casefold()
                if "error" in lowered or "exception" in lowered or "traceback" in lowered:
                    errors.append(f"{log_path.name}: {line[:500]}")
                    if len(errors) >= 20:
                        return errors
        return errors


def _modified(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _process_memory_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return None
    return None
