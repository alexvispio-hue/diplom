from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_SCRIPT = PROJECT_ROOT / "ui" / "streamlit_app.py"
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the local Handwriting OCR application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.25)
        return connection.connect_ex((host, port)) == 0


def url_is_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_url(url: str, process: subprocess.Popen | None, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if url_is_ready(url):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Процесс завершился до запуска: {url}")
        time.sleep(0.25)
    raise RuntimeError(f"Сервис не запустился за {timeout:.0f} секунд: {url}")


def start_process(command: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("a", encoding="utf-8")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    log_stream.close()
    return process


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def run(host: str, backend_port: int, ui_port: int, open_browser: bool) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    backend_health = f"http://{host}:{backend_port}/api/health"
    ui_url = f"http://{host}:{ui_port}"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["OCR_API_URL"] = f"http://{host}:{backend_port}/api"

    backend_process = None
    ui_process = None
    try:
        if port_is_open(host, backend_port):
            if not url_is_ready(backend_health):
                raise RuntimeError(f"Порт {backend_port} занят другим приложением.")
        else:
            backend_process = start_process(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    host,
                    "--port",
                    str(backend_port),
                ],
                RUNTIME_DIR / "backend.log",
                env,
            )
            wait_for_url(backend_health, backend_process)

        if port_is_open(host, ui_port):
            if not url_is_ready(ui_url):
                raise RuntimeError(f"Порт {ui_port} занят другим приложением.")
        else:
            ui_process = start_process(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(UI_SCRIPT),
                    "--server.address",
                    host,
                    "--server.port",
                    str(ui_port),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ],
                RUNTIME_DIR / "ui.log",
                env,
            )
            wait_for_url(ui_url, ui_process)

        print(f"Handwriting OCR: {ui_url}")
        print(f"API: http://{host}:{backend_port}/docs")
        print("Закройте это окно или нажмите Ctrl+C для остановки приложения.")
        if open_browser:
            webbrowser.open(ui_url)

        while backend_process is not None or ui_process is not None:
            for process, name in ((backend_process, "backend"), (ui_process, "UI")):
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"{name} неожиданно завершился. Проверьте data/runtime/*.log.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_process(ui_process)
        stop_process(backend_process)


def main() -> None:
    args = parse_args()
    try:
        run(args.host, args.backend_port, args.ui_port, not args.no_browser)
    except RuntimeError as exc:
        print(f"Ошибка запуска: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
