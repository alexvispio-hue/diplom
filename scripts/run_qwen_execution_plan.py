import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import UTC, datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_JSON = PROJECT_ROOT / "data" / "runtime" / "qwen_execution_plan.json"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "data" / "runtime" / "qwen_execution_results.json"


def load_plan(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Qwen execution plan not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_steps(plan: dict, step_id: str | None, run_all: bool) -> list[dict]:
    steps = plan.get("steps", [])
    if run_all:
        return steps
    if step_id is None:
        return []
    matches = [step for step in steps if step.get("id") == step_id]
    if not matches:
        available = ", ".join(step.get("id", "?") for step in steps)
        raise ValueError(f"Unknown step: {step_id}. Available steps: {available}.")
    return matches


def evaluate_step(step: dict, allow_network: bool) -> tuple[str, str]:
    blocked_by = step.get("blocked_by") or []
    if blocked_by:
        return "blocked", f"Step has unresolved inputs: {', '.join(blocked_by)}"
    if step.get("requires_network") and not allow_network:
        return "blocked", "Step requires network. Re-run with --allow-network to execute it."
    return "ready", ""


def command_for_step(step: dict) -> str | list[str]:
    command_args = step.get("command_args")
    if command_args:
        return [str(part) for part in command_args]
    return step["command"]


def run_step(
    step: dict,
    *,
    allow_network: bool,
    dry_run: bool,
    runner=subprocess.run,
) -> dict:
    gate_status, reason = evaluate_step(step, allow_network)
    result = {
        "id": step.get("id"),
        "title": step.get("title"),
        "command": step.get("command"),
        "command_args": step.get("command_args"),
        "requires_network": bool(step.get("requires_network")),
        "status": gate_status,
        "reason": reason,
        "returncode": None,
        "duration_seconds": 0.0,
        "stdout": "",
        "stderr": "",
    }
    if gate_status != "ready":
        return result
    if dry_run:
        result["status"] = "dry_run"
        result["reason"] = "Command was not executed because --dry-run is enabled."
        return result

    started = time.monotonic()
    command = command_for_step(step)
    completed = runner(
        command,
        cwd=PROJECT_ROOT,
        shell=isinstance(command, str),
        capture_output=True,
        text=True,
    )
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    result["returncode"] = completed.returncode
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    result["status"] = "passed" if completed.returncode == 0 else "failed"
    result["reason"] = ""
    return result


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_runs = []
    if path.exists():
        try:
            existing_payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_payload = {}
        existing_runs = existing_payload.get("runs") or []
        if not existing_runs and existing_payload.get("results"):
            existing_runs = [{"created_at": None, "results": existing_payload["results"]}]

    run = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": results,
    }
    payload = {
        "results": results,
        "runs": [*existing_runs, run],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run steps from the generated Qwen execution plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--step", help="Run a single step by id.")
    parser.add_argument("--all", action="store_true", help="Run all steps from the plan.")
    parser.add_argument("--list", action="store_true", help="Print available steps and exit.")
    parser.add_argument("--allow-network", action="store_true", help="Allow steps marked as network-dependent.")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute commands; only show what would run.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_plan(args.plan)
    if args.list:
        for step in plan.get("steps", []):
            print(f"{step['id']}: network={step.get('requires_network', False)} command={step.get('command')}")
        return 0
    if not args.all and not args.step:
        raise SystemExit("Pass --list, --step <id>, or --all.")

    steps = select_steps(plan, args.step, args.all)
    results = [
        run_step(step, allow_network=args.allow_network, dry_run=args.dry_run)
        for step in steps
    ]
    write_results(results, args.results)
    if args.json:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"{result['id']}: {result['status']}")
            if result["reason"]:
                print(f"  {result['reason']}")
        print(f"results: {args.results}")
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
