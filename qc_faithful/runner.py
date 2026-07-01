from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .results import parse_backtest_dir


def find_lean() -> Optional[str]:
    return shutil.which("lean")


def docker_running() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        proc = subprocess.run([docker, "info"], capture_output=True,
                              text=True, timeout=20)
        return proc.returncode == 0
    except Exception:
        return False


def preflight() -> tuple[bool, str]:
    if find_lean() is None:
        return False, (
            "Lean CLI not found. Install it with:\n"
            "    pip install lean\n"
        )
    if not docker_running():
        return False, (
            "Docker is not available\n"
        )
    return True, ""


_WORKSPACE_CONFIG = {
    "data-folder": "data",
    "job-user-id": "0",
    "api-access-token": "",
    "job-organization-id": "",
}

_PROJECT_CONFIG = {
    "algorithm-language": "Python",
    "parameters": {},
    "description": "Faithful local backtest",
    "cloud-id": 0,
}


def _ensure_workspace(workspace: Path, data_folder: Optional[str]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    lean_json = workspace / "lean.json"
    if not lean_json.exists():
        cfg = dict(_WORKSPACE_CONFIG)
        if data_folder:
            cfg["data-folder"] = os.path.abspath(data_folder)
        else:
            (workspace / "data").mkdir(exist_ok=True)
        with open(lean_json, "w") as f:
            json.dump(cfg, f, indent=4)


def _create_project(workspace: Path, name: str, code: str) -> Path:
    project_dir = workspace / name
    project_dir.mkdir(parents=True, exist_ok=True)
    with open(project_dir / "main.py", "w") as f:
        f.write(code)
    with open(project_dir / "config.json", "w") as f:
        json.dump(_PROJECT_CONFIG, f, indent=4)
    return project_dir


def _latest_backtest_dir(project_dir: Path) -> Optional[Path]:
    root = project_dir / "backtests"
    if not root.is_dir():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _run_capture(cmd, cwd: str, timeout: int, stream: bool) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    timed_out = {"hit": False}

    def _kill():
        timed_out["hit"] = True
        proc.kill()

    timer = threading.Timer(timeout, _kill) if timeout else None
    if timer:
        timer.start()

    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if stream:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
        if timer:
            timer.cancel()

    if timed_out["hit"]:
        raise subprocess.TimeoutExpired(cmd, timeout)
    return proc.returncode, "".join(lines)


# Run
def run_lean_backtest(
    code: str,
    workspace_dir: Optional[str] = None,
    data_folder: Optional[str] = None,
    timeout: int = 600,
    stream: bool = False,
    cleanup_project: bool = False,
) -> dict:
    """
    Run a faithful local LEAN backtest

    workspace_dir: an existing Lean CLI workspace (created with `lean init`).
                    This is where your data lives
    data_folder: path to a LEAN-format data folder to use instead.

    Returns a dict: success, error, trade_count, total_return, sharrpe_ratio,
    max_drawdown, win_rate, annual_return, execution_time, stdout, statistics.
    """
    result = {
        "success": False,
        "error": None,
        "trade_count": 0,
        "total_return": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "win_rate": None,
        "annual_return": None,
        "execution_time": None,
        "stdout": "",
        "statistics": {},
    }

    lean = find_lean()
    if lean is None:
        result["error"] = "Lean CLI not found, install it"
        return result

    use_temp = workspace_dir is None
    workspace = Path(workspace_dir) if workspace_dir else Path(
        tempfile.mkdtemp(prefix="qc_faithful_"))

    t0 = time.time()
    project_dir = None
    try:
        _ensure_workspace(workspace, data_folder)
        project_name = f"strategy_{int(t0 * 1000) % 1_000_000}_{uuid.uuid4().hex[:8]}"
        project_dir = _create_project(workspace, project_name, code)

        if stream:
            print("\n  Running the LEAN engine via Docker")
            print("  " + "-" * 64, flush=True)

        returncode, output = _run_capture(
            [lean, "backtest", project_name],
            cwd=str(workspace), timeout=timeout, stream=stream,
        )

        if stream:
            print("  " + "-" * 64)
        result["execution_time"] = time.time() - t0
        result["stdout"] = output[-4000:]

        if returncode != 0:
            match = re.search(r"(?:Error|Exception)[:\s]+(.+)", output)
            result["error"] = (match.group(1).strip() if match
                               else f"lean backtest exited with code {returncode}")
            return result

        run_dir = _latest_backtest_dir(project_dir)
        if run_dir is None:
            result["error"] = "lean ran but xid not produced any backtest output directory"
            return result

        parsed = parse_backtest_dir(run_dir)
        result.update(parsed)
        result["execution_time"] = time.time() - t0

    except subprocess.TimeoutExpired:
        result["error"] = f"lean backtest timed out after {timeout}s"
    except FileNotFoundError:
        result["error"] = "lean CLI not found, install it"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if use_temp:
            shutil.rmtree(workspace, ignore_errors=True)
        elif cleanup_project and project_dir is not None:
            shutil.rmtree(project_dir, ignore_errors=True)

    return result
