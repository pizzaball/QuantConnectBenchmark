"""
Stage 1 compiled: ast.parse() succeeds
Stage 2 validated: QCAlgorithm subclass + Initialize() present
Stage 3 backtested: the real LEAN engine ran it 
Stage 4 traded: LEAN reports at least one order
"""

from __future__ import annotations

from .compile_check import clean_code, compile_check, validate_qc_structure
from .runner import preflight, run_lean_backtest, find_lean, docker_running

__all__ = [
    "check",
    "clean_code",
    "compile_check",
    "validate_qc_structure",
    "preflight",
    "run_lean_backtest",
    "find_lean",
    "docker_running",
    "evaluate_strategy",
    "build_feedback",
    "StrategyGenerator",
    "GenerationResult",
    "compute_metrics",
    "StrategyJudge",
    "create_strategy_judge",
]

def check(
    code: str,
    workspace_dir: str | None = None,
    data_folder: str | None = None,
    timeout: int = 600,
    require_lean: bool = True,
    stream: bool = False,
    cleanup_project: bool = False,
) -> dict:
    """
    Returns a dict with:
        compiled: bool
        validated: bool
        environment_ok: bool | None (None until reached)
        backtested: bool
        traded: bool
        trade_count: int
        total_return: float | None
        sharpe_ratio: float | None
        max_drawdown: float | None
        win_rate: float | None
        annual_return: float | None
        execution_time: float | None
        error: str | None
        statistics: dict (raw LEAN statistics)
    """
    result: dict = {
        "compiled": False,
        "validated": False,
        "environment_ok": None,
        "backtested": False,
        "traded": False,
        "trade_count": 0,
        "total_return": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "win_rate": None,
        "annual_return": None,
        "execution_time": None,
        "error": None,
        "statistics": {},
    }

    code = clean_code(code)

    # Stage 1: compilation
    ok, msg = compile_check(code)
    if not ok:
        result["error"] = msg
        return result
    result["compiled"] = True

    # Stage 2: structure
    ok, msg = validate_qc_structure(code)
    if not ok:
        result["error"] = msg
        return result
    result["validated"] = True

    # Environment predefinition
    ready, msg = preflight()
    result["environment_ok"] = ready
    if not ready:
        if require_lean:
            result["error"] = msg
            return result
        result["error"] = msg
        return result

    # Stage 3: faithful backtest
    run = run_lean_backtest(
        code,
        workspace_dir=workspace_dir,
        data_folder=data_folder,
        timeout=timeout,
        stream=stream,
        cleanup_project=cleanup_project,
    )
    result["execution_time"] = run.get("execution_time")
    result["statistics"] = run.get("statistics", {})

    if not run.get("success"):
        result["error"] = run.get("error", "Backtest failed with unknown error")
        return result

    result["backtested"] = True
    for k in ("trade_count", "total_return", "sharpe_ratio",
              "max_drawdown", "win_rate", "annual_return"):
        result[k] = run.get(k, result[k])

    # Stage 4: trades
    if result["trade_count"] < 1:
        result["error"] = "Strategy ran but executed zero trades"
        return result
    result["traded"] = True

    return result

def __getattr__(name: str):
    if name in ("evaluate_strategy", "build_feedback"):
        from . import reward
        return getattr(reward, name)
    if name in ("StrategyGenerator", "GenerationResult", "compute_metrics"):
        from . import generator
        return getattr(generator, name)
    if name in ("StrategyJudge", "create_strategy_judge"):
        from . import judge
        return getattr(judge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
