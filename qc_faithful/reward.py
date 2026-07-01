"""
The five-stage pipeline mirrors QuantCodeBench, but stage 3 is the real LEAN engine

  1. compiled    — ast.parse() succeeds
  2. validated   — QCAlgorithm subclass + Initialize()
  3. backtested  — the LEAN engine ran it without error
  4. traded      — LEAN reports at least one order
  5. judge       — (optional) an LLM judge confirms the code matches the task

`reward` is 1.0 iff the strategy compiled, validated, backtested, traded, and passed the judge
"""

from __future__ import annotations

from typing import Optional

from . import check


def evaluate_strategy(
    code: str,
    workspace_dir: Optional[str] = None,
    data_folder: Optional[str] = None,
    timeout: int = 600,
    judge=None,
    task_description: str = "",
    stream: bool = False,
) -> dict:
    """Run the full pipeline against the real LEAN engine and score it"""
    result = check(
        code,
        workspace_dir=workspace_dir,
        data_folder=data_folder,
        timeout=timeout,
        stream=stream,
        cleanup_project=True,  # benchmark runs shouldn't litter the workspace
    )
    result["reward"] = 0.0
    result["judge_passed"] = None

    if not result["traded"]:
        return result

    if judge is not None and task_description:
        try:
            verdict = judge.evaluate(code, task_description)
            result["judge_passed"] = bool(verdict.get("passed", False))
            result["judge_reason"] = verdict.get("reason", "")
            result["reward"] = 1.0 if result["judge_passed"] else 0.0
        except Exception:
            result["judge_passed"] = None
            result["reward"] = 1.0
    else:
        result["reward"] = 1.0

    return result


def build_feedback(result: dict, task_description: str = "") -> str:
    if not result.get("compiled"):
        return (
            "Your code has a Python syntax error and cannot be parsed\n"
            f"Error: {result.get('error')}\n"
            "Fix the syntax and return valid Python"
        )

    if not result.get("validated"):
        return (
            "Your code compiles but lacks the required QuantConnect structure\n"
            f"Issue: {result.get('error')}\n\n"
            "A valid strategy must define a class inheriting from QCAlgorithm and "
            "implement initialize(self) meethodd"
        )

    if result.get("environment_ok") is False:
        return (
            "The LEAN engine environment is not available, the strategy could "
            f"not be backtested.\n{result.get('error')}"
        )

    if not result.get("backtested"):
        return (
            "Your strategy has valid structure but the LEAN engine raised an error "
            "during the backtesting.\n"
            f"Error: {result.get('error')}\n\n"
        )

    if not result.get("traded"):
        return (
            "Your strategy ran in the LEAN engine but placed zero orders\n\n"
        )

    if result.get("judge_passed") is False:
        reason = result.get("judge_reason", "")
        return (
            "Your strategy executes and trades, but a reviewer judged that it does "
            "not implement the requested logic.\n"
            f"Reviewer note: {reason}\n\n"
            f"Task: {task_description}\n\n"
        )

    ret = result.get("total_return")
    tail = f"\nTotal return    : {ret * 100:.2f}%" if ret is not None else ""
    return (
        "Strategy passed all evaluation stages.\n"
        f"Trades executed : {result.get('trade_count')}{tail}"
    )
