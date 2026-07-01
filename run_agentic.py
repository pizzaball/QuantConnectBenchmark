"""
Agentic benchmark runner for QuantConnect-Faithful.

The model gets iterative feedback for up to --max-turns attempts per task, with
each attempt scored by the REAL LEAN engine.

Usage:
    python run_agentic.py \
        --api-key   sk-...                       \
        --model     gpt-4o-mini                  \
        --tasks     data/benchmark_tasks.json    \
        --workspace ~/lean-workspace             \
        --max-turns 10                           \
        --output    results/                     \
        [--base-url   https://...] [--judge-key sk-...] \
        [--batch-size 2] [--limit 10] [--verbose]

Requires the Lean CLI + a running Docker daemon
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from qc_faithful import preflight
from qc_faithful.generator import StrategyGenerator, compute_metrics
from qc_faithful.judge import create_strategy_judge


def parse_args():
    p = argparse.ArgumentParser(description="QuantConnect-Faithful — agentic")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="LLM API key (or set OPENAI_API_KEY)")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None)
    p.add_argument("--tasks", default="data/benchmark_tasks.json")
    p.add_argument("--workspace", default=None)
    p.add_argument("--data-folder", default=None)
    p.add_argument("--output", default="results")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Lower than single-shot becausee each task may use many turns")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge-key", default=None)
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--trajectories-dir", default="trajectories")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


async def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("error: --api-key (or OPENAI_API_KEY) is required")
        return 2

    ready, msg = preflight()
    if not ready:
        print(f"\n LEAN environment not ready:\n\n  {msg}\n")
        return 2

    with open(args.tasks) as f:
        tasks = json.load(f)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Loaded {len(tasks)} tasks · max_turns={args.max_turns}")

    judge = None
    if args.judge_key:
        judge = create_strategy_judge(api_key=args.judge_key,
                                      base_url=args.base_url,
                                      model=args.judge_model)

    gen = StrategyGenerator(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        judge=judge,
        backtest_workspace=args.workspace,
        backtest_data_folder=args.data_folder,
        backtest_timeout=args.timeout,
        trajectories_dir=args.trajectories_dir,
        verbose=args.verbose,
    )

    print(f"Agentic model={args.model} real LEAN engine")
    t0 = time.time()
    results = await gen.run_batch(tasks=tasks, max_turns=args.max_turns,
                                  batch_size=args.batch_size)
    elapsed = time.time() - t0

    metrics = compute_metrics(results)

    turns_dist: Counter = Counter()
    for r in results:
        if r.reward == 1.0:
            turns_dist[r.turns] += 1

    diff_groups: dict[str, list] = {}
    for task, res in zip(tasks[: len(results)], results):
        diff_groups.setdefault(task.get("difficulty", "unknown"), []).append(res)
    diff_metrics = {d: compute_metrics(g) for d, g in diff_groups.items()}

    output = {
        "model": args.model,
        "engine": "lean",
        "mode": "agentic",
        "max_turns": args.max_turns,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 1),
        "overall": metrics,
        "by_difficulty": diff_metrics,
        "turns_to_success": {str(k): v for k, v in sorted(turns_dist.items())},
        "tasks": [
            {
                "task_id": r.task_id,
                "reward": r.reward,
                "compiled": r.compiled,
                "validated": r.validated,
                "backtested": r.backtested,
                "traded": r.traded,
                "judge_passed": r.judge_passed,
                "total_return": round(r.total_return * 100, 2) if r.total_return is not None else None,
                "sharpe_ratio": r.sharpe_ratio,
                "trade_count": r.trade_count,
                "turns": r.turns,
                "is_looped": r.is_looped,
                "error": r.error,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }
            for r in results
        ],
    }

    Path(args.output).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) / f"{args.model.replace('/', '_')}_agentic_t{args.max_turns}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Model: {args.model}")
    print(f"Engine: real LEAN  (agentic)")
    print(f"Max turns: {args.max_turns}")
    print(f"Tasks: {metrics.get('n', 0)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("-" * 60)
    print(f"Compilation rate: {metrics.get('compilation_rate', 0) * 100:.1f}%")
    print(f"Validation rate: {metrics.get('validation_rate', 0) * 100:.1f}%")
    print(f"Backtest rate: {metrics.get('backtest_rate', 0) * 100:.1f}%")
    print(f"Trade rate: {metrics.get('trade_rate', 0) * 100:.1f}%")
    print(f"Reward: {metrics.get('reward', 0) * 100:.1f}%")
    print(f"Avg turns: {metrics.get('avg_turns', 0):.2f}")
    print(f"Loop rate: {metrics.get('loop_rate', 0) * 100:.1f}%")
    if metrics.get("judge_pass_rate"):
        print(f"Judge pass rate: {metrics['judge_pass_rate'] * 100:.1f}%")
    print("-" * 60)
    print("Turns to success:")
    for turn, count in sorted(turns_dist.items()):
        print(f"    Turn {turn}: {count:3d}  {'█' * count}")
    print("-" * 60)
    for diff, dm in diff_metrics.items():
        print(f"  [{diff:8s}]  compile={dm.get('compilation_rate', 0)*100:.0f}%  "
              f"trade={dm.get('trade_rate', 0)*100:.0f}%  reward={dm.get('reward', 0)*100:.0f}%")
    print("=" * 60)
    print(f"\n  Results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
