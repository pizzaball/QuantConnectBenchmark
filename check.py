"""
Stage 1: ast.parse() — Python syntax is valid
Stage 2: structure check — QCAlgorithm subclass + Initialize() present
Stage 3: lean backtest — the REAL LEAN engine
Stage 4: trade count — does LEAN report any orders

Requires the Lean CLI (`pip install lean`) and a running Docker daemon.

Usage:
    python3 check.py examples/sma_crossover.py --workspace ~/lean-workspace
    python3 check.py my_strategy.py --data-folder /path/to/lean/data
    cat my_strategy.py | python3 check.py - --workspace ~/lean-workspace
    python3 check.py my_strategy.py --check-env     # just verify lean + Docker
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qc_faithful import check, preflight


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", nargs="?", help="Path to a .py strategy file, or '-' for stdin")
    p.add_argument("--workspace", help="Existing `lean init` workspace (where your data lives)")
    p.add_argument("--data-folder", help="Path to a LEAN-format data folder")
    p.add_argument("--timeout", type=int, default=600,
                   help="Seconds to allow the backtest (default: 600)")
    p.add_argument("--check-env", action="store_true",
                   help="Only verify the Lean CLI + Docker are available, then exit")
    args = p.parse_args()

    if args.check_env:
        ready, msg = preflight()
        print(f"\n  {'✓' if ready else '✗'}  LEAN environment ready")
        if not ready:
            print(f"\n  {msg}\n")
            return 2
        print()
        return 0

    if not args.file:
        p.error("a strategy file (or '-') is required unless --check-env is given")

    code = sys.stdin.read() if args.file == "-" else open(args.file).read()

    result = check(
        code,
        workspace_dir=args.workspace,
        data_folder=args.data_folder,
        timeout=args.timeout,
        stream=True,
    )

    stages = [
        ("compiled", "Compilation  (ast.parse)"),
        ("validated", "QC structure (QCAlgorithm + Initialize)"),
        ("backtested", "Backtest ran (real LEAN engine)"),
        ("traded", "Trades executed"),
    ]
    print()
    for key, label in stages:
        print(f"  {'✓' if result[key] else '✗'}  {label}")

    if result["trade_count"]:
        print(f"\n  Trades: {result['trade_count']}")
        rows = [
            ("Total return ", result["total_return"]),
            ("Annual return", result["annual_return"]),
            ("Sharpe ratio ", result["sharpe_ratio"]),
            ("Max drawdown ", result["max_drawdown"]),
            ("Win rate     ", result["win_rate"]),
        ]
        for label, val in rows:
            if val is None:
                continue
            if label.strip() == "Sharpe ratio":
                print(f"  {label} : {val:+.2f}")
            else:
                print(f"  {label} : {val * 100:+.2f}%")

    if result["error"]:
        print(f"\n  Error  : {result['error']}")

    print()

    """Exit codes: 
                  0 = compiled+validated+backtested+traded;
                  1 = a strategy stage failed
                  2 = environment not ready (can't run faithful backtest)
    """
    if result["environment_ok"] is False:
        return 2
    return 0 if result["traded"] else 1


if __name__ == "__main__":
    sys.exit(main())
