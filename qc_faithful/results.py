"""
Parse a LEAN backtest result directory into a clean statistics dict.

LEAN writes results to `<project>/backtests/<timestamp>/`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _ci_get(d: dict, *names: str, default=None):
    lowered = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in lowered:
            return lowered[n.lower()]
    return default


def _to_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace(",", "")
    is_pct = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100.0 if is_pct else v


def _to_int(raw: Any) -> int:
    try:
        return int(float(str(raw).replace(",", "")))
    except (ValueError, TypeError):
        return 0


def find_result_json(backtest_dir: Path) -> Optional[Path]:
    candidates = [
        f for f in backtest_dir.glob("*.json")
        if not any(tag in f.name.lower()
                   for tag in ("order-events", "alpha", "data-monitor"))
    ]
    if not candidates:
        return None

    def has_stats(p: Path) -> bool:
        try:
            with open(p) as f:
                data = json.load(f)
            return _ci_get(data, "Statistics", "statistics") is not None
        except Exception:
            return False

    with_stats = sorted(
        (p for p in candidates if has_stats(p)),
        key=lambda p: p.stat().st_mtime,
    )
    if with_stats:
        return with_stats[-1]
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_statistics(data: dict) -> dict:
    stats = _ci_get(data, "Statistics", "statistics", default={}) or {}

    total_orders = _to_int(
        _ci_get(stats, "Total Orders", "Total Trades", default="0")
    )
    return {
        "trade_count": total_orders,
        "total_return": _to_float(
            _ci_get(stats, "Net Profit", "Total Return", "Total Net Profit")
        ),
        "sharpe_ratio": _to_float(_ci_get(stats, "Sharpe Ratio")),
        "max_drawdown": _to_float(_ci_get(stats, "Drawdown")),
        "win_rate": _to_float(_ci_get(stats, "Win Rate")),
        "annual_return": _to_float(
            _ci_get(stats, "Compounding Annual Return", "Annual Return")
        ),
        "statistics": stats,
    }


def parse_backtest_dir(backtest_dir: Path) -> dict:
    """Load and parse the result JSON from one backtest run directory."""
    result_json = find_result_json(backtest_dir)
    if result_json is None:
        return {"success": False, "error": "No result JSON found in backtest output",
                "trade_count": 0, "statistics": {}}
    try:
        with open(result_json) as f:
            data = json.load(f)
    except Exception as exc:
        return {"success": False, "error": f"Could not read result JSON: {exc}",
                "trade_count": 0, "statistics": {}}

    parsed = parse_statistics(data)
    parsed["success"] = True
    parsed["result_file"] = str(result_json)
    return parsed
