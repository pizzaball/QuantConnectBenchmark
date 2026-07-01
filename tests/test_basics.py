"""
Tests that run without Docker or the Lean CLI: compilation, structure
validation, result parsing, and environment preflight behaviour

    python3 tests/test_basics.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc_faithful.compile_check import compile_check, validate_qc_structure
from qc_faithful.results import parse_statistics, parse_backtest_dir, find_result_json

HERE = Path(__file__).resolve().parent
_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}")


ok, _ = compile_check("class A(QCAlgorithm):\n    def Initialize(self): pass\n")
check("valid code compiles", ok)

ok, err = compile_check("class A(QCAlgorithm)\n  bad syntax")
check("syntax error caught", (not ok) and "SyntaxError" in err)

ok, _ = validate_qc_structure(
    "class A(QCAlgorithm):\n    def Initialize(self): pass\n")
check("valid structure passes", ok)

ok, _ = validate_qc_structure(
    "class A(QCAlgorithm):\n    def initialize(self): pass\n")
check("snake_case initialize passes", ok)

ok, _ = validate_qc_structure("x = 1\n")
check("missing QCAlgorithm rejected", not ok)

ok, _ = validate_qc_structure("class A(QCAlgorithm):\n    def foo(self): pass\n")
check("missing Initialize rejected", not ok)

with open(HERE / "sample_lean_result.json") as f:
    data = json.load(f)
parsed = parse_statistics(data)
check("parses Total Orders", parsed["trade_count"] == 6)
check("parses Net Profit as fraction", abs(parsed["total_return"] - 0.0418) < 1e-9)
check("parses Sharpe ratio", abs(parsed["sharpe_ratio"] - 0.512) < 1e-9)
check("parses Drawdown percent", abs(parsed["max_drawdown"] - 0.103) < 1e-9)
check("parses Win Rate", abs(parsed["win_rate"] - 0.60) < 1e-9)

camel = {"statistics": {"Total Orders": "2", "Net Profit": "1.5%"}}
parsed_camel = parse_statistics(camel)
check("camelCase statistics block parsed", parsed_camel["trade_count"] == 2)

with tempfile.TemporaryDirectory() as d:
    bt = Path(d)
    (bt / "12345.json").write_text(json.dumps(data))
    (bt / "12345-order-events.json").write_text("[]")
    (bt / "12345-alpha-results.json").write_text("{}")
    picked = find_result_json(bt)
    check("find_result_json picks the stats file", picked.name == "12345.json")

    res = parse_backtest_dir(bt)
    check("parse_backtest_dir success", res["success"] and res["trade_count"] == 6)

print(f"\n  {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
