"""
Stage 1 + 2: compilation and QuantConnect structure validation
"""

from __future__ import annotations

import ast
import re


def clean_code(code: str) -> str:
    """Strip markdown fences and <thinking> blocks"""
    code = re.sub(r"<thinking>.*?</thinking>", "", code, flags=re.DOTALL)
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
    code = re.sub(r"```python\s*\n?", "", code)
    code = re.sub(r"```\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"```", "", code)
    return code.strip()


_REQUIRED_PATTERNS = [
    (r"class\s+\w+\s*\(\s*QCAlgorithm\s*\)",
     "Strategy must define a class that inherits from QCAlgorithm"),
    (r"def\s+[Ii]nitialize\s*\(self",
     "Strategy must define the Initialize(self) / initialize(self) method"),
]


def compile_check(code: str) -> tuple[bool, str]:
    """Return (ok, error). Valid Python syntax via ast.parse()"""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"


def validate_qc_structure(code: str) -> tuple[bool, str]:
    """Return (ok, error). Requires a QCAlgorithm subclass + Initialize()"""
    for pattern, msg in _REQUIRED_PATTERNS:
        if not re.search(pattern, code, re.IGNORECASE):
            return False, msg
    return True, ""
