"""
LLM-based semantic judge for QuantConnect strategies.

Decides whether generated code implements the core idea of the task description (1 = implements it, 0 = ignores it)
"""

from __future__ import annotations

import json
import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None 


_JUDGE_SYSTEM = """\
You are an expert reviewer of QuantConnect / LEAN trading strategies.

You are given a task description and the Python code of a QCAlgorithm strategy.
Decide whether the code implements the CORE IDEA of the task.

Scoring:
  - 1 if the code implements the main logic, even if details or parameter values
    differ or are simplified.
  - 0 if it ignores the task: wrong indicators, inverted logic, or a trivial
    buy-and-hold unrelated to what was asked.
  - Be lenient: a reasonable, functional interpretation counts as 1.

Respond with ONLY a JSON object: {"score": 0 or 1, "reason": "..."}.
"""

_JUDGE_USER = """\
Task description:
{task_description}

Strategy code:
```python
{code}
```

Does this code implement the core idea of the task? JSON only.
"""

_HEURISTIC_KEYWORDS = [
    "sma", "ema", "rsi", "macd", "bollinger", "bb", "momentum", "stochastic",
    "crossover", "cross", "atr", "adx", "vwap", "obv", "volume", "breakout",
    "mean reversion", "trend", "signal", "setholdings", "set_holdings",
    "marketorder", "market_order", "liquidate", "buy", "sell",
]


def _heuristic_judge(code: str, task_description: str) -> dict:
    code_lower = code.lower()
    task_lower = task_description.lower()

    task_keywords = [kw for kw in _HEURISTIC_KEYWORDS if kw in task_lower]
    matches = [kw for kw in task_keywords if kw in code_lower]
    coverage = len(matches) / len(task_keywords) if task_keywords else 0.5

    has_class = "qcalgorithm" in code_lower
    has_orders = any(k in code_lower for k in
                     ["setholdings", "set_holdings", "marketorder",
                      "market_order", "liquidate"])

    score = 1 if (coverage >= 0.4 and has_class and has_orders) else 0
    return {
        "passed": bool(score),
        "score": score,
        "reason": f"Heuristic: {len(matches)}/{len(task_keywords)} task keywords matched",
        "source": "heuristic",
    }


class StrategyJudge:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self._model = model
        self._client = None
        if OpenAI is not None and api_key:
            kwargs: dict = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            try:
                self._client = OpenAI(**kwargs)
            except Exception:
                self._client = None

    def evaluate(self, code: str, task_description: str) -> dict:
        """Return {passed: bool, score: int, reason: str, source: str}."""
        if self._client is None:
            return _heuristic_judge(code, task_description)

        prompt = _JUDGE_USER.format(
            task_description=task_description,
            code=code[:6000],
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                score = int(data.get("score", 0))
                return {
                    "passed": score == 1,
                    "score": score,
                    "reason": data.get("reason", ""),
                    "source": "llm",
                }
        except Exception as exc:
            fallback = _heuristic_judge(code, task_description)
            fallback["llm_error"] = str(exc)
            return fallback

        return _heuristic_judge(code, task_description)


def create_strategy_judge(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> StrategyJudge:
    return StrategyJudge(api_key=api_key, base_url=base_url, model=model)
