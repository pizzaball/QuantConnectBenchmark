"""
LLM strategy generator 

Two modes (same generator, different max_turns):
  single-shot — one LLM call, no feedback        (max_turns = 1)
  agentic     — iterative refinement on feedback  (max_turns = N)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  

from .reward import evaluate_strategy, build_feedback


# Prompts

_SYSTEM_PROMPT = """\
You are an expert algorithmic trading developer for QuantConnect's LEAN engine.

Write a complete, self-contained Python strategy for QuantConnect.

REQUIRED:
- Exactly one class that inherits from QCAlgorithm.
- An Initialize(self) method (the only mandatory method).
- Start the file with: from AlgorithmImports import *
- Call self.SetStartDate / self.SetEndDate to match the requested window, and
  self.SetCash(100000).
- Add the requested symbol with self.AddEquity("TICKER", Resolution.Daily).

EXAMPLE shape:
```python
from AlgorithmImports import *

class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2020, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.sma = self.SMA(self.symbol, 20, Resolution.Daily)

    def OnData(self, data):
        if not self.sma.IsReady:
            return
        if not self.Portfolio.Invested:
            self.SetHoldings(self.symbol, 1.0)
```

RULES:
- Guard indicator access with `if not <indicator>.IsReady: return`.
- Do not import third-party libraries.
- Respond with ONLY the Python code block, nothing else.
"""

_USER_PROMPT = """\
Write a complete QuantConnect strategy for the following task:

{task_description}

Symbol : {ticker}
Start  : {start_date}
End    : {end_date}

Respond with only the Python code, no explanations.
"""

_REFINEMENT_PROMPT = """\
Your previous strategy attempt did not pass.

Evaluation feedback:
{feedback}

Original task:
{task_description}

Your previous code:
```python
{previous_code}
```

Fix every issue in the feedback and return a corrected strategy.
Respond with only the Python code, no explanations.
"""

_LOOP_DETECTION_THRESHOLD = 3


@dataclass
class GenerationResult:
    task_id: str
    code: str
    reward: float
    compiled: bool
    validated: bool
    backtested: bool
    traded: bool
    judge_passed: Optional[bool]
    total_return: Optional[float]
    trade_count: int
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    turns: int
    is_looped: bool
    environment_ok: Optional[bool]
    error: Optional[str]
    trajectory: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class StrategyGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        judge=None,
        backtest_workspace: Optional[str] = None,
        backtest_data_folder: Optional[str] = None,
        backtest_timeout: int = 600,
        trajectories_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        if AsyncOpenAI is None:
            raise RuntimeError("openai package required: pip install openai")
        self.model = model
        self.judge = judge
        self.workspace = backtest_workspace
        self.data_folder = backtest_data_folder
        self.timeout = backtest_timeout
        self.trajectories_dir = trajectories_dir
        self.verbose = verbose

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

        if trajectories_dir:
            Path(trajectories_dir).mkdir(parents=True, exist_ok=True)

    async def _call_llm(self, messages: list[dict], max_tokens: int = 4096):
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        return content, (usage.prompt_tokens or 0), (usage.completion_tokens or 0)

    @staticmethod
    def _extract_code(text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        m = re.search(r"```python\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        text = re.sub(r"```[a-z]*\n?", "", text)
        return text.replace("```", "").strip()

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _save_trajectory(self, task_id: str, trajectory: list):
        if not self.trajectories_dir:
            return
        path = Path(self.trajectories_dir) / f"{task_id}.jsonl"
        with open(path, "w") as f:
            for entry in trajectory:
                f.write(json.dumps(entry, default=str) + "\n")

    async def _evaluate(self, code: str, description: str) -> dict:
        return await asyncio.to_thread(
            evaluate_strategy,
            code,
            workspace_dir=self.workspace,
            data_folder=self.data_folder,
            timeout=self.timeout,
            judge=self.judge,
            task_description=description,
            stream=False,
        )

    async def generate_single(self, task: dict) -> GenerationResult:
        return await self._run_task(task, max_turns=1)

    async def generate_agentic(self, task: dict, max_turns: int = 10) -> GenerationResult:
        return await self._run_task(task, max_turns=max_turns)

    async def _run_task(self, task: dict, max_turns: int) -> GenerationResult:
        task_id = str(task.get("id", "unknown"))
        ticker = task.get("yf_symbol") or task.get("ticker", "SPY")
        start_date = task.get("start_date", "2015-01-01")
        end_date = task.get("end_date", "2020-01-01")
        description = task.get("description", "")

        trajectory: list = []
        code = ""
        prev_codes: list[str] = []
        total_in = total_out = 0
        result: Optional[dict] = None

        for turn in range(1, max_turns + 1):
            if turn == 1:
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_PROMPT.format(
                        task_description=description, ticker=ticker,
                        start_date=start_date, end_date=end_date)},
                ]
            else:
                feedback = build_feedback(result, description)  
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _REFINEMENT_PROMPT.format(
                        feedback=feedback, task_description=description,
                        previous_code=code)},
                ]

            self._log(f"[{task_id}] turn {turn}/{max_turns}")
            raw, in_tok, out_tok = await self._call_llm(messages)
            total_in += in_tok
            total_out += out_tok
            code = self._extract_code(raw)

            result = await self._evaluate(code, description)
            trajectory.append({"turn": turn, "code": code, "result": result})
            prev_codes.append(code)

            if result["reward"] == 1.0:
                self._log(f"[{task_id}] passed at turn {turn}")
                break

            # Environment failures won't be fixed by re-prompting — stop early.
            if result.get("environment_ok") is False:
                self._log(f"[{task_id}] LEAN environment unavailable — aborting")
                break

        assert result is not None
        is_looped = _detect_loop(prev_codes, _LOOP_DETECTION_THRESHOLD)
        self._save_trajectory(task_id, trajectory)

        return GenerationResult(
            task_id=task_id,
            code=code,
            reward=result["reward"],
            compiled=result["compiled"],
            validated=result["validated"],
            backtested=result["backtested"],
            traded=result["traded"],
            judge_passed=result.get("judge_passed"),
            total_return=result.get("total_return"),
            trade_count=result.get("trade_count", 0),
            sharpe_ratio=result.get("sharpe_ratio"),
            max_drawdown=result.get("max_drawdown"),
            turns=len(trajectory),
            is_looped=is_looped,
            environment_ok=result.get("environment_ok"),
            error=result.get("error"),
            trajectory=trajectory,
            input_tokens=total_in,
            output_tokens=total_out,
        )

    async def run_batch(
        self,
        tasks: list[dict],
        max_turns: int = 1,
        batch_size: int = 4,
        delay_between_batches: float = 0.5,
    ) -> list[GenerationResult]:
        """Evaluate tasks with bounded concurrency"""
        results: list[GenerationResult] = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            coros = [self._run_task(t, max_turns) for t in batch]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, Exception):
                    self._log(f"[batch] task error: {r!r}")
                else:
                    results.append(r)
            if i + batch_size < len(tasks):
                await asyncio.sleep(delay_between_batches)
        return results


# Helpers / metrics
def _detect_loop(codes: list[str], threshold: int) -> bool:
    if len(codes) < threshold:
        return False
    recent = codes[-threshold:]
    return len(set(recent)) == 1


def compute_metrics(results: list[GenerationResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def rate(attr: str) -> float:
        return sum(1 for r in results if getattr(r, attr)) / n

    judged = [r for r in results if r.judge_passed is not None]
    return {
        "n": n,
        "compilation_rate": rate("compiled"),
        "validation_rate": rate("validated"),
        "backtest_rate": rate("backtested"),
        "trade_rate": rate("traded"),
        "reward": sum(r.reward for r in results) / n,
        "judge_pass_rate": (sum(1 for r in judged if r.judge_passed) / len(judged)
                            if judged else 0.0),
        "loop_rate": rate("is_looped"),
        "avg_turns": sum(r.turns for r in results) / n,
        "avg_input_tokens": sum(r.input_tokens for r in results) / n,
        "avg_output_tokens": sum(r.output_tokens for r in results) / n,
    }
