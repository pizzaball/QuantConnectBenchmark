# QuantConnect-Faithful

Faithful QuantConnect strategy checking using the **LEAN engine**, run
**locally**.


## The four stages

| Stage | Check | How |
|-------|-------|-----|
| 1. `compiled`   | valid Python syntax | `ast.parse()` |
| 2. `validated`  | `QCAlgorithm` subclass + `Initialize()` | regex |
| 3. `backtested` | runs in the real engine | `lean backtest` (Docker) |
| 4. `traded`     | LEAN reports ≥ 1 order | parsed from LEAN's result JSON |

## Setup

```bash
pip install -r requirements.txt        # installs the Lean CLI
# Install Docker Desktop (or the docker daemon) and start it.

# Create a Lean workspace — this is where your data lives.
# `lean init` initializes the CURRENT directory (it takes no path argument),
# so make the folder and cd into it first:
mkdir -p ~/lean-workspace
cd ~/lean-workspace
lean init
```

Verify the environment is ready before you rely on it:

```bash
python3 check.py --check-env
```

## Usage

```bash
# Point --workspace at your `lean init` workspace so LEAN finds the data:
python3 check.py examples/sma_crossover.py --workspace ~/lean-workspace

# Or point directly at a LEAN-format data folder:
python3 check.py my_strategy.py --data-folder /path/to/lean/data

cat my_strategy.py | python3 check.py - --workspace ~/lean-workspace
```

Example output:

```
  Compilation  (ast.parse)
  QC structure (QCAlgorithm + Initialize)
  Backtest ran (real LEAN engine)
  Trades executed

  Trades        : 6
  Total return  : +4.18%
  Annual return : +4.18%
  Sharpe ratio  : +0.51
  Max drawdown  : +10.30%
  Win rate      : +60.00%
```

Exit codes (designed for CI):

- `0` — compiled **and** validated **and** backtested **and** traded
- `1` — a strategy stage failed (syntax, structure, runtime error, or zero trades)
- `2` — environment not ready (Lean CLI or Docker missing) — couldn't run the engine

## As a library

```python
from qc_faithful import check

result = check(open("my_strategy.py").read(), workspace_dir="~/lean-workspace")
print(result["traded"], result["trade_count"], result["sharpe_ratio"])
```

## Benchmark: single-shot & agentic

Beyond checking one file, this repo includes an LLM benchmark harness in the
spirit of [QuantCode-Bench](https://github.com/LimexAILab/QuantCode-Bench). An LLM is asked to write a QuantConnect strategy for each task; the strategy is then graded by the pipeline:

```
compiled → validated → backtested (LEAN) → traded → judged (optional LLM)
```

`reward = 1.0` only if it compiles, validates, backtests, trades, and (when a
judge key is given) the judge agrees the code matches the task.

**Two modes, same generator:**

- **single-shot** — one generation attempt per task, no feedback.
- **agentic** — up to `--max-turns` attempts, each re-prompted with structured
  feedback derived from the stage it failed.

```bash
# Single-shot over the bundled tasks, scored by real LEAN:
python run_single_shot.py \
    --api-key  $OPENAI_API_KEY \
    --model    gpt-4o-mini \
    --tasks    data/benchmark_tasks_multiframe.json \
    --workspace ~/lean-workspace \
    --output   results/

# Agentic with up to 10 feedback turns, plus an LLM judge:
python run_agentic.py \
    --api-key  $OPENAI_API_KEY \
    --judge-key $OPENAI_API_KEY \
    --model    gpt-4o-mini \
    --max-turns 10 \
    --workspace ~/lean-workspace \
    --batch-size 2
```

**Important — this is real LEAN, so it's slow and data-bound:**

- Each task spins up a LEAN Docker container, so keep `--batch-size` modest
  (default 4 for single-shot, 2 for agentic). Concurrent backtests run in worker
  threads, but too many containers will thrash your machine.
- The bundled `data/benchmark_tasks.json` (10 tasks across easy/medium/hard)
  uses **SPY/AAPL/IBM/BAC over 2008–2020**, which the free `lean init` sample
  data covers — so it runs out of the box. Add your own tasks/data to scale up
  toward QuantCode-Bench's 400-task scope.

You can also call the pipeline directly:

```python
from qc_faithful import evaluate_strategy
r = evaluate_strategy(code, workspace_dir="~/lean-workspace")
print(r["reward"], r["trade_count"])
```

## About the data

LEAN needs market data in its own folder format. Options:

- **`lean init`** bundles a small free sample dataset — a handful of symbols
  over a fixed window. Notably, **SPY daily covers 1998-01-02 → 2021-03-31**,
  so the bundled `examples/sma_crossover.py` uses 2010–2020 to stay inside it.
- **Bring your own** data in LEAN's CSV layout and point `--data-folder` at it.
- **Subscribe** to QuantConnect's data (this is the only part that costs money /
  uses a token — and only for *downloading* their data, not for running the
  engine).

A backtest that **runs but places zero trades** is usually a data-range
mismatch, not a strategy bug: if you ask for dates the data folder doesn't
cover, LEAN runs with no bars, indicators never warm up, and nothing trades.
Check what you actually have, e.g.:

```bash
unzip -p ~/lean-workspace/data/equity/usa/daily/spy.zip | head -1   # first bar
unzip -p ~/lean-workspace/data/equity/usa/daily/spy.zip | tail -1   # last bar
```

## Tests

The logic that doesn't need Docker — compilation, structure validation, and
LEAN result parsing — is covered by a dependency-free test:

```bash
python3 tests/test_basics.py
```
