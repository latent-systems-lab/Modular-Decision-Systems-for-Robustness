# MultiHedge

## Code and experiments for two publications

This repository contains the MultiHedge implementation and experimental code
supporting both publications:

1. Feliks Bańka and Jarosław A. Chudziak,
   [*MultiHedge: Adaptive Coordination via Retrieval-Augmented Control*](https://link.springer.com/chapter/10.1007/978-3-032-29924-6_39),
   in *Computational Science – ICCS 2026*, Lecture Notes in Computer Science,
   vol. 16784, pp. 446–454, Springer, 2026.
2. *When Memory Is Not Enough: What Drives Robustness in AI Decision Systems?
   Evidence from Sequential Financial Decision-Making*.

The [original conference-version repository](https://github.com/latent-systems-lab/MultiHedge)
is retained as the artifact accompanying the ICCS paper. This repository
contains the extended implementation and robustness experiments used in the
second paper.

MultiHedge is the financial evaluation system used in the study. It combines language-model-guided
portfolio reasoning, classical hedging strategies, and retrieval-augmented
memory reinforcement learning (RL) to stress-test stock/option allocations.
`main.py` orchestrates the full workflow—loading market data, calling the LLM for
regime assessment, coordinating the specialized hedging agents, and optionally
including the reflective memory-RL module to adapt allocations over time.

## 🧠 System overview
- **LLM-driven coordinator:** `main.py` wires CLI arguments into an
  `ExtendedSimulationConfig`, instantiates the `Coordinator`, and drives the
  simulation/plotting pipeline so you can programmatically toggle assets,
  hedging strategies, and memory-RL participation per run.
- **Optional retrieval-augmented RL:** `config.py` exposes `use_memory_rl` and
  related knobs so you can enable/disable the experience store and reflection
  loop without touching code; the same flag is forwarded via `--use-memory-rl`
  when running the simulator from the command line.
- **Structured outputs:** Runs can persist CSV/JSON summaries and Matplotlib
  charts inside `results/`, making it easy to compare multiple portfolios.
  Performance metrics are captured via `save_simulation_results`, and plots are
  written alongside those tables when `--generate-plots` is set.

## ⚙️ Prerequisites & installation
1. **Python**: Install Python 3.10+ (the project relies on modern typing and
   `pydantic` v2 features).
2. **Dependencies**: Install the runtime stack with:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # optional but recommended
   pip install -r requirements.txt
   ```
   The pinned packages cover data science utilities (NumPy, pandas, SciPy),
   plotting (Matplotlib), model serving (OpenAI + Google Generative AI SDKs),
   and RL baselines (Gym, Stable-Baselines3, PyTorch).
   `requirements.txt` contains supported dependency constraints rather than a
   historical bit-for-bit lockfile. See `REPRODUCING.md` for environment capture.
3. **Environment variables**: Copy `.env.example` to `.env`, fill in the LLM and
   market-data API keys, and adjust defaults (provider/model) as needed. The
   configuration layer reads `LLM_PROVIDER`, `LLM_MODEL`, and related secrets at
   runtime, so the simulator will automatically pick up your `.env` values.

## 📦 Data requirements
Place the following assets under `data/` before running simulations:
- **Price history**: One CSV per equity ticker with OHLCV columns
  (`data/AAPL.csv`, `data/NVDA.csv`, …).
- **Options chains**: A SQLite database per ticker (e.g., `data/aapl_options.db`)
  containing rows keyed by `QUOTE_DATE`, strike, option type, and greeks so the
  hedging agents can price/size strategies.
- **Macro/alternative data**: Earnings cadence/growth (`earnings.csv`), CPI
  releases (`cpi.csv`), VIX levels (`vix.csv`), and sentiment (`fear-greed-2011-2023.csv`).
  Real, non-synthetic daily OHLCV data in the research environment was sourced
  from Yahoo Finance. Historical option chains were sourced from OptionsDX and
  cannot be redistributed by the authors under the provider's commercial terms.
  The broad-market check uses SPY as an S&P 500 proxy. See `REPRODUCING.md` for
  the complete data and experiment map.

Need seed data quickly? Run the helper and it will synthesize every required
file for AAPL/NVDA/TSLA:
```bash
python scripts/generate_sample_data.py
```
The script logs each artifact it writes, so you can confirm `data/` is fully
populated before launching longer simulations.

## 🚀 Usage examples
### Run the multi-agent simulator
```bash
python main.py \
  --start-date 2023-01-03 \
  --end-date 2023-03-30 \
  --assets AAPL NVDA TSLA \
  --hedging-strategies collar straddle delta-neutral \
  --use-llm \
  --use-memory-rl True \
  --output-dir results/custom_run
```
Key flags:
- `--use-llm` / `--no-llm` toggle the language model allocator; pair with the
  environment variables in `.env` to switch between OpenAI and Gemini backends.
- `--use-memory-rl` accepts `True/False` so you can benchmark reflective memory
  loops vs. pure LLM guidance.
- `--equal-options-strategy-split` forces uniform capital splits across hedging
  agents; omit it to let the coordinator allocate dynamically.

### Inspect outputs
- **Directory layout**: By default, results live under
  `results/PAKDD/openai/` (configurable via `config.DEFAULT_OUTPUT_DIR`), so
  `--output-dir` lets you create scenario-specific folders such as
  `results/january_llm_off`.
- **Tables**: Each run emits `performance.csv` (daily KPIs), `summary.json`
  (aggregated returns/drawdown/breakdowns), and `final_portfolio.json` (end-state
  holdings) inside the chosen output directory.
- **Plots**: When `--generate-plots` is passed, `performance_plot.png` is saved
  alongside the tables, letting you visually compare equity curves.

### Monitor LLM usage
Every call to the language model appends metadata (timestamp, latency, token
counts, status) to `llm_usage.jsonl` inside the same output directory, enabling
post-run auditability. Analyze a single day’s
log slice via:
```bash
python scripts/llm_usage_one_day.py --date 2023-01-10 --usage-log results/custom_run/llm_usage.jsonl
```
The helper bootstraps a one-day simulation (respecting `config.py`) and prints
aggregate token/latency totals so you can verify usage-based cost budgets.

## 🧰 Auxiliary tooling
- `scripts/generate_sample_data.py`: quickly populate `data/` with synthetic
  equities, options, and macro indicators for development or demos.
- `scripts/llm_usage_one_day.py`: run a constrained trading day and summarise the
  accompanying `llm_usage.jsonl`, which is helpful for cost tracking or CI tests
  that need deterministic runtimes.

With the prerequisites satisfied, data staged, and environment variables set,
you can iterate on hedging strategies, compare LLM vs. rules-only baselines, and
inspect logs/plots in `results/` to validate every experiment end-to-end.
