# Reproducing the Paper Experiments

This repository accompanies the paper "Beyond Scale: The Role of Memory in
Robust Modular Decision Systems." The manuscript source is maintained outside
this repository; this repository contains the public code and experiment
workflow.

## Environment

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `LLM_PROVIDER`, `LLM_MODEL`, and the corresponding API key in `.env`.

## Data

The main historical experiments require local market data under `data/`:

- `AAPL.csv`, `NVDA.csv`, and `TSLA.csv` with daily OHLCV columns.
- `aapl_options.db`, `nvda_options.db`, and `tsla_options.db` SQLite option
  chains with an `options` table.
- `earnings.csv`, `cpi.csv`, `vix.csv`, and `fear-greed-2011-2023.csv` for
  macro and auxiliary signals.

The research runs used Yahoo Finance equity data and OptionsDX option chains.
Those raw vendor files are not tracked in git.

For development-only smoke checks, generate synthetic sample data:

```bash
python3 scripts/generate_sample_data.py
```

Synthetic data is useful for checking that the pipeline runs, but it will not
reproduce the paper tables.

## Main Simulator

```bash
python3 main.py \
  --start-date 2021-01-01 \
  --end-date 2023-12-31 \
  --assets AAPL TSLA NVDA \
  --hedging-strategies collar straddle delta-neutral \
  --use-llm \
  --use-memory-rl True \
  --output-dir results/main_full_llm
```

Outputs include `performance.csv`, `summary.json`, `final_portfolio.json`,
plots when enabled, and `llm_usage.jsonl`.

## Strategy and Model Ablations

The multi-model study runner exposes named groups, so source files do not need
to be edited between runs:

```bash
python3 experiments/run_multi_model_studies.py --group individual
python3 experiments/run_multi_model_studies.py --group pairs
python3 experiments/run_multi_model_studies.py --group providers
python3 experiments/run_multi_model_studies.py --group gpt-sweep
```

To run the main strategy/model ablation batch:

```bash
python3 experiments/run_multi_model_studies.py --group paper-ablation
```

These runs require valid LLM credentials and the historical market data
described above.
