# Modular Decision Systems for Robustness

Public companion repository for the paper **"Beyond Scale: The Role of Memory
in Robust Modular Decision Systems."**

The code implements a modular financial-control testbed used to study how
robust behavior changes when retrieval memory, coordinator model capacity, and
execution constraints are varied independently. The system combines structured
LLM-based coordination, retrieval-augmented episodic memory, and deterministic
hedging modules.

This repository is a code and experiment companion, not a bundled data release.
The historical vendor datasets used for the paper are not stored in git.

## Repository Layout

- `main.py` - command-line entry point for simulations.
- `config.py` - default experiment settings.
- `src/` - simulator, agents, memory, models, and utilities.
- `experiments/` - packaged experiment runners.
- `scripts/` - data generation and LLM usage utilities.
- `REPRODUCING.md` - reproduction workflow and data requirements.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with the LLM credentials for the backend you want to use.

For a development-only synthetic data setup:

```bash
python3 scripts/generate_sample_data.py
```

For the paper workflow and required historical data layout, see
[REPRODUCING.md](REPRODUCING.md).

## Example Run

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

## Experiment Runners

Strategy and model ablations:

```bash
python3 experiments/run_multi_model_studies.py --group paper-ablation
```

Available groups are:

- `individual`
- `all-strategies`
- `pairs`
- `providers`
- `gpt-sweep`
- `paper-ablation`

Generated outputs are written under `results/`. Keep raw data, generated
results, local `.env` files, and memory stores out of git.
