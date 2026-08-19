# Paper Experiment Manifest

This file maps the manuscript analyses to executable entry points and records
the configuration values that should accompany reproduced outputs.

## Main configuration

| Item | Setting |
| --- | --- |
| Evaluation period | 2021-01-01 to 2023-12-31 |
| Memory period | 2016-2020 |
| Assets | AAPL, TSLA, NVDA |
| Initial capital | USD 1,000,000 |
| Rebalance frequency | 90 trading days |
| Primary coordinator | OpenAI GPT-5 |
| Main decoding | `T=0.0` |
| Stochastic ablation | GPT-5, `T=0.7` |
| Retrieval encoder | `all-MiniLM-L6-v2` |
| Retrieval index | FAISS `IndexFlatL2` |
| Retrieved neighbors | `k=5` |
| Global Python/NumPy seed | 42 |

## Analysis-to-runner map

| Analysis | Command |
| --- | --- |
| Full system | `python main.py --controller-mode full_llm --use-llm --use-memory-rl True` |
| Memory isolation | `python experiments/run_memory_controller_ablations.py --random-runs 3` |
| Coordinator choice and stochasticity | `python experiments/run_multi_model_studies.py --group gpt-sweep` |
| Execution modules | `python experiments/run_multi_model_studies.py --group individual`, `--group pairs`, and `--group all-strategies` |
| Rules-only selector | `python experiments/run_h2_rules_only_selector.py` |
| SPY 2022 robustness | `python experiments/run_spy_1y_sanity.py` |
| Synthetic three-regime check | `python experiments/run_synthetic_3_regime.py --seed 42 --export-data True` |
| Temporal disruption | `python experiments/run_scrambled_timeline_placebo.py --seed 42` |
| Bootstrap evidence | `python scripts/paired_block_bootstrap_evidence.py --results-dir results --n-bootstrap 10000 --block-size 20` |

The memory-isolation runner records its run specifications in `metadata.json`.
Random-retrieval repetitions use seeds 42, 43, and 44 by default. The temporal-
disruption runner derives each permutation from the supplied seed and run index.

## Hosted inference variability

At `T=0.0`, the input data, prompt templates, and deterministic execution rules
remain fixed. Hosted LLM inference is not guaranteed to be bitwise deterministic,
and the recorded repeated calls returned different structured outputs. Because
decisions update the subsequent portfolio state, those differences propagate
through the sequential trace. They are the source of the full-system run-level
dispersion; the three runs are not different market splits.

## Environment record

`requirements.txt` specifies supported dependency constraints. A complete
historical `pip freeze` was not retained, so the repository does not claim a
bit-for-bit reconstruction of the original package environment. A new run should
save `python --version` and `python -m pip freeze` in its output directory.
