"""Run controlled memory-controller ablations with a fixed prebuilt memory buffer."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config
from main import ensure_output_directory, save_simulation_results
from src.core.coordinator import Coordinator
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig, MemoryRLConfig
from src.utils import llm_utils

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "memory_controller_ablations"
BUILD_START = pd.Timestamp("2020-01-01").date()
BUILD_END = pd.Timestamp("2020-12-31").date()
EVAL_START = pd.Timestamp("2021-01-01").date()
EVAL_END = pd.Timestamp("2023-12-31").date()
ASSETS = ["AAPL", "TSLA", "NVDA"]
REBALANCE_FREQUENCY_DAYS = 90
EMERGENCY_THRESHOLD = 0.1


@dataclass
class RunRecord:
    mode: str
    run_name: str
    seed: int
    start_date: str
    end_date: str
    output_dir: str
    final_value: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades_executed: int


def _apply_temperature_override(temperature: float = 0.0) -> None:
    """Force JSON-structured LLM calls to use a fixed temperature."""

    original_genai = llm_utils.get_llm_response_json_genai
    original_openai = llm_utils.get_llm_response_json_openai

    def _patched(prompt, response_model, temperature_override: float = temperature):
        provider = getattr(config, "LLM_PROVIDER", "openai").lower()
        if provider in {"gemini", "google"}:
            return original_genai(prompt, response_model, temperature_override)
        if provider in {"gpt", "openai"}:
            return original_openai(prompt, response_model, temperature_override)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    llm_utils.get_llm_response = _patched


def _build_config(
    *,
    start_date,
    end_date,
    controller_mode: str,
    use_memory_rl: bool,
    memory_dir: Path,
    allow_memory_updates: bool,
    random_seed: int,
) -> ExtendedSimulationConfig:
    memory_config = MemoryRLConfig(
        memory_dir=str(memory_dir),
        save_frequency=1,
        allow_memory_updates=allow_memory_updates,
        random_seed=random_seed,
    )
    memory_config.memory_config.memory_dir = str(memory_dir)
    memory_config.memory_config.portfolio_memory_config.memory_dir = str(memory_dir)
    memory_config.memory_config.asset_memory_config.memory_dir = str(memory_dir)
    memory_config.memory_config.hedging_memory_config.memory_dir = str(memory_dir)

    return ExtendedSimulationConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=config.INITIAL_CAPITAL,
        assets=ASSETS,
        rebalance_frequency=REBALANCE_FREQUENCY_DAYS,
        emergency_threshold=EMERGENCY_THRESHOLD,
        use_llm=True,
        controller_mode=controller_mode,
        hedging_strategies=config.HEDGING_STRATEGIES,
        use_memory_rl=use_memory_rl,
        equal_options_strategy_split=config.EQUAL_OPTIONS_STRATEGY_SPLIT,
        rule_based_strategy_split=False,
        max_options_allocation_ratio=config.MAX_OPTIONS_ALLOCATION_RATIO,
        max_collar_options_weight=config.MAX_COLLAR_OPTIONS_WEIGHT,
        allow_short_selling=config.ALLOW_SHORT_SELLING,
        equity_transaction_cost_bps=10.0,
        options_transaction_cost_bps=20.0,
        slippage_bps=2.0,
        memory_rl_config=memory_config,
    )


def _run_single(
    *,
    sim_config: ExtendedSimulationConfig,
    output_dir: Path,
) -> RunRecord:
    ensure_output_directory(str(output_dir))
    coordinator = Coordinator(sim_config)
    result = coordinator.run_simulation()
    save_simulation_results(result, str(output_dir))

    return RunRecord(
        mode=sim_config.controller_mode,
        run_name=output_dir.name,
        seed=sim_config.memory_rl_config.random_seed if sim_config.memory_rl_config else 42,
        start_date=str(sim_config.start_date),
        end_date=str(sim_config.end_date),
        output_dir=str(output_dir),
        final_value=result.final_portfolio.total_value,
        total_return=result.total_return,
        annualized_return=result.annualized_return,
        max_drawdown=result.max_drawdown,
        sharpe_ratio=result.sharpe_ratio,
        trades_executed=result.trades_executed,
    )


def _build_memory(memory_dir: Path) -> None:
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    build_config = _build_config(
        start_date=BUILD_START,
        end_date=BUILD_END,
        controller_mode="no_memory",
        use_memory_rl=True,
        memory_dir=memory_dir,
        allow_memory_updates=True,
        random_seed=42,
    )
    _run_single(sim_config=build_config, output_dir=memory_dir.parent / "memory_build")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled memory-controller ablations.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--random-runs", type=int, default=1)
    parser.add_argument("--skip-memory-build", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["full_llm", "retrieval_only", "random_retrieval", "no_memory"],
        default=["full_llm", "retrieval_only", "random_retrieval", "no_memory"],
        help="Controller modes to evaluate. Defaults to the full paper memory-ablation set.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature override for JSON-structured LLM calls.",
    )
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)
    _apply_temperature_override(args.temperature)

    output_dir: Path = args.output_dir
    ensure_output_directory(str(output_dir))
    memory_dir = output_dir / "shared_memory"

    if not args.skip_memory_build:
        _build_memory(memory_dir)

    run_specs: List[Dict[str, object]] = []
    for mode in args.modes:
        if mode == "random_retrieval":
            run_specs.extend(
                {"mode": mode, "seed": 42 + idx}
                for idx in range(args.random_runs)
            )
        else:
            run_specs.append({"mode": mode, "seed": 42})

    records: List[RunRecord] = []
    for spec in run_specs:
        mode = str(spec["mode"])
        seed = int(spec["seed"])
        run_name = f"{mode}_seed_{seed}"
        sim_config = _build_config(
            start_date=EVAL_START,
            end_date=EVAL_END,
            controller_mode=mode,
            use_memory_rl=(mode != "no_memory"),
            memory_dir=memory_dir,
            allow_memory_updates=False,
            random_seed=seed,
        )
        records.append(
            _run_single(
                sim_config=sim_config,
                output_dir=output_dir / run_name,
            )
        )

    results_df = pd.DataFrame([asdict(record) for record in records])
    results_path = output_dir / "summary.csv"
    results_df.to_csv(results_path, index=False)

    metadata = {
        "assets": ASSETS,
        "memory_construction_period": [str(BUILD_START), str(BUILD_END)],
        "evaluation_period": [str(EVAL_START), str(EVAL_END)],
        "transaction_costs_bps": {
            "equity": 10.0,
            "options": 20.0,
            "slippage": 2.0,
        },
        "llm_temperature": args.temperature,
        "rebalance_frequency_days": REBALANCE_FREQUENCY_DAYS,
        "emergency_threshold": EMERGENCY_THRESHOLD,
        "modes": args.modes,
        "random_runs": args.random_runs,
        "memory_dir": str(memory_dir),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(results_df.to_string(index=False))
    print(f"\nSummary saved to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
