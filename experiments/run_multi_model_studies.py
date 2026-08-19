"""Utility script to orchestrate multi-model hedging strategy studies."""

from __future__ import annotations

import asyncio
import argparse
import datetime
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Sequence
from uuid import uuid4

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config
from main import ensure_output_directory, save_simulation_results
from src.core.coordinator import Coordinator
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig
from src.utils import llm_utils

# Ensure deterministic simulations
np.random.seed(42)
random.seed(42)

OUTPUT_ROOT = config.RESULTS_DIR
EXPERIMENTS_ROOT = getattr(
    config, "PAKDD_OUTPUT_DIR", os.path.join(OUTPUT_ROOT, "experiments")
)
PAKDD_ROOT = EXPERIMENTS_ROOT
DEFAULT_GPT_MODEL = getattr(config, "DEFAULT_GPT_MODEL", "gpt-5")


@dataclass
class StudyRunSpec:
    """Description of a single simulation run."""

    name: str
    hedging_strategies: Sequence[str]
    provider: str = "openai"
    model: str = DEFAULT_GPT_MODEL
    use_memory_rl: bool = True
    temperature: float = 0.0
    output_dir: str | None = None

    def as_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["hedging_strategies"] = list(self.hedging_strategies)
        return data


def _apply_temperature_override(temperature: float) -> None:
    """Force llm_utils.get_llm_response to honour the requested temperature."""

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


def _run_simulation_worker(spec_dict: Dict[str, object]) -> Dict[str, object]:
    """Worker executed in a separate process for isolation."""

    provider = str(spec_dict["provider"]).lower()
    model = str(spec_dict["model"])
    hedging_strategies = list(spec_dict["hedging_strategies"])
    run_name = str(spec_dict["name"])
    use_memory_rl = bool(spec_dict.get("use_memory_rl", True))
    temperature = float(spec_dict.get("temperature", 0.0))
    run_output_dir = str(spec_dict.get("output_dir", os.path.join(OUTPUT_ROOT, run_name)))

    # Apply provider/model overrides locally to this process.
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    config.LLM_PROVIDER = provider
    config.LLM_MODEL = model

    config.DEFAULT_OUTPUT_DIR = run_output_dir

    _apply_temperature_override(temperature)

    sim_config = ExtendedSimulationConfig(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        initial_capital=config.INITIAL_CAPITAL,
        assets=config.ASSETS,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        emergency_threshold=config.EMERGENCY_THRESHOLD,
        use_llm=config.USE_LLM,
        hedging_strategies=hedging_strategies,
        use_memory_rl=use_memory_rl,
        equal_options_strategy_split=config.EQUAL_OPTIONS_STRATEGY_SPLIT,
        rule_based_strategy_split=False,
    )

    coordinator = Coordinator(sim_config)
    result = coordinator.run_simulation()

    summary = coordinator.get_performance_summary()
    final_value = result.final_portfolio.total_value if result else None

    ensure_output_directory(run_output_dir)
    coordinator.plot_performance(os.path.join(run_output_dir, "performance.png"))

    if result:
        save_simulation_results(result, run_output_dir)

    return {
        "run": run_name,
        "provider": provider,
        "model": model,
        "hedging_strategies": hedging_strategies,
        "use_memory_rl": use_memory_rl,
        "temperature": temperature,
        "final_value": final_value,
        "summary": summary,
        "output_dir": run_output_dir,
    }


def _default_executor(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max_workers)


async def _run_group(description: str, run_specs: Iterable[StudyRunSpec]) -> None:
    specs = list(run_specs)
    if not specs:
        print(f"No runs configured for {description}.")
        return

    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)

    group_output_root = None
    if any(spec.output_dir is None for spec in specs):
        session_prefix = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
        group_output_root = os.path.join(OUTPUT_ROOT, session_prefix)
        ensure_output_directory(group_output_root)

    max_workers = min(len(specs), os.cpu_count() or 1)
    loop = asyncio.get_running_loop()
    with _default_executor(max_workers=max_workers) as executor:
        tasks = []
        for spec in specs:
            spec_dict = spec.as_dict()
            if spec.output_dir is not None:
                run_output_dir = spec.output_dir
            else:
                if group_output_root is None:
                    raise RuntimeError("Group output root was not created for defaulted spec.")
                run_output_dir = os.path.join(group_output_root, spec.name)
            spec_dict["output_dir"] = run_output_dir
            tasks.append(loop.run_in_executor(executor, _run_simulation_worker, spec_dict))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            print(f"Run failed with error: {item}")
            continue

        print(f"Run '{item['run']}' ({item['provider']}:{item['model']}) completed.")
        print(f"  Hedging strategies: {', '.join(item['hedging_strategies'])}")
        print(f"  Use memory RL: {item['use_memory_rl']}")
        print(f"  Temperature: {item['temperature']}")
        if item.get("final_value") is not None:
            print(f"  Final portfolio value: ${item['final_value']:,.2f}")
        for metric, value in item.get("summary", {}).items():
            print(f"  {metric}: {value}")
        if item.get("output_dir"):
            print(f"  Results saved to: {item['output_dir']}")


async def run_individual_strategy_studies() -> None:
    """Run each hedging strategy individually using GPT."""

    strategies = ["collar", "straddle", "delta-neutral"]
    specs = [
        StudyRunSpec(
            name=f"single-{strategy.replace('-', '')}",
            hedging_strategies=[strategy],
            provider="openai",
            model=DEFAULT_GPT_MODEL,
            output_dir=os.path.join(EXPERIMENTS_ROOT, strategy),
        )
        for strategy in strategies
    ]
    await _run_group("Single strategy ablation (GPT)", specs)


async def run_all_strategy_study() -> None:
    """Run all hedging strategies together using GPT."""
    specs = [
        StudyRunSpec(
            name="all-strategies-30",
            hedging_strategies=["collar", "straddle", "delta-neutral"],
            provider="openai",
            model=DEFAULT_GPT_MODEL,
            output_dir=os.path.join(EXPERIMENTS_ROOT, "collar-straddle-delta-neutral"),
        )
    ]
    await _run_group("All strategies combined (GPT)", specs)


async def run_pair_strategy_studies() -> None:
    """Run pairwise combinations of hedging strategies using GPT."""

    base = ["collar", "straddle", "delta-neutral"]
    combos = combinations(base, 2)
    specs = [
        StudyRunSpec(
            name="pair-" + "-".join(part.replace("-", "") for part in combo),
            hedging_strategies=list(combo),
            provider="openai",
            model=DEFAULT_GPT_MODEL,
            output_dir=os.path.join(PAKDD_ROOT, "-".join(combo)),
        )
        for combo in combos
    ]
    await _run_group("Pairwise strategy ablation (GPT)", specs)


async def run_provider_comparison_study() -> None:
    """Compare GPT without memory RL against Gemini models."""

    specs = [
        StudyRunSpec(
            name="gpt-no-memory",
            hedging_strategies=config.HEDGING_STRATEGIES,
            provider="openai",
            model=DEFAULT_GPT_MODEL,
            use_memory_rl=False,
            output_dir=os.path.join(PAKDD_ROOT, "gpt-no-memory"),
        ),
        StudyRunSpec(
            name="gemini-25-flash",
            hedging_strategies=config.HEDGING_STRATEGIES,
            provider="gemini",
            model="gemini-2.5-flash-lite",
            output_dir=os.path.join(PAKDD_ROOT, "gemini-2-5-flash"),
        ),
        StudyRunSpec(
            name="gemini-flash-lite",
            hedging_strategies=config.HEDGING_STRATEGIES,
            provider="gemini",
            model="gemini-2.5-flash-lite",
            output_dir=os.path.join(PAKDD_ROOT, "gemini-2-5-flash-lite"),
        ),
    ]
    await _run_group("Provider comparison (GPT vs Gemini)", specs)


async def run_coordinator_choice_study() -> None:
    """Run the coordinator-choice and stochasticity variants reported in the paper."""

    specs = [
        StudyRunSpec(
            name="gpt-5-nano",
            hedging_strategies=config.HEDGING_STRATEGIES,
            provider="openai",
            model="gpt-5-nano",
            output_dir=os.path.join(PAKDD_ROOT, "gpt-5-nano"),
        ),
        StudyRunSpec(
            name="gemini-2-5-flash-lite",
            hedging_strategies=config.HEDGING_STRATEGIES,
            provider="gemini",
            model="gemini-2.5-flash-lite",
            output_dir=os.path.join(PAKDD_ROOT, "gemini-2-5-flash-lite"),
        ),
        StudyRunSpec(
            name="gpt-4o-mini",
            hedging_strategies=config.HEDGING_STRATEGIES,
            provider="openai",
            model="gpt-4o-mini",
            output_dir=os.path.join(PAKDD_ROOT, "gpt-4o-mini"),
        ),
        StudyRunSpec(
            name="gpt-5-t0-7",
            hedging_strategies=config.HEDGING_STRATEGIES,
            provider="openai",
            model=DEFAULT_GPT_MODEL,
            temperature=0.7,
            output_dir=os.path.join(PAKDD_ROOT, "gpt-5-t0-7"),
        ),
    ]
    await _run_group("Coordinator choice and stochasticity", specs)

async def run_spy_5day_test() -> None:
    """Run a simple 5-day test with SPY."""

    specs = [
        StudyRunSpec(
            name="spy-5day-test",
            hedging_strategies=["collar", "straddle", "delta-neutral"],
            provider="gemini",
            model=DEFAULT_GPT_MODEL,
            use_memory_rl=False,
            output_dir=os.path.join(EXPERIMENTS_ROOT, "spy-5day-test"),
        )
    ]
    await _run_group("SPY 5-day test (2024)", specs)

async def main() -> None:
    """Entrypoint for running selected study groups."""

    parser = argparse.ArgumentParser(description="Run MultiHedge study groups.")
    parser.add_argument(
        "--group",
        choices=[
            "spy-5day",
            "individual",
            "all-strategies",
            "pairs",
            "providers",
            "gpt-sweep",
            "paper-ablation",
        ],
        default="spy-5day",
        help="Study group to run. Use paper-ablation for the main strategy/model ablation groups.",
    )
    args = parser.parse_args()

    runners = {
        "spy-5day": [run_spy_5day_test],
        "individual": [run_individual_strategy_studies],
        "all-strategies": [run_all_strategy_study],
        "pairs": [run_pair_strategy_studies],
        "providers": [run_provider_comparison_study],
        "gpt-sweep": [run_coordinator_choice_study],
        "paper-ablation": [
            run_individual_strategy_studies,
            run_all_strategy_study,
            run_pair_strategy_studies,
            run_provider_comparison_study,
            run_coordinator_choice_study,
        ],
    }
    for runner in runners[args.group]:
        await runner()


if __name__ == "__main__":
    asyncio.run(main())
