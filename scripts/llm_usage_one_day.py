"""Command-line entry point for single-day LLM usage summaries.

This script runs the standard simulation pipeline for a single trading day and
summarises large language model (LLM) usage metrics that were collected during
that run. By default the configuration from :mod:`config` is used and the first
available trading day in the configured date range is selected.

Usage examples
--------------

Run for the next available trading day::

    python scripts/llm_usage_one_day.py

Run for a specific date::

    python scripts/llm_usage_one_day.py --date 2020-01-06

To override the output directory containing ``llm_usage.jsonl`` provide the
``--usage-log`` argument.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np
import random

# Ensure project root is on sys.path so top-level modules (like config.py)
# can be imported when running this script directly (python scripts/...).
# When a script is executed, sys.path[0] is the script's directory (scripts/),
# which is why the repository root (where config.py lives) isn't found.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ensure deterministic simulations and summarisation
np.random.seed(42)
random.seed(42)

import config
from src.core.coordinator import Coordinator
from src.core.data_manager import DataManager
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig
from src.utils.llm_utils import LLM_USAGE_LOG_FILENAME


@dataclass
class UsageTotals:
    """Simple container for usage summary statistics."""

    entries: int = 0
    latency_sum: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def update(self, entry: dict) -> None:
        latency = entry.get("latency_seconds")
        if latency is not None:
            self.latency_sum += float(latency)
        prompt = entry.get("prompt_tokens")
        if prompt is not None:
            self.prompt_tokens += int(prompt)
        completion = entry.get("candidates_tokens")
        if completion is not None:
            self.completion_tokens += int(completion)
        total = entry.get("total_tokens")
        if total is not None:
            self.total_tokens += int(total)
        self.entries += 1

    @property
    def latency_average(self) -> Optional[float]:
        if self.entries == 0:
            return None
        return self.latency_sum / self.entries


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date format '{value}'. Use YYYY-MM-DD."
        ) from exc


def _resolve_trading_day(
    data_manager: DataManager,
    assets: Iterable[str],
    start: date,
    end: date,
    requested: Optional[date] = None,
) -> date:
    """Return the first trading day with data within the configured range."""

    if requested and (requested < start or requested > end):
        raise ValueError(
            f"Requested date {requested} is outside the configured range {start} to {end}."
        )

    first_asset = next(iter(assets), None)
    if first_asset is None:
        raise ValueError("No assets configured; unable to determine trading day.")

    def has_data(target: date) -> bool:
        return data_manager.check_stock_data_exists(first_asset, target)

    current = requested or start
    while current <= end:
        if has_data(current):
            return current
        current += timedelta(days=1)

    raise RuntimeError(
        "Unable to locate a trading day with available stock data in the configured range."
    )


def _build_config(target_day: date) -> ExtendedSimulationConfig:
    """Construct an :class:`ExtendedSimulationConfig` for ``target_day``."""

    kwargs = dict(
        start_date=target_day,
        end_date=target_day,
        initial_capital=config.INITIAL_CAPITAL,
        assets=list(config.ASSETS),
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        emergency_threshold=config.EMERGENCY_THRESHOLD,
        use_llm=config.USE_LLM,
        hedging_strategies=list(config.HEDGING_STRATEGIES),
        use_memory_rl=getattr(config, "use_memory_rl", False),
        equal_options_strategy_split=getattr(
            config, "EQUAL_OPTIONS_STRATEGY_SPLIT", False
        ),
    )

    rule_based_split = getattr(config, "RULE_BASED_STRATEGY_SPLIT", None)
    if rule_based_split is not None:
        kwargs["rule_based_strategy_split"] = rule_based_split

    return ExtendedSimulationConfig(**kwargs)


def _load_usage_entries(log_path: str) -> List[dict]:
    if not os.path.exists(log_path):
        return []

    entries: List[dict] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _filter_entries_for_day(entries: Iterable[dict], target: date) -> List[dict]:
    target_str = target.isoformat()
    filtered: List[dict] = []
    for entry in entries:
        prompt = entry.get("prompt") or ""
        response = entry.get("response") or ""
        if target_str in prompt or target_str in response:
            filtered.append(entry)
    return filtered


def _summarise_usage(entries: Iterable[dict]) -> UsageTotals:
    totals = UsageTotals()
    for entry in entries:
        totals.update(entry)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single-day simulation and report LLM usage statistics."
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        help="Target trading day in YYYY-MM-DD format. Defaults to the next available trading day.",
    )
    parser.add_argument(
        "--usage-log",
        type=str,
        default=None,
        help="Path to the llm_usage.jsonl file. Defaults to config.DEFAULT_OUTPUT_DIR/llm_usage.jsonl.",
    )
    args = parser.parse_args()

    data_manager = DataManager(config.DATA_DIR)
    trading_day = _resolve_trading_day(
        data_manager=data_manager,
        assets=config.ASSETS,
        start=config.START_DATE,
        end=config.END_DATE,
        requested=args.date,
    )

    print(f"Running simulation for trading day {trading_day}...")
    sim_config = _build_config(trading_day)
    coordinator = Coordinator(sim_config)
    result = coordinator.run_simulation()
    print("Simulation completed. Final portfolio value:", result.final_portfolio.total_value)

    usage_log = args.usage_log or os.path.join(
        getattr(config, "DEFAULT_OUTPUT_DIR", "."), LLM_USAGE_LOG_FILENAME
    )
    entries = _load_usage_entries(usage_log)
    if not entries:
        print(f"No usage data found at {usage_log}.")
        return 0

    filtered_entries = _filter_entries_for_day(entries, trading_day)
    if not filtered_entries:
        print(f"No usage entries matched trading day {trading_day} in {usage_log}.")
        return 0

    totals = _summarise_usage(filtered_entries)

    print("\nLLM usage summary for", trading_day)
    print("Entries:", totals.entries)
    print("Total latency (s):", round(totals.latency_sum, 3))
    if totals.latency_average is not None:
        print("Average latency (s):", round(totals.latency_average, 3))
    print("Prompt tokens:", totals.prompt_tokens)
    print("Completion tokens:", totals.completion_tokens)
    print("Total tokens:", totals.total_tokens)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
