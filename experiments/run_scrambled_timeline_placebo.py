"""Scrambled-timeline placebo: block permutation on trading days with reruns."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from datetime import date
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from main import ensure_output_directory, save_simulation_results
from src.core.coordinator import Coordinator
from src.core.data_manager import DataManager
from src.core.data_provider import ScrambledCalendarDataProvider
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig

np.random.seed(42)
random.seed(42)


class ScrambledDataCache:
    """Handles creation and caching of scrambled data mappings."""

    def __init__(self, cache_dir: str = None):
        """Initialize cache directory."""
        if cache_dir is None:
            cache_dir = os.path.join(config.PAKDD_OUTPUT_DIR, ".scrambled_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create_permutation(self, data_list: List, cache_name: str, block_len: int, seed: int, run_idx: int) -> List:
        """
        1 & 2: Check if scrambled data exists. If yes load, if not create.
        """
        cache_file = self.cache_dir / f"{cache_name}.pkl"

        if cache_file.exists():
            print(f"[CACHE] Loading cached permutation from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"[CACHE] Creating new block permutation: {cache_file}")
            # ACTUALLY USE THE BLOCK LOGIC HERE
            permuted = build_scrambled_calendar(data_list, block_len, seed, run_idx)

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(permuted, f)
            print(f"[CACHE] Saved permutation to {cache_file}")

            return permuted

def compute_metrics(perf_df: pd.DataFrame) -> dict:
    returns = perf_df["daily_return"].to_numpy()
    idx = pd.to_datetime(perf_df["date"])
    monthly = pd.Series(perf_df["daily_return"].to_numpy(), index=idx).add(1).resample("ME").prod().sub(1)
    cvar_slice = returns[returns <= np.quantile(returns, 0.05)]
    return {
        "TotalReturn": float(np.prod(1 + returns) - 1),
        "Sharpe": float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-12)),
        "MaxDD": float(perf_df["drawdown"].max()),
        "CVaR95": float(cvar_slice.mean() if len(cvar_slice) else 0.0),
        "WorstMonth": float(monthly.min() if len(monthly) else 0.0),
    }


def run_with_dates(coordinator: Coordinator, trading_days: list[date]):
    coordinator.current_date = trading_days[0]
    coordinator.last_rebalance_date = None
    for d in trading_days:
        first = coordinator.config.assets[0]
        if not coordinator.data_manager.check_stock_data_exists(first, d):
            continue
        coordinator.run_single_day(d)

    final_portfolio = coordinator.portfolio_manager.get_portfolio_snapshot(trading_days[-1])
    return coordinator.performance_tracker.generate_simulation_result(
        coordinator.config,
        final_portfolio,
        coordinator,
    )


def build_scrambled_calendar(days: list[date], block_len: int, seed: int, run_idx: int) -> list[date]:
    blocks = [days[i : i + block_len] for i in range(0, len(days), block_len)]
    rng = np.random.default_rng(seed + run_idx)
    rng.shuffle(blocks)
    return [d for block in blocks for d in block]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", nargs="+", default=["AAPL"])
    p.add_argument("--start-date", type=lambda x: date.fromisoformat(x), default=date(2021, 1, 1))
    p.add_argument("--end-date", type=lambda x: date.fromisoformat(x), default=date(2023, 12, 31))
    p.add_argument("--block-len", type=int, default=10)
    p.add_argument("--n-scrambles", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=os.path.join(config.PAKDD_OUTPUT_DIR, "scrambled-timeline-placebo-17"))
    args = p.parse_args()

    base = DataManager("data")
    stock_df = base.load_stock_data_as_dataframe(args.assets[0], args.start_date, args.end_date)
    if stock_df.empty:
        raise FileNotFoundError("No stock data found in data/. Provide real historical files first.")
    trading_days = [d.date() for d in stock_df["Date"]]

    cfg = ExtendedSimulationConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=config.INITIAL_CAPITAL,
        assets=args.assets,
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        emergency_threshold=config.EMERGENCY_THRESHOLD,
        use_llm=True,
        use_memory_rl=False,
        hedging_strategies=config.HEDGING_STRATEGIES,
        equal_options_strategy_split=config.EQUAL_OPTIONS_STRATEGY_SPLIT,
    )

    output_dir = args.output_dir
    ensure_output_directory(output_dir)

    # Initialize cache for scrambled calendars
    cache = ScrambledDataCache()

    # Buy & Hold: (last close - first close) / first close
    bah_close = stock_df["Close"].to_numpy().astype(float)
    bah_total_return = float(bah_close[-1] / bah_close[0] - 1)
    bah_dates = [d.date() for d in stock_df["Date"]]
    bah_cumret = bah_close / bah_close[0] - 1

    # Phase 1: Pre-generate all scrambled calendars before any simulation runs
    print(f"\n=== Phase 1: Generating {args.n_scrambles} scrambled calendars ===")
    all_scrambled = []
    for i in range(args.n_scrambles):
        cache_name = f"scrambled_{args.start_date}_{args.end_date}_seed{args.seed}_block{args.block_len}_run{i}"
        scrambled = cache.get_or_create_permutation(
            trading_days,
            cache_name=cache_name,
            block_len=args.block_len,
            seed=args.seed,
            run_idx=i,
        )
        all_scrambled.append(scrambled)
    print(f"=== Phase 1 complete: {len(all_scrambled)} calendars ready ===\n")

    # Phase 2: Run simulations using the pre-generated scrambled calendars
    print(f"=== Phase 2: Running {args.n_scrambles} scrambled simulations ===")
    rows = []
    missing_logs = []
    run0_perf_df = None  # Captured for the plot; avoids re-running later

    for i, scrambled in enumerate(all_scrambled):
        provider = ScrambledCalendarDataProvider(
            base=base,
            trading_days=trading_days,
            scrambled_days=scrambled,
            log_prefix=f"run={i} ",
            scrambled_option_id_fix=True,
        )
        coord = Coordinator(cfg, data_provider=provider)
        res = run_with_dates(coord, trading_days)
        save_simulation_results(res, os.path.join(output_dir, f"scramble_{i:02d}"))

        perf_df = pd.DataFrame([d.model_dump() for d in res.daily_performances])
        if i == 0:
            run0_perf_df = perf_df

        metrics = compute_metrics(perf_df)
        metrics["run"] = i
        rows.append(metrics)
        missing_logs.extend(provider.missing_option_chain_logs)
    print(f"=== Phase 2 complete ===\n")

    scr_df = pd.DataFrame(rows)
    compact = pd.DataFrame(
        {
            "metric": ["Sharpe", "MaxDD", "CVaR95", "WorstMonth", "TotalReturn"],
            "buy_and_hold": ["N/A", "N/A", "N/A", "N/A", f"{bah_total_return:.4f}"],
            "scrambled_mean±std": [
                f"{scr_df[k].mean():.4f}±{scr_df[k].std(ddof=1):.4f}" for k in ["Sharpe", "MaxDD", "CVaR95", "WorstMonth", "TotalReturn"]
            ],
        }
    )

    detail_path = Path(output_dir) / "scrambled_timeline_placebo.csv"
    compact_path = Path(output_dir) / "scrambled_timeline_placebo_compact.csv"
    scr_df.to_csv(detail_path, index=False)
    compact.to_csv(compact_path, index=False)

    if missing_logs:
        with open(Path(output_dir) / "missing_option_chain.log", "w", encoding="utf-8") as f:
            f.write("\n".join(missing_logs) + "\n")

    print(f"Buy & Hold Total Return: {bah_total_return:.4f}")
    print(f"Scrambled Total Return:  {scr_df['TotalReturn'].mean():.4f}±{scr_df['TotalReturn'].std(ddof=1):.4f}")

    print(f"Scrambled Sharpe: {scr_df['Sharpe'].mean():.4f}±{scr_df['Sharpe'].std(ddof=1):.4f}")
    print(f"Scrambled MaxDD:  {scr_df['MaxDD'].mean():.4f}±{scr_df['MaxDD'].std(ddof=1):.4f}")

    # Plot using the already-executed run 0 results (no redundant coordinator setup)
    if run0_perf_df is not None:
        ax = plt.subplots(figsize=(12, 6))[1]
        ax.plot(bah_dates, bah_cumret, label="Buy & Hold", linewidth=2)
        ax.plot(run0_perf_df["date"], run0_perf_df["cumulative_return"], label="Scrambled (Run 0)", linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "baseline_vs_scramble_run0.png", dpi=100)
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
