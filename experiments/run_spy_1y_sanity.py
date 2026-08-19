"""SPY one-year sanity check with real option chains + Buy&Hold."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from main import ensure_output_directory, save_simulation_results
from src.core.coordinator import Coordinator
from src.core.data_manager import DataManager
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig
from src.utils import llm_utils

np.random.seed(42)
random.seed(42)


def _apply_temperature_override(temperature: float) -> None:
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


def compute_metrics(perf_df: pd.DataFrame) -> dict:
    r = perf_df["daily_return"].to_numpy()
    idx = pd.to_datetime(perf_df["date"])
    monthly = pd.Series(perf_df["daily_return"].to_numpy(), index=idx).add(1).resample("ME").prod().sub(1)
    cvar_slice = r[r <= np.quantile(r, 0.05)]
    return {
        "TotalReturn": float(np.prod(1 + r) - 1),
        "Sharpe": float(np.sqrt(252) * r.mean() / (r.std() + 1e-12)),
        "MaxDD": float(perf_df["drawdown"].max()),
        "CVaR95": float(cvar_slice.mean() if len(cvar_slice) else 0.0),
        "WorstMonth": float(monthly.min() if len(monthly) else 0.0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", type=lambda x: date.fromisoformat(x), default=date(2022, 1, 1))
    p.add_argument("--end-date", type=lambda x: date.fromisoformat(x), default=date(2022, 12, 31))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default=os.path.join(config.PAKDD_OUTPUT_DIR, "spy-1y-sanity-43"))
    args = p.parse_args()

    dm = DataManager("data")
    prices = dm.load_stock_data_as_dataframe("SPY", args.start_date, args.end_date)
    if prices.empty:
        raise FileNotFoundError("SPY.csv not found in data/. Real SPY data is required for sanity run.")

    _apply_temperature_override(args.temperature)

    cfg = ExtendedSimulationConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=config.INITIAL_CAPITAL,
        assets=["SPY"],
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        emergency_threshold=config.EMERGENCY_THRESHOLD,
        use_llm=True,
        use_memory_rl=False,
        hedging_strategies=config.HEDGING_STRATEGIES,
        equal_options_strategy_split=config.EQUAL_OPTIONS_STRATEGY_SPLIT,
    )
    coord = Coordinator(cfg, data_provider=dm)
    result = coord.run_simulation()

    output_dir = args.output_dir
    ensure_output_directory(output_dir)
    save_simulation_results(result, output_dir)

    mh = compute_metrics(pd.DataFrame([d.model_dump() for d in result.daily_performances]))

    close = prices["Close"].to_numpy()
    bh_ret = np.zeros(len(close))
    bh_ret[1:] = close[1:] / close[:-1] - 1
    bh_curve = np.cumprod(1 + bh_ret)
    bh_df = pd.DataFrame(
        {
            "date": prices["Date"],
            "daily_return": bh_ret,
            "drawdown": 1 - bh_curve / np.maximum.accumulate(bh_curve),
        }
    )
    bh = compute_metrics(bh_df)

    out = pd.DataFrame([
        {"Strategy": "MultiHedge", **mh},
        {"Strategy": "Buy&Hold", **bh},
    ])
    out_path = os.path.join(output_dir, "spy_1y_sanity_metrics.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved metrics to: {out_path}")
    # plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(prices["Date"], np.cumprod(1 + bh_ret), label="Buy&Hold")
    mh_returns = np.array(
        [d.daily_return for d in result.daily_performances]
    )
    plt.plot(
        prices["Date"],
        np.cumprod(1 + mh_returns),
        label="MultiHedge"
    )
    plt.legend()
    plt.title("SPY 1-Year Sanity Check: Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt_path = os.path.join(
        output_dir,
        "spy_1y_sanity_cumulative_returns.png"
    )
    plt.savefig(plt_path)
    plt.show()
    print(f"Saved cumulative returns plot to: {plt_path}")


if __name__ == "__main__":
    main()
