"""Synthetic 3-regime no-leakage experiment with MultiHedge + baselines."""

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
from src.core.data_provider import SyntheticRegimeDataProvider
from src.memory_rl.memory_rl_integration import ExtendedSimulationConfig

np.random.seed(42)
random.seed(42)


def metrics_from_returns(returns: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = 1 - equity / peak
    month = pd.Series(returns, index=dates).add(1).resample("ME").prod().sub(1)
    return {
        "TotalReturn": float(equity[-1] - 1),
        "Sharpe": float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-12)),
        "MaxDD": float(dd.max()),
        "CVaR95": float(returns[returns <= np.quantile(returns, 0.05)].mean()),
        "WorstMonth": float(month.min() if len(month) else 0.0),
    }


def buy_and_hold_returns(prices: np.ndarray) -> np.ndarray:
    r = np.zeros(len(prices))
    r[1:] = prices[1:] / prices[:-1] - 1
    return r


def collar_heuristic_returns(prices: np.ndarray) -> np.ndarray:
    spot_rets = buy_and_hold_returns(prices)
    hedged = np.clip(spot_rets, -0.02, 0.018)
    hedged[1:] -= 0.0006
    return hedged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=786, help="Number of trading days to simulate (default: 3 years)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticker", default="SYN")
    p.add_argument("--output-dir", default=os.path.join(config.PAKDD_OUTPUT_DIR, "synthetic-3-regime-16"))
    p.add_argument("--export-data", default=True, help="Export synthetic OHLCV CSV + options DB")
    p.add_argument("--data-dir", default=config.DATA_DIR, help="Target data directory for exports")
    args = p.parse_args()

    data_dir = args.data_dir
    stock_csv = os.path.join(data_dir, f"{args.ticker}.csv")
    options_db = os.path.join(data_dir, f"{args.ticker.lower()}_options.db")
    use_existing = args.export_data and os.path.exists(stock_csv) and os.path.exists(options_db)

    if use_existing:
        provider = DataManager(data_dir)
        stock_df = provider.load_stock_data_as_dataframe(args.ticker)
    else:
        provider = SyntheticRegimeDataProvider(
            ticker=args.ticker,
            length=args.length,
            seed=args.seed,
            start_date=date(2021, 1, 4),
        )
        stock_df = provider.df.copy()
        if args.export_data:
            paths = provider.export_market_data(data_dir)
            print(f"Exported synthetic data to: {paths}")

    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    days = [d.date() for d in stock_df["Date"]]
    cfg = ExtendedSimulationConfig(
        start_date=days[0],
        end_date=days[-1],
        initial_capital=config.INITIAL_CAPITAL,
        assets=[args.ticker],
        rebalance_frequency=config.REBALANCE_FREQUENCY,
        emergency_threshold=config.EMERGENCY_THRESHOLD,
        use_llm=True,
        use_memory_rl=False,
        hedging_strategies=config.HEDGING_STRATEGIES,
        equal_options_strategy_split=config.EQUAL_OPTIONS_STRATEGY_SPLIT,
    )

    coord = Coordinator(cfg, data_provider=provider)
    coord.run_simulation()

    final_portfolio = coord.portfolio_manager.get_portfolio_snapshot(days[-1])
    result = coord.performance_tracker.generate_simulation_result(cfg, final_portfolio, coord)

    output_dir = args.output_dir
    ensure_output_directory(output_dir)
    save_simulation_results(result, output_dir)

    mh_perf = pd.DataFrame([d.model_dump() for d in result.daily_performances])
    mh_ret = mh_perf["daily_return"].to_numpy()
    dates = pd.DatetimeIndex(stock_df["Date"])

    prices = stock_df["Close"].to_numpy()
    bh_ret = buy_and_hold_returns(prices)
    collar_ret = collar_heuristic_returns(prices)

    table = pd.DataFrame(
        [
            {"Strategy": "MultiHedge", **metrics_from_returns(mh_ret, dates)},
            {"Strategy": "Buy&Hold", **metrics_from_returns(bh_ret, dates)},
            {
                "Strategy": "ProtectiveCollarHeuristic",
                **metrics_from_returns(collar_ret, dates),
            },
        ]
    )
    table_path = os.path.join(output_dir, "synthetic_3_regime_metrics.csv")
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved metrics to: {table_path}")

    # plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(dates, np.cumsum(mh_ret), label="MultiHedge")
    plt.plot(dates, np.cumsum(bh_ret), label="Buy&Hold")
    plt.plot(dates, np.cumsum(collar_ret), label="ProtectiveCollarHeuristic")
    plt.xlabel("Date")
    plt.ylabel("Total Return")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "synthetic_3_regime_returns.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved returns plot to: {plot_path}")


if __name__ == "__main__":
    main()
