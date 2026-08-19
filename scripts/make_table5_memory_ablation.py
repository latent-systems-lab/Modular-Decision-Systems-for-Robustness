#!/usr/bin/env python3
"""Aggregate memory-controller ablation results for paper Table 5.

The recovered experiment directories live under results/experiments/results-*.
If a target directory contains summary.json, values are read from it. Otherwise,
metrics are computed from portfolio_history.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


RETRIEVAL_ONLY_RUN = "results-29"
RANDOM_RETRIEVAL_RUNS = ("results-30", "results-21", "results-14")

# IMPORTANT SOURCE NOTE
# ---------------------
# The paper's final benchmark/memory-table numbers for Full Pipeline/MultiHedge
# do NOT come from the currently checked-out results/final/multihedge copy files.
# They come from the historical paper-results folder:
#
#   results/pakdd_final/multihedge/
#
# In this checkout that folder may be absent. The script therefore falls back to
# reading those exact files from git history at HISTORICAL_COMMIT via:
#
#   git show 88215eeb76dca627de609379a3339b67aa155837:<path>
#
# Correct Full Pipeline files:
#   results/pakdd_final/multihedge/portfolio_history.json
#   results/pakdd_final/multihedge/portfolio_history_copy_compressed.json
#   results/pakdd_final/multihedge/portfolio_history_copy_2_compressed.json
#
# These reproduce the paper row:
#   SR 1.69 ± 0.04, TR 152.08 ± 25.26, MD 16.22 ± 3.28.
HISTORICAL_COMMIT = "88215eeb76dca627de609379a3339b67aa155837"
FULL_PIPELINE_PATHS = (
    Path("results") / "pakdd_final" / "multihedge" / "portfolio_history.json",
    Path("results") / "pakdd_final" / "multihedge" / "portfolio_history_copy_compressed.json",
    Path("results") / "pakdd_final" / "multihedge" / "portfolio_history_copy_2_compressed.json",
)
NO_MEMORY_PATHS = (
    Path("results") / "pakdd_final" / "ablation" / "NoMemory" / "portfolio_history.json",
    Path("results") / "pakdd_final" / "ablation" / "NoMemory" / "portfolio_history copy.json",
    Path("results") / "pakdd_final" / "ablation" / "NoMemory" / "portfolio_history copy 2.json",
)
DEFAULT_BASE_DIRS = (
    Path("results") / "experiments",
    Path("results") / "memory_controller_ablations",
)

METRICS = (
    ("SR", "annualized_sharpe", False),
    ("Total Return (%)", "total_return", True),
    ("Max Drawdown (%)", "max_drawdown", True),
    ("CVaR 0.95 (%)", "cvar_0.95", True),
    ("TDD >10% (%)", "time_in_drawdown_10", True),
)


def _find_run_dir(run_name: str, base_dirs: tuple[Path, ...]) -> Path:
    for base_dir in base_dirs:
        candidate = base_dir / run_name
        if candidate.exists():
            return candidate
    searched = ", ".join(str(base_dir / run_name) for base_dir in base_dirs)
    raise FileNotFoundError(f"Could not find {run_name}. Searched: {searched}")


def _read_portfolio_records(history_path: Path) -> list[dict]:
    if not history_path.exists():
        git_path = history_path.as_posix()
        try:
            payload_text = subprocess.run(
                ["git", "show", f"{HISTORICAL_COMMIT}:{git_path}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise FileNotFoundError(
                f"Could not find {history_path} locally or in {HISTORICAL_COMMIT}"
            ) from exc
        payload = json.loads(payload_text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("history", "portfolio_history", "records", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported portfolio history structure in {history_path}")

    if history_path.suffix.lower() == ".jsonl":
        records = []
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("history", "portfolio_history", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported portfolio history structure in {history_path}")


def _load_history(history_path: Path) -> pd.DataFrame:
    rows = []
    for item in _read_portfolio_records(history_path):
        if not isinstance(item, dict) or "date" not in item:
            continue
        value = item.get("total_value", item.get("portfolio_value", item.get("value")))
        if value is None:
            continue
        rows.append(
            {
                "date": pd.to_datetime(item["date"]),
                "portfolio_value": float(value),
            }
            )

    history = pd.DataFrame(rows)
    if history.empty:
        raise ValueError(f"No usable portfolio rows in {history_path}")
    history = history.sort_values("date").reset_index(drop=True)
    daily_return = history["portfolio_value"].pct_change().fillna(0.0)
    history = history.assign(daily_return=daily_return)
    return history


def _annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if len(returns) < 2 or std == 0.0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(252.0))


def _cvar_95_loss_magnitude(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cutoff = returns.quantile(0.05)
    tail_mean = returns[returns <= cutoff].mean()
    if pd.isna(tail_mean):
        return 0.0
    return float(abs(tail_mean))


def _time_in_drawdown_10(values: pd.Series) -> float:
    running_peak = values.cummax()
    drawdown = (running_peak - values) / running_peak
    return float((drawdown >= 0.10).mean())


def _metrics_from_history(run_dir: Path) -> dict[str, float]:
    return _metrics_from_history_path(run_dir / "portfolio_history.jsonl")


def _metrics_from_history_path(history_path: Path) -> dict[str, float]:
    history = _load_history(history_path)
    values = history["portfolio_value"].astype(float)
    returns = history["daily_return"].astype(float)
    running_peak = values.cummax()
    drawdown = (running_peak - values) / running_peak
    return {
        "annualized_sharpe": _annualized_sharpe(returns),
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "max_drawdown": float(drawdown.max()),
        "cvar_0.95": _cvar_95_loss_magnitude(returns),
        "time_in_drawdown_10": _time_in_drawdown_10(values),
    }


def _metrics_from_summary(run_dir: Path) -> dict[str, float]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    metrics = {
        "annualized_sharpe": float(
            summary.get("annualized_sharpe", summary.get("sharpe_ratio", 0.0))
        ),
        "total_return": float(summary.get("total_return", 0.0)),
        "max_drawdown": float(summary.get("max_drawdown", 0.0)),
    }

    if "cvar_0.95" in summary:
        metrics["cvar_0.95"] = abs(float(summary["cvar_0.95"]))
    elif "cvar_95" in summary:
        metrics["cvar_0.95"] = abs(float(summary["cvar_95"]))

    if "time_in_drawdown_10" in summary:
        metrics["time_in_drawdown_10"] = float(summary["time_in_drawdown_10"])

    missing = {"cvar_0.95", "time_in_drawdown_10"} - set(metrics)
    if missing and (run_dir / "portfolio_history.jsonl").exists():
        metrics.update(_metrics_from_history(run_dir))
    elif missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"{run_dir / 'summary.json'} is missing: {missing_text}")

    return metrics


def load_run_metrics(run_name: str, base_dirs: tuple[Path, ...]) -> dict[str, float]:
    run_dir = _find_run_dir(run_name, base_dirs)
    if (run_dir / "summary.json").exists():
        metrics = _metrics_from_summary(run_dir)
    else:
        metrics = _metrics_from_history(run_dir)
    metrics["run_name"] = run_name
    return metrics


def load_history_metrics(label: str, history_path: Path) -> dict[str, float]:
    metrics = _metrics_from_history_path(history_path)
    metrics["run_name"] = label
    return metrics


def load_history_metric_runs(label: str, history_paths: tuple[Path, ...]) -> list[dict[str, float]]:
    runs = []
    for index, history_path in enumerate(history_paths, start=1):
        runs.append(load_history_metrics(f"{label}_{index}", history_path))
    return runs


def _format_value(value: float, as_percent: bool) -> str:
    if as_percent:
        return f"{value * 100.0:.2f}"
    return f"{value:.3f}"


def _format_mean_std(values: list[float], as_percent: bool) -> str:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{_format_value(mean, as_percent)} ± {_format_value(std, as_percent)}"


def build_table(base_dirs: tuple[Path, ...]) -> pd.DataFrame:
    full_pipeline_runs = load_history_metric_runs("full_pipeline", FULL_PIPELINE_PATHS)
    no_memory_runs = load_history_metric_runs("no_memory", NO_MEMORY_PATHS)
    retrieval = load_run_metrics(RETRIEVAL_ONLY_RUN, base_dirs)
    random_runs = [load_run_metrics(run_name, base_dirs) for run_name in RANDOM_RETRIEVAL_RUNS]

    rows = []
    for label, key, as_percent in METRICS:
        rows.append(
            {
                "Metric": label,
                "Full Pipeline": _format_mean_std(
                    [float(run[key]) for run in full_pipeline_runs],
                    as_percent,
                ),
                "No Memory": _format_mean_std(
                    [float(run[key]) for run in no_memory_runs],
                    as_percent,
                ),
                "Retrieval-Only": _format_value(float(retrieval[key]), as_percent),
                "Random Retrieval": _format_mean_std(
                    [float(run[key]) for run in random_runs],
                    as_percent,
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the paper Table 5 aggregation for memory ablations."
    )
    parser.add_argument(
        "--base-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Directory containing results-* folders. Can be passed multiple times. "
            "Defaults to results/experiments and results/memory_controller_ablations."
        ),
    )
    args = parser.parse_args()

    base_dirs = tuple(args.base_dir) if args.base_dir else DEFAULT_BASE_DIRS
    table = build_table(base_dirs)

    print("\nTable 5: Memory Controller Ablation")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
