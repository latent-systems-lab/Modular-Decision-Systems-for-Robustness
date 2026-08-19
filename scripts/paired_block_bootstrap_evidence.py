#!/usr/bin/env python3
"""Paired circular block-bootstrap evidence for strategy robustness claims.

The script accepts either:

1. A single wide CSV with one date column and one column per strategy.
2. Multiple CSVs passed as ``Strategy Name=path/to/file.csv``.

Inputs may be daily returns or portfolio values. Portfolio values are converted
to daily returns before alignment and metric computation.

By default this script uses the paper's recovered ``results/pakdd_final``
inputs where available. That keeps the bootstrap evidence consistent with the
mean ± std values reported in the paper tables.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


FULL_PIPELINE = "Full Pipeline"
BASELINES = (
    "Buy & Hold",
    "PPO",
    "No Memory",
    "Retrieval-Only",
    "Random Retrieval",
)
BOOTSTRAP_METRICS = ("Sharpe", "MD", "CVaR", "Total Return")
DEFAULT_OUTPUT_CSV = Path("bootstrap_results.csv")
DEFAULT_START_DATE = "2021-01-04"
DEFAULT_END_DATE = "2023-12-29"
N_BOOTSTRAP = 10_000
BLOCK_SIZE = 20
SEED = 42

# If you prefer one wide file, set CSV_PATH to e.g. Path("data/returns.csv").
# Expected columns: date, Full Pipeline, Buy & Hold, PPO, No Memory,
# Retrieval-Only, Random Retrieval. Leave as None to use the result files below.
CSV_PATH: Path | None = None
INPUT_KIND = "values"
DATE_COL = "date"
OUTPUT_CSV = DEFAULT_OUTPUT_CSV
RESULTS_DIR = Path("results")
PAPER_RESULTS_DIR = RESULTS_DIR / "pakdd_final"



def returns_from_values(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert portfolio value series to daily returns.

    The first row has no prior value from which to compute a return, so it is
    left as NaN and dropped later after all strategies are aligned.
    """
    numeric_values = values.astype(float)
    return numeric_values.pct_change()


def _normalize_strategy_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").split()).strip()


def _canonical_strategy_name(name: str) -> str:
    normalized = _normalize_strategy_name(name)
    key = normalized.lower()
    aliases = {
        "full pipeline": "Full Pipeline",
        "buy hold": "Buy & Hold",
        "buy and hold": "Buy & Hold",
        "buy & hold": "Buy & Hold",
        "ppo": "PPO",
        "no memory": "No Memory",
        "retrieval only": "Retrieval-Only",
        "random retrieval": "Random Retrieval",
    }
    return aliases.get(key, str(name).strip())


def _display_name_from_path_name(name: str) -> str:
    aliases = {
        "bh": "Buy & Hold",
        "ew": "Equal Weight",
        "ppo": "PPO",
        "finrl": "FinRL",
        "finagent": "FinAgent",
        "multihedge": "Full Pipeline",
        "NoMemory": "No Memory",
    }
    if name in aliases:
        return aliases[name]
    return name.replace("_", " ").replace("-", " ").strip()


def _portfolio_files_in_dir(path: Path) -> list[Path]:
    patterns = (
        "performance.csv",
        "*portfolio_value*.csv",
        "*portfolio_values*.csv",
        "*account_value*.csv",
        "*values*.csv",
        "portfolio_history*.json",
        "portfolio_history*.jsonl",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    unique = sorted({file for file in files if file.is_file()})
    return unique


def discover_strategy_files(results_dir: Path = RESULTS_DIR) -> dict[str, str | list[str]]:
    """Discover default strategy inputs from the paper results directory."""
    strategy_files: dict[str, str | list[str]] = {}
    paper_dir = results_dir / "pakdd_final"
    base_dir = paper_dir if paper_dir.exists() else results_dir / "final"

    full_pipeline_runs = [
        paper_dir / "multihedge" / "portfolio_history.json",
        paper_dir / "multihedge" / "portfolio_history_copy_compressed.json",
        paper_dir / "multihedge" / "portfolio_history_copy_2_compressed.json",
    ]
    existing_full_pipeline_runs = [str(path) for path in full_pipeline_runs if path.exists()]
    if existing_full_pipeline_runs:
        strategy_files[FULL_PIPELINE] = existing_full_pipeline_runs
    else:
        full_pipeline = base_dir / "multihedge" / "portfolio_history.json"
        if full_pipeline.exists():
            strategy_files[FULL_PIPELINE] = str(full_pipeline)

    benchmark_dir = base_dir / "benchmark"
    if benchmark_dir.exists():
        for child in sorted(benchmark_dir.iterdir()):
            if not child.is_dir():
                continue
            files = _portfolio_files_in_dir(child)
            if not files:
                continue
            name = _canonical_strategy_name(_display_name_from_path_name(child.name))
            strategy_files[name] = [str(path) for path in files] if len(files) > 1 else str(files[0])

    ablation_dir = base_dir / "ablation"
    if ablation_dir.exists():
        for child in sorted(ablation_dir.iterdir()):
            if not child.is_dir():
                continue
            files = _portfolio_files_in_dir(child)
            if not files:
                continue
            name = _canonical_strategy_name(_display_name_from_path_name(child.name))
            strategy_files[name] = [str(path) for path in files] if len(files) > 1 else str(files[0])

    retrieval_only = results_dir / "experiments" / "results-29"
    if retrieval_only.exists():
        strategy_files["Retrieval-Only"] = str(retrieval_only)

    random_retrieval = [
        results_dir / "experiments" / "results-30",
        results_dir / "experiments" / "results-21",
        results_dir / "experiments" / "results-14",
    ]
    existing_random_retrieval = [str(path) for path in random_retrieval if path.exists()]
    if existing_random_retrieval:
        strategy_files["Random Retrieval"] = existing_random_retrieval

    return strategy_files


def _resolve_column_case_insensitive(columns: Sequence[str], requested: str) -> str | None:
    if requested in columns:
        return requested
    requested_lower = requested.lower()
    for column in columns:
        if str(column).lower() == requested_lower:
            return str(column)
    return None


def _read_csv_with_date(path: Path, date_col: str) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    resolved_date_col = _resolve_column_case_insensitive(frame.columns, date_col)
    if resolved_date_col is None:
        raise ValueError(f"{path} is missing required date column {date_col!r}")
    frame = frame.copy()
    frame[resolved_date_col] = pd.to_datetime(frame[resolved_date_col])
    return frame.sort_values(resolved_date_col), resolved_date_col


def _read_json_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("history", "portfolio_history", "records", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Could not find a record list in {path}")


def _read_table_with_date(path: Path, date_col: str) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".csv":
        return _read_csv_with_date(path, date_col)
    if path.suffix.lower() in {".json", ".jsonl"}:
        frame = pd.DataFrame(_read_json_records(path))
        resolved_date_col = _resolve_column_case_insensitive(frame.columns, date_col)
        if resolved_date_col is None:
            raise ValueError(f"{path} is missing required date column {date_col!r}")
        frame = frame.copy()
        frame[resolved_date_col] = pd.to_datetime(frame[resolved_date_col])
        return frame.sort_values(resolved_date_col), resolved_date_col
    raise ValueError(f"Unsupported input file type for {path}. Use CSV, JSON, or JSONL.")


def _pick_series_column(frame: pd.DataFrame, date_col: str, requested_col: str | None, kind: str) -> str:
    if requested_col:
        if requested_col not in frame.columns:
            raise ValueError(f"Requested {kind} column {requested_col!r} was not found")
        return requested_col

    candidates_by_kind = {
        "returns": ("daily_return", "return", "returns", "ret", "strategy_return"),
        "values": (
            "portfolio_value",
            "total_value",
            "value",
            "equity",
            "wealth",
            "nav",
            "strategy_value",
        ),
    }
    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in candidates_by_kind[kind]:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    numeric_cols = [
        column
        for column in frame.columns
        if column != date_col and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Could not infer {kind} column. Pass --series-column for per-strategy "
        "CSVs, or --strategy-columns for a wide CSV."
    )


def _parse_strategy_file_specs(specs: Sequence[str]) -> dict[str, Path]:
    strategy_files: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --strategy-file value {spec!r}. Expected 'Strategy Name=path.csv'."
            )
        strategy, path_text = spec.split("=", 1)
        strategy_name = _canonical_strategy_name(strategy)
        if not strategy_name:
            raise ValueError(f"Invalid empty strategy name in {spec!r}")
        strategy_files[strategy_name] = Path(path_text).expanduser()
    return strategy_files


def _resolve_input_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        "performance.csv",
        "portfolio_history.jsonl",
        "portfolio_history.json",
        "test_portfolio_values.csv",
    )
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported portfolio file found in {path}")


def _series_from_path(
    path_like: str | Path,
    *,
    input_kind: str,
    date_col: str,
    series_col: str | None,
) -> pd.Series:
    path = _resolve_input_file(Path(path_like).expanduser())
    frame, resolved_date_col = _read_table_with_date(path, date_col)
    column = _pick_series_column(frame, resolved_date_col, series_col, input_kind)
    return frame.set_index(resolved_date_col)[column].astype(float).sort_index()


def _mean_normalized_value_series(
    paths: Sequence[str | Path],
    *,
    date_col: str,
    series_col: str | None,
) -> pd.Series:
    normalized = []
    for path in paths:
        series = _series_from_path(
            path,
            input_kind="values",
            date_col=date_col,
            series_col=series_col,
        )
        normalized.append(series / series.iloc[0])
    joined = pd.concat(normalized, axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError("Cannot build mean trajectory; repeated runs have no overlapping dates.")
    return joined.mean(axis=1) * 1_000_000.0


def _returns_from_path_spec(
    path_spec: str | Path | Sequence[str | Path],
    *,
    input_kind: str,
    date_col: str,
    series_col: str | None,
    start_date: str | None,
    end_date: str | None,
) -> list[pd.Series]:
    """Load each run separately so run-level table metrics stay reproducible."""
    paths: Sequence[str | Path]
    if isinstance(path_spec, (list, tuple)):
        paths = path_spec
    else:
        paths = [path_spec]

    run_returns = []
    for path in paths:
        series = _series_from_path(
            path,
            input_kind=input_kind,
            date_col=date_col,
            series_col=series_col,
        )
        if start_date:
            series = series[series.index >= pd.Timestamp(start_date)]
        if end_date:
            series = series[series.index <= pd.Timestamp(end_date)]
        if input_kind == "values":
            returns = returns_from_values(series)
        else:
            returns = series
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            raise ValueError(f"No usable returns in {path}")
        run_returns.append(returns.astype(float))
    return run_returns


def table_consistent_observed_improvements(
    strategy_files: Mapping[str, str | Path | Sequence[str | Path]],
    *,
    full_strategy: str,
    baselines: Sequence[str],
    input_kind: str,
    date_col: str,
    series_col: str | None,
    start_date: str | None,
    end_date: str | None,
) -> dict[tuple[str, str], float]:
    """Observed improvements from mean run-level metrics, matching paper tables.

    This differs from the bootstrap estimand when repeated runs are first
    averaged into one daily trajectory. In particular, maximum drawdown is
    nonlinear, so MD(mean trajectory) generally differs from mean(MD(run)).
    """
    metrics_by_strategy: dict[str, dict[str, float]] = {}
    needed = [full_strategy, *baselines]
    for strategy in needed:
        path_spec = strategy_files[strategy]
        runs = _returns_from_path_spec(
            path_spec,
            input_kind=input_kind,
            date_col=date_col,
            series_col=series_col,
            start_date=start_date,
            end_date=end_date,
        )
        run_metrics = [_compute_metrics_np(run.to_numpy(dtype=float)) for run in runs]
        metrics_by_strategy[strategy] = {
            metric: float(np.mean([run[metric] for run in run_metrics]))
            for metric in ("Sharpe", "MD", "VaR", "CVaR", "Total Return")
        }

    observed = {}
    full_metrics = metrics_by_strategy[full_strategy]
    for baseline in baselines:
        comparison = f"{full_strategy} vs {baseline}"
        improvements = _metric_improvements(full_metrics, metrics_by_strategy[baseline])
        for metric, value in improvements.items():
            observed[(comparison, metric)] = value
    return observed


def load_and_align_data(
    *,
    wide_csv: Path | None = None,
    strategy_files: Mapping[str, str | Path | Sequence[str | Path]] | None = None,
    input_kind: str = "returns",
    date_col: str = "date",
    series_col: str | None = None,
    strategy_columns: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load strategy return/value data, align by date, and drop missing rows.

    Parameters
    ----------
    wide_csv:
        Optional CSV containing all strategies as columns.
    strategy_files:
        Optional mapping from strategy name to one CSV per strategy.
    input_kind:
        Either ``returns`` or ``values``.
    date_col:
        Name of the date column in each CSV.
    series_col:
        Column to read from each per-strategy CSV. If omitted, a common return
        or value column name is inferred.
    strategy_columns:
        Strategy columns to read from a wide CSV. If omitted, all numeric
        non-date columns are used.
    """
    if input_kind not in {"returns", "values"}:
        raise ValueError("input_kind must be either 'returns' or 'values'")
    if bool(wide_csv) == bool(strategy_files):
        raise ValueError("Pass exactly one of wide_csv or strategy_files")

    if wide_csv:
        frame, resolved_date_col = _read_table_with_date(Path(wide_csv), date_col)
        if strategy_columns:
            missing = sorted(set(strategy_columns) - set(frame.columns))
            if missing:
                raise ValueError(f"{wide_csv} is missing strategy columns: {missing}")
            data = frame[[resolved_date_col, *strategy_columns]].copy()
        else:
            numeric_cols = [
                column
                for column in frame.columns
                if column != resolved_date_col and pd.api.types.is_numeric_dtype(frame[column])
            ]
            if not numeric_cols:
                raise ValueError(f"{wide_csv} has no numeric strategy columns")
            data = frame[[resolved_date_col, *numeric_cols]].copy()

        data = data.set_index(resolved_date_col).sort_index()
        data = data.rename(columns={column: _canonical_strategy_name(column) for column in data.columns})
    else:
        series_by_strategy: list[pd.Series] = []
        assert strategy_files is not None
        for strategy, path_spec in strategy_files.items():
            if isinstance(path_spec, (list, tuple)):
                if input_kind != "values":
                    raise ValueError("Repeated-run path lists are supported only with --input-kind values")
                series = _mean_normalized_value_series(
                    path_spec,
                    date_col=date_col,
                    series_col=series_col,
                ).rename(strategy)
            else:
                series = _series_from_path(
                    path_spec,
                    input_kind=input_kind,
                    date_col=date_col,
                    series_col=series_col,
                ).rename(strategy)
            series_by_strategy.append(series)
        data = pd.concat(series_by_strategy, axis=1).sort_index()

    if start_date:
        data = data[data.index >= pd.Timestamp(start_date)]
    if end_date:
        data = data[data.index <= pd.Timestamp(end_date)]

    if input_kind == "values":
        data = returns_from_values(data)

    aligned = data.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if aligned.empty:
        raise ValueError("No overlapping non-missing return observations after alignment")
    if len(aligned) < 2:
        raise ValueError("At least two aligned daily return observations are required")

    duplicate_cols = aligned.columns[aligned.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"Duplicate strategy columns after name normalization: {duplicate_cols}")
    return aligned.astype(float)


def compute_metrics(returns: pd.Series | pd.DataFrame) -> dict[str, float] | pd.DataFrame:
    """Compute Sharpe, MD, VaR, CVaR, and Total Return from daily returns.

    VaR and CVaR are computed on the loss scale, using losses = -returns.
    Maximum Drawdown is computed as a positive drawdown magnitude from
    cumulative wealth starting at 1.0.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {column: compute_metrics(returns[column]) for column in returns.columns}
        ).T

    clean_returns = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean_returns.empty:
        raise ValueError("Cannot compute metrics for an empty return series")

    std = clean_returns.std(ddof=1)
    sharpe = 0.0 if len(clean_returns) < 2 or std == 0.0 or np.isnan(std) else (
        clean_returns.mean() / std * np.sqrt(252.0)
    )

    wealth = (1.0 + clean_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = (running_peak - wealth) / running_peak
    max_drawdown = max(float(drawdown.max()), 0.0)

    losses = -clean_returns
    var95 = float(losses.quantile(0.95))
    tail_losses = losses[losses >= var95]
    cvar95 = float(tail_losses.mean()) if not tail_losses.empty else var95
    total_return = float(wealth.iloc[-1] - 1.0)

    return {
        "Sharpe": float(sharpe),
        "MD": max_drawdown,
        "VaR": var95,
        "CVaR": cvar95,
        "Total Return": total_return,
    }


def _compute_metrics_np(returns: np.ndarray) -> dict[str, float]:
    """Fast NumPy metric implementation for bootstrap inner loops."""
    clean_returns = returns[np.isfinite(returns)]
    if clean_returns.size == 0:
        raise ValueError("Cannot compute metrics for an empty return array")

    if clean_returns.size < 2:
        sharpe = 0.0
    else:
        std = float(np.std(clean_returns, ddof=1))
        sharpe = 0.0 if std == 0.0 or np.isnan(std) else (
            float(np.mean(clean_returns)) / std * float(np.sqrt(252.0))
        )

    wealth = np.cumprod(1.0 + clean_returns)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = (running_peak - wealth) / running_peak
    max_drawdown = max(float(np.max(drawdown)), 0.0)

    losses = -clean_returns
    var95 = float(np.quantile(losses, 0.95))
    tail_losses = losses[losses >= var95]
    cvar95 = float(np.mean(tail_losses)) if tail_losses.size else var95
    total_return = float(wealth[-1] - 1.0)

    return {
        "Sharpe": float(sharpe),
        "MD": max_drawdown,
        "VaR": var95,
        "CVaR": cvar95,
        "Total Return": total_return,
    }


def circular_block_bootstrap_indices(
    n_obs: int,
    block_size: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one circular moving-block bootstrap index vector."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    generator = rng if rng is not None else np.random.default_rng()
    n_blocks = int(np.ceil(n_obs / block_size))
    starts = generator.integers(0, n_obs, size=n_blocks)
    offsets = np.arange(block_size)
    indices = (starts[:, None] + offsets[None, :]) % n_obs
    return indices.ravel()[:n_obs]


def _metric_improvements(full_metrics: Mapping[str, float], baseline_metrics: Mapping[str, float]) -> dict[str, float]:
    return {
        "Sharpe": full_metrics["Sharpe"] - baseline_metrics["Sharpe"],
        "MD": baseline_metrics["MD"] - full_metrics["MD"],
        "CVaR": baseline_metrics["CVaR"] - full_metrics["CVaR"],
        "Total Return": full_metrics["Total Return"] - baseline_metrics["Total Return"],
    }


def paired_block_bootstrap_comparison(
    returns: pd.DataFrame,
    *,
    full_strategy: str,
    baseline_strategy: str,
    n_bootstrap: int = 10_000,
    block_size: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Full Pipeline against one baseline with paired block bootstrap."""
    missing = {full_strategy, baseline_strategy} - set(returns.columns)
    if missing:
        raise ValueError(f"Missing strategy columns: {sorted(missing)}")

    pair_returns = returns[[full_strategy, baseline_strategy]].to_numpy(dtype=float)
    n_obs = pair_returns.shape[0]
    rng = np.random.default_rng(seed)

    observed = _metric_improvements(
        _compute_metrics_np(pair_returns[:, 0]),
        _compute_metrics_np(pair_returns[:, 1]),
    )
    bootstrap_improvements = {metric: np.empty(n_bootstrap, dtype=float) for metric in BOOTSTRAP_METRICS}

    for iteration in range(n_bootstrap):
        indices = circular_block_bootstrap_indices(n_obs, block_size=block_size, rng=rng)
        sample = pair_returns[indices, :]
        full_metrics = _compute_metrics_np(sample[:, 0])
        baseline_metrics = _compute_metrics_np(sample[:, 1])
        improvements = _metric_improvements(full_metrics, baseline_metrics)
        for metric in BOOTSTRAP_METRICS:
            bootstrap_improvements[metric][iteration] = improvements[metric]

    comparison = f"{full_strategy} vs {baseline_strategy}"
    rows = []
    for metric in BOOTSTRAP_METRICS:
        distribution = bootstrap_improvements[metric]
        prob_le_zero = float(np.mean(distribution <= 0.0))
        prob_ge_zero = float(np.mean(distribution >= 0.0))
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "observed_improvement": observed[metric],
                "bootstrap_mean": float(np.mean(distribution)),
                "ci_2_5": float(np.quantile(distribution, 0.025)),
                "ci_97_5": float(np.quantile(distribution, 0.975)),
                "positive_resample_pct": float(np.mean(distribution > 0.0) * 100.0),
                "p_one_sided": prob_le_zero,
                "p_two_sided": min(1.0, 2.0 * min(prob_le_zero, prob_ge_zero)),
            }
        )
    return pd.DataFrame(rows)


def _paired_run_return_frames(
    strategy_files: Mapping[str, str | Path | Sequence[str | Path]],
    *,
    full_strategy: str,
    baseline_strategy: str,
    input_kind: str,
    date_col: str,
    series_col: str | None,
    start_date: str | None,
    end_date: str | None,
) -> list[pd.DataFrame]:
    full_runs = _returns_from_path_spec(
        strategy_files[full_strategy],
        input_kind=input_kind,
        date_col=date_col,
        series_col=series_col,
        start_date=start_date,
        end_date=end_date,
    )
    baseline_runs = _returns_from_path_spec(
        strategy_files[baseline_strategy],
        input_kind=input_kind,
        date_col=date_col,
        series_col=series_col,
        start_date=start_date,
        end_date=end_date,
    )

    n_pairs = max(len(full_runs), len(baseline_runs))
    if len(full_runs) not in {1, n_pairs} or len(baseline_runs) not in {1, n_pairs}:
        raise ValueError(
            f"Cannot pair {len(full_runs)} {full_strategy} runs with "
            f"{len(baseline_runs)} {baseline_strategy} runs."
        )

    frames = []
    for index in range(n_pairs):
        full = full_runs[index if len(full_runs) > 1 else 0].rename(full_strategy)
        baseline = baseline_runs[index if len(baseline_runs) > 1 else 0].rename(baseline_strategy)
        pair = pd.concat([full, baseline], axis=1, join="inner").dropna()
        if len(pair) < 2:
            raise ValueError(f"Not enough overlapping returns for pair {full_strategy} vs {baseline_strategy}")
        frames.append(pair.astype(float))
    return frames


def paired_block_bootstrap_comparison_run_level(
    strategy_files: Mapping[str, str | Path | Sequence[str | Path]],
    *,
    full_strategy: str,
    baseline_strategy: str,
    input_kind: str,
    date_col: str,
    series_col: str | None,
    start_date: str | None,
    end_date: str | None,
    n_bootstrap: int = 10_000,
    block_size: int = 20,
    seed: int = 42,
    avg_trajectory_observed: Mapping[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Paired block bootstrap averaged over run-level pairs.

    For repeated evaluations, this matches the table estimand: metrics are
    computed per run pair, improvements are computed per pair, and the reported
    improvement is the mean across pairs. Single-run baselines are reused
    against each repeated Full Pipeline run.
    """
    pair_frames = _paired_run_return_frames(
        strategy_files,
        full_strategy=full_strategy,
        baseline_strategy=baseline_strategy,
        input_kind=input_kind,
        date_col=date_col,
        series_col=series_col,
        start_date=start_date,
        end_date=end_date,
    )
    rng = np.random.default_rng(seed)
    comparison = f"{full_strategy} vs {baseline_strategy}"

    observed_by_pair = []
    for pair in pair_frames:
        arr = pair[[full_strategy, baseline_strategy]].to_numpy(dtype=float)
        observed_by_pair.append(
            _metric_improvements(
                _compute_metrics_np(arr[:, 0]),
                _compute_metrics_np(arr[:, 1]),
            )
        )
    observed = {
        metric: float(np.mean([pair_obs[metric] for pair_obs in observed_by_pair]))
        for metric in BOOTSTRAP_METRICS
    }

    bootstrap_improvements = {metric: np.empty(n_bootstrap, dtype=float) for metric in BOOTSTRAP_METRICS}
    arrays = [pair[[full_strategy, baseline_strategy]].to_numpy(dtype=float) for pair in pair_frames]
    for iteration in range(n_bootstrap):
        iteration_improvements = {metric: [] for metric in BOOTSTRAP_METRICS}
        for arr in arrays:
            indices = circular_block_bootstrap_indices(arr.shape[0], block_size=block_size, rng=rng)
            sample = arr[indices, :]
            improvements = _metric_improvements(
                _compute_metrics_np(sample[:, 0]),
                _compute_metrics_np(sample[:, 1]),
            )
            for metric in BOOTSTRAP_METRICS:
                iteration_improvements[metric].append(improvements[metric])
        for metric in BOOTSTRAP_METRICS:
            bootstrap_improvements[metric][iteration] = float(np.mean(iteration_improvements[metric]))

    rows = []
    for metric in BOOTSTRAP_METRICS:
        distribution = bootstrap_improvements[metric]
        prob_le_zero = float(np.mean(distribution <= 0.0))
        prob_ge_zero = float(np.mean(distribution >= 0.0))
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "observed_improvement": observed[metric],
                "observed_improvement_table_consistent": observed[metric],
                "observed_improvement_avg_trajectory": (
                    np.nan
                    if avg_trajectory_observed is None
                    else avg_trajectory_observed.get((comparison, metric), np.nan)
                ),
                "bootstrap_mean": float(np.mean(distribution)),
                "ci_2_5": float(np.quantile(distribution, 0.025)),
                "ci_97_5": float(np.quantile(distribution, 0.975)),
                "positive_resample_pct": float(np.mean(distribution > 0.0) * 100.0),
                "p_one_sided": prob_le_zero,
                "p_two_sided": min(1.0, 2.0 * min(prob_le_zero, prob_ge_zero)),
                "n_run_pairs": len(pair_frames),
            }
        )
    return pd.DataFrame(rows)


def run_all_comparisons_run_level(
    strategy_files: Mapping[str, str | Path | Sequence[str | Path]],
    *,
    full_strategy: str,
    baselines: Sequence[str],
    input_kind: str,
    date_col: str,
    series_col: str | None,
    start_date: str | None,
    end_date: str | None,
    n_bootstrap: int,
    block_size: int,
    seed: int,
    avg_trajectory_observed: Mapping[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    frames = []
    total = len(baselines)
    for offset, baseline in enumerate(baselines):
        print(
            f"Run-level bootstrap {offset + 1}/{total}: {full_strategy} vs {baseline} "
            f"({n_bootstrap:,} resamples)",
            flush=True,
        )
        frames.append(
            paired_block_bootstrap_comparison_run_level(
                strategy_files,
                full_strategy=full_strategy,
                baseline_strategy=baseline,
                input_kind=input_kind,
                date_col=date_col,
                series_col=series_col,
                start_date=start_date,
                end_date=end_date,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
                seed=seed + offset,
                avg_trajectory_observed=avg_trajectory_observed,
            )
        )
    return pd.concat(frames, ignore_index=True)


def apply_table_consistent_observed_values(
    results: pd.DataFrame,
    observed: Mapping[tuple[str, str], float],
) -> pd.DataFrame:
    """Replace observed values with table-consistent run-level observed values.

    The bootstrap distribution is still computed on average daily trajectories.
    We keep that original observed value in a separate column for transparency.
    """
    updated = results.copy()
    updated["observed_improvement_avg_trajectory"] = updated["observed_improvement"]
    updated["observed_improvement_table_consistent"] = [
        observed.get((row.comparison, row.metric), row.observed_improvement)
        for row in updated.itertuples(index=False)
    ]
    updated["observed_improvement"] = updated["observed_improvement_table_consistent"]

    ordered = [
        "comparison",
        "metric",
        "observed_improvement",
        "observed_improvement_table_consistent",
        "observed_improvement_avg_trajectory",
        "bootstrap_mean",
        "ci_2_5",
        "ci_97_5",
        "positive_resample_pct",
        "p_one_sided",
        "p_two_sided",
    ]
    return updated[ordered]


def run_all_comparisons(
    returns: pd.DataFrame,
    *,
    full_strategy: str = FULL_PIPELINE,
    baselines: Sequence[str] | None = None,
    n_bootstrap: int = 10_000,
    block_size: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all requested Full Pipeline vs baseline bootstrap comparisons."""
    if baselines is None:
        priority = [baseline for baseline in BASELINES if baseline in returns.columns]
        remaining = [
            strategy
            for strategy in returns.columns
            if strategy != full_strategy and strategy not in priority
        ]
        baselines = [*priority, *remaining]

    missing = [strategy for strategy in (full_strategy, *baselines) if strategy not in returns.columns]
    if missing:
        available = ", ".join(returns.columns)
        raise ValueError(f"Missing required strategies {missing}. Available columns: {available}")

    frames = []
    total = len(baselines)
    for offset, baseline in enumerate(baselines):
        print(
            f"Bootstrap {offset + 1}/{total}: {full_strategy} vs {baseline} "
            f"({n_bootstrap:,} resamples)",
            flush=True,
        )
        frames.append(
            paired_block_bootstrap_comparison(
                returns,
                full_strategy=full_strategy,
                baseline_strategy=baseline,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
                seed=seed + offset,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _paper_summary(results: pd.DataFrame) -> str:
    lines = []
    for comparison, comparison_results in results.groupby("comparison", sort=False):
        by_metric = comparison_results.set_index("metric")
        md = by_metric.loc["MD"]
        cvar = by_metric.loc["CVaR"]
        lines.append(
            f"For {comparison}, improvements are positive in "
            f"{md['positive_resample_pct']:.1f}% of bootstrap resamples for MD and "
            f"{cvar['positive_resample_pct']:.1f}% for CVaR; the corresponding "
            f"one-sided bootstrap p-values are {md['p_one_sided']:.4f} and "
            f"{cvar['p_one_sided']:.4f}, respectively."
        )
    return "\n".join(lines)


def print_key_ieee_summary(results: pd.DataFrame) -> None:
    """Print a compact robustness summary focused on downside metrics."""
    key_pairs = [
        "Full Pipeline vs Buy & Hold",
        "Full Pipeline vs PPO",
        "Full Pipeline vs No Memory",
        "Full Pipeline vs Retrieval-Only",
        "Full Pipeline vs Random Retrieval",
    ]
    key_metrics = ["MD", "CVaR"]

    print("\nKey IEEE Access robustness summary:")
    for comparison in key_pairs:
        subset = results[results["comparison"] == comparison]
        if subset.empty:
            continue
        print(f"\n{comparison}:")
        for metric in key_metrics:
            metric_rows = subset[subset["metric"] == metric]
            if metric_rows.empty:
                continue
            row = metric_rows.iloc[0]
            print(
                f"  {metric}: observed improvement={row['observed_improvement']:.4f}, "
                f"95% CI=[{row['ci_2_5']:.4f}, {row['ci_97_5']:.4f}], "
                f"positive={row['positive_resample_pct']:.1f}%, "
                f"p_one_sided={row['p_one_sided']:.4f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run paired circular block-bootstrap robustness comparisons."
    )
    parser.add_argument("--wide-csv", type=Path, default=CSV_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--input-kind", choices=["returns", "values"], default=INPUT_KIND)
    parser.add_argument("--date-col", default=DATE_COL)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.wide_csv is not None:
        print(f"Using wide CSV: {args.wide_csv}")
        wide_csv = args.wide_csv
        strategy_files = None
    else:
        wide_csv = None
        strategy_files = discover_strategy_files(args.results_dir)
        print("Using discovered strategy files from results:")
        for strategy, path in strategy_files.items():
            print(f"  {strategy}: {path}")

    returns = load_and_align_data(
        wide_csv=wide_csv,
        strategy_files=strategy_files,
        input_kind=args.input_kind,
        date_col=args.date_col,
        series_col=None,
        strategy_columns=None,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    metrics = compute_metrics(returns)
    print("\nStrategy metrics on aligned daily returns:")
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))

    baselines = [baseline for baseline in BASELINES if baseline in returns.columns]

    if strategy_files is not None:
        avg_trajectory_observed = {}
        full_avg_metrics = _compute_metrics_np(returns[FULL_PIPELINE].to_numpy(dtype=float))
        for baseline in baselines:
            comparison = f"{FULL_PIPELINE} vs {baseline}"
            improvements = _metric_improvements(
                full_avg_metrics,
                _compute_metrics_np(returns[baseline].to_numpy(dtype=float)),
            )
            for metric, value in improvements.items():
                avg_trajectory_observed[(comparison, metric)] = value

        results = run_all_comparisons_run_level(
            strategy_files,
            full_strategy=FULL_PIPELINE,
            baselines=baselines,
            input_kind=args.input_kind,
            date_col=args.date_col,
            series_col=None,
            start_date=args.start_date,
            end_date=args.end_date,
            n_bootstrap=args.n_bootstrap,
            block_size=args.block_size,
            seed=args.seed,
            avg_trajectory_observed=avg_trajectory_observed,
        )
    else:
        results = run_all_comparisons(
            returns,
            baselines=baselines,
            n_bootstrap=args.n_bootstrap,
            block_size=args.block_size,
            seed=args.seed,
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    print("\nBootstrap results:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPaper summary:")
    print(_paper_summary(results))
    print_key_ieee_summary(results)
    print(f"\nSaved: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
