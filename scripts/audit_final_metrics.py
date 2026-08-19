#!/usr/bin/env python3
"""Audit performance metrics for every strategy-like folder in results/final.

This is intentionally diagnostic: it scans available final result files,
computes paper-style metrics per run/file, then aggregates mean and standard
deviation by folder. It helps catch cases where files were copied or folders
were named in a way that does not match the paper table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_FINAL_DIR = Path("results") / "final"
OUTPUT_DIR = Path("results") / "final" / "metrics_audit"
PER_RUN_CSV = OUTPUT_DIR / "final_metrics_per_run.csv"
SUMMARY_CSV = OUTPUT_DIR / "final_metrics_summary.csv"
SUMMARY_MARKDOWN = OUTPUT_DIR / "final_metrics_summary.md"
START_DATE = "2021-01-04"
END_DATE = "2023-12-29"
TRADING_DAYS = 252.0


def _display_name(path: Path) -> str:
    aliases = {
        "bh": "Buy & Hold",
        "ew": "Equal Weight",
        "ppo": "PPO",
        "finrl": "FinRL",
        "finagent": "FinAgent",
        "multihedge": "Full Pipeline",
        "NoMemory": "No Memory",
    }
    return aliases.get(path.name, path.name.replace("_", " ").replace("-", " "))


def _strategy_group_for_file(path: Path) -> str | None:
    rel = path.relative_to(RESULTS_FINAL_DIR)
    if not rel.parts:
        return None
    if rel.parts[0] == "plots":
        return None
    if rel.parts[0] == "benchmark" and len(rel.parts) >= 2:
        return _display_name(RESULTS_FINAL_DIR / rel.parts[0] / rel.parts[1])
    if rel.parts[0] == "ablation" and len(rel.parts) >= 2:
        return _display_name(RESULTS_FINAL_DIR / rel.parts[0] / rel.parts[1])
    if rel.parts[0] == "multihedge":
        return "Full Pipeline"
    return _display_name(RESULTS_FINAL_DIR / rel.parts[0])


def _candidate_files(root: Path) -> list[Path]:
    patterns = (
        "**/performance.csv",
        "**/*portfolio_value*.csv",
        "**/*portfolio_values*.csv",
        "**/*account_value*.csv",
        "**/*values*.csv",
        "**/portfolio_history*.json",
        "**/portfolio_history*.jsonl",
    )
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(path for path in files if "plots" not in path.parts)


def _read_json_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("history", "portfolio_history", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON portfolio structure in {path}")


def _normalise_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    lower_to_original = {str(col).lower(): col for col in frame.columns}
    date_col = lower_to_original.get("date")
    value_col = None
    for candidate in ("portfolio_value", "total_value", "value", "account_value"):
        if candidate in lower_to_original:
            value_col = lower_to_original[candidate]
            break
    if value_col is None and len(frame.columns) >= 2:
        value_col = frame.columns[1]
    if date_col is None or value_col is None:
        raise ValueError(f"Could not identify date/value columns in {source}")

    cleaned = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "portfolio_value": pd.to_numeric(frame[value_col], errors="coerce"),
        }
    )
    cleaned = cleaned.dropna(subset=["date", "portfolio_value"])
    cleaned = cleaned.sort_values("date").drop_duplicates("date", keep="last")
    if cleaned.empty:
        raise ValueError(f"No usable portfolio observations in {source}")
    return cleaned


def load_value_series(path: Path) -> pd.Series:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        records = []
        for item in _read_json_records(path):
            if not isinstance(item, dict) or "date" not in item:
                continue
            value = item.get("portfolio_value", item.get("total_value", item.get("value")))
            if value is not None:
                records.append({"date": item["date"], "portfolio_value": value})
        frame = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    cleaned = _normalise_frame(frame, path)
    series = pd.Series(
        cleaned["portfolio_value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(cleaned["date"]),
        name="portfolio_value",
    )
    return series.sort_index()


def crop_series(values: pd.Series) -> pd.Series:
    cropped = values.copy()
    cropped = cropped[cropped.index >= pd.Timestamp(START_DATE)]
    cropped = cropped[cropped.index <= pd.Timestamp(END_DATE)]
    if cropped.empty:
        raise ValueError("No observations inside audit date range")
    return cropped


def _annualized_return(total_return: float, values: pd.Series) -> float:
    days = (values.index.max() - values.index.min()).days
    if days <= 0:
        return 0.0
    return float((1.0 + total_return) ** (365.0 / days) - 1.0)


def _max_consecutive_losses(returns: pd.Series) -> int:
    max_run = 0
    current = 0
    for value in returns:
        if value < 0.0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def compute_metrics(values: pd.Series) -> dict[str, float]:
    values = crop_series(values).astype(float)
    returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    annualized_return = _annualized_return(total_return, values)

    std = returns.std(ddof=1)
    sharpe = 0.0 if len(returns) < 2 or std == 0.0 or np.isnan(std) else (
        returns.mean() / std * np.sqrt(TRADING_DAYS)
    )

    downside = returns[returns < 0.0]
    downside_std = downside.std(ddof=1)
    sortino = 0.0 if len(downside) < 2 or downside_std == 0.0 or np.isnan(downside_std) else (
        returns.mean() / downside_std * np.sqrt(TRADING_DAYS)
    )

    drawdown = values / values.cummax() - 1.0
    max_drawdown = float(-drawdown.min())
    calmar = 0.0 if max_drawdown == 0.0 else float(annualized_return / max_drawdown)

    losses = -returns
    var95 = float(losses.quantile(0.95))
    cvar95 = float(losses[losses >= var95].mean())

    monthly_returns = values.resample("ME").last().pct_change().dropna()
    worst_month = float(monthly_returns.min()) if not monthly_returns.empty else 0.0
    drawdown_vol = float(drawdown.std(ddof=1))
    time_in_drawdown_10 = float((drawdown <= -0.10).mean())

    return {
        "observations": float(len(values)),
        "start_value": float(values.iloc[0]),
        "final_value": float(values.iloc[-1]),
        "SR": float(sharpe),
        "SoR": float(sortino),
        "CaR": float(calmar),
        "TR_pct": total_return * 100.0,
        "MD_pct": max_drawdown * 100.0,
        "VaR_pct": var95 * 100.0,
        "CVaR_pct": cvar95 * 100.0,
        "WM_pct": worst_month * 100.0,
        "DDv_pct": drawdown_vol * 100.0,
        "TDD_pct": time_in_drawdown_10 * 100.0,
        "MCL": float(_max_consecutive_losses(returns)),
    }


def collect_per_run_metrics() -> pd.DataFrame:
    rows = []
    for path in _candidate_files(RESULTS_FINAL_DIR):
        strategy = _strategy_group_for_file(path)
        if strategy is None:
            continue
        try:
            values = load_value_series(path)
            metrics = compute_metrics(values)
        except Exception as exc:
            rows.append(
                {
                    "strategy": strategy,
                    "file": str(path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        rows.append(
            {
                "strategy": strategy,
                "file": str(path),
                "status": "ok",
                "error": "",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    if np.isnan(std) or std == 0.0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    ok = per_run[per_run["status"] == "ok"].copy()
    metric_cols = (
        "SR",
        "SoR",
        "CaR",
        "TR_pct",
        "MD_pct",
        "VaR_pct",
        "CVaR_pct",
        "WM_pct",
        "DDv_pct",
        "TDD_pct",
        "MCL",
    )
    rows = []
    for strategy, group in ok.groupby("strategy", sort=True):
        row = {"strategy": strategy, "runs": int(len(group))}
        for metric in metric_cols:
            mean = float(group[metric].mean())
            std = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = _format_mean_std(mean, std, decimals=2 if metric != "SR" else 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run = collect_per_run_metrics()
    summary = build_summary(per_run)

    per_run.to_csv(PER_RUN_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_MARKDOWN.write_text(summary.to_markdown(index=False), encoding="utf-8")

    print("\nPer-strategy summary from results/final:")
    display_cols = [
        "strategy",
        "runs",
        "SR",
        "SoR",
        "CaR",
        "TR_pct",
        "MD_pct",
        "VaR_pct",
        "CVaR_pct",
        "WM_pct",
        "DDv_pct",
        "TDD_pct",
        "MCL",
    ]
    print(summary[display_cols].to_string(index=False))

    errors = per_run[per_run["status"] != "ok"]
    if not errors.empty:
        print("\nFiles skipped due to errors:")
        print(errors[["strategy", "file", "error"]].to_string(index=False))

    print(f"\nSaved per-run metrics: {PER_RUN_CSV}")
    print(f"Saved summary metrics: {SUMMARY_CSV}")
    print(f"Saved markdown summary: {SUMMARY_MARKDOWN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
