#!/usr/bin/env python3
"""Print memory ablation and experiment-history stats as tables."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable
from datetime import datetime

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("results") / "memory_controller_ablations"
DEFAULT_EXPERIMENTS_DIR = Path("results") / "experiments"
MODES = ("retrieval_only", "random_retrieval")


def _format_timestamp(ts: float | None) -> str:
    if not ts or ts <= 0:
        return ""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _file_birth_epoch(path: Path) -> float | None:
    """Return filesystem birth time when available through GNU stat."""
    try:
        result = subprocess.run(
            ["stat", "-c", "%W", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        value = float(result.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _max_drawdown(values: pd.Series) -> float:
    running_peak = values.cummax()
    drawdown = (running_peak - values) / running_peak
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _annualized_return(total_return: float, dates: pd.Series) -> float:
    if dates.empty:
        return 0.0
    days = (dates.max() - dates.min()).days
    if days <= 0:
        return 0.0
    return float((1.0 + total_return) ** (365.0 / days) - 1.0)


def _annualized_sharpe(returns: pd.Series) -> float:
    returns = returns.fillna(0.0)
    std = returns.std(ddof=1)
    if len(returns) < 2 or std == 0.0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(252))


def _cvar_95(returns: pd.Series) -> float:
    returns = returns.fillna(0.0)
    if returns.empty:
        return 0.0
    cutoff = returns.quantile(0.05)
    tail = returns[returns <= cutoff]
    return float(tail.mean()) if not tail.empty else 0.0


def _worst_month(returns: pd.Series, dates: pd.Series) -> float:
    if returns.empty:
        return 0.0
    monthly = (
        pd.Series(returns.fillna(0.0).to_numpy(), index=pd.to_datetime(dates))
        .add(1.0)
        .resample("ME")
        .prod()
        .sub(1.0)
    )
    return float(monthly.min()) if not monthly.empty else 0.0


def _mode_from_name(name: str) -> str | None:
    for mode in MODES:
        if name.startswith(mode):
            return mode
    return None


def _iter_run_dirs(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists():
        return
    for child in sorted(results_dir.iterdir()):
        if child.is_dir() and _mode_from_name(child.name):
            yield child


def _portfolio_history_to_performance(history_path: Path) -> tuple[pd.DataFrame, int]:
    records = []
    skipped_lines = 0
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue
            if "date" not in item or "total_value" not in item:
                skipped_lines += 1
                continue
            records.append(
                {
                    "date": item["date"],
                    "portfolio_value": float(item["total_value"]),
                }
            )

    perf = pd.DataFrame(records)
    if perf.empty:
        return perf, skipped_lines
    perf = perf.copy()
    perf.loc[:, "date"] = pd.to_datetime(perf["date"])
    perf = perf.sort_values("date").reset_index(drop=True)
    perf.loc[:, "daily_return"] = perf["portfolio_value"].pct_change().fillna(0.0)
    return perf, skipped_lines


def _row_from_performance(
    *,
    source: str,
    mode: str,
    run_name: str,
    perf: pd.DataFrame,
    trades_executed: int = 0,
) -> dict:
    returns = perf["daily_return"].astype(float)
    values = perf["portfolio_value"].astype(float)
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    return {
        "source": source,
        "mode": mode,
        "run_name": run_name,
        "start_date": perf["date"].min().date().isoformat(),
        "end_date": perf["date"].max().date().isoformat(),
        "days": int(len(perf)),
        "start_value": float(values.iloc[0]),
        "final_value": float(values.iloc[-1]),
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, perf["date"]),
        "max_drawdown": _max_drawdown(values),
        "annualized_sharpe": _annualized_sharpe(returns),
        "cvar_95": _cvar_95(returns),
        "worst_month": _worst_month(returns, perf["date"]),
        "trades_executed": int(trades_executed),
    }


def collect_stats(results_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in _iter_run_dirs(results_dir):
        perf_path = run_dir / "performance.csv"
        if not perf_path.exists():
            continue

        perf = pd.read_csv(perf_path, parse_dates=["date"])
        if perf.empty:
            continue

        rows.append(
            _row_from_performance(
                source="memory_controller_ablations",
                mode=_mode_from_name(run_dir.name) or "unknown",
                run_name=run_dir.name,
                perf=perf,
                trades_executed=int(_load_summary(run_dir / "summary.json").get("trades_executed", 0)),
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["mode", "total_return"], ascending=[True, False]).reset_index(drop=True)


def _ablation_reference_rows(results_dir: Path) -> list[dict]:
    refs = []
    if not results_dir.exists():
        return refs
    for perf_path in sorted(results_dir.glob("*/performance.csv")):
        run_dir = perf_path.parent
        perf = pd.read_csv(perf_path, parse_dates=["date"])
        if perf.empty:
            continue
        values = perf["portfolio_value"].astype(float)
        refs.append(
            {
                "run_name": run_dir.name,
                "mode": _mode_from_name(run_dir.name) or run_dir.name,
                "start_date": perf["date"].min().date().isoformat(),
                "end_date": perf["date"].max().date().isoformat(),
                "days": int(len(perf)),
                "start_value": float(values.iloc[0]),
                "final_value": float(values.iloc[-1]),
            }
        )
    return refs


def _match_ablation_reference(row: dict, refs: list[dict]) -> tuple[str, str]:
    for ref in refs:
        if (
            row["start_date"] == ref["start_date"]
            and row["end_date"] == ref["end_date"]
            and row["days"] == ref["days"]
            and abs(row["start_value"] - ref["start_value"]) < 0.05
            and abs(row["final_value"] - ref["final_value"]) < 0.05
        ):
            return ref["mode"], ref["run_name"]
    return "", ""


def _classify_llm_prompt(prompt: str) -> str:
    if "provide your hedging strategy decision" in prompt or "Current Hedging State" in prompt:
        return "hedging"
    if "provide your asset allocation decision" in prompt or "stock/options allocation plan" in prompt:
        return "asset"
    if "provide your allocation decision" in prompt or "capital allocation plan" in prompt:
        return "portfolio"
    return "market"


def _load_llm_events(experiments_dir: Path) -> list[dict]:
    llm_log = experiments_dir / "results" / "llm_usage.jsonl"
    events = []
    if not llm_log.exists():
        return events
    with llm_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = item.get("timestamp")
            if not timestamp:
                continue
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            events.append(
                {
                    "timestamp": ts,
                    "prompt_type": _classify_llm_prompt(item.get("prompt", "") or ""),
                    "status": item.get("status", ""),
                }
            )
    return sorted(events, key=lambda item: item["timestamp"])


def _llm_counts_for_run(events: list[dict], start_ts: float, end_ts: float) -> dict:
    counts = {
        "llm_market_prompts": 0,
        "llm_portfolio_prompts": 0,
        "llm_asset_prompts": 0,
        "llm_hedging_prompts": 0,
        "llm_errors": 0,
        "llm_first_prompt_time": "",
        "llm_last_prompt_time": "",
        "llm_duration_seconds": np.nan,
    }
    key_by_type = {
        "market": "llm_market_prompts",
        "portfolio": "llm_portfolio_prompts",
        "asset": "llm_asset_prompts",
        "hedging": "llm_hedging_prompts",
    }
    matching_events = []
    for event in events:
        if start_ts < event["timestamp"] <= end_ts:
            matching_events.append(event)
            counts[key_by_type[event["prompt_type"]]] += 1
            if event["status"] == "error":
                counts["llm_errors"] += 1
    if matching_events:
        first_ts = matching_events[0]["timestamp"]
        last_ts = matching_events[-1]["timestamp"]
        counts["llm_first_prompt_time"] = _format_timestamp(first_ts)
        counts["llm_last_prompt_time"] = _format_timestamp(last_ts)
        counts["llm_duration_seconds"] = round(last_ts - first_ts, 2)
    return counts


def _inferred_controller(counts: dict) -> str:
    synthesis_count = (
        counts["llm_portfolio_prompts"]
        + counts["llm_asset_prompts"]
        + counts["llm_hedging_prompts"]
    )
    if counts["llm_market_prompts"] > 0 and synthesis_count == 0:
        return "retrieval_only_like"
    if synthesis_count > 0:
        return "llm_controller_like"
    return "unknown"


def _duration_bucket(duration_seconds: float) -> str:
    if pd.isna(duration_seconds):
        return "unknown"
    if duration_seconds < 120:
        return "short"
    if duration_seconds < 400:
        return "medium"
    return "long"


def _duration_controller(duration_seconds: float) -> str:
    bucket = _duration_bucket(duration_seconds)
    if bucket == "short":
        return "retrieval_only_like"
    if bucket == "long":
        return "llm_controller_like"
    return "ambiguous"


def _final_controller(row: dict) -> str:
    matched_mode = row.get("matched_mode") or ""
    prompt_controller = row.get("prompt_inferred_controller") or row.get("inferred_controller") or ""
    duration_controller = row.get("duration_inferred_controller") or ""

    if matched_mode in {"retrieval_only", "random_retrieval", "memory_build"}:
        return matched_mode
    if prompt_controller == duration_controller and prompt_controller in {
        "retrieval_only_like",
        "llm_controller_like",
    }:
        return prompt_controller
    if prompt_controller in {"retrieval_only_like", "llm_controller_like"}:
        return prompt_controller
    if duration_controller in {"retrieval_only_like", "llm_controller_like"}:
        return duration_controller
    return "unknown"


def _classification_note(row: dict) -> str:
    matched_mode = row.get("matched_mode") or ""
    prompt_controller = row.get("prompt_inferred_controller") or row.get("inferred_controller") or ""
    duration_controller = row.get("duration_inferred_controller") or ""

    if matched_mode == "random_retrieval" and (
        prompt_controller == "retrieval_only_like"
        or duration_controller == "retrieval_only_like"
    ):
        return "matched_random_but_fast_no_synthesis"
    if prompt_controller and duration_controller and prompt_controller != duration_controller:
        return "prompt_duration_disagree"
    if not matched_mode:
        return "inferred_only"
    return "matched"


def collect_experiment_histories(experiments_dir: Path, ablation_refs: list[dict]) -> pd.DataFrame:
    rows = []
    if not experiments_dir.exists():
        return pd.DataFrame(rows)

    llm_events = _load_llm_events(experiments_dir)
    run_dirs = sorted(
        experiments_dir.glob("results-*"),
        key=lambda path: (path / "portfolio_history.jsonl").stat().st_mtime
        if (path / "portfolio_history.jsonl").exists()
        else 0.0,
    )
    for run_dir in run_dirs:
        history_path = run_dir / "portfolio_history.jsonl"
        if not history_path.exists():
            continue
        birth_ts = _file_birth_epoch(history_path)
        mtime = history_path.stat().st_mtime
        start_ts = birth_ts or 0.0
        perf, skipped_lines = _portfolio_history_to_performance(history_path)
        if perf.empty:
            continue
        row = _row_from_performance(
            source="experiments",
            mode="experiments_history",
            run_name=run_dir.name,
            perf=perf,
        )
        matched_mode, matched_run = _match_ablation_reference(row, ablation_refs)
        if matched_mode:
            row["matched_mode"] = matched_mode
            row["matched_run"] = matched_run
            row["mode"] = matched_mode
        else:
            row["matched_mode"] = ""
            row["matched_run"] = ""
        row["file_birth_time"] = _format_timestamp(birth_ts)
        row["file_mtime"] = _format_timestamp(mtime)
        row["file_duration_seconds"] = (
            round(mtime - birth_ts, 2) if birth_ts and mtime >= birth_ts else np.nan
        )
        row["duration_bucket"] = _duration_bucket(row["file_duration_seconds"])
        row["duration_inferred_controller"] = _duration_controller(
            row["file_duration_seconds"]
        )
        llm_counts = _llm_counts_for_run(llm_events, start_ts - 5.0, mtime + 5.0)
        row.update(llm_counts)
        row["prompt_inferred_controller"] = _inferred_controller(llm_counts)
        row["final_inferred_controller"] = _final_controller(row)
        row["classification_note"] = _classification_note(row)
        if not matched_mode:
            row["mode"] = row["final_inferred_controller"]
        row["skipped_lines"] = skipped_lines
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("total_return", ascending=False).reset_index(drop=True)


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    percent_prefixes = (
        "total_return",
        "annualized_return",
        "max_drawdown",
        "cvar_95",
        "worst_month",
    )
    money_prefixes = ("start_value", "final_value")
    sharpe_prefixes = ("annualized_sharpe",)

    for col in formatted.columns:
        if col.startswith(percent_prefixes):
            formatted[col] = formatted[col].map(
                lambda value: "" if pd.isna(value) else f"{value * 100:.2f}%"
            )
        elif col.startswith(money_prefixes):
            formatted[col] = formatted[col].map(
                lambda value: "" if pd.isna(value) else f"${value:,.2f}"
            )
        elif col.startswith(sharpe_prefixes):
            formatted[col] = formatted[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    return formatted


def _group_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "final_value",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "annualized_sharpe",
        "cvar_95",
        "worst_month",
        "trades_executed",
    ]
    grouped = df.groupby("mode")[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.insert(1, "runs", df.groupby("mode").size().reindex(grouped["mode"]).to_numpy())
    return grouped


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _experiment_recovery_table(stats: pd.DataFrame) -> pd.DataFrame:
    experiments = stats[stats["source"] == "experiments"].copy()
    if experiments.empty:
        return experiments
    columns = [
        "run_name",
        "start_date",
        "end_date",
        "days",
        "final_value",
        "total_return",
        "file_duration_seconds",
        "duration_bucket",
        "llm_portfolio_prompts",
        "llm_asset_prompts",
        "llm_hedging_prompts",
        "matched_run",
        "prompt_inferred_controller",
        "duration_inferred_controller",
        "final_inferred_controller",
        "classification_note",
    ]
    existing = [col for col in columns if col in experiments.columns]
    return experiments[existing].sort_values(
        ["final_inferred_controller", "file_duration_seconds", "run_name"],
        ascending=[True, True, True],
    )


def _ranking_mode(row: pd.Series) -> str:
    prompt_controller = row.get("prompt_inferred_controller", "")
    duration_controller = row.get("duration_inferred_controller", "")
    final_controller = row.get("final_inferred_controller", "")
    mode = row.get("mode", "")

    if prompt_controller == "retrieval_only_like" or duration_controller == "retrieval_only_like":
        return "retrieval_only"
    if final_controller in {"retrieval_only", "retrieval_only_like"}:
        return "retrieval_only"
    if mode == "retrieval_only":
        return "retrieval_only"
    return "random_retrieval"


def _ranking_note(row: pd.Series) -> str:
    note = row.get("classification_note", "")
    if note:
        return note
    final_controller = row.get("final_inferred_controller", "")
    if final_controller in {"retrieval_only_like", "llm_controller_like"}:
        return final_controller
    return row.get("matched_run", "") or "direct_ablation"


def _ranking_table(
    stats: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    base = stats[stats["source"] == "experiments"].copy()
    if base.empty:
        base = stats.copy()
    if start_date:
        base = base[base["start_date"] == start_date].copy()
    if end_date:
        base = base[base["end_date"] == end_date].copy()
    if base.empty:
        return base

    ranked = pd.DataFrame()
    ranked["mode"] = base.apply(_ranking_mode, axis=1)
    ranked["run_name"] = base["run_name"]
    ranked["total_return"] = base["total_return"]
    ranked["annualized_return"] = base["annualized_return"]
    ranked["annualized_sharpe"] = base["annualized_sharpe"]
    ranked["max_drawdown"] = base["max_drawdown"]
    ranked["final_value"] = base["final_value"]
    ranked["start_date"] = base["start_date"]
    ranked["end_date"] = base["end_date"]
    if "file_duration_seconds" in base.columns:
        ranked["duration_seconds"] = base["file_duration_seconds"]
    if "llm_portfolio_prompts" in base.columns:
        ranked["llm_portfolio"] = base["llm_portfolio_prompts"].fillna(0).astype(int)
        ranked["llm_asset"] = base["llm_asset_prompts"].fillna(0).astype(int)
        ranked["llm_hedging"] = base["llm_hedging_prompts"].fillna(0).astype(int)
    ranked["evidence"] = base.apply(_ranking_note, axis=1)
    return ranked.sort_values("total_return", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print memory ablation and experiment-history statistics."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument(
        "--include-experiments",
        action="store_true",
        help="Also scan results/experiments/results-* portfolio histories.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--ranking-only",
        action="store_true",
        help="Print only the clean retrieval_only vs random_retrieval ranking table.",
    )
    parser.add_argument(
        "--ranking-start-date",
        default=None,
        help="Optional start_date filter for the ranking table, for example 2021-01-04.",
    )
    parser.add_argument(
        "--ranking-end-date",
        default=None,
        help="Optional end_date filter for the ranking table, for example 2023-12-29.",
    )
    args = parser.parse_args()

    ablation_refs = _ablation_reference_rows(args.results_dir)
    frames = [collect_stats(args.results_dir)]
    if args.include_experiments:
        frames.append(collect_experiment_histories(args.experiments_dir, ablation_refs))
    frames = [frame for frame in frames if not frame.empty]
    stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if stats.empty:
        print("No runs found.")
        return 1
    stats = stats.sort_values(["source", "mode", "total_return"], ascending=[True, True, False])

    output_csv = args.output_csv or args.results_dir / "retrieval_ablation_stats.csv"
    output_md = args.output_md or args.results_dir / "retrieval_ablation_stats.md"

    stats.to_csv(output_csv, index=False)
    display = _format_table(stats)
    _write_markdown(display, output_md)

    summary = _group_summary(stats)
    summary_csv = output_csv.with_name(output_csv.stem + "_by_mode.csv")
    summary.to_csv(summary_csv, index=False)

    recovery = _experiment_recovery_table(stats)
    recovery_csv = output_csv.with_name(output_csv.stem + "_experiment_recovery.csv")
    if not recovery.empty:
        recovery.to_csv(recovery_csv, index=False)

    ranking = _ranking_table(
        stats,
        start_date=args.ranking_start_date,
        end_date=args.ranking_end_date,
    )
    ranking_csv = output_csv.with_name(output_csv.stem + "_ranking.csv")
    ranking_md = output_md.with_name(output_md.stem + "_ranking.md")
    ranking_txt = output_md.with_name(output_md.stem + "_ranking.txt")
    if not ranking.empty:
        ranking.to_csv(ranking_csv, index=False)
        formatted_ranking = _format_table(ranking)
        _write_markdown(formatted_ranking, ranking_md)
        ranking_txt.write_text(
            formatted_ranking.to_string(index=False) + "\n",
            encoding="utf-8",
        )

    if not ranking.empty:
        print("\nRanking by Total Return")
        print(_format_table(ranking).to_string(index=False))
    if not args.ranking_only:
        print("\nRetrieval Ablation Runs")
        print(display.to_string(index=False))
        if not recovery.empty:
            print("\nExperiment Mode Recovery")
            print(_format_table(recovery).to_string(index=False))
        print("\nBy-Mode Summary")
        print(_format_table(summary).to_string(index=False))
    print(f"\nSaved per-run CSV: {output_csv}")
    print(f"Saved per-run Markdown: {output_md}")
    print(f"Saved by-mode CSV: {summary_csv}")
    if not recovery.empty:
        print(f"Saved experiment recovery CSV: {recovery_csv}")
    if not ranking.empty:
        print(f"Saved ranking CSV: {ranking_csv}")
        print(f"Saved ranking Markdown: {ranking_md}")
        print(f"Saved ranking TXT: {ranking_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
