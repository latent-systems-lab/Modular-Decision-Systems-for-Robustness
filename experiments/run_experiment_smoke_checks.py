"""Run 2-trading-day smoke checks for newly added experiment scripts."""

from __future__ import annotations

import os
import sqlite3
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "smoke_checks"


@dataclass
class SmokeResult:
    name: str
    command: str
    returncode: int
    output_path: Path


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


def _black_scholes(spot: float, strike: float, t: float, rate: float, sigma: float, is_call: bool) -> tuple[float, float, float, float, float, float]:
    if t <= 0:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        delta = 1.0 if (is_call and spot >= strike) else (-1.0 if not is_call and strike > spot else 0.0)
        return intrinsic, delta, 0.0, 0.0, 0.0, 0.0

    sigma = max(sigma, 1e-6)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if is_call:
        price = spot * _norm_cdf(d1) - strike * np.exp(-rate * t) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = strike * np.exp(-rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0

    gamma = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi) / (spot * sigma * np.sqrt(t))
    vega = spot * np.sqrt(t) * np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    theta = -(spot * np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi) * sigma) / (2 * np.sqrt(t))
    rho = strike * t * np.exp(-rate * t) * (_norm_cdf(d2) if is_call else -_norm_cdf(-d2))
    return max(price, 0.01), delta, gamma, vega, theta, rho


def ensure_smoke_market_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2024-01-02", periods=5)

    def write_stock_csv(ticker: str, start_price: float) -> pd.DataFrame:
        close = np.array([start_price, start_price * 1.003, start_price * 0.998, start_price * 1.01, start_price * 1.006])
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.full(len(dates), 1_000_000, dtype=int),
            }
        )
        frame.to_csv(DATA_DIR / f"{ticker}.csv", index=False)
        return frame

    aapl_df = write_stock_csv("AAPL", 190.0)
    spy_df = write_stock_csv("SPY", 470.0)

    def write_option_db(ticker: str, stock_df: pd.DataFrame) -> None:
        db_path = DATA_DIR / f"{ticker.lower()}_options.db"
        rows = []
        rate = 0.01
        sigma = 0.22
        for _, row in stock_df.iterrows():
            quote_date = pd.Timestamp(row["Date"])
            spot = float(row["Close"])
            for dte in (7, 14, 30):
                expire = quote_date + timedelta(days=dte)
                t = dte / 365.0
                for mult in (0.95, 1.0, 1.05):
                    strike = round(spot * mult, 2)
                    for is_call in (0, 1):
                        px, delta, gamma, vega, theta, rho = _black_scholes(spot, strike, t, rate, sigma, bool(is_call))
                        rows.append(
                            {
                                "QUOTE_DATE": quote_date.strftime("%Y-%m-%d 00:00:00"),
                                "option_id": f"{ticker}_{quote_date.date()}_{expire.date()}_{is_call}_{strike:.2f}",
                                "is_call": is_call,
                                "STRIKE": strike,
                                "EXPIRE_DATE": expire.strftime("%Y-%m-%d 00:00:00"),
                                "UNDERLYING_LAST": spot,
                                "DTE": dte,
                                "LAST": px,
                                "BID": px * 0.99,
                                "ASK": px * 1.01,
                                "VOLUME": 100,
                                "IV": sigma,
                                "DELTA": delta,
                                "GAMMA": gamma,
                                "VEGA": vega,
                                "THETA": theta,
                                "RHO": rho,
                                "moneyness": spot / strike,
                            }
                        )

        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(rows).to_sql("options", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quote_date ON options (QUOTE_DATE)")

    write_option_db("AAPL", aapl_df)
    write_option_db("SPY", spy_df)


def run_cmd(name: str, cmd: list[str]) -> SmokeResult:
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    child_env = os.environ.copy()
    child_env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=child_env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    log_path.write_text(proc.stdout + "\n\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    return SmokeResult(name=name, command=" ".join(cmd), returncode=proc.returncode, output_path=log_path)


def main() -> int:
    ensure_smoke_market_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checks = [
        (
            "synthetic_3_regime",
            [
                sys.executable,
                "experiments/run_synthetic_3_regime.py",
                "--length",
                "10",
                "--seed",
                "123",
                "--output-dir",
                str(RESULTS_DIR / "synthetic_3_regime"),
            ],
        ),
        (
            "scrambled_timeline_placebo",
            [
                sys.executable,
                "experiments/run_scrambled_timeline_placebo.py",
                "--assets",
                "AAPL",
                "--start-date",
                "2024-01-02",
                "--end-date",
                "2024-01-08",
                "--n-scrambles",
                "1",
                "--block-len",
                "2",
                "--output-dir",
                str(RESULTS_DIR / "scrambled_timeline_placebo"),
            ],
        ),
        (
            "spy_1y_sanity",
            [
                sys.executable,
                "experiments/run_spy_1y_sanity.py",
                "--start-date",
                "2024-01-02",
                "--end-date",
                "2024-01-08",
                "--output-dir",
                str(RESULTS_DIR / "spy_1y_sanity"),
            ],
        ),
    ]

    results = [run_cmd(name, cmd) for name, cmd in checks]
    summary_rows = []
    failed = False
    for r in results:
        status = "PASS" if r.returncode == 0 else "FAIL"
        summary_rows.append({"check": r.name, "status": status, "returncode": r.returncode, "log": str(r.output_path)})
        if r.returncode != 0:
            failed = True

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
