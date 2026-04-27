# Rule-based baseline backtests.
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------------
# SETTINGS
# ----------------------------------
TICKERS          = ["AAPL", "NVDA", "TSLA"]
DATA_START       = "2021-01-01"       
DATA_END         = "2023-12-31"
BACKTEST_START   = pd.Timestamp("2021-01-01")
INITIAL_CAPITAL  = 1_000_000
REB_FREQ         = "MS"                
LOOKBACK_MV      = 60                  
LOOKBACK_ZMR     = 21                  
LOOKBACK_TSM     = 252                 
OUT_BASE         = Path("results/baseline/rule-based2")

# ----------------------------------
# DATA (load from local CSV files in `data/` instead of yfinance)
# ----------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_local_prices(tickers, data_dir, start=None, end=None):
    """Load `Close` prices for each ticker from data/{TICKER}.csv and
    return a DataFrame of Close prices (columns = tickers).

    Expects CSVs with a `Date` column and `Close` column (like the repo's
    CSVs). Missing files or completely empty results raise SystemExit with a
    helpful message.
    """
    frames = {}
    missing = []
    for t in tickers:
        fpath = data_dir / f"{t}.csv"
        if not fpath.exists():
            missing.append(t)
            continue
        try:
            df = pd.read_csv(fpath, parse_dates=["Date"])
        except Exception as e:
            raise SystemExit(f"Failed to read {fpath}: {e}")
        if "Close" not in df.columns:
            raise SystemExit(f"{fpath} does not contain a 'Close' column.")
        df = df.set_index("Date").sort_index()
        # restrict to requested window if provided
        if start is not None:
            df = df.loc[df.index >= pd.to_datetime(start)]
        if end is not None:
            df = df.loc[df.index <= pd.to_datetime(end)]
        if df.empty:
            raise SystemExit(f"Price file for {t} is empty after filtering by date ({start}..{end}).")
        frames[t] = df["Close"].rename(t)

    if missing:
        raise SystemExit(f"Missing local price files for tickers: {missing}.\n"
                         "Place CSV files named TICKER.csv under the `data/` folder.")

    # combine into a single DataFrame (outer join), then drop rows where all assets are NA
    prices = pd.concat(frames.values(), axis=1, keys=frames.keys())
    prices = prices.sort_index()
    prices = prices.loc[(prices.index >= pd.to_datetime(DATA_START)) & (prices.index <= pd.to_datetime(DATA_END))]
    prices = prices.dropna(how="all")
    if prices.empty:
        raise SystemExit("Combined `prices` is empty after loading local CSVs. Check files and date range.")
    return prices


prices = load_local_prices(TICKERS, DATA_DIR, start=DATA_START, end=DATA_END)

rets = prices.pct_change().dropna()

# ----------------------------------
# HELPERS
# ----------------------------------
def first_dates(idx, freq):
    """Return first trading day in each freq period."""
    return idx.to_series().resample(freq).first().dropna().index

def mv_weights(r):
    """Minimum-variance via inverse variance (long-only, sums to 1)."""
    # annualise variances
    variances = r.var() * 252
    inv_var   = 1 / variances
    w         = inv_var / inv_var.sum()
    return w

def zmr_weights(prices_slice):
    """Contrarian: 100 % weight to worst 1-month performer."""
    perf_1m = prices_slice.pct_change().iloc[-1]  # last 1-month return
    worst   = perf_1m.idxmin()
    w = pd.Series(0.0, index=prices_slice.columns)
    w[worst] = 1.0
    return w

def tsm_weights(prices_slice):
    """Time-series momentum: equal weight to assets with pos 12-m return."""
    perf_12m = prices_slice.iloc[-1] / prices_slice.iloc[-252] - 1
    longers  = perf_12m[perf_12m > 0].index.tolist()
    w = pd.Series(0.0, index=prices_slice.columns)
    if longers:                # at least one long signal
        w[longers] = 1 / len(longers)
    # if none positive -> 100 % cash (all zeros)
    return w

def ew_weights(prices_slice=None):
    """Equal-weighted portfolio across the current universe.

    If `prices_slice` is provided, use its columns as the universe. Otherwise
    fall back to the global TICKERS list. Returns a long-only weight Series
    that sums to 1 (or all zeros if universe is empty).
    """
    if prices_slice is None:
        universe = TICKERS
    else:
        universe = list(prices_slice.columns)

    if not universe:
        return pd.Series(0.0, index=TICKERS)

    w = pd.Series(0.0, index=TICKERS)
    per = 1.0 / len(universe)
    for t in universe:
        # only set weights for known tickers to keep index consistent
        if t in w.index:
            w[t] = per
    return w

# Note: 'BH' (buy-and-hold) is handled specially below (shares are bought once
# at the start with equal capital per asset and then held; weights are not
# rebalanced).
STRAT_FUNCS = {"MV": mv_weights, "ZMR": zmr_weights, "EW": ew_weights, "BH": None}

# ----------------------------------
# BACKTEST LOOP
# ----------------------------------
results = {}
_summary_rows = []

for strat, weight_func in STRAT_FUNCS.items():
    backtest_idx = rets.loc[BACKTEST_START:].index
    if backtest_idx.empty:
        raise SystemExit(f"No return data available for backtest period starting {BACKTEST_START}.\n"
                         "Ensure `prices` covers this period and that the yfinance download succeeded.")

    port_val = pd.Series(index=backtest_idx, dtype=float)
    # initialise first observation (guarded)
    if port_val.size == 0:
        raise SystemExit(f"Backtest index is empty for strategy {strat}. Cannot initialise portfolio values.")
    port_val.iloc[0] = INITIAL_CAPITAL
    weights   = pd.Series(0.0, index=TICKERS)
    rebal_dates = first_dates(rets.loc[BACKTEST_START:].index, REB_FREQ)
    # Special setup for buy-and-hold: compute shares bought at first backtest date
    shares = None
    if strat == "BH":
        # find first price row corresponding to the backtest start index
        first_bt_date = backtest_idx[0]
        try:
            price0 = prices.loc[first_bt_date]
        except KeyError:
            # find the first available price date on/after the backtest date
            dates = prices.index
            pos = dates.searchsorted(first_bt_date)
            if pos >= len(dates):
                raise SystemExit(f"No price data available on/after first backtest date {first_bt_date} for BH strategy.")
            price0 = prices.iloc[pos]

        available = price0.dropna()
        if available.empty:
            raise SystemExit(f"No valid prices on the first backtest date {first_bt_date} to initialise BH strategy.")
        universe = [t for t in available.index if t in TICKERS]
        if not universe:
            raise SystemExit("No tickers with valid prices on BH start date.")
        per_cap = INITIAL_CAPITAL / len(universe)
        shares = pd.Series(0.0, index=prices.columns)
        for t in universe:
            shares[t] = per_cap / float(price0[t])

    for t, date in enumerate(port_val.index):
        # Rebalance?
        if date in rebal_dates:
            # slice up to the day before the rebalance date
            past_prices = prices.loc[:date - pd.Timedelta(days=1)]
            past_rets   = rets.loc[:date - pd.Timedelta(days=1)]
            if strat == "MV":
                if len(past_rets) >= LOOKBACK_MV:
                    weights = weight_func(past_rets.iloc[-LOOKBACK_MV:])
            elif strat == "ZMR":
                if len(past_prices) >= LOOKBACK_ZMR:
                    weights = weight_func(past_prices.iloc[-(LOOKBACK_ZMR+1):])
            elif strat == "TSM":
                if len(past_prices) >= LOOKBACK_TSM:
                    weights = weight_func(past_prices)
            elif strat == "EW":
                # Equal-weight doesn't require a lookback; use current universe
                weights = weight_func(past_prices)
            # (weights already long-only / sum to 1 or 0)

        # Daily mark-to-market
        if strat == "BH":
            # mark-to-market using fixed shares
            pv = (prices.loc[date].reindex(shares.index).fillna(0.0) * shares).sum()
            port_val.iloc[t] = pv
            continue

        daily_ret = (rets.loc[date] * weights).sum()
        if t:  # from day 1 onward
            port_val.iloc[t] = port_val.iloc[t-1] * (1 + daily_ret)

    # ----------------------------------
    # SAVE
    # ----------------------------------
    out_dir = OUT_BASE / strat
    out_dir.mkdir(parents=True, exist_ok=True)
    (port_val
        .reset_index()
        .rename(columns={"index": "date", 0: "portfolio_value"})
        .to_csv(out_dir / "performance.csv", index=False, float_format="%.2f")
    )

    results[strat] = port_val

    # ----- metrics & plot per-strategy -----
    def compute_metrics(series):
        """Compute portfolio performance metrics.

        Returns a dict with:
        - final_value
        - total_return (decimal)
        - total_return_pct (percent)
        - cagr (decimal)
        - ann_vol (decimal)
        - volatility_pct (percent)
        - sharpe
        - sortino
        - max_drawdown (decimal, negative)
        - max_drawdown_pct (percent, positive)
        - calmar
        """
        final = series.iloc[-1]
        start = series.index[0]
        end = series.index[-1]
        years = (end - start).days / 365.25

        total_ret = final / INITIAL_CAPITAL - 1
        total_return_pct = total_ret * 100

        cagr = (final / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan

        # daily returns from portfolio values
        daily_rets = series.pct_change().dropna()

        # Annualised volatility
        ann_vol = daily_rets.std() * np.sqrt(252) if not daily_rets.empty else np.nan
        volatility_pct = ann_vol * 100 if not np.isnan(ann_vol) else np.nan

        # Annualised mean return (from arithmetic mean of daily returns)
        ann_mean_ret = daily_rets.mean() * 252 if not daily_rets.empty else np.nan

        # Sharpe ratio (assumes risk-free rate = 0)
        sharpe = ann_mean_ret / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan

        # Sortino ratio: use downside deviation (target = 0)
        downside = daily_rets[daily_rets < 0]
        if not downside.empty:
            downside_std = downside.std() * np.sqrt(252)
            sortino = ann_mean_ret / downside_std if downside_std and not np.isnan(downside_std) else np.nan
        else:
            # No downside returns -> undefined (set to nan)
            sortino = np.nan

        # Drawdown / Max drawdown
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax
        mdd = drawdown.min() if not drawdown.empty else np.nan
        max_drawdown_pct = (abs(mdd) * 100) if not np.isnan(mdd) else np.nan

        # Calmar ratio: CAGR / max_drawdown (use absolute drawdown). If MDD==0 -> nan
        if not np.isnan(cagr) and mdd != 0 and not np.isnan(mdd):
            calmar = cagr / abs(mdd) if abs(mdd) > 0 else np.nan
        else:
            calmar = np.nan

        return {
            "total_return_pct": total_return_pct,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "calmar": calmar,
        }

    metrics = compute_metrics(port_val)
    _summary_rows.append({"strategy": strat, **metrics})

    # plot equity curve
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        port_val.plot(ax=ax)
        ax.set_title(f"Equity curve - {strat}")
        ax.set_ylabel("Portfolio value")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / "equity_curve.png")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to create plot for {strat}: {e}")

# ----- combined outputs -----
if _summary_rows:
    summary_df = pd.DataFrame(_summary_rows).set_index("strategy").sort_index()
    # save summary
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_BASE / "summary.csv", float_format="%.6f")

    # combined equity plot (normalized to initial capital)
    try:
        all_pf = pd.DataFrame(results)
        norm = all_pf / INITIAL_CAPITAL
        fig, ax = plt.subplots(figsize=(10, 6))
        norm.plot(ax=ax)
        ax.set_title("Normalized equity (all strategies)")
        ax.set_ylabel("Value (multiple of initial)")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(OUT_BASE / "all_strategies_equity.png")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to create combined plot: {e}")

    # print summary to console (rounded for readability)
    with pd.option_context('display.float_format', '{:,.4f}'.format):
        print("\nSummary metrics:")
        print(summary_df)

print("Finished!  CSV files written under:", OUT_BASE.resolve())
