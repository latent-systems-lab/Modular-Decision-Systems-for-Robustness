"""Custom data-provider wrappers for placebo/synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import os
import sqlite3
from math import erf, exp, log, sqrt
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core.data_manager import DataManager
from src.models.pydantic_models import OptionData, StockData


def normalize_option_type_to_int(option_type: Union[int, str]) -> int:
    """Convert option_type to int (0=put, 1=call), handling multiple formats."""
    if isinstance(option_type, int):
        return option_type
    s = str(option_type).lower().strip()
    if s in ('call', 'c', '1'):
        return 1
    if s in ('put', 'p', '0'):
        return 0
    raise ValueError(f"Invalid option_type: {option_type}")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * np.pi)


def black_scholes_european(
    spot: float, strike: float, t_years: float, rate: float, sigma: float, is_call: bool
) -> Dict[str, float]:
    if t_years <= 0:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        return {"price": intrinsic, "delta": 1.0 if (is_call and spot > strike) else -1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sigma = max(sigma, 1e-4)
    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt(t_years))
    d2 = d1 - sigma * sqrt(t_years)
    if is_call:
        price = spot * _norm_cdf(d1) - strike * exp(-rate * t_years) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = strike * exp(-rate * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0

    gamma = _norm_pdf(d1) / (spot * sigma * sqrt(t_years))
    vega = spot * sqrt(t_years) * _norm_pdf(d1)
    theta = -(spot * _norm_pdf(d1) * sigma) / (2 * sqrt(t_years))
    rho = strike * t_years * exp(-rate * t_years) * (_norm_cdf(d2) if is_call else -_norm_cdf(-d2))
    return {"price": max(price, 0.01), "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


class ScrambledCalendarDataProvider:
    """Remaps requested dates to a deterministic scrambled trading calendar."""

    def __init__(
        self,
        base: DataManager,
        trading_days: List[date],
        scrambled_days: List[date],
        log_prefix: str = "",
        scrambled_option_id_fix: bool = False,
    ):
        self.base = base
        self.trading_days = trading_days
        self.scrambled_days = scrambled_days
        self.date_map = {logical: real for logical, real in zip(trading_days, scrambled_days)}
        self.inverse_map = {real: logical for logical, real in self.date_map.items()}
        self.last_valid_option_day: Dict[str, date] = {}
        self.missing_option_chain_logs: List[str] = []
        self.log_prefix = log_prefix
        self.scrambled_option_id_fix = scrambled_option_id_fix
        self.option_key_by_id: Dict[str, tuple] = {}
        self.option_id_by_key: Dict[tuple, Dict[date, str]] = {}

    def _rewrite_option_id_date(self, option_id: str, target_date: date) -> Optional[str]:
        if not self.scrambled_option_id_fix:
            return None
        if not option_id:
            return None
        tokens = re.findall(r"\d{4}-\d{2}-\d{2}", option_id)
        if not tokens:
            return None
        new_date = target_date.isoformat()
        if tokens[0] == new_date:
            return None
        return re.sub(r"\d{4}-\d{2}-\d{2}", new_date, option_id, count=1)

    def _option_key_from_row(self, ticker: str, option: OptionData) -> tuple:
        expire = option.expire_date if isinstance(option.expire_date, date) else option.expire_date.date()
        strike = round(float(option.strike), 4)
        try:
            opt_type = normalize_option_type_to_int(option.option_type)
        except ValueError:
            opt_type = int(option.option_type)
        return (ticker, opt_type, strike, expire)

    def _lookup_option(self, option_id: str, ticker: str, target_date: date):
        option_row = self.base.get_option_data_by_id(option_id, ticker, target_date)
        if option_row is not None:
            return option_row

        if not self.scrambled_option_id_fix:
            return None

        rewritten = self._rewrite_option_id_date(option_id, target_date)
        if rewritten:
            option_row = self.base.get_option_data_by_id(rewritten, ticker, target_date)
            if option_row is not None:
                return option_row

        key = self.option_key_by_id.get(option_id)
        if key is not None:
            alt_id = self.option_id_by_key.get(key, {}).get(target_date)
            if alt_id and alt_id != option_id:
                option_row = self.base.get_option_data_by_id(alt_id, ticker, target_date)
                if option_row is not None:
                    return option_row

        return None

    def remap_date(self, requested_date: date) -> date:
        return self.date_map.get(requested_date, requested_date)

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def check_stock_data_exists(self, ticker: str, target_date: date) -> bool:
        return self.base.check_stock_data_exists(ticker, self.remap_date(target_date))

    def load_stock_data(self, ticker: str, start_date: date = None, end_date: date = None) -> List[StockData]:
        return self.base.load_stock_data(ticker, start_date=start_date, end_date=end_date)

    def get_stock_price(self, ticker: str, target_date: date) -> Optional[float]:
        return self.base.get_stock_price(ticker, self.remap_date(target_date))

    def get_technical_indicators(self, ticker: str, target_date: date) -> Dict[str, float]:
        return self.base.get_technical_indicators(ticker, self.remap_date(target_date))

    def load_options_data(self, ticker: str, quote_date: date, **kwargs) -> List[OptionData]:
        mapped_date = self.remap_date(quote_date)
        options = self.base.load_options_data(ticker, mapped_date, **kwargs)
        if options:
            self.last_valid_option_day[ticker] = mapped_date
            if self.scrambled_option_id_fix:
                for opt in options:
                    key = self._option_key_from_row(ticker, opt)
                    self.option_key_by_id[opt.option_id] = key
                    self.option_id_by_key.setdefault(key, {})[mapped_date] = opt.option_id
            return options

        last_valid = self.last_valid_option_day.get(ticker)
        if last_valid is None:
            self.missing_option_chain_logs.append(f"{self.log_prefix}missing option chain for {ticker} @ {mapped_date}; skip")
            return []

        self.missing_option_chain_logs.append(f"{self.log_prefix}missing option chain for {ticker} @ {mapped_date}; forward-filled from {last_valid}")
        return self.base.load_options_data(ticker, last_valid, **kwargs)

    def get_option_data_by_id(self, option_id: str, ticker: str, quote_date: date):
        mapped_date = self.remap_date(quote_date)
        option_row = self._lookup_option(option_id, ticker, mapped_date)
        if option_row is not None:
            self.last_valid_option_day[ticker] = mapped_date
            return option_row

        last_valid = self.last_valid_option_day.get(ticker)
        if last_valid is None:
            self.missing_option_chain_logs.append(f"{self.log_prefix}missing option id {option_id} for {ticker} @ {mapped_date}; skip")
            return None

        self.missing_option_chain_logs.append(f"{self.log_prefix}missing option id {option_id} for {ticker} @ {mapped_date}; forward-filled from {last_valid}")
        return self._lookup_option(option_id, ticker, last_valid)


@dataclass
class _RegimeSpec:
    drift: float
    vol: float
    iv_offset: float


class SyntheticRegimeDataProvider:
    """3-regime synthetic market + on-the-fly option-chain generation."""

    regimes = {
        0: _RegimeSpec(drift=0.0003, vol=0.008, iv_offset=0.00),
        1: _RegimeSpec(drift=-0.0001, vol=0.020, iv_offset=0.10),
        2: _RegimeSpec(drift=-0.0012, vol=0.030, iv_offset=0.18),
    }

    def __init__(self, ticker: str = "SYN", length: int = 756, seed: int = 42, start_date: date = date(2021, 1, 4)):
        self.ticker = ticker
        self.length = length
        self.seed = seed
        self.start_date = start_date
        self.rng = np.random.default_rng(seed)
        self.transition = np.array([
            [0.95, 0.04, 0.01],
            [0.08, 0.86, 0.06],
            [0.10, 0.25, 0.65],
        ])
        self.base_iv = 0.17
        self.rf = 0.01
        self.strike_multipliers = [0.9, 0.95, 1.0, 1.05, 1.1]
        self._option_cache: Dict[tuple, List[OptionData]] = {}
        self._option_lookup: Dict[tuple, Dict[str, float]] = {}
        self._build_series()
        # Pre-compute a fixed strike grid from the initial price so option IDs
        # remain consistent across trading days (strikes don't shift with spot).
        p0 = self.df["Close"].iloc[0]
        self._fixed_strikes = sorted(set(round(p0 * m, 2) for m in self.strike_multipliers))
        # Pre-compute fixed expiration dates (biweekly cycle) so the same
        # option contract persists across multiple trading days.
        self._fixed_expiries: List[date] = []
        exp = self.start_date + timedelta(days=14)
        horizon_end = self.trading_days[-1] + timedelta(days=90)
        while exp <= horizon_end:
            self._fixed_expiries.append(exp)
            exp += timedelta(days=14)

    def _build_series(self) -> None:
        idx = pd.bdate_range(self.start_date, periods=self.length)
        regime_states = np.zeros(self.length, dtype=int)
        price = np.zeros(self.length, dtype=float)
        price[0] = 100.0
        for i in range(1, self.length):
            prev_state = regime_states[i - 1]
            regime_states[i] = self.rng.choice([0, 1, 2], p=self.transition[prev_state])
            spec = self.regimes[regime_states[i]]
            ret = spec.drift + spec.vol * self.rng.normal()
            price[i] = max(1.0, price[i - 1] * exp(ret))

        self.df = pd.DataFrame({"Date": idx, "Close": price, "Regime": regime_states})
        self.df["Open"] = self.df["Close"].shift(1).fillna(self.df["Close"]) * (1 + 0.001 * self.rng.normal(size=self.length))
        self.df["High"] = self.df[["Open", "Close"]].max(axis=1) * 1.005
        self.df["Low"] = self.df[["Open", "Close"]].min(axis=1) * 0.995
        self.df["Volume"] = (1_000_000 * (1 + 0.1 * self.rng.normal(size=self.length))).clip(200_000).astype(int)
        self.date_to_row = {d.date(): r for d, r in zip(self.df["Date"], self.df.to_dict("records"))}
        self.trading_days = [d.date() for d in self.df["Date"]]

    def check_stock_data_exists(self, ticker: str, target_date: date) -> bool:
        return ticker == self.ticker and target_date in self.date_to_row

    def load_stock_data(self, ticker: str, start_date: date = None, end_date: date = None) -> List[StockData]:
        if ticker != self.ticker:
            return []
        data = self.df.copy()
        if start_date:
            data = data[data["Date"] >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data["Date"] <= pd.Timestamp(end_date)]
        return [StockData(date=r.Date.date(), open=float(r.Open), high=float(r.High), low=float(r.Low), close=float(r.Close), volume=int(r.Volume)) for r in data.itertuples(index=False)]

    def get_stock_price(self, ticker: str, target_date: date) -> Optional[float]:
        row = self.date_to_row.get(target_date)
        return None if ticker != self.ticker or row is None else float(row["Close"])

    def get_technical_indicators(self, ticker: str, target_date: date) -> Dict[str, float]:
        from src.utils.financial_utils import calculate_rsi, calculate_moving_average, calculate_bollinger_bands, calculate_macd, calculate_atr
        hist = [d for d in self.trading_days if d <= target_date]
        closes = [self.date_to_row[d]["Close"] for d in hist]
        highs = [self.date_to_row[d]["High"] for d in hist]
        lows = [self.date_to_row[d]["Low"] for d in hist]
        lb, mb, ub = calculate_bollinger_bands(closes)
        macd, signal, histo = calculate_macd(closes)
        sma50 = calculate_moving_average(closes, 50)
        sma200 = calculate_moving_average(closes, 200)
        cur = closes[-1] if closes else 0.0
        return {
            "rsi_14": calculate_rsi(closes, 14),
            "sma_20": calculate_moving_average(closes, 20),
            "sma_50": sma50,
            "sma_200": sma200,
            "bollinger_lower": lb,
            "bollinger_middle": mb,
            "bollinger_upper": ub,
            "macd_line": macd,
            "macd_signal": signal,
            "macd_histogram": histo,
            "atr_14": calculate_atr(highs, lows, closes),
            "price_vs_sma_50": (cur / sma50 - 1.0) if sma50 else 0.0,
            "price_vs_sma_200": (cur / sma200 - 1.0) if sma200 else 0.0,
        }

    def _chain_for_day(self, quote_date: date) -> List[OptionData]:
        key = (self.ticker, quote_date)
        if key in self._option_cache:
            return self._option_cache[key]
        spot = self.get_stock_price(self.ticker, quote_date)
        if spot is None:
            return []
        state = int(self.date_to_row[quote_date]["Regime"])
        sigma = self.base_iv + self.regimes[state].iv_offset
        rows: List[OptionData] = []
        # Iterate over fixed expiration dates; include if 5 <= DTE <= 60
        for expire in self._fixed_expiries:
            dte = (expire - quote_date).days
            if dte < 5 or dte > 60:
                continue
            t = dte / 365.0
            for strike in self._fixed_strikes:
                for is_call in (0, 1):
                    bs = black_scholes_european(spot, strike, t, self.rf, sigma, is_call=bool(is_call))
                    option_type_str = "C" if is_call else "P"
                    oid = f"{self.ticker}_{option_type_str}_{strike:.2f}_{expire.strftime('%Y%m%d')}"
                    opt = OptionData(
                        quote_date=quote_date,
                        option_id=oid,
                        option_type=is_call,
                        strike=strike,
                        expire_date=expire,
                        dte=dte,
                        last=float(bs["price"]),
                        bid=float(bs["price"] * 0.99),
                        ask=float(bs["price"] * 1.01),
                        delta=float(bs["delta"]),
                        gamma=float(bs["gamma"]),
                        vega=float(bs["vega"]),
                        theta=float(bs["theta"]),
                        volume=1000.0,
                        rho=float(bs["rho"]),
                        moneyness=float(spot / strike),
                        underlying_last=float(spot),
                        iv=float(sigma),
                    )
                    rows.append(opt)
                    self._option_lookup[(oid, quote_date)] = {
                        "LAST": opt.last,
                        "BID": opt.bid,
                        "ASK": opt.ask,
                        "IV": opt.iv,
                        "DELTA": opt.delta,
                    }
        self._option_cache[key] = rows
        return rows

    def load_options_data(self, ticker: str, quote_date: date, **kwargs) -> List[OptionData]:
        return self._chain_for_day(quote_date) if ticker == self.ticker else []

    def get_option_data_by_id(self, option_id: str, ticker: str, quote_date: date):
        if ticker != self.ticker:
            return None
        if (option_id, quote_date) not in self._option_lookup:
            self._chain_for_day(quote_date)
        result = self._option_lookup.get((option_id, quote_date))
        if result is not None:
            return result
        # Fuzzy fallback — match by option type + closest strike when the
        # LLM hallucinates an ID not in the available chain.
        parts = option_id.split('_')
        req_type = None
        req_strike = None
        if len(parts) >= 3 and parts[1].upper() in ('C', 'CALL', 'P', 'PUT'):
            req_type = 'C' if parts[1].upper() in ('C', 'CALL') else 'P'
            try:
                req_strike = float(parts[2])
            except ValueError:
                pass
        if req_type is None or req_strike is None:
            return None
        best, best_diff = None, float('inf')
        for (oid, qd), data in self._option_lookup.items():
            if qd != quote_date:
                continue
            oparts = oid.split('_')
            if len(oparts) >= 3 and oparts[1] == req_type:
                try:
                    s = float(oparts[2])
                except ValueError:
                    continue
                diff = abs(s - req_strike)
                if diff < best_diff:
                    best_diff = diff
                    best = data
        if best is not None and best_diff < req_strike * 0.02:
            return best
        # Last resort: synthesize option price using Black-Scholes
        # Extract strike and type from hallucinated ID, calculate price for today
        spot = self.get_stock_price(self.ticker, quote_date)
        if spot is None or req_strike is None:
            return None
        state = int(self.date_to_row[quote_date]["Regime"])
        sigma = self.base_iv + self.regimes[state].iv_offset
        # Use closest available expiry (30 DTE default)
        expire = quote_date + timedelta(days=30)
        dte = 30
        t = dte / 365.0
        is_call = req_type == 'C'
        bs = black_scholes_european(spot, req_strike, t, self.rf, sigma, is_call=is_call)
        return {
            "LAST": bs["price"],
            "BID": bs["price"] * 0.99,
            "ASK": bs["price"] * 1.01,
            "IV": sigma,
            "DELTA": bs["delta"],
        }

    def export_market_data(self, data_dir: str) -> Dict[str, str]:
        os.makedirs(data_dir, exist_ok=True)
        stock_path = os.path.join(data_dir, f"{self.ticker}.csv")
        stock_df = self.df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        stock_df.to_csv(stock_path, index=False)

        option_rows = []
        for quote_date in self.trading_days:
            for opt in self._chain_for_day(quote_date):
                option_rows.append(
                    {
                        "QUOTE_DATE": f"{opt.quote_date:%Y-%m-%d} 00:00:00",
                        "option_id": opt.option_id,
                        "is_call": opt.option_type,
                        "STRIKE": opt.strike,
                        "EXPIRE_DATE": f"{opt.expire_date:%Y-%m-%d} 00:00:00",
                        "UNDERLYING_LAST": opt.underlying_last,
                        "DTE": opt.dte,
                        "LAST": opt.last,
                        "BID": opt.bid,
                        "ASK": opt.ask,
                        "VOLUME": opt.volume,
                        "IV": opt.iv,
                        "DELTA": opt.delta,
                        "GAMMA": opt.gamma,
                        "VEGA": opt.vega,
                        "THETA": opt.theta,
                        "RHO": opt.rho,
                        "moneyness": opt.moneyness,
                    }
                )

        options_db_path = os.path.join(data_dir, f"{self.ticker.lower()}_options.db")
        options_df = pd.DataFrame(option_rows)
        with sqlite3.connect(options_db_path) as conn:
            options_df.to_sql("options", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quote_date ON options (QUOTE_DATE)")

        return {"stock_csv": stock_path, "options_db": options_db_path}
