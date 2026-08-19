"""Training utilities for reinforcement-learning baseline strategies."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple, Type

import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3 import A2C, DDPG, DQN, PPO

class PortfolioEnv(gym.Env):
    """Portfolio environment optimised for Sortino ratio reward."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, initial_capital: float = 1e6, window: int = 90) -> None:
        super().__init__()
        prices_ext = prices.copy()
        prices_ext["CASH"] = 1.0
        self.prices = prices_ext.sort_index()
        self.dates = self.prices.index.to_list()
        self.n_steps = len(self.prices) - 1
        self.n_assets = self.prices.shape[1]
        self.initial_capital = initial_capital
        self.window = window

        self.action_space = spaces.Box(-1.0, 1.0, (self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, np.inf, (self.n_assets,), dtype=np.float32)

        self.step_idx = 0
        self.portfolio_value = initial_capital
        self.returns: list[float] = []
        self.price_rel = np.ones(self.n_assets, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.step_idx = 0
        self.portfolio_value = self.initial_capital
        self.returns.clear()
        self._update_price_rel()
        return self.price_rel

    def _update_price_rel(self) -> None:
        p0 = self.prices.iloc[self.step_idx].values
        p1 = self.prices.iloc[self.step_idx + 1].values
        self.price_rel = (p1 / p0).astype(np.float32)

    def step(self, action):  # type: ignore[override]
        weights = np.abs(action)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones_like(weights) / self.n_assets

        growth = float((weights * self.price_rel).sum())
        ret = growth - 1.0
        self.returns.append(ret)
        self.portfolio_value *= growth

        reward = 0.0
        if len(self.returns) >= self.window:
            recent = np.array(self.returns[-self.window:], dtype=np.float64)
            mu = recent.mean()
            downside = recent[recent < 0]
            downside_std = downside.std(ddof=0) if downside.size > 0 else 1e-6
            sortino = (mu / downside_std) * np.sqrt(252)
            reward = float(np.clip(sortino, -5, 5))

        self.step_idx += 1
        done = self.step_idx >= self.n_steps
        if not done:
            self._update_price_rel()
            obs = self.price_rel
        else:
            obs = np.zeros_like(self.price_rel)

        info = {"portfolio_value": self.portfolio_value, "date": self.dates[self.step_idx]}
        return obs, reward, done, info

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise NotImplementedError("Only human render mode is supported.")
        dt = self.dates[self.step_idx]
        print(f"{dt} | PV: {self.portfolio_value:.2f}")


class DiscretePortfolioEnv(PortfolioEnv):
    """
    Override only the action space & step mapping for DQN:
    Discrete(n_assets) → one‐hot weight vector.
    """
    def __init__(self, prices: pd.DataFrame, initial_capital: float = 1e6):
        super().__init__(prices, initial_capital)
        self.action_space = spaces.Discrete(self.n_assets)

    def step(self, action: int):  # type: ignore[override]
        one_hot = np.zeros(self.n_assets, dtype=np.float32)
        one_hot[action] = 1.0
        return super().step(one_hot)


def load_data(tickers: Iterable[str], start: str, end: str, folder: str = "data") -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        path = os.path.join(folder, f"{ticker}.csv")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df = df.sort_index().loc[start:end, ["Close"]].rename(columns={"Close": ticker})
        frames.append(df)
    return pd.concat(frames, axis=1).dropna()


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    pv = df["portfolio_value"]
    rets = pv.pct_change().dropna()
    annual_factor = np.sqrt(252)
    ret_std = rets.std()
    sharpe = (rets.mean() / ret_std * annual_factor) if ret_std > 0 else 0.0
    downside = rets[rets < 0]
    downside_std = downside.std() if not downside.empty else 0.0
    sortino = (rets.mean() / downside_std * annual_factor) if downside_std > 0 else 0.0
    cummax = pv.cummax()
    drawdown = (pv / cummax - 1).min()
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    calmar = cagr / abs(drawdown) if drawdown < 0 else 0.0
    return {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Total Return (%)": (pv.iloc[-1] / pv.iloc[0] - 1) * 100,
        "Max Drawdown (%)": drawdown * -100,
        "Volatility (%)": ret_std * annual_factor * 100 if ret_std > 0 else 0.0,
        "Final Value ($)": pv.iloc[-1],
    }


AlgorithmSpec = Tuple[Type[Any], Type[PortfolioEnv]]


def train_and_evaluate(
    tickers: Optional[Iterable[str]] = None,
    train_range: Tuple[str, str] = ("2016-01-01", "2020-12-31"),
    test_range: Tuple[str, str] = ("2021-01-01", "2023-12-31"),
    total_timesteps: int = 50_000,
    run_id: Optional[str] = None,
    algorithms: Optional[Dict[str, AlgorithmSpec]] = None,
    models_dir: str = "models",
    results_dir: str = "results/baseline",
) -> Dict[str, Dict[str, float]]:
    tickers = list(tickers or ("TSLA", "AAPL", "NVDA"))
    train_prices = load_data(tickers, *train_range)
    test_prices = load_data(tickers, *test_range)

    algo_map: Dict[str, AlgorithmSpec] = algorithms or {
        "PPO": (PPO, PortfolioEnv),
        "A2C": (A2C, PortfolioEnv),
        "DDPG": (DDPG, PortfolioEnv),
        "DQN": (DQN, DiscretePortfolioEnv),
    }

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics: Dict[str, Dict[str, float]] = {}
    for name, (algo_cls, env_cls) in algo_map.items():
        print(f"\n=== Training {name} ===")
        train_env = env_cls(train_prices)
        test_env = env_cls(test_prices)

        model = algo_cls("MlpPolicy", train_env, verbose=0)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        model_path = os.path.join(models_dir, f"{name.lower()}_model_{resolved_run_id}")
        model.save(model_path)

        obs = test_env.reset()
        done = False
        records = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = test_env.step(action)
            records.append((info["date"], info["portfolio_value"]))

        df = pd.DataFrame(records, columns=["Date", "portfolio_value"]).set_index("Date")
        result_path = os.path.join(results_dir, f"{name.lower()}_test_portfolio_values_{resolved_run_id}.csv")
        df.to_csv(result_path)

        strat_metrics = compute_metrics(df)
        metrics[name] = strat_metrics
        print(f"Results for {name}:")
        for key, value in strat_metrics.items():
            print(f"  {key}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
