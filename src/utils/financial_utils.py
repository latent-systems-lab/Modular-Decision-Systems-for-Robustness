"""
Financial utility functions for technical analysis and financial calculations.
"""

import numpy as np
import pandas as pd
from math import exp, log, sqrt
from typing import Dict, List, Optional, Tuple, Union

from scipy.stats import norm


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI) for a series of prices.

    Args:
        prices: List of closing prices.
        period: The period over which to calculate RSI (default: 14).

    Returns:
        The RSI value (0-100).
    """
    if len(prices) < period + 1:
        return 50.0  # Default to neutral if not enough data

    # Calculate price changes
    deltas = np.diff(prices)

    # Create arrays for gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate average gains and losses
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate RS and RSI
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_moving_average(prices: List[float], period: int) -> float:
    """
    Calculate the simple moving average for a series of prices.

    Args:
        prices: List of closing prices.
        period: The period over which to calculate the moving average.

    Returns:
        The moving average value.
    """
    if len(prices) < period:
        return prices[-1]  # Return the last price if not enough data

    return np.mean(prices[-period:])


def calculate_exponential_moving_average(prices: List[float], period: int) -> float:
    """
    Calculate the exponential moving average for a series of prices.

    Args:
        prices: List of closing prices.
        period: The period over which to calculate the EMA.

    Returns:
        The EMA value.
    """
    if len(prices) < period:
        return prices[-1]  # Return the last price if not enough data

    # Convert to numpy array for calculations
    prices_array = np.array(prices)

    # Calculate EMA
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()

    # Calculate EMA using convolution
    ema = np.convolve(prices_array, weights, mode='valid')[-1]

    return ema


def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    """
    Calculate Bollinger Bands for a series of prices.

    Args:
        prices: List of closing prices.
        period: The period over which to calculate the bands (default: 20).
        num_std: Number of standard deviations for the bands (default: 2.0).

    Returns:
        Tuple of (lower_band, middle_band, upper_band).
    """
    if len(prices) < period:
        # Not enough data, return the last price as middle band and approximate the others
        last_price = prices[-1]
        return (last_price * 0.95, last_price, last_price * 1.05)

    # Calculate middle band (SMA)
    middle_band = calculate_moving_average(prices, period)

    # Calculate standard deviation
    std_dev = np.std(prices[-period:])

    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    return (lower_band, middle_band, upper_band)


def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a series of prices.

    Args:
        prices: List of closing prices.
        fast_period: Period for the fast EMA (default: 12).
        slow_period: Period for the slow EMA (default: 26).
        signal_period: Period for the signal line (default: 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    if len(prices) < slow_period:
        return (0.0, 0.0, 0.0)  # Not enough data

    # Calculate fast and slow EMAs
    fast_ema = calculate_exponential_moving_average(prices, fast_period)
    slow_ema = calculate_exponential_moving_average(prices, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD line)
    # For simplicity, we'll approximate this since we don't have historical MACD values
    signal_line = macd_line * 0.9  # Approximation

    # Calculate histogram
    histogram = macd_line - signal_line

    return (macd_line, signal_line, histogram)


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """
    Calculate the Average True Range (ATR) for a series of prices.

    Args:
        highs: List of high prices.
        lows: List of low prices.
        closes: List of closing prices.
        period: The period over which to calculate ATR (default: 14).

    Returns:
        The ATR value.
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return (highs[-1] - lows[-1])  # Return the last day's range if not enough data

    # Calculate true ranges
    tr_values = []

    for i in range(1, len(closes)):
        # Current high - current low
        range1 = highs[i] - lows[i]
        # Current high - previous close (absolute)
        range2 = abs(highs[i] - closes[i-1])
        # Current low - previous close (absolute)
        range3 = abs(lows[i] - closes[i-1])

        # True range is the maximum of these three values
        tr = max(range1, range2, range3)
        tr_values.append(tr)

    # Calculate ATR (simple average of true ranges)
    atr = np.mean(tr_values[-period:])

    return atr


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    annualize: bool = True,
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: List of period returns (not cumulative).
        risk_free_rate: Annual risk-free rate when annualize=True.
        periods_per_year: Number of return periods per year.
        annualize: Whether to return an annualized Sharpe ratio.

    Returns:
        The Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0  # Not enough data

    period_risk_free_rate = risk_free_rate / periods_per_year if annualize else risk_free_rate
    excess_returns = [r - period_risk_free_rate for r in returns]

    # Calculate mean and standard deviation
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)  # Use sample standard deviation

    # Avoid division by zero
    if std_dev == 0:
        return 0.0

    # Calculate Sharpe ratio
    sharpe_ratio = mean_excess_return / std_dev
    if annualize:
        sharpe_ratio *= np.sqrt(periods_per_year)

    return sharpe_ratio


def calculate_drawdown(cumulative_returns: List[float]) -> Tuple[float, int, int]:
    """
    Calculate the maximum drawdown for a series of cumulative returns.

    Args:
        cumulative_returns: List of cumulative returns.

    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx).
    """
    if len(cumulative_returns) < 2:
        return (0.0, 0, 0)  # Not enough data

    # Initialize variables
    max_drawdown = 0.0
    peak_idx = 0
    trough_idx = 0
    current_peak_idx = 0

    # Iterate through returns to find maximum drawdown
    for i in range(1, len(cumulative_returns)):
        # Check if current value is a new peak
        if cumulative_returns[i] > cumulative_returns[current_peak_idx]:
            current_peak_idx = i
        else:
            # Calculate drawdown from current peak
            drawdown = (cumulative_returns[current_peak_idx] - cumulative_returns[i]) / cumulative_returns[current_peak_idx]

            # Update maximum drawdown if current drawdown is larger
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                peak_idx = current_peak_idx
                trough_idx = i

    return (max_drawdown, peak_idx, trough_idx)


def calculate_option_greeks(
    option_type: str,
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,  # in years
    risk_free_rate: float,
    implied_volatility: float
) -> Dict[str, float]:
    """
    Calculate option Greeks using the Black-Scholes model.
    This is a simplified implementation and should not be used for actual trading.

    Args:
        option_type: 'call' or 'put'.
        underlying_price: Current price of the underlying asset.
        strike_price: Strike price of the option.
        time_to_expiry: Time to expiration in years.
        risk_free_rate: Annual risk-free interest rate.
        implied_volatility: Implied volatility of the option.

    Returns:
        Dictionary containing the calculated Greeks.
    """
    # Avoid division by zero or negative time
    if time_to_expiry <= 0:
        return {
            'delta': 1.0 if option_type == 'call' and underlying_price > strike_price else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    # Calculate d1 and d2
    d1 = (np.log(underlying_price / strike_price) +
          (risk_free_rate + 0.5 * implied_volatility ** 2) * time_to_expiry) / (implied_volatility * np.sqrt(time_to_expiry))
    d2 = d1 - implied_volatility * np.sqrt(time_to_expiry)

    # Calculate option price and Greeks based on option type
    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (underlying_price * implied_volatility * np.sqrt(time_to_expiry))
        theta = -(underlying_price * norm.pdf(d1) * implied_volatility) / (2 * np.sqrt(time_to_expiry)) - risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        vega = underlying_price * np.sqrt(time_to_expiry) * norm.pdf(d1) / 100  # Divided by 100 to get the change per 1% change in IV
        rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100  # Divided by 100 to get the change per 1% change in interest rate
    else:  # put
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (underlying_price * implied_volatility * np.sqrt(time_to_expiry))
        theta = -(underlying_price * norm.pdf(d1) * implied_volatility) / (2 * np.sqrt(time_to_expiry)) + risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        vega = underlying_price * np.sqrt(time_to_expiry) * norm.pdf(d1) / 100
        rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate the Black–Scholes price for a call option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float('nan')
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate the Black–Scholes price for a put option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float('nan')
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def sanitize_option_price(
    option_type: str,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.01,
    sigma: float = 0.25,
    tolerance: float = 3.0,
) -> float:
    """Validate option price using the Black–Scholes model and replace outliers."""

    # Validate strike price reasonableness (should be within 50% to 200% of spot)
    # Strikes far outside this range indicate data errors (e.g., expiration used as strike)
    if K <= 0 or S <= 0:
        return market_price

    strike_ratio = K / S
    if strike_ratio < 0.5 or strike_ratio > 2.0:
        # Strike is unrealistic - likely a data error (e.g., expiration date used as strike)
        # Return market price without Black-Scholes correction
        print(
            f"⚠️  Invalid strike detected: K={K:.2f}, S={S:.2f} (ratio={strike_ratio:.2f}). "
            f"Skipping Black-Scholes validation - using market price {market_price:.2f}"
        )
        return market_price

    option_type_lower = option_type.lower()
    if option_type_lower == "call":
        theoretical = black_scholes_call(S, K, T, r, sigma)
    else:
        theoretical = black_scholes_put(S, K, T, r, sigma)

    if np.isnan(theoretical) or theoretical <= 0:
        return market_price

    if abs(market_price - theoretical) > tolerance * max(theoretical, 1e-3):
        option_type_label = option_type_lower.upper()
        print(
            f"⚠️  Outlier detected for {option_type_label}: "
            f"{market_price:.2f} → {theoretical:.2f} (BS correction)"
        )
        return theoretical

    return market_price


def calculate_portfolio_metrics(
    positions: Dict[str, Dict],
    prices: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate portfolio metrics based on current positions and prices.

    Args:
        positions: Dictionary mapping tickers to position details.
        prices: Dictionary mapping tickers to current prices.
        weights: Optional dictionary mapping tickers to portfolio weights.

    Returns:
        Dictionary containing portfolio metrics.
    """
    # Calculate portfolio value and weights if not provided
    portfolio_value = 0.0
    position_values = {}

    for ticker, position in positions.items():
        if ticker in prices:
            shares = position.get('shares', 0)
            current_price = prices[ticker]
            position_value = shares * current_price
            position_values[ticker] = position_value
            portfolio_value += position_value

    # Calculate weights if not provided
    if weights is None:
        weights = {}
        for ticker, value in position_values.items():
            weights[ticker] = value / portfolio_value if portfolio_value > 0 else 0.0

    # Calculate weighted average metrics
    weighted_beta = 0.0
    weighted_dividend_yield = 0.0

    for ticker, weight in weights.items():
        position = positions.get(ticker, {})
        weighted_beta += position.get('beta', 1.0) * weight
        weighted_dividend_yield += position.get('dividend_yield', 0.0) * weight

    return {
        'portfolio_value': portfolio_value,
        'weighted_beta': weighted_beta,
        'weighted_dividend_yield': weighted_dividend_yield
    }


def calculate_option_payoff(
    option_positions: List[Dict],
    underlying_price: float
) -> float:
    """
    Calculate the payoff of option positions at a given underlying price.

    Args:
        option_positions: List of option position dictionaries.
        underlying_price: Price of the underlying asset at expiration.

    Returns:
        Total payoff of the option positions.
    """
    total_payoff = 0.0

    for position in option_positions:
        option_type = position.get('option_type', 'call')
        strike = position.get('strike', 0.0)
        contracts = position.get('contracts', 0)
        premium = position.get('premium_per_contract', 0.0)
        action = position.get('action', 'buy')

        # Calculate intrinsic value at expiration
        if option_type == 'call':
            intrinsic_value = max(0, underlying_price - strike)
        else:  # put
            intrinsic_value = max(0, strike - underlying_price)

        # Calculate payoff based on action (buy or sell)
        if action == 'buy':
            position_payoff = (intrinsic_value - premium) * contracts * 100  # Each contract is for 100 shares
        else:  # sell
            position_payoff = (premium - intrinsic_value) * contracts * 100

        total_payoff += position_payoff

    return total_payoff
