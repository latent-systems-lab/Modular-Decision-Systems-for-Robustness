"""
Pydantic data models for the portfolio risk-management simulation system.
These models define the structure of data used throughout the application.
"""

from typing import Any, Dict, List, Optional, Union, Literal, Tuple
from pydantic import BaseModel, Field, computed_field
from datetime import date, datetime
import config
from src.models.llm_response_models import OptionContractResponse


class StockData(BaseModel):
    """Model representing stock price data for a single day."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class OptionData(BaseModel):
    """Model representing option data for a single contract."""
    quote_date: date
    option_id: str
    option_type: int
    strike: float
    expire_date: date
    dte: int  # Days to expiration
    last: float
    bid: float
    ask: float
    delta: float
    moneyness: float
    underlying_last: float
    iv: float  # Implied Volatility
    gamma: float
    theta: float
    vega: float
    rho: float
    volume: int



class SplitResponse(BaseModel):
    stock_percentage: float = Field(ge=0.0, le=1.0)
    options_percentage: float = Field(ge=0.0, le=1.0)
    rationale: str
            
class CapitalAllocation(BaseModel):
    """Model representing capital allocation for assets in the portfolio."""
    asset: Literal["AAPL", "NVDA", "TSLA"]
    amount: float
    percentage: float = Field(ge=0.0, le=1.0)
    action: Literal["buy", "sell"]


class PortfolioAllocation(BaseModel):
    """Model representing the complete portfolio allocation across assets."""
    total_capital: float
    allocations: List[CapitalAllocation]
    cash_reserve: float
    rationale: str

    
class HedgingStrategies(BaseModel):
    collar: float = Field(ge=0.0, le=1.0, default=0.0)
    straddle: float = Field(ge=0.0, le=1.0, default=0.0)
    delta_neutral: float = Field(ge=0.0, le=1.0, default=0.0)

class AssetSplit(BaseModel):
    """Model representing the split between stock and options for an asset."""
    asset: str
    total_allocation: float
    stock_percentage: float = Field(ge=0.0, le=1.0)
    stocks_action: Literal["buy", "sell"]
    options_percentage: float = Field(ge=0.0, le=1.0)
    stock_amount: float
    options_amount: float
    rationale: str
    hedging_strategies: HedgingStrategies


class OptionPosition(BaseModel):
    """Model representing an option position."""
    asset: str
    option_type: Literal["call", "put"]
    option_id: str
    strike: float
    expiration_date: date
    contracts: int
    premium_per_contract: float
    total_premium: float = 0.0
    paid_amount: float = 0.0         # The total amount paid for the option contracts
    market_value: float = 0.0        # The current market value of the option position
    action: Literal["buy", "sell"]
    strategy: Literal["collar", "straddle", "delta-neutral"]
    unrealized_pnl: float = 0.0


class StockPosition(BaseModel):
    """Model representing a stock position."""
    asset: str
    shares: int
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float = 0.0


class Trade(BaseModel):
    """Model representing a trade execution."""
    asset: str
    trade_date: date
    trade_type: Literal["stock", "option"]
    action: Literal["buy", "sell", "expire"]
    quantity: int  # shares or contracts
    price: float
    total_value: float
    fees: float = 0.0
    stock_position: Optional[StockPosition] = None
    option_position: Optional[OptionPosition] = None


class Portfolio(BaseModel):
    """Model representing the complete portfolio state."""
    date: date
    cash: float
    stock_positions: List[StockPosition] = []
    option_positions: List[OptionPosition] = []
    total_value: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class DailyPerformance(BaseModel):
    """Model representing daily performance metrics."""
    date: date
    portfolio_value: float
    daily_return: float
    daily_pnl: float
    cumulative_return: float
    cumulative_pnl: float
    drawdown: float = 0.0
    asset_pnl_breakdown: Any = {}



class LLMRequest(BaseModel):
    """Model representing a request to the LLM API."""
    prompt: str
    model: str = Field(default_factory=lambda: getattr(config, "LLM_MODEL", "gemini-2.5-flash"))
    temperature: float = 0.3
    max_tokens: int = 1000
    response_format: Optional[Dict] = None
    system_prompt: Optional[str] = Field(default_factory=lambda: getattr(config, "LLM_SYSTEM_PROMPT", None))


class LLMResponse(BaseModel):
    """Model representing a response from the LLM API."""
    content: str
    parsed_json: Optional[Dict] = None
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    candidates_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None
    model: Optional[str] = None
    prompt: Optional[str] = None


class HedgingStrategy(BaseModel):
    """Model representing a hedging strategy recommendation."""
    asset: str
    strategy_type: str  # e.g., "collar", "straddle", "delta-neutral"
    rationale: str
    option_positions: List[OptionPosition]
    estimated_cost: float
    estimated_protection: float  # percentage of downside protection
    confidence: float = Field(ge=0.0, le=1.0)


# Add this to your pydantic_models.py
class CombinedAllocationSplitResponse(BaseModel):
    """Model for combined portfolio allocation and asset splits response."""
    allocations: List[CapitalAllocation]
    asset_splits: List[AssetSplit]
    cash_reserve: float
    rationale: str
        
class MarketRegime(BaseModel):
    """Model representing market regime analysis for an asset."""

    asset: str
    regime: Literal["bullish", "bearish", "neutral", "volatile"]
    sentiment: float = Field(ge=-1.0, le=1.0)
    outlook: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    narrative: Optional[str] = None
    days_to_next_earnings: Optional[int]
    revenue_growth: Optional[float]

    @computed_field(return_type=str)
    @property
    def sentiment_label(self) -> str:
        """Categorical representation of the sentiment score."""

        if self.sentiment >= 0.25:
            return "positive"
        if self.sentiment <= -0.25:
            return "negative"
        return "neutral"

    @computed_field(return_type=str)
    @property
    def outlook_label(self) -> str:
        """Categorical representation of the outlook score."""

        if self.outlook >= 0.25:
            return "optimistic"
        if self.outlook <= -0.25:
            return "defensive"
        return "balanced"

    @computed_field(return_type=str)
    @property
    def short_term_outlook(self) -> str:
        """Human readable short-term outlook derived from the outlook score."""

        if self.narrative:
            return self.narrative

        label = self.outlook_label
        if label == "optimistic":
            return "Positive momentum expected to continue in the near term."
        if label == "defensive":
            return "Guarded outlook—expect choppy or negative price action."
        return "Range-bound behaviour expected; monitor for breakout cues."

    def as_tuple(self) -> Tuple[str, float, float, float]:
        """Return the regime summary as the (regime, sentiment, outlook, confidence) tuple."""

        return (self.regime, self.sentiment, self.outlook, self.confidence)

# Create a custom response model for batch analysis
class BatchMarketAnalysisResponse(BaseModel):
    analyses: List[MarketRegime]


class MemoryRLConfig(BaseModel):
    """Configuration for memory-based RL integration."""
    memory_dir: str = "memory"
    enable_reflection: bool = True
    reflection_frequency: int = 5  # Generate reflection every N experiences
    save_frequency: int = 10  # Save memory to disk every N experiences
    max_experiences: int = 5  # Maximum number of experiences to include in prompts
    include_temporal_patterns: bool = True
    include_market_context: bool = True
    return_weight: float = 0.5
    risk_weight: float = 0.3
    benchmark_weight: float = 0.1
    cost_weight: float = 0.1
    use_long_term_rewards: bool = True
    use_what_if_comparison: bool = False

class SimulationConfig(BaseModel):
    """Model representing simulation configuration parameters."""
    start_date: date
    end_date: date
    initial_capital: float
    assets: List[str]
    rebalance_frequency: int = 30  # days
    emergency_threshold: float = 0.1  # 10% drawdown triggers emergency rebalance
    use_llm: bool = True
    hedging_strategies: List[str] = []
    rule_based_strategy_split: bool = False

class SimulationResult(BaseModel):
    """Model representing the final simulation results."""
    config: SimulationConfig
    final_portfolio: Portfolio
    daily_performances: List[DailyPerformance]
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades_executed: int
    average_position_duration: float  # days
    best_performing_asset: str
    worst_performing_asset: str
    profit_loss_breakdown: Optional[Dict]
    portfolio_summary: Optional[str]
