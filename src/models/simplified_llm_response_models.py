"""
Simplified Pydantic models for LLM responses in the portfolio risk-management simulation system.
These models define the structure of data expected from LLM API calls.
"""

from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import date

class MarketAnalysisResponse(BaseModel):
    """Model for structured market analysis response from LLM."""
    regime: Literal["bullish", "bearish", "neutral", "volatile"]
    sentiment: Literal["positive", "negative", "neutral"]
    short_term_outlook: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

class AssetAllocationResponse(BaseModel):
    """Model for structured asset allocation response from LLM."""
    asset: str
    allocation_percentage: float = Field(ge=0.0, le=1.0)
    rationale: str

class PortfolioAllocationResponse(BaseModel):
    """Model for structured portfolio allocation response from LLM."""
    allocations: List[AssetAllocationResponse]
    cash_reserve_percentage: float = Field(ge=0.0, le=1.0)
    overall_strategy: str

class AssetSplitResponse(BaseModel):
    """Model for structured asset split response from LLM."""
    stock_percentage: float = Field(ge=0.0, le=1.0)
    options_percentage: float = Field(ge=0.0, le=1.0)
    rationale: str

class SimplifiedOptionContractResponse(BaseModel):
    """
    Simplified model for option contract selection from LLM.
    Only includes essential fields that the LLM needs to provide.
    Other fields will be populated from available options data.
    """
    option_type: Literal["call", "put"]
    option_id: str  # Will be used to look up full option details
    contracts: int
    action: Literal["buy", "sell"]

class HedgingStrategyResponse(BaseModel):
    """Model for structured hedging strategy response from LLM."""
    asset: str
    strategy_type: Literal["collar", "straddle", "delta-neutral"]
    option_contracts: List[SimplifiedOptionContractResponse]
    estimated_cost: float
    rationale: str

class CollarStrategyResponse(BaseModel):
    """Model for structured collar strategy response from LLM."""
    protective_put: SimplifiedOptionContractResponse
    covered_call: SimplifiedOptionContractResponse
    net_cost: float
    downside_protection: float = Field(ge=0.0, le=1.0)
    upside_cap: float
    rationale: str

class StraddleStrategyResponse(BaseModel):
    """Model for structured straddle strategy response from LLM."""
    call_option: SimplifiedOptionContractResponse
    put_option: SimplifiedOptionContractResponse
    total_cost: float
    breakeven_points: List[float]
    expected_profit_on_move: float
    # rationale: str

class DeltaNeutralStrategyResponse(BaseModel):
    """Model for structured delta-neutral strategy response from LLM."""
    option_contracts: List[SimplifiedOptionContractResponse]
    net_delta: float
    total_cost: float
    expected_behavior: str
    rationale: str
