"""
LLM Prompt Templates for memory-based reinforcement learning.
Provides templates for constructing prompts that incorporate past experiences.
"""

import os
from typing import Dict, List, Optional, Union, Literal, Any
import logging
from datetime import date, datetime, timedelta

from .experience_memory_store import (
    Experience, PortfolioExperience, AssetExperience, HedgingExperience,
    ExperienceMemoryStore
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptConfig:
    """Configuration for prompt templates."""
    
    def __init__(
        self,
        max_experiences: int = 5,
        include_reflection: bool = True,
        include_market_context: bool = True,
        include_system_instructions: bool = True,
        max_prompt_length: int = 40000,
        temperature: float = 0.3,
        format_instructions: bool = True
    ):
        """
        Initialize prompt configuration.
        
        Args:
            max_experiences: Maximum number of experiences to include in prompt.
            include_reflection: Whether to include reflections in experiences.
            include_market_context: Whether to include market context.
            include_system_instructions: Whether to include system instructions.
            max_prompt_length: Maximum length of prompt in characters.
            temperature: Temperature for LLM generation.
            format_instructions: Whether to include format instructions.
        """
        self.max_experiences = max_experiences
        self.include_reflection = include_reflection
        self.include_market_context = include_market_context
        self.include_system_instructions = include_system_instructions
        self.max_prompt_length = max_prompt_length
        self.temperature = temperature
        self.format_instructions = format_instructions

class PromptTemplates:
    """
    Templates for constructing prompts for LLM-based decision making.
    """
    
    def __init__(self, config: PromptConfig = None):
        """
        Initialize prompt templates.
        
        Args:
            config: Configuration for prompt templates.
        """
        self.config = config or PromptConfig()
    
    def create_portfolio_prompt(
        self,
        current_date: date,
        portfolio_value: float,
        cash: float,
        assets: List[str],
        market_analysis: Dict[str, Dict[str, Any]],
        current_positions: Dict[str, Dict[str, Any]],
        experiences: List[PortfolioExperience],
        emergency: float = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for portfolio-level decision making.
        
        Args:
            current_date: Current date.
            portfolio_value: Current portfolio value.
            cash: Available cash.
            assets: List of assets.
            market_analysis: Market analysis for each asset.
            current_positions: Current positions.
            experiences: List of relevant past experiences.
            additional_context: Additional context to include.
            
        Returns:
            Formatted prompt.
        """
        # System instructions
        system_instructions = self._get_portfolio_system_instructions() if self.config.include_system_instructions else ""
        
        # Current state
        current_state = self._format_portfolio_state(
            current_date=current_date,
            portfolio_value=portfolio_value,
            cash=cash,
            assets=assets,
            market_analysis=market_analysis,
            current_positions=current_positions
        )
        
        # Market context
        market_context = self._format_market_context(market_analysis, emergency) if self.config.include_market_context else ""
        
        # Past experiences
        past_experiences = self._format_experiences(experiences, self.config.max_experiences)
        
        # Format instructions
        format_instructions = self._get_portfolio_format_instructions() if self.config.format_instructions else ""
        
        # Additional context
        additional_context_text = f"\nAdditional Context:\n{additional_context}\n" if additional_context else ""
        
        # Combine all sections
        prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
        
        # Trim if needed
        if len(prompt) > self.config.max_prompt_length:
            logger.warning(f"Prompt exceeds max length ({len(prompt)} > {self.config.max_prompt_length}). Trimming experiences.")
            # Recalculate with fewer experiences
            max_experiences = max(1, self.config.max_experiences - 2)
            past_experiences = self._format_experiences(experiences, max_experiences)
            prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
            
            # If still too long, trim market context
            if len(prompt) > self.config.max_prompt_length and self.config.include_market_context:
                logger.warning("Still too long. Trimming market context.")
                market_context = ""
                prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
        
        return prompt
    
    def create_asset_prompt(
        self,
        current_date: date,
        asset: str,
        allocation_amount: float,
        market_analysis: Dict[str, Any],
        hedging_strategies: List[str],
        current_price: float,
        volatility: float,
        experiences: List[AssetExperience],
        additional_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for asset-level decision making.
        
        Args:
            current_date: Current date.
            asset: Asset ticker.
            allocation_amount: Amount allocated to this asset.
            market_analysis: Market analysis for this asset.
            current_price: Current price of the asset.
            volatility: Volatility of the asset.
            experiences: List of relevant past experiences.
            additional_context: Additional context to include.
            
        Returns:
            Formatted prompt.
        """
        # System instructions
        system_instructions = self._get_asset_system_instructions() if self.config.include_system_instructions else ""
        
        # Current state
        current_state = self._format_asset_state(
            current_date=current_date,
            asset=asset,
            allocation_amount=allocation_amount,
            market_analysis=market_analysis,
            current_price=current_price,
            volatility=volatility,
            hedging_strategies=hedging_strategies
        )
        
        # Market context
        market_context = self._format_asset_market_context(market_analysis) if self.config.include_market_context else ""
        
        # Past experiences
        past_experiences = self._format_experiences(experiences, self.config.max_experiences)
        
        # Format instructions
        format_instructions = self._get_asset_format_instructions() if self.config.format_instructions else ""
        
        # Additional context
        additional_context_text = f"\nAdditional Context:\n{additional_context}\n" if additional_context else ""
        
        # Combine all sections
        prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
        
        # Trim if needed
        if len(prompt) > self.config.max_prompt_length:
            logger.warning(f"Prompt exceeds max length ({len(prompt)} > {self.config.max_prompt_length}). Trimming experiences.")
            # Recalculate with fewer experiences
            max_experiences = max(1, self.config.max_experiences - 2)
            past_experiences = self._format_experiences(experiences, max_experiences)
            prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
            
            # If still too long, trim market context
            if len(prompt) > self.config.max_prompt_length and self.config.include_market_context:
                logger.warning("Still too long. Trimming market context.")
                market_context = ""
                prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
        
        return prompt
    
    def create_hedging_prompt(
        self,
        current_date: date,
        asset: str,
        strategy_type: str,
        stock_position: Dict[str, Any],
        available_options: List[Dict[str, Any]],
        available_capital: float,
        market_analysis: Dict[str, Any],
        experiences: List[HedgingExperience],
        additional_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for hedging-level decision making.
        
        Args:
            current_date: Current date.
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            stock_position: Current stock position.
            available_options: Available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.
            experiences: List of relevant past experiences.
            additional_context: Additional context to include.
            
        Returns:
            Formatted prompt.
        """
        # System instructions
        system_instructions = self._get_hedging_system_instructions(strategy_type) if self.config.include_system_instructions else ""
        
        # Current state
        current_state = self._format_hedging_state(
            current_date=current_date,
            asset=asset,
            strategy_type=strategy_type,
            stock_position=stock_position,
            available_options=available_options,
            available_capital=available_capital,
            market_analysis=market_analysis
        )
        
        # Market context
        market_context = self._format_asset_market_context(market_analysis) if self.config.include_market_context else ""
        
        # Past experiences
        past_experiences = self._format_experiences(experiences, self.config.max_experiences)
        
        # Format instructions
        format_instructions = self._get_hedging_format_instructions() if self.config.format_instructions else ""
        
        # Additional context
        additional_context_text = f"\nAdditional Context:\n{additional_context}\n" if additional_context else ""
        
        # Combine all sections
        prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
        
        # Trim if needed
        if len(prompt) > self.config.max_prompt_length:
            logger.warning(f"Prompt exceeds max length ({len(prompt)} > {self.config.max_prompt_length}). Trimming experiences.")
            # Recalculate with fewer experiences
            max_experiences = max(1, self.config.max_experiences - 2)
            past_experiences = self._format_experiences(experiences, max_experiences)
            prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
            
            # If still too long, trim market context
            if len(prompt) > self.config.max_prompt_length and self.config.include_market_context:
                logger.warning("Still too long. Trimming market context.")
                market_context = ""
                prompt = f"{system_instructions}\n\n{current_state}\n\n{market_context}{additional_context_text}\n\n{past_experiences}\n\n{format_instructions}"
        
        return prompt
    
    def create_reflection_prompt(
        self,
        experience: Experience,
        similar_experiences: List[Experience] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for generating reflections on experiences.
        
        Args:
            experience: Experience to reflect on.
            similar_experiences: List of similar past experiences.
            additional_context: Additional context to include.
            
        Returns:
            Formatted prompt.
        """
        # System instructions
        system_instructions = self._get_reflection_system_instructions()
        
        # Experience details
        experience_text = experience.to_text()
        
        # Similar experiences
        similar_experiences_text = ""
        if similar_experiences:
            similar_experiences_text = "Similar Past Experiences:\n"
            for i, exp in enumerate(similar_experiences[:3]):  # Limit to 3 similar experiences
                similar_experiences_text += f"Experience {i+1}:\n{exp.to_text()}\n"
        
        # Additional context
        additional_context_text = f"\nAdditional Context:\n{additional_context}\n" if additional_context else ""
        
        # Format instructions
        format_instructions = """
Please provide a thoughtful reflection on this experience, considering:
1. What worked well and what didn't
2. Why the outcome occurred (market conditions, timing, strategy selection, etc.)
3. What could have been done differently
4. What lessons can be applied to future decisions
5. How this experience relates to similar past experiences

Your reflection should be concise but insightful, focusing on actionable lessons.
"""
        
        # Combine all sections
        prompt = f"{system_instructions}\n\n{experience_text}\n\n{similar_experiences_text}{additional_context_text}\n\n{format_instructions}"
        
        return prompt
    
    def _format_experiences(self, experiences: List[Experience], max_experiences: int) -> str:
        """
        Format a list of experiences for inclusion in a prompt.
        
        Args:
            experiences: List of experiences.
            max_experiences: Maximum number of experiences to include.
            
        Returns:
            Formatted experiences text.
        """
        if not experiences:
            return "Past Experiences: None available."
        
        # Limit number of experiences
        experiences = experiences[:max_experiences]
        
        # Format experiences
        text = "Past Experiences:\n\n"
        for i, exp in enumerate(experiences):
            text += f"Experience {i+1}:\n{exp.to_text()}\n\n"
        
        return text
    
    def _format_portfolio_state(
        self,
        current_date: date,
        portfolio_value: float,
        cash: float,
        assets: List[str],
        market_analysis: Dict[str, Dict[str, Any]],
        current_positions: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Format the current portfolio state.
        
        Args:
            current_date: Current date.
            portfolio_value: Current portfolio value.
            cash: Available cash.
            assets: List of assets.
            market_analysis: Market analysis for each asset.
            current_positions: Current positions.
            
        Returns:
            Formatted state text.
        """
        # Calculate cash percentage
        cash_percentage = (cash / portfolio_value) * 100 if portfolio_value > 0 else 100.0
        
        # Format current positions
        positions_text = ""
        for asset in assets:
            if asset in current_positions:
                position = current_positions[asset]
                positions_text += f"  {asset}: {position.get('shares', 0)} shares, Market Value: ${position.get('market_value', 0):.2f}, "
                positions_text += f"Average Price: ${position.get('average_price', 0):.2f}, Current Price: ${position.get('current_price', 0):.2f}\n"
            else:
                positions_text += f"  {asset}: No position\n"
        
        # Format state
        text = f"Current Portfolio State (Date: {current_date}):\n"
        text += f"Portfolio Value: ${portfolio_value:.2f}\n"
        text += f"Cash: ${cash:.2f} ({cash_percentage:.2f}%)\n"
        text += f"Available Assets: {', '.join(assets)}\n"
        text += "Current Positions:\n"
        text += positions_text
        
        return text
    
    def _format_asset_state(
        self,
        current_date: date,
        asset: str,
        allocation_amount: float,
        market_analysis: Dict[str, Any],
        current_price: float,
        volatility: float,
        hedging_strategies: List[str]
    ) -> str:
        """
        Format the current asset state.
        
        Args:
            current_date: Current date.
            asset: Asset ticker.
            allocation_amount: Amount allocated to this asset.
            market_analysis: Market analysis for this asset.
            current_price: Current price of the asset.
            volatility: Volatility of the asset.
            
        Returns:
            Formatted state text.
        """
        # Format state
        text = f"Current Asset State (Date: {current_date}):\n"
        text += f"Asset: {asset}\n"
        text += f"Allocation Amount: ${allocation_amount:.2f}\n"
        text += f"Current Price: ${current_price:.2f}\n"
        text += f"Volatility: {volatility:.2f}\n"
        text += f"Market Regime: {market_analysis.get('regime', 'unknown')}\n"
        text += (
            f"Sentiment: {market_analysis.get('sentiment_label', 'unknown')} "
            f"(score {market_analysis.get('sentiment', 0.0):.2f})\n"
        )
        text += (
            f"Short-term Outlook: {market_analysis.get('short_term_outlook', 'unknown')} "
            f"(score {market_analysis.get('outlook', 0.0):.2f})\n"
        )
        text += f"Available Hedging Strategies: {', '.join(hedging_strategies)}\n"
        
        return text
    
    def _format_hedging_state(
        self,
        current_date: date,
        asset: str,
        strategy_type: str,
        stock_position: Dict[str, Any],
        available_options: List[Dict[str, Any]],
        available_capital: float,
        market_analysis: Dict[str, Any]
    ) -> str:
        """
        Format the current hedging state.
        
        Args:
            current_date: Current date.
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            stock_position: Current stock position.
            available_options: Available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.
            
        Returns:
            Formatted state text.
        """
        # Format stock position
        stock_text = f"Shares: {stock_position.get('shares', 0)}, "
        stock_text += f"Market Value: ${stock_position.get('market_value', 0):.2f}, "
        stock_text += f"Average Price: ${stock_position.get('average_price', 0):.2f}, "
        stock_text += f"Current Price: ${stock_position.get('current_price', 0):.2f}"
        
        # Format available options
        options_text = ""
        for i, option in enumerate(available_options[:]):  # Limit to 10 options
            option_type = "Call" if option.get('option_type', 1) == 1 else "Put"
            options_text += f"Option ID: {option.get('option_id', 'N/A')}"
            options_text += f"  {i+1}. {option_type} {option.get('strike', 0):.2f} "
            options_text += f"Exp: {option.get('expire_date', current_date)}, "
            options_text += f"Premium: ${option.get('last', 0):.2f}, "
            options_text += f"IV: {option.get('implied_volatility', 0):.2f}\n"
        
        # Format state
        text = f"Current Hedging State (Date: {current_date}):\n"
        text += f"Asset: {asset}\n"
        text += f"Strategy Type: {strategy_type}\n"
        text += f"Available Capital: ${available_capital:.2f}\n - use all of it."
        text += f"Stock Position: {stock_text}\n"
        text += "Available Options:\n"
        text += options_text
        text += f"Market Regime: {market_analysis.get('regime', 'unknown')}\n"
        text += (
            f"Sentiment: {market_analysis.get('sentiment_label', 'unknown')} "
            f"(score {market_analysis.get('sentiment', 0.0):.2f})\n"
        )
        text += (
            f"Short-term Outlook: {market_analysis.get('short_term_outlook', 'unknown')} "
            f"(score {market_analysis.get('outlook', 0.0):.2f})\n"
        )
        
        return text
    
    def _format_market_context(self, market_analysis: Dict[str, Dict[str, Any]], emergency: str = float) -> str:
        """
        Format market context for multiple assets.
        
        Args:
            market_analysis: Market analysis for each asset.
            
        Returns:
            Formatted market context text.
        """
        text = "Market Context:\n"
        
        for asset, analysis in market_analysis.items():
            analysis = analysis.model_dump()
            text += f"  {asset}: Regime: {analysis.get('regime', 'unknown')}, "
            text += f"Sentiment: {analysis.get('sentiment', 'unknown')}, "
            text += f"Outlook: {analysis.get('short_term_outlook', 'unknown')}, "
            text += f"Confidence: {analysis.get('confidence', 0):.2f}\n"
            text += f"Days to Next Earnings: {analysis.get('days_to_next_earnings', 'unknown')}\n"
            text += f"Quarterly Revenue Growth Forecast: {analysis.get('revenue_growth', 'unknown')}\n"
        if emergency:
            text += f"  Emergency: Drawdown reached {emergency} since last rebalancing.\n"
        return text
    
    def _format_asset_market_context(self, market_analysis: Dict[str, Any]) -> str:
        """
        Format market context for a single asset.
        
        Args:
            market_analysis: Market analysis for this asset.
            
        Returns:
            Formatted market context text.
        """
        text = "Market Context:\n"
        text += f"  Regime: {market_analysis.get('regime', 'unknown')}\n"
        text += (
            f"  Sentiment: {market_analysis.get('sentiment_label', 'unknown')} "
            f"(score {market_analysis.get('sentiment', 0.0):.2f})\n"
        )
        text += (
            f"  Short-term Outlook: {market_analysis.get('short_term_outlook', 'unknown')} "
            f"(score {market_analysis.get('outlook', 0.0):.2f})\n"
        )
        text += f"  Confidence: {market_analysis.get('confidence', 0):.2f}\n"
        text += f"  Days to Next Earnings: {market_analysis.get('days_to_next_earnings', 'unknown')}\n"
        text += f"  Quarterly Revenue Growth Forecast: {market_analysis.get('revenue_growth', 'unknown')}\n"
        return text
    
    def _get_portfolio_system_instructions(self) -> str:
        """
        Get system instructions for portfolio-level decision making.
        
        Returns:
            System instructions text.
        """
        return """
You are an expert portfolio risk manager. Your task is to allocate capital among different assets based on the current market conditions and portfolio state. You should consider both the current state and past experiences to make optimal decisions.

Your goal is to maximize risk-adjusted returns while maintaining appropriate diversification. You should consider market regimes, asset correlations, and the specific characteristics of each asset.

Based on the current portfolio state, market context, and past experiences, provide a capital allocation plan for the portfolio.
"""
    #Do not leave cash in the portfolio unless in extreme market conditions. Try to allocate some capital to each asset.
    def _get_asset_system_instructions(self) -> str:
        """
        Get system instructions for asset-level decision making.
        
        Returns:
            System instructions text.
        """
        return """
You are an expert asset manager. Your task is to decide how to split the allocated capital between stock and options for a specific asset. You should consider both the current state and past experiences to make optimal decisions.

Your goal is to balance potential returns with risk management. In bullish regimes, you might allocate more to stock, while in bearish or volatile regimes, you might allocate more to options for protection.

Based on the current asset state, market context, and past experiences, provide a stock/options allocation plan for this asset. 

If you predict negative return you can short sell by setting stock_action = 'sell'.
"""
    def _get_hedging_system_instructions(self, strategy_type: str) -> str:
        """
        Get system instructions for hedging-level decision making.
        
        Args:
            strategy_type: Type of hedging strategy.
            
        Returns:
            System instructions text.
        """
        base_instructions = """
You are an expert options strategist. Your task is to implement a specific options strategy to hedge or enhance a stock position. You should consider both the current state and past experiences to make optimal decisions.
IMPORTANT: Each option contract represents 100 shares. Calculate the cost per contract and ensure the total cost is within the available capital.
 For the estimated cost, add the cost for the put option and the cost for the call option together.
    - For example, for 4 contracts with a put ask price of $13.55 and a call ask price of $11.50, the calculation should be:
  (4 * 13.55 * 100) + (4 * 11.50 * 100) = $10,020.
    Use ONLY the option_ids provided in the Available Options list.
    Make sure option_ids are valid and correspond to the selected options.
    Remember to provide number of contracts after calculating the cost.
"""
        
        strategy_specific = ""
        if strategy_type == "collar":
            strategy_specific = """
For a collar strategy, you should select put options to buy for downside protection and call options to sell for premium collection. The goal is to limit downside risk while partially funding the protection by giving up some upside potential.
"""
        elif strategy_type == "straddle":
            strategy_specific = """
For a straddle strategy, you should select put and call options with the same strike price and expiration. The goal is to profit from significant price movements in either direction, particularly in volatile market conditions.
"""
        elif strategy_type == "delta_neutral":
            strategy_specific = """
For a delta-neutral strategy, you should select options positions that offset the delta of the stock position. The goal is to create a position that is insensitive to small price movements, allowing you to profit from changes in volatility or time decay.
"""

        return base_instructions + strategy_specific + """
Based on the current hedging state, market context, and past experiences, provide a specific options strategy implementation. Remember to use all available capital.
"""
    
    def _get_reflection_system_instructions(self) -> str:
        """
        Get system instructions for reflection generation.
        
        Returns:
            System instructions text.
        """
        return """
You are an expert financial analyst specializing in post-trade analysis and reflection. Your task is to analyze a trading decision and its outcome, extracting valuable lessons and insights.

You should consider the market conditions, the decision rationale, and the actual outcome to provide a thoughtful reflection on what worked, what didn't, and what could be improved in future decisions.

Your reflection should be specific, insightful, and actionable, focusing on concrete lessons rather than general principles.
"""
    
    def _get_portfolio_format_instructions(self) -> str:
        """
        Get format instructions for portfolio-level decision making.
        
        Returns:
            Format instructions text.
        """
        return """
Please provide your allocation decision in the following JSON format:

```
{
  "total_capital": 1000000,
  "allocations": [
    {
      "asset": "AAPL",
      "amount": 200000,
      "percentage": 0.2
    },
    {
      "asset": "TSLA",
      "amount": 150000,
      "percentage": 0.15
    },
    ...
  ],
  "cash_reserve": 100000,
  "rationale": "Detailed explanation of your allocation strategy..."
}
```

Ensure that:
1. The sum of all allocation percentages plus cash reserve percentage equals 1.0
2. The allocations are appropriate for the current market conditions
3. The rationale explains your decision-making process, including how you've incorporated lessons from past experiences
"""
    
    def _get_asset_format_instructions(self) -> str:
        """
        Get format instructions for asset-level decision making.
        
        Returns:
            Format instructions text.
        """
        return """
Please provide your asset allocation decision in the following JSON format:

```
{
  "asset": "AAPL",
  "total_allocation": 200000,
  "stock_percentage": 0.7,
  "stocks_action": "sell",
  "options_percentage": 0.3,
  "stock_amount": 140000,
  "options_amount": 60000,
  "rationale": "Detailed explanation of your allocation strategy...",
  "hedging_strategies": {
    "collar": 0.2,
    "straddle": 0.4,
    "delta_neutral": 0.4,
  }
}
```

Guidelines on strategy usage:
### DECISION RULES
• Collar when High volatility + low momentum.
• Straddle when Event window (eg. close to earnings) or IV spike.
• Delta‑neutral puts when Negative trend or skew.
• No hedge (stocks only) when Calm, bullish market.

Ensure that:
1. If you do not want to allocate everyhing you can leave some cash, You can also short sell stocks, by setting stock_action = 'sell' and allocating to stock (more than 0.0).
2. The sum of hedging strategy percentages equals 1.0.
3. The allocation is appropriate for the current market conditions
4. The rationale explains your decision-making process, including how you've incorporated lessons from past experiences
"""
    
    def _get_hedging_format_instructions(self) -> str:
        """
        Get format instructions for hedging-level decision making.
        
        Returns:
            Format instructions text.
        """
        return """
Please provide your hedging strategy decision in the following JSON format:

```
{
  "asset": "AAPL",
  "strategy_type": "collar",
  "rationale": "Detailed explanation of your hedging strategy...",
  "option_positions": [
    {
      "option_type": "put",
      "option_id": "AAPL240419P190",
      "strike": 190.0,
      "expiration_date": "2024-04-19",
      "contracts": 10,
      "premium_per_contract": 5.0,
      "action": "buy",
      "rationale": "Downside protection..."
    },
    {
      "option_type": "call",
      "option_id": "AAPL240419C210",
      "strike": 210.0,
      "expiration_date": "2024-04-19",
      "contracts": 10,
      "premium_per_contract": 4.0,
      "action": "sell",
      "rationale": "Premium collection..."
    }
  ],
  "estimated_cost": 1000.0,
  "estimated_protection": 0.1,
  "confidence": 0.8
}
```

Ensure that:
1. The option positions are appropriate for the specified strategy type
2. The positions are feasible given the available capital
3. The rationale explains your decision-making process, including how you've incorporated lessons from past experiences
4. The estimated cost and protection are realistic
"""
