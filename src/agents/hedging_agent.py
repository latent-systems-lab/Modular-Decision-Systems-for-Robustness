"""
Hedging Agent for the hedge fund simulation system.
Abstract class for all hedging agents.
"""

from datetime import date
from typing import Dict, List

# Import base agent
from src.agents.base_agent import BaseAgent

# Import models and utilities
from src.models.pydantic_models import HedgingStrategy
from src.models.llm_response_models import HedgingStrategyResponse


class HedgingAgent(BaseAgent):
    """
    Abstract base class for all hedging agents.
    Provides common functionality for hedging strategies.
    """

    def __init__(self, data_manager):
        """
        Initialize the HedgingAgent.

        Args:
            data_manager: DataManager instance for accessing stock and options data.
        """
        super().__init__(data_manager)
        self.strategy_type = "hedging"
        self.performance_history = []

    def get_action(
        self,
        ticker: str,
        current_date: date,
        stock_position: Dict,
        available_capital: float,
        market_analysis: Dict,
    ) -> Dict:
        """
        Get recommended hedging action.

        Args:
            ticker: Stock ticker symbol.
            current_date: Current simulation date.
            stock_position: Dictionary with details of the stock position.
            available_capital: Available capital for this asset.
            market_analysis: Market analysis for this asset.

        Returns:
            Dictionary containing recommended option positions and rationale.
        """
        # Get available options
        available_options = self.get_available_options(ticker, current_date)

        if not available_options:
            return {
                "option_positions": [],
                "rationale": "No suitable options available for hedging.",
                "estimated_cost": 0.0,
                "estimated_protection": 0.0,
            }

        # Select options for hedging strategy
        hedging_strategy = self.select_options(
            ticker,
            stock_position,
            available_options,
            available_capital,
            market_analysis,
        )

        return {
            "option_positions": hedging_strategy.option_positions,
            "rationale": hedging_strategy.rationale,
            "estimated_cost": hedging_strategy.estimated_cost,
            "estimated_protection": hedging_strategy.estimated_protection,
        }

    def update_performance(self, performance_metrics: Dict) -> None:
        """
        Update agent's internal state based on performance metrics.

        Args:
            performance_metrics: Dictionary of performance metrics.
        """
        self.performance_history.append(performance_metrics)

    def select_options(
        self,
        ticker: str,
        stock_position: Dict,
        available_options: List[Dict],
        available_capital: float,
        market_analysis: Dict,
    ) -> HedgingStrategy:
        """
        Select options for hedging strategy.
        To be implemented by subclasses.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            available_options: List of available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.

        Returns:
            HedgingStrategy object containing selected option positions.
        """
        raise NotImplementedError("Subclasses must implement select_options method")

    def build_hedging_prompt(
        self,
        ticker: str,
        stock_position: Dict,
        available_options: List[Dict],
        market_analysis: Dict,
        strategy_description: str,
        available_capital: float,
    ) -> str:
        """
        Build a prompt for LLM to select options for hedging.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            available_options: List of available options.
            market_analysis: Market analysis for this asset.
            strategy_description: Description of the hedging strategy.

        Returns:
            Formatted prompt string.
        """
        prompt = f"""
        You are a hedge fund options strategist. Design a {self.strategy_type} hedging strategy for the following position:

        Asset: {ticker}
        Current Position: {stock_position.get('shares', 0)} shares at ${stock_position.get('average_price', 0):.2f} (Market Value: ${stock_position.get('market_value', 0):,.2f})
        Current Price: ${stock_position.get('current_price', 0):.2f}

        You have ${available_capital:,.2f} to allocate to this options strategy.
        Given this substantial capital, recommend an appropriate number of contracts (try to use all of it).

        Market Analysis:
        - Regime: {market_analysis.get('regime', 'neutral')}
        - Sentiment Score: {market_analysis.get('sentiment', 0.0):.2f} ({market_analysis.get('sentiment_label', 'neutral')})
        - Outlook Score: {market_analysis.get('outlook', 0.0):.2f} ({market_analysis.get('outlook_label', 'balanced')})
        - Short-term Outlook: {market_analysis.get('short_term_outlook', 'stable')}

        Available Options:
        """
        # Add available options data (limit to 10 to keep prompt size reasonable)
        # Filter available options to get the first call and first put
        call_options = [
            opt for opt in available_options if opt.get("option_type", 1) == 1
        ]
        put_options = [
            opt for opt in available_options if opt.get("option_type", 1) == 0
        ]
        filtered_options = [call_options[0], put_options[0]]
        for option in filtered_options[:2]:
            option_type = "call" if option.get("option_type", 1) == 1 else "put"
            prompt += f"""
            - {option_type.upper()} {option.get('strike', 0):.2f} exp {option.get('quote_date', '')} ({option.get('dte', 0)} DTE)
              Option ID: {option.get('option_id', 'N/A')}
              Last: ${option.get('last', 0):.2f}, Bid: ${option.get('bid', 0):.2f}, Ask: ${option.get('ask', 0):.2f}
              IV: {option.get('implied_volatility', 0):.2f}, Delta: {option.get('delta', 0):.2f}
            """

        # Add strategy-specific instructions
        prompt += f"""
        {strategy_description}

        For your recommended strategy, provide:
        1. Specific option contracts to use (strike, expiration, type)
        2. Number of contracts
        3. Estimated cost/premium
        4. Expected protection/payoff profile
        5. Rationale for this strategy
        """

        return prompt

    def parse_llm_response(self, response, ticker: str) -> HedgingStrategyResponse:
        """
        Parse LLM response into a HedgingStrategy object.

        Args:
            response: LLM response object.
            ticker: Stock ticker symbol.

        Returns:
            HedgingStrategy object.
        """
        if isinstance(response, HedgingStrategyResponse):
            # Ensure asset field is set correctly
            response.asset = ticker
            return response
        else:
            # Return empty strategy if parsing fails
            return HedgingStrategyResponse(
                asset=ticker,
                strategy_type=self.strategy_type,
                rationale="Failed to parse LLM response.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.0,
            )
