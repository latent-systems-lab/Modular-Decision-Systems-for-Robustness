"""
Updated Straddle Agent for the hedge fund simulation system.
Implements a straddle strategy using near-the-money calls and puts.
Uses simplified LLM response models and option lookup mechanism.
"""

from datetime import date
from typing import Dict, List, Type

import config

# Import hedging agent
from src.agents.hedging_agent import HedgingAgent

# Import models and utilities
from src.models.pydantic_models import OptionPosition, HedgingStrategy, StockPosition
from src.models.simplified_llm_response_models import HedgingStrategyResponse
from src.utils.llm_utils import get_llm_response
from src.utils.option_lookup import convert_hedging_strategy_response


class StraddleAgent(HedgingAgent):
    """
    Implements a straddle strategy using near-the-money calls and puts.
    A straddle involves buying both a call and a put at the same strike price,
    typically at-the-money, to profit from significant price movement in either direction.
    """

    def __init__(self, data_manager):
        """
        Initialize the StraddleAgent.

        Args:
            data_manager: DataManager instance for accessing stock and options data.
        """
        super().__init__(data_manager)
        self.strategy_type = "straddle"

    def get_response_model(self) -> Type:
        """
        Get the appropriate response model for this agent.

        Returns:
            Pydantic model class for LLM response.
        """
        return HedgingStrategyResponse

    def select_options(
        self,
        ticker: str,
        stock_position: StockPosition,
        available_options: List[Dict],
        available_capital: float,
        market_analysis: Dict,
        use_llm: bool = True,
    ) -> HedgingStrategy:
        """
        Select options for a straddle strategy.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            available_options: List of available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.
            use_llm: Whether to use LLM for option selection.

        Returns:
            HedgingStrategy object containing selected option positions.
        """
        # Change stock position to dict if it's a StockPosition object
        if isinstance(stock_position, StockPosition):
            stock_position = stock_position.dict()

        # Get current price
        current_price = stock_position.get(
            "current_price", available_options[0].get("underlying_price", 100)
        )

        # Get current date from available options
        current_date = (
            available_options[0].get("quote_date", date.today())
            if available_options
            else date.today()
        )

        # Prefilter options for the straddle strategy
        prefiltered_options = self.prefilter_straddle_options(
            available_options, current_price
        )

        if use_llm:
            # Build prompt for LLM
            strategy_description = """
            Design a straddle strategy by selecting:
            1. At-the-money call options
            2. At-the-money put options with the same strike and expiration

            The goal is to profit from significant price movement in either direction.
            This strategy is most effective when you expect high volatility but are uncertain
            about the direction of the price movement.

            For each option, provide:
            1. The option_id from the available options list
            2. The number of contracts to buy
            """

            prompt = self.build_hedging_prompt(
                ticker,
                stock_position,
                prefiltered_options,
                market_analysis,
                strategy_description,
                available_capital,
            )

            # Get response from LLM
            response = get_llm_response(prompt, HedgingStrategyResponse)

            # Convert response to HedgingStrategy using option lookup
            if isinstance(response, HedgingStrategyResponse):
                return convert_hedging_strategy_response(
                    response=response,
                    available_options=prefiltered_options,
                    ticker=ticker,
                    available_capital=available_capital,
                    current_date=current_date,
                )
            else:
                # Fall back to rule-based approach if LLM fails
                return self._rule_based_selection(
                    ticker,
                    stock_position,
                    prefiltered_options,
                    available_capital,
                    market_analysis,
                    current_date,
                )
        else:
            # Rule-based option selection
            return self._rule_based_selection(
                ticker,
                stock_position,
                prefiltered_options,
                available_capital,
                market_analysis,
                current_date,
            )

    def _rule_based_selection(
        self,
        ticker: str,
        stock_position: Dict,
        available_options: List[Dict],
        available_capital: float,
        market_analysis: Dict,
        current_date: date = None,
    ) -> HedgingStrategy:
        """
        Rule-based selection of options for a straddle strategy.
        This version simply picks the first call and first put from the available options.
        The number of contracts is computed based on available capital and the sum of ask prices.
        The expiration date is derived from the option's 'expire_date' if available,
        otherwise computed as current_date + timedelta(days=dte).

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with stock position details.
            available_options: List of available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.
            current_date: Current simulation date.

        Returns:
            HedgingStrategy object containing selected option positions.
        """
        if current_date is None:
            current_date = date.today()

        if not available_options:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="straddle",
                rationale="No options available for straddle strategy.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.5,
            )

        # Get current price
        current_price = stock_position.get("current_price", 0)
        if current_price == 0:
            current_price = available_options[0].get("underlying_price", 100)

        # Filter options into calls and puts
        call_options = [opt for opt in available_options if opt.get("option_type") == 1]
        put_options = [opt for opt in available_options if opt.get("option_type") == 0]

        if not call_options or not put_options:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="straddle",
                rationale="Insufficient call or put options available for straddle strategy.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.5,
            )

        # Select the first call and first put
        selected_call = call_options[0]
        selected_put = put_options[0]

        # Get ask prices
        call_price = selected_call.get("ask", 1)
        put_price = selected_put.get("ask", 1)

        contract_size = getattr(config, "OPTION_CONTRACT_SIZE", 100)
        contract_cost = (call_price + put_price) * contract_size

        # Determine number of contracts we can afford
        num_contracts = int(available_capital // contract_cost) if contract_cost else 0
        if num_contracts < 1:
            num_contracts = 1

        call_expiration = selected_call.get("expire_date", None)
        put_expiration = selected_put.get("expire_date", None)

        # Create OptionPosition objects for call and put
        call_position = OptionPosition(
            asset=ticker,
            option_type="call",
            option_id=selected_call.get("option_id"),
            strike=selected_call.get("strike"),
            expiration_date=call_expiration,
            contracts=num_contracts,
            premium_per_contract=call_price,
            total_premium=call_price * num_contracts * contract_size,
            action="buy",
            strategy="straddle",
            paid_amount=call_price * num_contracts * contract_size,
            market_value=call_price * num_contracts * contract_size,
        )
        put_position = OptionPosition(
            asset=ticker,
            option_type="put",
            option_id=selected_put.get("option_id"),
            strike=selected_put.get("strike"),
            expiration_date=put_expiration,
            contracts=num_contracts,
            premium_per_contract=put_price,
            total_premium=put_price * num_contracts * contract_size,
            action="buy",
            strategy="straddle",
            paid_amount=put_price * num_contracts * contract_size,
            market_value=put_price * num_contracts * contract_size,
        )

        option_positions = [call_position, put_position]
        total_cost = contract_cost * num_contracts

        rationale = (
            f"Rule-based straddle strategy: selected call (ID {selected_call.get('option_id')}) and put (ID {selected_put.get('option_id')}). "
            f"Number of contracts chosen: {num_contracts}. Total estimated cost: ${total_cost:.2f}."
        )

        return HedgingStrategy(
            asset=ticker,
            strategy_type="straddle",
            rationale=rationale,
            option_positions=option_positions,
            estimated_cost=total_cost,
            estimated_protection=1.0,
            confidence=0.7,
        )

    def prefilter_straddle_options(self, available_options, current_price):
        """
        Prefilter options for a straddle strategy.
        Only include options with 30-60 days to expiration and strikes
        within a small percentage (e.g. 5%) of the current price.
        Returns at most 10 options.
        """
        filtered = []
        current_price = available_options[0].get("underlying_last", current_price)
        min_dte = getattr(config, "STRADDLE_MIN_DTE", 38)
        max_dte = getattr(config, "STRADDLE_MAX_DTE", 60)
        atm_tolerance = getattr(config, "STRADDLE_ATM_TOLERANCE", 0.05)

        for opt in available_options:
            dte = opt.get("dte", 0)
            strike = opt.get("strike", 0)
            if min_dte <= dte <= max_dte and abs(strike - current_price) / current_price <= atm_tolerance:
                filtered.append(opt)
        filtered.sort(key=lambda x: abs(x.get("strike") - current_price))
        # choose 5 calls and 5 puts
        calls = [opt for opt in filtered if opt.get("option_type") == 1]
        puts = [opt for opt in filtered if opt.get("option_type") == 0]
        calls = calls[:10]
        puts = puts[:10]
        filtered = calls + puts
        return filtered[:20]

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
            available_capital: Available capital for options.

        Returns:
            Formatted prompt string.
        """
        if isinstance(market_analysis, dict) is False:
            market_analysis = market_analysis.dict()
        prompt = f"""
        You are a hedge fund options strategist. Design a {self.strategy_type} hedging strategy for the following position:

        Asset: {ticker}
        Current Position: {stock_position.get('shares', 0)} shares at ${stock_position.get('average_price', 0):.2f} (Market Value: ${stock_position.get('market_value', 0):,.2f})
        Current Price: ${stock_position.get('current_price', 0):.2f}
        Available Capital for Options: ${available_capital:,.2f} - use all of it.

        IMPORTANT: Ensure that the total cost of the recommended option contracts does not exceed the available capital. For example, if the ask price is $10 per share, then each contract costs $10 * 100 = $1,000. In that case, if the available capital is $5,000, recommend no more than 5 contracts.
        Remember to multiply the number of contracts by 100 to get the total cost.
        Note: Each option contract represents 100 shares. Calculate the cost per contract and ensure the total cost is within the available capital.

        Market Analysis:
        - Regime: {market_analysis.get('regime', 'neutral')}
        - Sentiment Score: {market_analysis.get('sentiment', 0.0):.2f} ({market_analysis.get('sentiment_label', 'neutral')})
        - Outlook Score: {market_analysis.get('outlook', 0.0):.2f} ({market_analysis.get('outlook_label', 'balanced')})
        - Short-term Outlook: {market_analysis.get('short_term_outlook', 'stable')}

        Available Options:
        """

        # Add available options data
        for i, option in enumerate(available_options):
            option_type = "call" if option.get("option_type") == 1 else "put"
            prompt += f"""
            {i+1}. {option_type.upper()} {option.get('strike', 0):.2f} exp {option.get('expire_date', '')} ({option.get('dte', 0)} DTE)
              Option ID: {option.get('option_id', 'N/A')}
              Last: ${option.get('last', 0):.2f}, Bid: ${option.get('bid', 0):.2f}, Ask: ${option.get('ask', 0):.2f}
              Delta: {option.get('delta', 0):.2f}
            """

        # Add strategy-specific instructions
        prompt += f"""
        {strategy_description}

        Your response should be a JSON object with the following structure:
        {{
            "asset": "{ticker}",
            "strategy_type": "straddle",
            "option_contracts": [
                {{
                    "option_type": "call",
                    "option_id": "option_id_from_list",
                    "contracts": number_of_contracts,
                    "action": "buy"
                }},
                {{
                    "option_type": "put",
                    "option_id": "option_id_from_list",
                    "contracts": number_of_contracts,
                    "action": "buy"
                }}
            ],
            "estimated_cost": total_cost_estimate,
            "rationale": "Your rationale for the strategy, quantity of contracts, and option selection.",
        }}

        IMPORTANT: Use only option_ids from the available options list above.
        """

        return prompt
