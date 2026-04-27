from datetime import date, timedelta
from typing import Dict, List, Type

# Import hedging agent base class
from src.agents.hedging_agent import HedgingAgent

# Import models and utilities
from src.models.pydantic_models import OptionPosition, HedgingStrategy, StockPosition
from src.models.simplified_llm_response_models import HedgingStrategyResponse
from src.utils.llm_utils import get_llm_response
from src.utils.option_lookup import convert_hedging_strategy_response


class DeltaNeutralAgent(HedgingAgent):
    """
    Implements a delta-neutral strategy to offset directional risk.
    A delta-neutral strategy involves creating a position with a net delta of zero,
    making the portfolio insensitive to small price movements in the underlying asset.
    """

    def __init__(self, data_manager):
        """
        Initialize the DeltaNeutralAgent.

        Args:
            data_manager: DataManager instance for accessing stock and options data.
        """
        super().__init__(data_manager)
        self.strategy_type = "delta-neutral"

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
        Select options for a delta-neutral strategy.

        Args:
            ticker: Stock ticker symbol.
            stock_position: StockPosition object.
            available_options: List of available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.
            use_llm: Whether to use LLM for option selection.

        Returns:
            HedgingStrategy object containing selected option positions.
        """
        # Convert stock_position to dict if needed.
        if isinstance(stock_position, StockPosition):
            stock_position = stock_position.dict()

        current_date = (
            available_options[0].get("quote_date", date.today())
            if available_options
            else date.today()
        )

        # Prefilter options for delta-neutral strategy.
        prefiltered_options = self.prefilter_delta_neutral_options(available_options)

        if use_llm:
            # Build prompt for LLM with explicit capital constraint.
            prompt = self.build_adjusted_delta_prompt(
                ticker,
                stock_position,
                prefiltered_options,
                market_analysis,
                available_capital,
            )

            # Get response from LLM.
            response = get_llm_response(prompt, HedgingStrategyResponse)
            if isinstance(response, HedgingStrategyResponse):
                options = convert_hedging_strategy_response(
                    response=response,
                    available_options=prefiltered_options,
                    ticker=ticker,
                    available_capital=available_capital,
                    current_date=current_date,
                )
                return options
            else:
                return self._rule_based_selection(
                    ticker,
                    stock_position,
                    prefiltered_options,
                    available_capital,
                    market_analysis,
                    current_date,
                )
        else:
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
        Rule-based selection of options for a delta-neutral strategy.
        This version computes the number of contracts needed to offset stock delta,
        then scales the positions if the total cost exceeds available capital.

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
                strategy_type="delta-neutral",
                rationale="No options available for delta-neutral strategy.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.5,
            )

        # Get current stock position details.
        shares = stock_position.get("shares", 0)
        current_price = stock_position.get("current_price", 0)
        if current_price == 0:
            current_price = available_options[0].get("underlying_price", 100)

        # Assume stock delta is 1 per share.
        stock_delta = shares

        # Filter options by type.
        calls = [opt for opt in available_options if opt.get("option_type") == "call"]
        puts = [opt for opt in available_options if opt.get("option_type") == "put"]

        if not calls or not puts:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="delta-neutral",
                rationale="Insufficient call or put options available.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.5,
            )

        # Sort options by delta proximity.
        calls.sort(key=lambda x: abs(x.get("delta", 0) - 0.5))
        puts.sort(key=lambda x: abs(x.get("delta", 0) + 0.5))

        positive_delta_option = calls[0]
        negative_delta_option = puts[0]

        pos_delta = positive_delta_option.get("delta", 0.5)
        neg_delta = negative_delta_option.get("delta", -0.5)

        # Compute required contracts to offset delta.
        if stock_delta > 0:
            put_contracts_needed = int(stock_delta / (abs(neg_delta) * 100)) + 1
            remaining_delta = stock_delta + (neg_delta * put_contracts_needed * 100)
            call_contracts_needed = 0
            if remaining_delta < 0:
                call_contracts_needed = min(
                    5, int(abs(remaining_delta) / (pos_delta * 100)) + 1
                )
        else:
            call_contracts_needed = int(abs(stock_delta) / (pos_delta * 100)) + 1
            remaining_delta = stock_delta + (pos_delta * call_contracts_needed * 100)
            put_contracts_needed = 0
            if remaining_delta > 0:
                put_contracts_needed = min(
                    5, int(remaining_delta / (abs(neg_delta) * 100)) + 1
                )

        # Compute tentative total cost.
        call_cost = positive_delta_option.get("ask", 0) * call_contracts_needed * 100
        put_cost = negative_delta_option.get("ask", 0) * put_contracts_needed * 100
        total_cost = call_cost + put_cost

        # If total cost exceeds available capital, scale down proportionally.
        if total_cost > available_capital and total_cost > 0:
            scale_factor = available_capital / total_cost
            call_contracts_needed = max(0, int(call_contracts_needed * scale_factor))
            put_contracts_needed = max(0, int(put_contracts_needed * scale_factor))
            call_cost = (
                positive_delta_option.get("ask", 0) * call_contracts_needed * 100
            )
            put_cost = negative_delta_option.get("ask", 0) * put_contracts_needed * 100
            total_cost = call_cost + put_cost

        # Create OptionPosition objects.
        option_positions = []
        if call_contracts_needed > 0:
            call_expiration = (
                positive_delta_option.get("quote_date")
                + timedelta(days=positive_delta_option.get("dte"))
                if positive_delta_option.get("quote_date")
                and positive_delta_option.get("dte")
                else current_date + timedelta(days=30)
            )
            call_position = OptionPosition(
                asset=ticker,
                option_type="call",
                option_id=positive_delta_option.get("option_id"),
                strike=positive_delta_option.get("strike"),
                expiration_date=call_expiration,
                contracts=call_contracts_needed,
                premium_per_contract=positive_delta_option.get("ask"),
                total_premium=positive_delta_option.get("ask")
                * call_contracts_needed
                * 100,
                action="buy",
                strategy="delta-neutral",
                paid_amount=positive_delta_option.get("ask")
                * call_contracts_needed
                * 100,
                market_value=positive_delta_option.get("ask")
                * call_contracts_needed
                * 100,
            )
            option_positions.append(call_position)
        if put_contracts_needed > 0:
            put_expiration = (
                negative_delta_option.get("quote_date")
                + timedelta(days=negative_delta_option.get("dte"))
                if negative_delta_option.get("quote_date")
                and negative_delta_option.get("dte")
                else current_date + timedelta(days=30)
            )
            put_position = OptionPosition(
                asset=ticker,
                option_type="put",
                option_id=negative_delta_option.get("option_id"),
                strike=negative_delta_option.get("strike"),
                expiration_date=put_expiration,
                contracts=put_contracts_needed,
                premium_per_contract=negative_delta_option.get("ask"),
                total_premium=negative_delta_option.get("ask")
                * put_contracts_needed
                * 100,
                action="buy",
                strategy="delta-neutral",
                paid_amount=negative_delta_option.get("ask")
                * put_contracts_needed
                * 100,
                market_value=negative_delta_option.get("ask")
                * put_contracts_needed
                * 100,
            )
            option_positions.append(put_position)

        rationale = (
            f"Rule-based delta-neutral strategy: For a stock position of {shares} shares (delta = {stock_delta}), "
            f"selected call option (ID {positive_delta_option.get('option_id')}) with {call_contracts_needed} contracts "
            f"and put option (ID {negative_delta_option.get('option_id')}) with {put_contracts_needed} contracts. "
            f"Total estimated cost: ${total_cost:.2f}."
        )

        return HedgingStrategy(
            asset=ticker,
            strategy_type="delta-neutral",
            rationale=rationale,
            option_positions=option_positions,
            estimated_cost=total_cost,
            estimated_protection=0.8,  # Default value; adjust as needed
            confidence=0.7,  # Default value; adjust as needed
        )

    def prefilter_delta_neutral_options(self, available_options) -> List[Dict]:
        """
        Prefilter options for a delta-neutral strategy using only long puts.
        Selects puts with appropriate deltas (near -0.5) and 30-60 DTE.
        These puts can be paired with long stock positions to achieve delta neutrality.
        """
        filtered_puts = []

        for opt in available_options:
            # Skip any calls, we only want puts
            if opt.get("option_type") != 0:
                continue

            dte = opt.get("dte", 0)
            if not (65 <= dte <= 90):
                continue

            delta = opt.get("delta", 0)

            # Select puts with delta -0.7 to -0.3 (centered around -0.5)
            if -0.55 <= delta <= -0.45:
                # Store the put with its deviation from ideal delta (-0.5)
                filtered_puts.append((abs(delta + 0.5), opt))

        # Sort puts by how close they are to the ideal delta of -0.5
        filtered_puts.sort(key=lambda x: x[0])
        if not filtered_puts:
            # choose random 10 puts if no ideal delta found
            filtered_puts = [
                (0, opt) for opt in available_options if opt.get("option_type") == 0
            ][:20]
        # Return the top 10 puts (or fewer if not available)
        return [opt[1] for opt in filtered_puts[:20]]

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
        Build a prompt for the LLM to select options for a delta-neutral hedging strategy.

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
        prompt = f"""
        You are a portfolio options strategist. Design a delta-neutral hedging strategy for the following position:
        
        Asset: {ticker}
        Current Position: {stock_position.get('shares', 0)} shares at ${stock_position.get('average_price', 0):.2f} 
        (Market Value: ${stock_position.get('market_value', 0):,.2f})
        Current Price: ${stock_position.get('current_price', 0):.2f}
        Available Capital for Options: ${available_capital:,.2f} - you can use all of it, but do not exceed this capital.
        Adjust delta position to current market analysis, if analysis is very negative you may even try shorting stock with negative delta (eg. buy twice delta-neutral quantity).
        If market is bullish you may not advise buying options or only some of them.
        Basicly, adjust delta position based on market analysis to achieve highest portfolio return long term.

        IMPORTANT: Remember that one option contract should represents 100 shares. For example, if an option's ask price is $10,
        the cost per whole contract is $10 * 100 = $1,000. Ensure that the total cost of all recommended option contracts does not exceed the available capital.
        Remember to multiply the number of contracts by 100 to get the total cost.

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
            "strategy_type": "delta-neutral",
            "option_contracts": [
                {{
                    "option_type": "put",
                    "option_id": "option_id_from_list",
                    "contracts": number_of_contracts,
                    "action": "buy"
                }}
            ],
            "estimated_cost": total_cost_estimate,
            "rationale": "Explain your rationale for the recommended strategy, including how you determined the number of contracts."
        }}
        
        IMPORTANT: Use only option_ids from the available options list above.
        """
        return prompt

    def build_adjusted_delta_prompt(
        self,
        ticker: str,
        stock_position: Dict,
        available_options: List[Dict],
        market_analysis: Dict,
        available_capital: float,
    ) -> str:
        """
        Build a prompt for the LLM to design an 'adjusted delta' hedging strategy,
        allowing net delta to vary based on market conditions.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position (shares, average_price, market_value, current_price).
            available_options: List of available options data (each a dict).
            market_analysis: Market analysis for this asset (regime, sentiment, short_term_outlook).
            available_capital: How much capital can be used for options.

        Returns:
            Prompt string instructing the LLM on how to produce a flexible hedging strategy.
        """
        market_analysis = market_analysis.dict()
        # Extract data from the analysis dict, with defaults
        regime = market_analysis.get("regime", "neutral")
        sentiment_score = market_analysis.get("sentiment", 0.0)
        sentiment_label = market_analysis.get("sentiment_label", "neutral")
        outlook_score = market_analysis.get("outlook", 0.0)
        outlook = market_analysis.get("short_term_outlook", "stable")

        prompt = f"""
    You are an advanced portfolio options strategist. 
    Your goal is to design an 'adjusted delta' strategy for the following stock position, 
    meaning the final net delta (stock delta + any option deltas) can be positive, negative, or near zero, 
    depending on how you interpret the current market analysis:

    Asset: {ticker}
    Current Stock Position: {stock_position.get('shares', 0)} shares at ${stock_position.get('average_price', 0):.2f} 
                        (Market Value: ${stock_position.get('market_value', 0):,.2f})
    Current Stock Price: ${stock_position.get('current_price', 0):.2f}

    Available Capital for Options: ${available_capital:,.2f} 
    (Do not exceed this amount in total premium or margin requirements.)

    Market Analysis:
    - Regime: {regime}
    - Sentiment: {sentiment_label} (score {sentiment_score:.2f})
    - Short-term Outlook: {outlook} (score {outlook_score:.2f})

    You can select any number of option contracts (calls or puts, buy or sell) from the list below. 
    You may choose to buy calls if you want to increase net delta, or buy puts / sell calls to reduce it, or do nothing at all. 
    Remember:
    - 1 option contract = 100 shares.
    - If an option's ask price is $X, the cost per contract is 100 * X. 
    - Your total option costs must not exceed ${available_capital:,.2f}.

    ## Guidance:
    - If the analysis is strongly bullish, you might choose to remain net long or even add additional positive delta (e.g., buying calls, or selling puts) to capitalize on the upside.
    - If the analysis is bearish, you might reduce net delta or go negative delta by buying puts, selling calls, or opening bear spreads.
    - If neutral, you might aim near zero delta or partial hedges, etc.
    - It's okay to do nothing if that best suits the bullish scenario (i.e., no options purchased if we want to remain simply net long stock).
    - It's also okay to over-hedge if you see strong downside risk.

    Below is a list of available options you can use (id, type, strike, price, delta, etc.):

    """
        # Append the list of available options
        for i, opt in enumerate(available_options, start=1):
            opt_type_str = "call" if opt.get("option_type") == 1 else "put"
            prompt += (
                f"{i}. {opt_type_str.upper()} Strike {opt.get('strike', 0):.2f}, "
                f"Exp {opt.get('expire_date', 'N/A')} ({opt.get('dte', 0)} DTE)\n"
                f"   Option ID: {opt.get('option_id', 'N/A')}, "
                f"   Last: ${opt.get('last',0):.2f}, Bid: ${opt.get('bid',0):.2f}, Ask: ${opt.get('ask',0):.2f}, "
                f"   Delta: {opt.get('delta',0):.2f}\n"
            )

        # Instructions for the final JSON structure:
        prompt += f"""
    Now design an 'adjusted delta' strategy that can have net delta positive, negative, or near zero, 
    depending on how you interpret the above market analysis. 
    Address these points:
    1. Which options (if any) to trade (buy or sell, calls or puts).
    2. The number of contracts.
    3. The total cost (ensure it doesn't exceed ${available_capital:,.2f}).
    4. The expected net delta after these option trades (how it changes from the stock position alone).
    5. The rationale for why you chose that net delta stance (e.g., bullish, bearish, or partial hedge).

    IMPORTANT:
    - Do NOT always force net delta = 0. 
    - If you believe a bullish scenario is likely, net delta could be substantially positive (the stock alone is already positive). 
    - If you believe a bearish scenario is likely, net delta could be partially or fully negative.
    - If neutral, near zero delta is fine.
    - Provide a short rationale describing your reasoning in context of the market outlook.
    NEVER EXCEED the available capital in total premium = ${available_capital:,.2f}).
    """

        return prompt
