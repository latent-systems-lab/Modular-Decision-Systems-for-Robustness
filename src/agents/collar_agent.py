from datetime import date, timedelta
from typing import Dict, List

import config

# Import hedging agent base class
from src.agents.hedging_agent import HedgingAgent

# Import models and utilities
from src.models.pydantic_models import OptionPosition, HedgingStrategy, StockPosition
from src.models.simplified_llm_response_models import HedgingStrategyResponse
from src.utils.llm_utils import get_llm_response
from src.utils.option_lookup import convert_hedging_strategy_response


class CollarAgent(HedgingAgent):
    """
    Implements a collar strategy using protective puts and covered calls.
    A collar involves buying a protective put and selling a covered call to protect against downside while offsetting costs.
    """

    def __init__(self, data_manager):
        """
        Initialize the CollarAgent.

        Args:
            data_manager: DataManager instance for accessing stock and options data.
        """
        super().__init__(data_manager)
        self.strategy_type = "collar"

    def select_options(
        self,
        ticker: str,
        stock_position: Dict,
        available_options: List[Dict],
        available_capital: float,
        market_analysis: Dict,
        use_llm: bool = True,
    ) -> HedgingStrategy:
        """
        Select options for a collar strategy.

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
        # Convert stock_position to dict if needed.
        if isinstance(stock_position, StockPosition):
            stock_position = stock_position.dict()
        current_price = stock_position.get(
            "current_price", available_options[0].get("underlying_price", 100)
        )
        current_date = (
            available_options[0].get("quote_date", date.today())
            if available_options
            else date.today()
        )
        prefiltered_options = self.prefilter_collar_options(
            available_options, current_price
        )
        if prefiltered_options == []:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="collar",
                rationale="No suitable options available for collar strategy.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.5,
            )

        if use_llm:
            put_lower_ratio = getattr(config, "COLLAR_PUT_STRIKE_LOWER_RATIO", 0.85)
            put_upper_ratio = getattr(config, "COLLAR_PUT_STRIKE_UPPER_RATIO", 0.90)
            call_lower_ratio = getattr(config, "COLLAR_CALL_STRIKE_LOWER_RATIO", 1.10)
            call_upper_ratio = getattr(config, "COLLAR_CALL_STRIKE_UPPER_RATIO", 1.20)
            min_dte = getattr(config, "COLLAR_MIN_DTE", 30)
            max_dte = getattr(config, "COLLAR_MAX_DTE", 60)

            put_lower_pct = abs((1 - put_upper_ratio) * 100)
            put_upper_pct = abs((1 - put_lower_ratio) * 100)
            call_lower_pct = abs((call_lower_ratio - 1) * 100)
            call_upper_pct = abs((call_upper_ratio - 1) * 100)

            strategy_description = f"""
            Design a collar strategy by selecting:
            1. A protective put with a strike price approximately {put_lower_pct:.0f}-{put_upper_pct:.0f}% below the current stock price.
            2. A covered call with a strike price approximately {call_lower_pct:.0f}-{call_upper_pct:.0f}% above the current stock price.

            The strikes should ideally be balanced (roughly equidistant from the current price).
            Use options with {min_dte}-{max_dte} days to expiration.
            IMPORTANT: The available capital for options is provided. Ensure that the total cost of the recommended option contracts does not exceed this capital.
            For example, if the ask price is $10 per share (i.e. $1,000 per contract), then if available capital is $5,000.
            """
            prompt = self.build_adjusted_hedging_prompt(
                ticker,
                stock_position,
                prefiltered_options,
                market_analysis,
                strategy_description,
                available_capital,
            )

            response = get_llm_response(prompt, HedgingStrategyResponse)
            if isinstance(response, HedgingStrategyResponse):
                strategy = convert_hedging_strategy_response(
                    response=response,
                    available_options=prefiltered_options,
                    ticker=ticker,
                    available_capital=available_capital,
                    current_date=current_date,
                )
                # Present short legs before long legs for readability
                strategy.option_positions.sort(
                    key=lambda pos: 0 if pos.action == "sell" else 1
                )
                return strategy
            else:
                return self._rule_based_selection(
                    ticker,
                    stock_position,
                    prefiltered_options,
                    available_capital,
                    market_analysis,
                )
        else:
            return self._rule_based_selection(
                ticker,
                stock_position,
                prefiltered_options,
                available_capital,
                market_analysis,
            )

    def _rule_based_selection(
        self,
        ticker: str,
        stock_position: Dict,
        available_options: List[Dict],
        available_capital: float,
        market_analysis: Dict,
    ) -> HedgingStrategy:
        """
        Rule-based selection of options for a collar strategy.

        This method:
          - Selects a protective put with a strike 10-15% below the current price.
          - Selects a covered call with a strike 10-20% above the current price.
          - Calculates the number of contracts based on the stock position (shares/100).
          - Computes the net cost (put cost minus call premium) and scales contracts if the total cost exceeds available capital.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            available_options: List of available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.

        Returns:
            HedgingStrategy object containing selected option positions.
        """
        if not available_options:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="collar",
                rationale="No options available for collar strategy.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.5,
            )

        current_price = stock_position.get(
            "current_price", available_options[0].get("underlying_price", 100)
        )

        put_lower = current_price * getattr(
            config, "COLLAR_PUT_STRIKE_LOWER_RATIO", 0.85
        )
        put_upper = current_price * getattr(
            config, "COLLAR_PUT_STRIKE_UPPER_RATIO", 0.90
        )
        call_lower = current_price * getattr(
            config, "COLLAR_CALL_STRIKE_LOWER_RATIO", 1.10
        )
        call_upper = current_price * getattr(
            config, "COLLAR_CALL_STRIKE_UPPER_RATIO", 1.20
        )

        puts = [
            opt
            for opt in available_options
            if opt.get("option_type") == 0
            and put_lower <= opt.get("strike", 0) <= put_upper
        ]
        calls = [
            opt
            for opt in available_options
            if opt.get("option_type") == 1
            and call_lower <= opt.get("strike", 0) <= call_upper
        ]

        if not puts or not calls:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="collar",
                rationale="Insufficient suitable puts or calls for collar strategy.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.5,
            )

        puts.sort(key=lambda x: x.get("strike"))
        calls.sort(key=lambda x: x.get("strike"))

        protective_put = puts[0]
        covered_call = calls[0]

        # Determine number of contracts (typically stock_position shares / 100)
        shares = stock_position.get("shares", 0)
        contracts = max(1, int(shares / 100))

        contract_size = getattr(config, "OPTION_CONTRACT_SIZE", 100)
        put_cost = protective_put.get("ask", 0) * contracts * contract_size
        call_premium = covered_call.get("bid", 0) * contracts * contract_size
        net_cost = put_cost - call_premium

        # Scale contracts if cost exceeds available capital
        if put_cost > available_capital:
            scale = available_capital / put_cost
            contracts = max(1, int(contracts * scale))
            put_cost = protective_put.get("ask", 0) * contracts * contract_size
            call_premium = covered_call.get("bid", 0) * contracts * contract_size
            net_cost = put_cost - call_premium

        put_expiration = (
            (
                protective_put.get("quote_date")
                + timedelta(days=protective_put.get("dte"))
            )
            if protective_put.get("quote_date") and protective_put.get("dte")
            else None
        )
        call_expiration = (
            (covered_call.get("quote_date") + timedelta(days=covered_call.get("dte")))
            if covered_call.get("quote_date") and covered_call.get("dte")
            else None
        )

        put_position = OptionPosition(
            asset=ticker,
            option_type="put",
            option_id=protective_put.get("option_id"),
            strike=protective_put.get("strike"),
            expiration_date=put_expiration,
            contracts=contracts,
            premium_per_contract=protective_put.get("ask"),
            total_premium=protective_put.get("ask") * contracts * contract_size,
            action="buy",
            strategy="collar",
            paid_amount=put_cost,
        )
        call_position = OptionPosition(
            asset=ticker,
            option_type="call",
            option_id=covered_call.get("option_id"),
            strike=covered_call.get("strike"),
            expiration_date=call_expiration,
            contracts=contracts,
            premium_per_contract=covered_call.get("bid"),
            total_premium=covered_call.get("bid") * contracts * contract_size,
            action="sell",
            strategy="collar",
            paid_amount=-call_premium,
        )

        rationale = (
            f"Rule-based collar strategy: protective put at ${protective_put.get('strike'):.2f} "
            f"and covered call at ${covered_call.get('strike'):.2f} for {contracts} contracts. "
            f"Net cost (put cost minus call premium): ${net_cost:.2f}."
        )

        return HedgingStrategy(
            asset=ticker,
            strategy_type="collar",
            rationale=rationale,
            option_positions=[put_position, call_position],
            estimated_cost=net_cost,
            estimated_protection=(current_price - protective_put.get("strike"))
            / current_price,
            confidence=0.8,
        )

    def prefilter_collar_options(
        self, available_options: List[Dict], current_price: float
    ) -> List[Dict]:
        """Return puts and calls that fall within the configured collar bands."""

        if not available_options:
            return []

        current_price = available_options[0].get("underlying_price", current_price)
        min_dte = getattr(config, "COLLAR_MIN_DTE", 30)
        max_dte = getattr(config, "COLLAR_MAX_DTE", 60)

        put_lower = current_price * getattr(
            config, "COLLAR_PUT_STRIKE_LOWER_RATIO", 0.85
        )
        put_upper = current_price * getattr(
            config, "COLLAR_PUT_STRIKE_UPPER_RATIO", 0.90
        )
        call_lower = current_price * getattr(
            config, "COLLAR_CALL_STRIKE_LOWER_RATIO", 1.10
        )
        call_upper = current_price * getattr(
            config, "COLLAR_CALL_STRIKE_UPPER_RATIO", 1.20
        )

        puts: List[Dict] = []
        calls: List[Dict] = []
        for option in available_options:
            dte = option.get("dte", 0)
            if not (min_dte <= dte <= max_dte):
                continue
            strike = option.get("strike", 0)
            if option.get("option_type") == 0 and put_lower <= strike <= put_upper:
                puts.append(option)
            elif option.get("option_type") == 1 and call_lower <= strike <= call_upper:
                calls.append(option)

        puts.sort(key=lambda opt: abs(opt.get("strike", 0) - (put_lower + put_upper) / 2))
        calls.sort(key=lambda opt: abs(opt.get("strike", 0) - (call_lower + call_upper) / 2))

        return (puts[:10] + calls[:10]) if puts and calls else []

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
        Build a prompt for LLM to select options for a collar hedging strategy.

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
        if hasattr(market_analysis, "dict"):
            market_analysis = market_analysis.dict()

        put_lower_ratio = getattr(config, "COLLAR_PUT_STRIKE_LOWER_RATIO", 0.85)
        put_upper_ratio = getattr(config, "COLLAR_PUT_STRIKE_UPPER_RATIO", 0.90)
        call_lower_ratio = getattr(config, "COLLAR_CALL_STRIKE_LOWER_RATIO", 1.10)
        call_upper_ratio = getattr(config, "COLLAR_CALL_STRIKE_UPPER_RATIO", 1.20)
        min_dte = getattr(config, "COLLAR_MIN_DTE", 30)
        max_dte = getattr(config, "COLLAR_MAX_DTE", 60)

        put_lower_pct = abs((1 - put_upper_ratio) * 100)
        put_upper_pct = abs((1 - put_lower_ratio) * 100)
        call_lower_pct = abs((call_lower_ratio - 1) * 100)
        call_upper_pct = abs((call_upper_ratio - 1) * 100)

        prompt = f"""
        You are a hedge fund options strategist. Design a collar hedging strategy for the following position:

        Asset: {ticker}
        Current Position: {stock_position.get('shares', 0)} shares at ${stock_position.get('average_price', 0):.2f} (Market Value: ${stock_position.get('market_value', 0):,.2f})
        Current Price: ${stock_position.get('current_price', 0):.2f}
        Available Capital for Options: ${available_capital:,.2f} - use all of it, but do not exceed this capital.

        IMPORTANT: For a collar strategy, you must recommend:
        - A protective put with a strike price approximately {put_lower_pct:.0f}-{put_upper_pct:.0f}% below the current stock price.
        - A covered call with a strike price approximately {call_lower_pct:.0f}-{call_upper_pct:.0f}% above the current stock price.
        The strikes should be balanced (ideally equidistant from the current price) so that the call premium partially or fully offsets the cost of the put.
        Use options with {min_dte}-{max_dte} days to expiration.
        Remember that each contract represents 100 shares; calculate the cost per contract and ensure that the total cost does not exceed the available capital.

        Market Analysis:
        - Regime: {market_analysis.get('regime', 'neutral')}
        - Sentiment Score: {market_analysis.get('sentiment', 0.0):.2f} ({market_analysis.get('sentiment_label', 'neutral')})
        - Outlook Score: {market_analysis.get('outlook', 0.0):.2f} ({market_analysis.get('outlook_label', 'balanced')})
        - Short-term Outlook: {market_analysis.get('short_term_outlook', 'stable')}

        {strategy_description}

        Your response should be a JSON object with the following structure:
        {{
            "asset": "{ticker}",
            "strategy_type": "collar",
            "option_contracts": [
                {{
                    "option_type": "put",
                    "option_id": "option_id_from_list",
                    "contracts": number_of_contracts,
                    "action": "buy"
                }},
                {{
                    "option_type": "call",
                    "option_id": "option_id_from_list",
                    "contracts": number_of_contracts,
                    "action": "sell"
                }}
            ],
            "estimated_cost": total_cost_estimate,
            "rationale": "Explain your rationale for the recommended strategy, including how you determined the option strikes and the number of contracts."
        }}

        IMPORTANT: Use only option_ids from the available options list above. Be reasonable in your selections and quantities.
        """
        return prompt

    def build_adjusted_hedging_prompt(
        self,
        ticker: str,
        stock_position: Dict,
        available_options: List[Dict],
        market_analysis: Dict,
        strategy_description: str,
        available_capital: float,
    ) -> str:
        """
        Build a prompt for the LLM to select options for a collar hedging strategy.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            available_options: List of available options.
            market_analysis: Market analysis for this asset.
            strategy_description: Additional instructions or guidelines for the strategy.
            available_capital: Available capital for options.

        Returns:
            A formatted prompt string.
        """
        market_analysis = market_analysis.dict()
        prompt = f"""
    You are an experienced hedge fund options strategist tasked with designing a collar hedging strategy for the asset "{ticker}".

    Position Details:
    - Shares: {stock_position.get('shares', 0)}
    - Average Price: ${stock_position.get('average_price', 0):.2f}
    - Market Value: ${stock_position.get('market_value', 0):,.2f}
    - Current Price: ${stock_position.get('current_price', 0):.2f}

    Available Options Capital: ${available_capital:,.2f}
    (Do not exceed this amount in total cost.)

    Collar Strategy Requirements:
    1. Recommend a protective put with a strike price approximately 10-15% below the current stock price.
    2. Recommend a covered call with a strike price approximately 10-20% above the current stock price.
    3. The strikes should be balanced (ideally equidistant from the current price) so that the premium received from the call partially or fully offsets the put's cost.
    4. Use options with an expiration between 30 to 60 days.
    5. Note: Each option contract represents 100 shares. Calculate the cost per contract and ensure the total cost is within the available capital.
    6. For the estimated cost, add the cost for the put option and the cost for the call option together.
    - For example, for 4 contracts with a put ask price of $13.55 and a call ask price of $11.50, the calculation should be:
  (4 * 13.55 * 100) + (4 * 11.50 * 100) = $10,020.

    Market Analysis:
    - Regime: {market_analysis.get('regime', 'neutral')}
    - Sentiment Score: {market_analysis.get('sentiment', 0.0):.2f} ({market_analysis.get('sentiment_label', 'neutral')})
    - Short-term Outlook: {market_analysis.get('short_term_outlook', 'stable')} (score {market_analysis.get('outlook', 0.0):.2f})

    Available Options:
    """
        # Append a numbered list of available options.
        for i, option in enumerate(available_options, start=1):
            option_type_str = "Call" if option.get("option_type", 1) == 1 else "Put"
            prompt += f"""
    {i}. {option_type_str} Option:
    - Option ID: {option.get('option_id', 'N/A')}
    - Strike: ${option.get('strike', 0):.2f}
    - Expiration Date: {option.get('expire_date', 'N/A')} ({option.get('dte', 0)} days to expiration)
    - Last Price: ${option.get('last', 0):.2f}
    - Bid: ${option.get('bid', 0):.2f} | Ask: ${option.get('ask', 0):.2f}
    - Implied Volatility: {option.get('implied_volatility', 0):.2f} | Delta: {option.get('delta', 0):.2f}
    """
        # Append additional strategy instructions and JSON response format.
        prompt += f"""

    Additional Strategy Instructions:
    {strategy_description}

    IMPORTANT: Use ONLY the option_ids provided in the Available Options list.
    Make sure option_ids are valid and correspond to the selected options.
    Explain your rationale for selecting the option strikes, contract numbers and calculations.
    DO NOT EXCEED the available capital in your recommendations = ${available_capital:,.2f}.
    """
        return prompt
