"""
Asset Manager for the portfolio risk-management simulation system.
Responsible for managing per-asset logic including stock vs. options split and hedging strategies.
"""

from datetime import date
from typing import Dict, List, Optional

# Import models and utilities
from src.models.pydantic_models import (
    AssetSplit,
    MarketRegime,
    HedgingStrategy,
    SplitResponse,
)
from src.utils.llm_utils import get_llm_response

# Import agents
from src.agents.collar_agent import CollarAgent
from src.agents.straddle_agent import StraddleAgent
from src.agents.delta_neutral_agent import DeltaNeutralAgent


class AssetManager:
    """
    Manages per-asset allocation and hedging strategies.
    """

    def __init__(self, data_manager):
        """
        Initialize the AssetManager.

        Args:
            data_manager: DataManager instance for accessing stock and options data.
        """
        self.data_manager = data_manager

        # Initialize hedging agents
        self.collar_agent = CollarAgent(
            data_manager,
        )
        self.straddle_agent = StraddleAgent(data_manager)
        self.delta_neutral_agent = DeltaNeutralAgent(data_manager)

    def determine_stock_options_split(
        self,
        ticker: str,
        allocation_amount: float,
        market_analysis: MarketRegime,
        current_date: date,
        use_llm: bool = True,
    ) -> AssetSplit:
        """
        Determine how to split allocation between stock and options for an asset.

        Args:
            ticker: Stock ticker symbol.
            allocation_amount: Total amount allocated to this asset.
            market_analysis: MarketRegime object for this asset.
            current_date: Current simulation date.
            use_llm: Whether to use LLM for split decisions.

        Returns:
            AssetSplit object containing the split decision.
        """
        if use_llm:
            # Build prompt for LLM
            prompt = f"""
            You are a portfolio manager deciding how to split ${allocation_amount:,.2f} allocated to {ticker} between stock and options.
            
            Market Analysis for {ticker}:
            - Market Regime: {market_analysis.regime}
            - Sentiment Score: {market_analysis.sentiment:.2f} ({market_analysis.sentiment_label})
            - Outlook Score: {market_analysis.outlook:.2f} ({market_analysis.outlook_label})
            - Short-term Outlook: {market_analysis.short_term_outlook}
            - Confidence: {market_analysis.confidence:.2f}
            
            Based on this analysis, determine:
            1. What percentage should be allocated to stock vs. options?
            2. Provide a brief rationale for this decision.
            
            Consider:
            - In bullish regimes, you might allocate more to stock
            - In bearish regimes, you might allocate more to protective options
            - In volatile regimes, you might use options for both protection and opportunity
            - Options typically involve higher risk but can provide leverage or protection
            - Do not keep too much cash (max 20% if bearish), do not allocate more than 10% to options
            """

            # Get response from LLM
            response = get_llm_response(prompt, SplitResponse)

            # Check if response is valid
            # If the model returns percentages (e.g., 20/80), convert to decimals
            if (
                isinstance(response, dict)
                and "stock_percentage" in response
                and "options_percentage" in response
            ):
                stock_percentage = response["stock_percentage"]
                options_percentage = response["options_percentage"]
                # Ensure percentages sum to 1
                if stock_percentage + options_percentage != 1.0:
                    response["stock_percentage"] = float(stock_percentage / 100)
                    response["options_percentage"] = float(options_percentage / 100)
            if isinstance(response, SplitResponse):
                # Create AssetSplit object
                return AssetSplit(
                    asset=ticker,
                    total_allocation=allocation_amount,
                    stock_percentage=response.stock_percentage,
                    options_percentage=response.options_percentage,
                    stock_amount=allocation_amount * response.stock_percentage,
                    options_amount=allocation_amount * response.options_percentage,
                )
            else:
                # Fall back to rule-based approach if LLM fails
                print(
                    f"LLM split decision failed for {ticker}, falling back to rule-based approach"
                )
                return self._rule_based_split(
                    ticker, allocation_amount, market_analysis
                )
        else:
            # Use rule-based approach
            return self._rule_based_split(ticker, allocation_amount, market_analysis)

    def _rule_based_split(
        self, ticker: str, allocation_amount: float, market_analysis: MarketRegime
    ) -> AssetSplit:
        """
        Perform rule-based stock/options split without using LLM.

        Args:
            ticker: Stock ticker symbol.
            allocation_amount: Total amount allocated to this asset.
            market_analysis: MarketRegime object for this asset.

        Returns:
            AssetSplit object containing the split decision.
        """
        # Default split (70% stock, 30% options)
        stock_percentage = 0.7
        options_percentage = 0.3

        # Adjust based on market regime
        if market_analysis.regime == "bullish":
            # In bullish regime, allocate more to stock
            stock_percentage = 0.8
            options_percentage = 0.2
            rationale = "Bullish regime favors higher stock allocation with smaller options component for leverage."
        elif market_analysis.regime == "bearish":
            # In bearish regime, allocate more to options for protection
            stock_percentage = 0.6
            options_percentage = 0.4
            rationale = "Bearish regime requires more allocation to protective options to hedge downside risk."
        elif market_analysis.regime == "volatile":
            # In volatile regime, use more options for both protection and opportunity
            stock_percentage = 0.5
            options_percentage = 0.5
            rationale = "Volatile regime benefits from equal stock/options split to capture volatility and provide protection."
        else:  # neutral
            rationale = "Neutral regime suggests standard 70/30 stock/options split for balanced exposure."

        # Calculate amounts
        stock_amount = allocation_amount * stock_percentage
        options_amount = allocation_amount * options_percentage

        return AssetSplit(
            asset=ticker,
            total_allocation=allocation_amount,
            stock_percentage=stock_percentage,
            options_percentage=options_percentage,
            stock_amount=stock_amount,
            options_amount=options_amount,
            rationale=rationale,
        )

    def select_hedging_strategy(
        self,
        ticker: str,
        stock_position: Dict,
        options_allocation: float,
        market_analysis: MarketRegime,
        current_date: date,
        strategy_type: str = "optimal",
        use_llm: bool = True,
    ) -> HedgingStrategy:
        """
        Select an appropriate hedging strategy for an asset.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            options_allocation: Amount allocated to options.
            market_analysis: MarketRegime object for this asset.
            current_date: Current simulation date.
            strategy_type: Type of hedging strategy to focus on.
            use_llm: Whether to use LLM for strategy selection.

        Returns:
            HedgingStrategy object containing the selected strategy.
        """
        # Get available options
        available_options = self._get_available_options(ticker, current_date)

        if not available_options:
            # No options available, return empty strategy
            return HedgingStrategy(
                asset=ticker,
                strategy_type="none",
                rationale="No options available for this asset.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.0,
            )

        normalized_strategy = self._normalize_strategy_name(strategy_type)
        market_analysis_dict = (
            market_analysis.dict()
            if hasattr(market_analysis, "dict")
            else market_analysis
        )
        if normalized_strategy in {"rules_only", "rules-only"}:
            return self.select_rule_based_hedging_strategy(
                ticker=ticker,
                stock_position=stock_position,
                options_allocation=options_allocation,
                market_analysis=market_analysis,
                current_date=current_date,
                available_options=available_options,
                strategies=["collar", "straddle", "delta-neutral"],
                use_llm=use_llm,
            )

        # Select strategy based on type or market conditions
        if normalized_strategy == "collar" or (
            normalized_strategy == "optimal" and market_analysis.regime == "bullish"
        ):
            return self.collar_agent.select_options(
                ticker=ticker,
                stock_position=stock_position,
                available_options=available_options,
                available_capital=options_allocation,
                market_analysis=market_analysis_dict,
                use_llm=use_llm,
            )
        elif normalized_strategy == "straddle" or (
            normalized_strategy == "optimal" and market_analysis.regime == "volatile"
        ):
            return self.straddle_agent.select_options(
                ticker=ticker,
                stock_position=stock_position,
                available_options=available_options,
                available_capital=options_allocation,
                market_analysis=market_analysis_dict,
                use_llm=use_llm,
            )
        elif normalized_strategy == "delta_neutral" or (
            normalized_strategy == "optimal" and market_analysis.regime == "neutral"
        ):
            return self.delta_neutral_agent.select_options(
                ticker=ticker,
                stock_position=stock_position,
                available_options=available_options,
                available_capital=options_allocation,
                market_analysis=market_analysis_dict,
                use_llm=use_llm,
            )
        else:  # Default to protective put for bearish regime
            # For protective put, we'll use a simplified version of the collar strategy
            # (just the put side without selling calls)
            hedging_result = self.collar_agent.select_options(
                ticker=ticker,
                stock_position=stock_position,
                available_options=available_options,
                available_capital=options_allocation,
                market_analysis=market_analysis_dict,
                use_llm=use_llm,
            )

            # Filter to keep only put positions
            put_positions = [
                pos
                for pos in hedging_result.option_positions
                if pos.option_type == "put"
            ]

            return HedgingStrategy(
                asset=ticker,
                strategy_type="protective-put",
                rationale="Protective put strategy to hedge downside risk in bearish market.",
                option_positions=put_positions,
                estimated_cost=sum(pos.total_premium for pos in put_positions),
                estimated_protection=hedging_result.estimated_protection,
                confidence=hedging_result.confidence,
            )

    def _normalize_strategy_name(self, strategy: str) -> str:
        """Normalize strategy names to snake_case identifiers."""

        return strategy.lower().replace("-", "_").replace(" ", "_")

    def _get_agent_for_strategy(self, strategy: str):
        """Return the agent responsible for the provided strategy name."""

        if strategy == "collar":
            return self.collar_agent
        if strategy == "straddle":
            return self.straddle_agent
        if strategy in {"delta_neutral", "deltaneutral"}:
            return self.delta_neutral_agent
        return None

    def _determine_rule_based_weights(
        self, strategies: List[str], market_analysis: MarketRegime
    ) -> Dict[str, float]:
        """Derive allocation weights for the rule-based splitter."""

        if not strategies:
            return {}

        regime = "neutral"
        if market_analysis:
            if isinstance(market_analysis, dict):
                regime = market_analysis.get("regime", "neutral")
            else:
                regime = getattr(market_analysis, "regime", "neutral")

        # Base weights evenly distributed across all requested strategies.
        weights = {
            self._normalize_strategy_name(s): 1.0 / len(strategies) for s in strategies
        }

        # Adjust weights depending on the prevailing regime. The adjustments
        # preserve the focus on the three supported strategies while tilting
        # toward the risk posture best suited for the environment.
        adjustments = {
            "bearish": {"collar": 0.35, "straddle": 0.20, "delta_neutral": 0.45},
            "volatile": {"collar": 0.25, "straddle": 0.45, "delta_neutral": 0.30},
            "bullish": {"collar": 0.45, "straddle": 0.35, "delta_neutral": 0.20},
        }

        if regime in adjustments:
            override = adjustments[regime]
            total = sum(override.values())
            if total > 0:
                weights.update(
                    {k: v / total for k, v in override.items() if k in weights}
                )

        # Ensure weights sum to one after adjustments.
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def select_rule_based_hedging_strategy(
        self,
        ticker: str,
        stock_position: Dict,
        options_allocation: float,
        market_analysis: MarketRegime,
        current_date: date,
        available_options: Optional[List[Dict]] = None,
        strategies: Optional[List[str]] = None,
        use_llm: bool = False,
    ) -> HedgingStrategy:
        """
        Apply the rules-based selector using a deterministic split across the
        collar, straddle, and delta-neutral agents.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            options_allocation: Capital allocated to options.
            market_analysis: Market regime analysis for the asset.
            current_date: Trading date used to source option chains.
            available_options: Optional pre-fetched options data.
            strategies: Optional list restricting which strategies participate.
            use_llm: Flag indicating whether LLMs are enabled for downstream agents.

        Returns:
            HedgingStrategy object representing the selected hedging approach.
        """
        if options_allocation <= 0:
            regime_name = (
                getattr(market_analysis, "regime", "neutral")
                if market_analysis
                else "neutral"
            )
            return HedgingStrategy(
                asset=ticker,
                strategy_type="combined",
                rationale=f"Rule-based selector could not allocate capital for {regime_name} regime.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.0,
            )

        if available_options is None:
            available_options = self._get_available_options(ticker, current_date)
        if not available_options:
            regime_name = (
                getattr(market_analysis, "regime", "neutral")
                if market_analysis
                else "neutral"
            )
            return HedgingStrategy(
                asset=ticker,
                strategy_type="combined",
                rationale=f"Rule-based selector found no options to deploy capital for {regime_name} regime.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.0,
            )

        active_strategies = strategies or ["collar", "straddle", "delta-neutral"]
        normalized_strategies = [
            self._normalize_strategy_name(s) for s in active_strategies
        ]
        weights = self._determine_rule_based_weights(
            normalized_strategies, market_analysis
        )

        market_dict = (
            market_analysis.dict()
            if hasattr(market_analysis, "dict")
            else market_analysis
        )

        combined_option_positions = []
        total_estimated_cost = 0.0
        estimated_protections = []
        confidences = []
        rationales = []

        for strategy_name in normalized_strategies:
            allocation_factor = weights.get(strategy_name, 0.0)
            if allocation_factor <= 0:
                continue

            agent = self._get_agent_for_strategy(strategy_name)
            if agent is None:
                rationales.append(
                    f"{strategy_name.replace('_', ' ').title()}: unsupported strategy requested by rule-based splitter."
                )
                continue

            capital = options_allocation * allocation_factor
            if capital <= 0:
                continue

            try:
                result = agent.select_options(
                    ticker=ticker,
                    stock_position=stock_position,
                    available_options=available_options,
                    available_capital=capital,
                    market_analysis=(
                        market_dict
                        if isinstance(market_dict, dict)
                        else market_analysis
                    ),
                    use_llm=use_llm,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                rationales.append(
                    f"{strategy_name.replace('_', ' ').title()}: encountered error during selection ({exc})."
                )
                continue

            if result is None:
                rationales.append(
                    f"{strategy_name.replace('_', ' ').title()}: no recommendation returned by agent."
                )
                continue

            combined_option_positions.extend(result.option_positions)
            total_estimated_cost += result.estimated_cost
            estimated_protections.append(result.estimated_protection)
            confidences.append(result.confidence)
            display_name = strategy_name.replace("_", " ").title()
            rationales.append(f"{display_name}: {result.rationale}")

        if not combined_option_positions:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="combined",
                rationale="Rule-based selector could not generate any collar, straddle, or delta-neutral positions.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.0,
            )

        avg_estimated_protection = (
            sum(estimated_protections) / len(estimated_protections)
            if estimated_protections
            else 0.0
        )
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return HedgingStrategy(
            asset=ticker,
            strategy_type="combined",
            rationale=(
                "Rule-based selector blended collar, straddle, and delta-neutral hedges "
                "according to the prevailing market regime. " + " | ".join(rationales)
            ),
            option_positions=combined_option_positions,
            estimated_cost=total_estimated_cost,
            estimated_protection=avg_estimated_protection,
            confidence=avg_confidence,
        )

    def _get_available_options(self, ticker: str, current_date: date) -> List[Dict]:
        """
        Get available options for a specific ticker and date.

        Args:
            ticker: Stock ticker symbol.
            current_date: Current simulation date.

        Returns:
            List of option data dictionaries.
        """
        # Try to load options from database
        options_data = self.data_manager.load_options_data(
            ticker=ticker,
            quote_date=current_date,
            min_dte=7,  # At least 1 week to expiration
            max_dte=60,  # At most 2 months to expiration
        )

        # Convert to list of dictionaries
        return [option.dict() for option in options_data]

    def select_all_hedging_strategies(
        self,
        ticker: str,
        stock_position: Dict,
        options_allocation: Dict,
        market_analysis: MarketRegime,
        current_date: date,
        strategy_type: List[str],
        use_llm: bool = True,
    ) -> HedgingStrategy:
        """
        Select options using all three hedging strategies concurrently.
        Allocates 33% of the available options capital to each strategy.

        Args:
            ticker: Stock ticker symbol.
            stock_position: Dictionary with details of the stock position.
            options_allocation: Total capital allocated for options for this asset.
            market_analysis: MarketRegime object for this asset.
            current_date: Current simulation date.
            use_llm: Whether to use LLM-based selection.

        Returns:
            A combined HedgingStrategy object containing aggregated option positions,
            the total estimated cost, combined rationales, and averaged protection/confidence.
        """
        combined_option_positions = []
        total_estimated_cost = 0.0
        rationales = []
        estimated_protections = []
        confidences = []
        # Retrieve available options for the asset
        available_options = self._get_available_options(ticker, current_date)
        if not available_options:
            return HedgingStrategy(
                asset=ticker,
                strategy_type="none",
                rationale="No available options for hedging.",
                option_positions=[],
                estimated_cost=0.0,
                estimated_protection=0.0,
                confidence=0.0,
            )

        # Mapping from normalized strategy name to the appropriate agent call.
        for strat in strategy_type:
            s = self._normalize_strategy_name(strat)
            result = None

            if s == "collar":
                capital = options_allocation.get("collar", 0.0)
                if capital <= 0:
                    continue
                result = self.collar_agent.select_options(
                    ticker,
                    stock_position,
                    available_options,
                    capital,
                    market_analysis,
                    use_llm,
                )
            elif s == "straddle":
                capital = options_allocation.get("straddle", 0.0)
                if capital <= 0:
                    continue
                result = self.straddle_agent.select_options(
                    ticker,
                    stock_position,
                    available_options,
                    capital,
                    market_analysis,
                    use_llm,
                )
            elif s in ("deltaneutral", "delta_neutral"):
                capital = options_allocation.get("delta_neutral", 0.0)
                if capital <= 0:
                    continue
                result = self.delta_neutral_agent.select_options(
                    ticker,
                    stock_position,
                    available_options,
                    capital,
                    market_analysis,
                    use_llm,
                )
            else:
                print(f"Strategy '{strat}' is not supported; skipping.")

            if result is not None:
                combined_option_positions.extend(result.option_positions)
                total_estimated_cost += result.estimated_cost
                rationales.append(f"{strat.capitalize()}: {result.rationale}")
                estimated_protections.append(result.estimated_protection)
                confidences.append(result.confidence)

        combined_rationale = " | ".join(rationales)
        avg_estimated_protection = (
            sum(estimated_protections) / len(estimated_protections)
            if estimated_protections
            else 0.0
        )
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return HedgingStrategy(
            asset=ticker,
            strategy_type="combined",
            rationale=combined_rationale,
            option_positions=combined_option_positions,
            estimated_cost=total_estimated_cost,
            estimated_protection=avg_estimated_protection,
            confidence=avg_confidence,
        )

    def bullish_options(
        self, available_options: List[Dict], current_price: float
    ) -> List[Dict]:
        """
        Pre‑filter available options for a bullish/momentum strategy,
        but return a mix across several strike and DTE buckets.

        - Only calls
        - Strike bands: 10–20%, 20–30%, 30–40% OTM
        - DTE bands: 60–75 days, 76–90 days
        - Within each bucket, pick the single contract with the lowest IV
        """
        # 1) only calls
        calls = [opt for opt in available_options if opt["option_type"] == 1]

        # 2) define strike bands as % of spot
        strike_bands = [
            (1.10, 1.20),
            (1.20, 1.30),
            (1.30, 1.40),
        ]

        # 3) define DTE bands
        dte_bands = [
            (60, 75),
            (76, 90),
        ]

        selected = []
        # 4) scan buckets
        for st_min, st_max in strike_bands:
            for dte_min, dte_max in dte_bands:
                bucket = [
                    opt
                    for opt in calls
                    if (st_min * current_price)
                    <= opt["strike"]
                    <= (st_max * current_price)
                    and dte_min <= opt.get("dte", 0) <= dte_max
                ]
                if not bucket:
                    continue
                # pick the lowest-IV contract from this bucket
                bucket_sorted = sorted(bucket, key=lambda o: o.get("iv", float("inf")))
                selected.extend(bucket_sorted[:2])

        # 5) if no buckets yielded anything, fall back to original broad filter
        if not selected:
            # 60–90 DTE, 10–40% OTM sorted by IV
            lower, upper = current_price * 1.10, current_price * 1.40
            fallback = [
                opt
                for opt in calls
                if 60 <= opt.get("dte", 0) <= 90 and lower <= opt["strike"] <= upper
            ]
            fallback.sort(key=lambda o: o.get("iv", float("inf")))
            # take the cheapest 3
            selected = fallback[:3]

        return selected
