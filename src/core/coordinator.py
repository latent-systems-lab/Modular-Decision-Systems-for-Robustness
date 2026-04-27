"""
Coordinator for the portfolio risk-management simulation system.
Orchestrates the entire simulation process and manages interactions between components.
Includes integration with memory-based reinforcement learning.
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Import models and utilities
import os
from src.utils.llm_utils import build_combined_allocation_split_prompt, get_llm_response
from src.models.pydantic_models import (
    AssetSplit,
    CombinedAllocationSplitResponse,
    SimulationConfig,
    Portfolio,
    MarketRegime,
    CapitalAllocation,
    PortfolioAllocation,
    SimulationResult,
    HedgingStrategy,
)
from src.core.data_manager import DataManager
from src.core.market_analyzer import MarketAnalyzer
from src.core.portfolio_manager import PortfolioManager
from src.core.asset_manager import AssetManager
from src.core.performance_tracker import PerformanceTracker

# Import memory-based RL components
from src.memory_rl.memory_rl_integration import MemoryRLIntegration
from src.memory_rl.llm_client import LLMClient


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Coordinator:
    """
    Coordinates the simulation process and manages interactions between components.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the Coordinator.

        Args:
            config: SimulationConfig object containing simulation parameters.
        """
        self.config = config
        self.data_dir = "data"

        # Initialize components
        self.data_manager = DataManager(self.data_dir)
        self.market_analyzer = MarketAnalyzer(self.data_manager)
        self.portfolio_manager = PortfolioManager(
            self.data_manager, config.initial_capital
        )
        self.asset_manager = AssetManager(self.data_manager)
        self.performance_tracker = PerformanceTracker(
            config.initial_capital, config.start_date
        )
        self.market_analysis = None

        # Initialize memory-based RL if enabled
        self.memory_rl = None
        if hasattr(config, "use_memory_rl") and config.use_memory_rl:
            # Create memory directory if it doesn't exist
            memory_dir = "memory"
            os.makedirs(memory_dir, exist_ok=True)

            # Initialize LLM client
            llm_client = self._get_llm_client()

            # Initialize memory-based RL integration
            self.memory_rl = MemoryRLIntegration(
                llm_client=llm_client,
                data_manager=self.data_manager,
                config=config.memory_rl_config,
            )
            logger.info(
                "Memory-based RL initialized with config: %s", config.memory_rl_config
            )

        # Simulation state
        self.current_date = config.start_date
        self.last_rebalance_date = None
        self.last_rebalance_value = None
        self.emergency_rebalance_triggered = False
        self.emergency_rebalance_drawndown = None

    def _get_llm_client(self) -> Any:
        """
        Get LLM client for memory-based RL.

        Returns:
            LLM client instance.
        """
        return LLMClient()

    def run_simulation(self) -> SimulationResult:
        """
        Run the complete simulation from start_date to end_date.

        Returns:
            SimulationResult object containing simulation results.
        """
        logger.info(
            "Starting simulation from %s to %s",
            self.config.start_date,
            self.config.end_date,
        )

        # Initialize simulation
        self.current_date = self.config.start_date
        self.last_rebalance_date = None

        # Main simulation loop
        while self.current_date <= self.config.end_date:

            first_asset = self.config.assets[0] if self.config.assets else None
            if first_asset:
                has_stock_data = self.data_manager.check_stock_data_exists(
                    ticker=first_asset, target_date=self.current_date
                )

                if not has_stock_data:
                    logger.info(
                        "Skipping %s - no stock data for %s",
                        self.current_date,
                        first_asset,
                    )
                    self.current_date += timedelta(days=1)
                    continue

            logger.info("Processing date: %s", self.current_date)

            # Run single day simulation
            self.run_single_day(self.current_date)

            # Move to next day
            self.current_date += timedelta(days=1)

        # Generate final portfolio snapshot
        final_portfolio = self.portfolio_manager.get_portfolio_snapshot(
            self.config.end_date
        )
        self.portfolio_manager.sell_all_option_positions(self.current_date)
        self.portfolio_manager.sell_all_stocks_position(self.current_date)
        logger.info(
            "All positions sold. Updated cash: $%.2f",
            self.portfolio_manager.cash,
        )
        pnl_breakdown = self.portfolio_manager.get_profit_loss_breakdown(True, True)

        # Generate simulation result
        result = self.performance_tracker.generate_simulation_result(
            self.config, final_portfolio, self
        )
        self.performance_tracker.convert_jsonl_to_json()
        # Save memory-based RL experiences if enabled
        if self.memory_rl:
            self.memory_rl.save_memory()
            logger.info("Memory-based RL experiences saved.")

        logger.info("Simulation completed successfully.")
        return result

    def run_single_day(self, current_date: date) -> None:
        """
        Run simulation for a single day.

        Args:
            current_date: Date to simulate.
        """
        # Update current date
        self.current_date = current_date

        # Check if rebalancing is needed
        should_rebalance = self._should_rebalance()
        portfolio = None

        if should_rebalance:
            logger.info("Rebalancing portfolio on %s", current_date)
            try:
                # Store pre-rebalance portfolio for memory-based RL
                pre_rebalance_portfolio = self.portfolio_manager.get_portfolio_snapshot(
                    current_date
                )

                # Perform rebalance
                self._perform_rebalance()
                self.last_rebalance_date = current_date

                # Only update last_rebalance_value here, after actual rebalance
                portfolio = self.portfolio_manager.get_portfolio_snapshot(current_date)
                self.last_rebalance_value = portfolio.total_value

                # Update memory-based RL with rebalance results if enabled
                if (
                    self.memory_rl
                    and hasattr(self.config, "use_memory_rl")
                    and self.config.use_memory_rl
                ):
                    self._update_memory_rl_experiences(
                        current_date, pre_rebalance_portfolio, portfolio
                    )

                self.emergency_rebalance_triggered = False
                self.emergency_rebalance_drawndown = None
            except Exception as e:
                logger.error(
                    "Error performing rebalance on %s: %s", current_date, str(e)
                )
        else:
            # Just update positions with current prices
            self.portfolio_manager.update_positions(current_date)
            # Get portfolio snapshot
            portfolio = self.portfolio_manager.get_portfolio_snapshot(current_date)

        if portfolio is None:
            logger.warning("Skipping %s - no portfolio data available", current_date)
            portfolio = self.portfolio_manager.get_portfolio_snapshot(current_date)

        # Update performance metrics
        pnl_breakdown = self.portfolio_manager.get_profit_loss_breakdown(False, True)
        self.performance_tracker.update(portfolio, pnl_breakdown)
        self.performance_tracker.append_portfolio_snapshot(portfolio)
        logger.info(
            "Portfolio snapshot | date=%s | total_value=%.2f",
            current_date,
            portfolio.total_value,
        )

        # Check for emergency rebalance trigger
        self._check_emergency_trigger(portfolio)

    def _should_rebalance(self) -> bool:
        """
        Determine if portfolio rebalancing is needed.

        Returns:
            True if rebalancing is needed, False otherwise.
        """
        # Always rebalance on first day
        if self.last_rebalance_date is None:
            return True

        # Rebalance if emergency trigger was activated
        if self.emergency_rebalance_triggered:
            return True

        # Rebalance based on configured frequency
        days_since_last_rebalance = (self.current_date - self.last_rebalance_date).days
        if days_since_last_rebalance >= self.config.rebalance_frequency:
            return True

        return False

    def _perform_rebalance(self) -> None:
        """
        Perform portfolio rebalancing with optimized API calls.
        """
        # Sell all existing option positions first
        self.portfolio_manager.sell_all_option_positions(self.current_date)
        self.portfolio_manager.sell_all_stocks_position(self.current_date)
        logger.info(
            "All positions sold. Updated cash: $%.2f",
            self.portfolio_manager.cash,
        )
        pnl_breakdown = self.portfolio_manager.get_profit_loss_breakdown()

        # Step 1: Analyze market conditions (using batch analysis)
        market_analysis = self.market_analyzer.analyze_assets_batch(
            self.config.assets, self.current_date, use_llm=self.config.use_llm
        )
        self.market_analysis = market_analysis

        # Step 2 & 3 Combined: Get allocation and split
        if (
            self.memory_rl
            and hasattr(self.config, "use_memory_rl")
            and self.config.use_memory_rl
        ):
            # Use memory-based RL for allocation and split
            allocation, asset_splits = self._get_memory_rl_allocation_and_split(
                self.config.assets, market_analysis, self.current_date
            )
        else:
            # Use existing LLM approach
            allocation, asset_splits = self.allocate_and_split_combined(
                self.config.assets, market_analysis, self.current_date
            )

        # Create stock-only allocations based on the splits
        stock_allocations = []
        for asset_allocation in allocation.allocations:
            ticker = asset_allocation.asset
            if asset_allocation.amount <= 0:
                continue

            # Get the corresponding split info
            asset_split = asset_splits.get(ticker)
            if not asset_split:
                continue

            # Create stock allocation
            stock_allocation = CapitalAllocation(
                asset=ticker,
                amount=asset_split.stock_amount,
                percentage=asset_allocation.percentage * (asset_split.stock_percentage),
                action=asset_split.stocks_action,
            )
            stock_allocations.append(stock_allocation)

        # Create portfolio allocation for stock trades
        stock_portfolio_allocation = PortfolioAllocation(
            total_capital=sum(a.amount for a in stock_allocations),
            allocations=stock_allocations,
            allocation_date=self.current_date,
            cash_reserve=0.0,
            rationale="Stock-only allocation based on combined market analysis and asset split.",
        )

        # Execute stock trades
        executed_trades = self.portfolio_manager.execute_trades(
            stock_portfolio_allocation, self.current_date
        )

        # Apply hedging strategies for options portion
        rule_based_split = getattr(self.config, "rule_based_strategy_split", False)
        num_strategies = len(self.config.hedging_strategies)

        for ticker, asset_split in asset_splits.items():
            # Skip if options allocation is too small or no stock position exists
            if (
                asset_split.options_amount <= 1000
                or ticker not in self.portfolio_manager.stock_positions
            ):
                continue

            if rule_based_split:
                hedging_strategy = (
                    self.asset_manager.select_rule_based_hedging_strategy(
                        ticker=ticker,
                        stock_position=self.portfolio_manager.stock_positions[
                            ticker
                        ].dict(),
                        options_allocation=asset_split.options_amount,
                        market_analysis=market_analysis[ticker],
                        current_date=self.current_date,
                        strategies=self.config.hedging_strategies,
                        use_llm=self.config.use_llm,
                    )
                )
            else:
                # Build options_allocations using the strategies defined in config.
                options_allocations = {}
                for strat in self.config.hedging_strategies:
                    normalized = strat.lower().replace("-", "_").replace(" ", "_")
                    if isinstance(asset_split.hedging_strategies, dict):
                        factor = asset_split.hedging_strategies.get(normalized, 0.0)
                    else:
                        # Otherwise assume it's an object with attributes
                        factor = getattr(
                            asset_split.hedging_strategies, normalized, 0.0
                        )
                    if factor == 0.0 and num_strategies == 1:
                        factor = 1.0
                    options_allocations[normalized] = (
                        asset_split.options_amount * factor
                    )

                # Use memory-based RL for hedging strategies if enabled
                if (
                    self.memory_rl
                    and hasattr(self.config, "use_memory_rl")
                    and self.config.use_memory_rl
                ):
                    hedging_strategy = self._get_memory_rl_hedging_strategy(
                        ticker,
                        self.portfolio_manager.stock_positions[ticker].dict(),
                        options_allocations,
                        market_analysis[ticker],
                        self.current_date,
                    )
                else:
                    # Use existing approach
                    try:
                        hedging_strategy = (
                            self.asset_manager.select_all_hedging_strategies(
                                ticker,
                                self.portfolio_manager.stock_positions[ticker].dict(),
                                options_allocations,
                                market_analysis[ticker],
                                self.current_date,
                                strategy_type=self.config.hedging_strategies,
                                use_llm=self.config.use_llm,
                            )
                        )
                    except Exception as e:
                        logger.error(
                            "Error selecting hedging strategy for %s: %s", ticker, str(e)
                        )
                        continue

            logger.info(
                "Options for %s: allocated $%.2f, using $%.2f",
                ticker,
                asset_split.options_amount,
                sum(op.paid_amount for op in hedging_strategy.option_positions),
            )

            # Add option positions to portfolio
            for option_position in hedging_strategy.option_positions:
                try:
                    self.portfolio_manager.execute_option_trade(
                        option_position, self.current_date
                    )
                except Exception as e:
                    logger.error(
                        "Error executing option trade for %s: %s",
                        option_position.asset,
                        str(e),
                    )

            used_options_capital = sum(
                op.paid_amount for op in hedging_strategy.option_positions
            )

    def _get_memory_rl_allocation_and_split(
        self,
        assets: List[str],
        market_analysis: Dict[str, MarketRegime],
        current_date: date,
    ) -> Tuple[PortfolioAllocation, Dict[str, AssetSplit]]:
        """
        Get portfolio allocation and asset splits using memory-based RL.

        Args:
            assets: List of assets.
            market_analysis: Market analysis for each asset.
            current_date: Current date.

        Returns:
            Tuple of (portfolio allocation, asset splits dictionary).
        """
        # Prepare current positions data
        current_positions = {}
        for ticker, position in self.portfolio_manager.stock_positions.items():
            current_positions[ticker] = {
                "shares": position.shares,
                "average_price": position.average_price,
                "market_value": position.market_value,
                "current_price": position.current_price,
            }

        # Get portfolio allocation using memory-based RL
        portfolio_allocation, decision_details = (
            self.memory_rl.get_portfolio_allocation(
                current_date=current_date,
                portfolio_value=self.portfolio_manager.cash,
                cash=self.portfolio_manager.cash,
                assets=assets,
                market_analysis=market_analysis,
                current_positions=current_positions,
                emergency=self.emergency_rebalance_drawndown,
            )
        )

        # Get asset splits for each allocated asset
        asset_splits = {}
        for allocation in portfolio_allocation.allocations:
            ticker = allocation.asset
            if allocation.amount <= 0:
                continue

            # Get asset split using memory-based RL
            asset_split, split_details = self.memory_rl.get_asset_allocation(
                current_date=current_date,
                asset=ticker,
                allocation_amount=allocation.amount,
                market_analysis=market_analysis.get(ticker, {}),
                current_price=self.data_manager.get_stock_price(ticker, current_date)
                or 0.0,
                volatility=self._get_asset_volatility(ticker, current_date),
                hedging_strategies=self.config.hedging_strategies,
            )

            asset_splits[ticker] = asset_split

        return portfolio_allocation, asset_splits

    def _get_memory_rl_hedging_strategy(
        self,
        asset: str,
        stock_position: Dict[str, Any],
        options_allocations: Dict[str, float],
        market_analysis: Dict[str, Any],
        current_date: date,
    ) -> HedgingStrategy:
        """
        Get hedging strategy using memory-based RL.

        Args:
            asset: Asset ticker.
            stock_position: Current stock position.
            options_allocations: Options allocations by strategy type.
            market_analysis: Market analysis for the asset.
            current_date: Current date.

        Returns:
            Combined hedging strategy (of type HedgingStrategy).
        """
        # Get available options for the asset
        available_options = self.data_manager.load_options_data(
            asset, current_date, min_dte=30, max_dte=92
        )
        available_options = [option.model_dump() for option in available_options]

        combined_option_positions = []
        total_estimated_cost = 0.0
        rationales = []
        estimated_protections = []
        confidences = []

        # Iterate over each strategy type and its allocated capital
        for strategy_type, allocation in options_allocations.items():
            if allocation <= 0:
                continue

            # Pre-filter available options using strategy-specific criteria.
            normalized_strategy = (
                strategy_type.lower().replace(" ", "").replace("-", "_")
            )
            if normalized_strategy in {"rulesonly", "rules_only"}:
                result = self.asset_manager.select_rule_based_hedging_strategy(
                    ticker=asset,
                    stock_position=stock_position,
                    options_allocation=allocation,
                    market_analysis=market_analysis,
                    current_date=current_date,
                    available_options=available_options,
                    strategies=self.config.hedging_strategies,
                    use_llm=self.config.use_llm,
                )
                if result is not None:
                    combined_option_positions.extend(result.option_positions)
                    total_estimated_cost += result.estimated_cost
                    rationales.append(f"Rule-based: {result.rationale}")
                    estimated_protections.append(result.estimated_protection)
                    confidences.append(result.confidence)
                continue
            if normalized_strategy == "collar":
                # Use the CollarAgent prefiltering method
                prefiltered_options = (
                    self.asset_manager.collar_agent.prefilter_collar_options(
                        available_options, stock_position.get("current_price", 100)
                    )
                )
            elif normalized_strategy == "delta_neutral":
                # Use the DeltaNeutralAgent prefiltering method
                prefiltered_options = self.asset_manager.delta_neutral_agent.prefilter_delta_neutral_options(
                    available_options
                )
            elif normalized_strategy == "straddle":
                # Use the StraddleAgent prefiltering method
                prefiltered_options = (
                    self.asset_manager.straddle_agent.prefilter_straddle_options(
                        available_options, stock_position.get("current_price", 100)
                    )
                )
            else:
                logger.warning("Strategy '%s' is not supported; skipping.", strategy_type)
                continue
            # Get hedging strategy using memory-based RL; assume it returns a tuple:
            # (result, strategy_details), where result has a compatible structure (e.g. a HedgingStrategy)
            if prefiltered_options == []:
                continue
            result, strategy_details = self.memory_rl.get_hedging_strategy(
                current_date=current_date,
                asset=asset,
                strategy_type=strategy_type,
                stock_position=stock_position,
                available_options=prefiltered_options,
                available_capital=allocation,
                market_analysis=market_analysis.model_dump(),
            )

            if result is not None:
                combined_option_positions.extend(result.option_positions)
                total_estimated_cost += result.estimated_cost
                rationales.append(f"{strategy_type.capitalize()}: {result.rationale}")
                estimated_protections.append(result.estimated_protection)
                confidences.append(result.confidence)

        combined_rationale = " | ".join(rationales)
        avg_estimated_protection = (
            sum(estimated_protections) / len(estimated_protections)
            if estimated_protections
            else 0.0
        )
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Now aggregate into a single HedgingStrategy output.
        return HedgingStrategy(
            asset=asset,
            strategy_type="combined",
            rationale=combined_rationale,
            option_positions=combined_option_positions,
            estimated_cost=total_estimated_cost,
            estimated_protection=avg_estimated_protection,
            confidence=avg_confidence,
        )

    def _get_asset_volatility(self, ticker: str, current_date: date) -> float:
        """
        Get asset volatility.

        Args:
            ticker: Asset ticker.
            current_date: Current date.

        Returns:
            Asset volatility.
        """
        # Get historical prices for the last 30 days
        start_date = current_date - timedelta(days=30)
        prices = self.data_manager.load_stock_data(ticker, start_date, current_date)
        prices = [price.model_dump() for price in prices]
        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1]["close"] > 0:
                returns.append(
                    (prices[i]["close"] - prices[i - 1]["close"])
                    / prices[i - 1]["close"]
                )
        # Calculate volatility (standard deviation of returns)
        if returns:
            import numpy as np

            volatility = np.std(returns)
            return volatility

        return 0.02  # Default volatility if no data

    def _update_memory_rl_experiences(
        self,
        current_date: date,
        pre_rebalance_portfolio: Portfolio,
        post_rebalance_portfolio: Portfolio,
    ) -> None:
        """
        Update memory-based RL with rebalance results.

        Args:
            current_date: Current date.
            pre_rebalance_portfolio: Portfolio before rebalancing.
            post_rebalance_portfolio: Portfolio after rebalancing.
        """
        # Skip if memory-based RL is not enabled
        if not self.memory_rl:
            return

        market_analysis = self.market_analysis
        # Determine market regime (simplified - use most common regime)
        regimes = [
            analysis.model_dump().get("regime", "neutral")
            for analysis in self.market_analysis.values()
        ]
        market_regime = max(set(regimes), key=regimes.count) if regimes else "neutral"

        # Calculate portfolio metrics
        portfolio_return = 0.0
        if pre_rebalance_portfolio and pre_rebalance_portfolio.total_value > 0:
            portfolio_return = (
                post_rebalance_portfolio.total_value
                - pre_rebalance_portfolio.total_value
            ) / pre_rebalance_portfolio.total_value

        # Get benchmark return (simplified)
        benchmark_return = 0.0  # Replace with actual benchmark calculation

        # Calculate portfolio volatility (simplified)
        portfolio_volatility = self._get_portfolio_volatility(current_date)

        # Estimate transaction costs: slippage from turnover + option trading costs
        turnover = self._calculate_portfolio_turnover(
            pre_rebalance_portfolio, post_rebalance_portfolio
        )
        slippage_cost = (
            turnover
            * 0.0002  # 2 bps per unit turnover
            * post_rebalance_portfolio.total_value
        )

        option_turnover_value = self._calculate_option_turnover_value(
            pre_rebalance_portfolio, post_rebalance_portfolio
        )
        option_transaction_cost = (
            option_turnover_value * 0.002
        )  # 20 bps on option turnover

        transaction_costs = slippage_cost + option_transaction_cost

        # Update portfolio results
        self.memory_rl.update_portfolio_results(
            date_val=current_date,
            assets=self.config.assets,
            state={
                "portfolio_value": pre_rebalance_portfolio.total_value,
                "cash": pre_rebalance_portfolio.cash,
                "market_regimes": {
                    asset: analysis.model_dump().get("regime", "neutral")
                    for asset, analysis in market_analysis.items()
                },
            },
            action={
                "allocations": [
                    {
                        "asset": pos.asset,
                        "amount": pos.market_value,
                        "percentage": pos.market_value
                        / post_rebalance_portfolio.total_value,
                    }
                    for pos in post_rebalance_portfolio.stock_positions
                ],
                "cash_reserve": post_rebalance_portfolio.cash,
            },
            previous_value=pre_rebalance_portfolio.total_value,
            current_value=post_rebalance_portfolio.total_value,
            running_peak=max(
                self.performance_tracker.peak_portfolio_value,
                pre_rebalance_portfolio.total_value,
            ),
            realized_variance=portfolio_volatility**2,
            benchmark_return=benchmark_return,
            transaction_costs=transaction_costs,
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            market_regime=market_regime,
        )

        # Update asset results for each asset
        for asset in self.config.assets:
            # Get asset positions
            pre_position = next(
                (
                    pos
                    for pos in pre_rebalance_portfolio.stock_positions
                    if pos.asset == asset
                ),
                None,
            )
            post_position = next(
                (
                    pos
                    for pos in post_rebalance_portfolio.stock_positions
                    if pos.asset == asset
                ),
                None,
            )

            if not post_position:
                continue

            # Calculate asset metrics
            stock_return = 0.0
            if pre_position and pre_position.market_value > 0:
                stock_return = (
                    post_position.market_value - pre_position.market_value
                ) / pre_position.market_value

            # Get options for this asset
            pre_options = [
                pos
                for pos in pre_rebalance_portfolio.option_positions
                if pos.asset == asset
            ]
            post_options = [
                pos
                for pos in post_rebalance_portfolio.option_positions
                if pos.asset == asset
            ]

            # Calculate options metrics
            options_return = 0.0
            pre_options_value = sum(pos.market_value for pos in pre_options)
            post_options_value = sum(pos.market_value for pos in post_options)
            if pre_options_value > 0:
                options_return = (
                    post_options_value - pre_options_value
                ) / pre_options_value

            # Calculate asset volatility
            asset_volatility = self._get_asset_volatility(asset, current_date)

            # Calculate stock and options percentages
            total_value = post_position.market_value + post_options_value
            stock_percentage = (
                post_position.market_value / total_value if total_value > 0 else 1.0
            )
            options_percentage = (
                post_options_value / total_value if total_value > 0 else 0.0
            )

            # Update asset results
            self.memory_rl.update_asset_results(
                date_val=current_date,
                asset=asset,
                state={
                    "allocation_amount": total_value,
                    "current_price": post_position.current_price,
                    "volatility": asset_volatility,
                    "market_regime": market_analysis.get(asset, {})
                    .model_dump()
                    .get("regime", "neutral"),
                },
                action={
                    "stock_percentage": stock_percentage,
                    "options_percentage": options_percentage,
                    "stock_amount": post_position.market_value,
                    "options_amount": post_options_value,
                },
                stock_return=stock_return,
                options_return=options_return,
                asset_volatility=asset_volatility,
                stock_percentage=stock_percentage,
                options_percentage=options_percentage,
                market_regime=market_analysis.get(asset, {})
                .model_dump()
                .get("regime", "neutral"),
            )

            # Update hedging results for each strategy type
            strategy_types = set(pos.strategy for pos in post_options if pos.strategy)
            for strategy_type in strategy_types:
                # Get options for this strategy
                strategy_options = [
                    pos for pos in post_options if pos.strategy == strategy_type
                ]

                if not strategy_options:
                    continue

                # Calculate strategy metrics
                options_value = sum(pos.market_value for pos in strategy_options)

                # Estimate option-specific costs (20 bps of turnover for the strategy)
                pre_strategy_options = [
                    pos for pos in pre_options if pos.strategy == strategy_type
                ]
                pre_strategy_value = sum(
                    pos.market_value for pos in pre_strategy_options
                )
                option_turnover_value = abs(options_value - pre_strategy_value)
                asset_total_value = post_position.market_value + post_options_value
                option_costs = 0.0
                if asset_total_value > 0:
                    option_costs = (0.002 * option_turnover_value) / asset_total_value

                # Calculate volatility metrics (simplified)
                combined_volatility = (
                    asset_volatility * 0.8
                )  # Simplified - assume 20% reduction
                stock_only_volatility = asset_volatility

                # Update hedging results
                self.memory_rl.update_hedging_results(
                    date_val=current_date,
                    asset=asset,
                    strategy_type=strategy_type,
                    state={
                        "stock_position": post_position.dict(),
                        "available_capital": options_value,
                        "market_regime": market_analysis.get(asset, {})
                        .model_dump()
                        .get("regime", "neutral"),
                    },
                    action={
                        "option_positions": [pos.dict() for pos in strategy_options]
                    },
                    stock_return=stock_return,
                    options_return=options_return,
                    combined_volatility=combined_volatility,
                    stock_only_volatility=stock_only_volatility,
                    option_costs=option_costs,
                    market_regime=market_analysis.get(asset, {})
                    .model_dump()
                    .get("regime", "neutral"),
                )

    def _get_portfolio_volatility(self, current_date: date) -> float:
        """
        Calculate portfolio volatility.

        Args:
            current_date: Current date.

        Returns:
            Portfolio volatility.
        """
        # Get daily returns for the last 30 days
        returns = []
        for i in range(min(30, len(self.performance_tracker.daily_performances))):
            if i > 0:
                prev_value = self.performance_tracker.daily_performances[
                    -(i + 1)
                ].portfolio_value
                curr_value = self.performance_tracker.daily_performances[
                    -i
                ].portfolio_value
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)

        # Calculate volatility (standard deviation of returns)
        if returns:
            import numpy as np

            volatility = np.std(returns)
            return volatility

        return 0.01  # Default volatility if no data

    def _calculate_portfolio_weights(
        self, portfolio: Optional[Portfolio]
    ) -> Dict[str, float]:
        """Calculate asset weights (including cash) for a portfolio snapshot."""

        if not portfolio or portfolio.total_value <= 0:
            return {}

        weights: Dict[str, float] = {}
        total_value = portfolio.total_value

        for position in portfolio.stock_positions:
            weights[position.asset] = weights.get(position.asset, 0.0) + (
                position.market_value / total_value
            )

        for position in portfolio.option_positions:
            weights[position.asset] = weights.get(position.asset, 0.0) + (
                position.market_value / total_value
            )

        if portfolio.cash > 0:
            weights["cash"] = portfolio.cash / total_value

        return weights

    def _calculate_portfolio_turnover(
        self,
        previous_portfolio: Optional[Portfolio],
        current_portfolio: Optional[Portfolio],
    ) -> float:
        """Estimate turnover between two portfolio states based on weight changes."""

        previous_weights = self._calculate_portfolio_weights(previous_portfolio)
        current_weights = self._calculate_portfolio_weights(current_portfolio)

        if not previous_weights and not current_weights:
            return 0.0

        assets = set(previous_weights) | set(current_weights)
        turnover = 0.0
        for asset in assets:
            turnover += abs(
                current_weights.get(asset, 0.0) - previous_weights.get(asset, 0.0)
            )

        return turnover

    def _calculate_option_turnover_value(
        self,
        previous_portfolio: Optional[Portfolio],
        current_portfolio: Optional[Portfolio],
    ) -> float:
        """Calculate total change in option market value between two portfolio states."""

        if not previous_portfolio and not current_portfolio:
            return 0.0

        def _aggregate_option_values(
            portfolio: Optional[Portfolio],
        ) -> Dict[str, float]:
            values: Dict[str, float] = {}
            if not portfolio:
                return values

            for position in portfolio.option_positions:
                values[position.asset] = (
                    values.get(position.asset, 0.0) + position.market_value
                )

            return values

        previous_values = _aggregate_option_values(previous_portfolio)
        current_values = _aggregate_option_values(current_portfolio)

        assets = set(previous_values) | set(current_values)
        turnover_value = 0.0
        for asset in assets:
            turnover_value += abs(
                current_values.get(asset, 0.0) - previous_values.get(asset, 0.0)
            )

        return turnover_value

    def _check_emergency_trigger(self, portfolio: Portfolio) -> None:
        """
        Check if emergency rebalancing should be triggered based on drawdown from last rebalance.

        Args:
            portfolio: Current portfolio state.
        """
        # Use last_rebalance_value instead of overall peak value
        if self.last_rebalance_value is None:
            return

        current_drawdown = (
            self.last_rebalance_value - portfolio.total_value
        ) / self.last_rebalance_value
        logger.info(
            "Drawdown check | last_rebalance_value=%.2f | current_value=%.2f | drawdown=%.4f",
            self.last_rebalance_value,
            portfolio.total_value,
            current_drawdown,
        )
        if current_drawdown > self.config.emergency_threshold:
            logger.warning(
                "Emergency rebalance triggered: Drawdown of %.2f%% from last rebalance exceeds threshold of %.2f%%",
                current_drawdown * 100,
                self.config.emergency_threshold * 100,
            )
            self.emergency_rebalance_triggered = True
            self.emergency_rebalance_drawndown = current_drawdown

    def allocate_and_split_combined(
        self,
        assets: List[str],
        market_analysis: Dict[str, MarketRegime],
        current_date: date,
    ) -> Tuple[PortfolioAllocation, Dict[str, AssetSplit]]:
        """Perform capital allocation and stock/options split in a single API call."""
        # Prepare current positions data
        current_positions = {}
        for ticker, position in self.portfolio_manager.stock_positions.items():
            current_positions[ticker] = {
                "shares": position.shares,
                "average_price": position.average_price,
                "market_value": position.market_value,
            }

        # Build combined prompt
        prompt = build_combined_allocation_split_prompt(
            available_capital=self.portfolio_manager.cash,
            assets=assets,
            market_analysis=market_analysis,
            current_positions=current_positions,
        )

        # Get response
        response = get_llm_response(prompt, CombinedAllocationSplitResponse)

        # Process response
        if isinstance(response, CombinedAllocationSplitResponse):
            # Create portfolio allocation object
            portfolio_allocation = PortfolioAllocation(
                total_capital=self.portfolio_manager.cash,
                allocations=response.allocations,
                cash_reserve=response.cash_reserve,
                rationale=response.rationale,
            )

            # Create asset split objects
            asset_splits = {}
            for split_response in response.asset_splits:
                # Find allocation amount for this asset
                ticker = split_response.asset
                allocation_amount = next(
                    (a.amount for a in response.allocations if a.asset == ticker), 0
                )

                asset_splits[ticker] = AssetSplit(
                    asset=ticker,
                    total_allocation=allocation_amount,
                    stocks_action=split_response.stocks_action,
                    stock_percentage=split_response.stock_percentage,
                    options_percentage=split_response.options_percentage,
                    stock_amount=allocation_amount * split_response.stock_percentage,
                    options_amount=allocation_amount
                    * split_response.options_percentage,
                    rationale=split_response.rationale,
                    hedging_strategies=split_response.hedging_strategies,
                )

            return portfolio_allocation, asset_splits
        else:
            # Fall back to separate approaches if combined call fails
            logger.warning(
                "Combined allocation and split failed, falling back to separate calls"
            )
            # [Fall back implementation here]

            # For now, return empty results
            return (
                PortfolioAllocation(
                    total_capital=self.portfolio_manager.cash,
                    allocations=[],
                    cash_reserve=self.portfolio_manager.cash,
                    rationale="Fallback allocation due to error.",
                ),
                {},
            )

    @property
    def daily_performances(self) -> List:
        """
        Get daily performance history.

        Returns:
            List of DailyPerformance objects.
        """
        return self.performance_tracker.daily_performances

    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get performance summary.

        Returns:
            Dictionary of performance metrics.
        """
        return self.performance_tracker.get_performance_summary()

    def get_performance_report(self) -> str:
        """
        Get performance report.

        Returns:
            Performance report string.
        """
        return self.performance_tracker.report()

    def plot_performance(self, output_file: str = None) -> None:
        """
        Plot performance metrics.

        Args:
            output_file: Optional file path to save the plot.
        """
        self.performance_tracker.plot_performance(output_file)
