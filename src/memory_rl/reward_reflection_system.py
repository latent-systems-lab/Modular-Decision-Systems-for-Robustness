"""
Reward and Reflection System for memory-based reinforcement learning.
Calculates rewards for decisions and generates reflections on experiences.
"""

import os
import json
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union, Literal, Tuple, Any
import logging
from pydantic import BaseModel, Field

from .experience_memory_store import (
    Experience, PortfolioExperience, AssetExperience, HedgingExperience,
    ExperienceMemoryStore
)
import re
from .prompt_templates import PromptTemplates, PromptConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardConfig(BaseModel):
    """Configuration for reward calculation."""
    return_weight: float = 0.5
    risk_weight: float = 0.3
    benchmark_weight: float = 0.1
    cost_weight: float = 0.1
    use_long_term_rewards: bool = True
    use_what_if_comparison: bool = False
    time_horizon: int = 30  # Days for long-term reward calculation
    lambda_sigma: float = 0.5
    lambda_drawdown: float = 0.5
    reward_clip_range: Tuple[float, float] = (-1.0, 1.0)

class ReflectionConfig(BaseModel):
    reflection: str

class RewardCalculator:
    """
    Base class for reward calculators.
    """

    def __init__(self, config: RewardConfig = None):
        """
        Initialize reward calculator.

        Args:
            config: Configuration for reward calculation.
        """
        self.config = config or RewardConfig()

    def calculate_reward(
        self,
        return_component: float = 0.0,
        risk_component: float = 0.0,
        benchmark_component: Optional[float] = None,
        cost_component: float = 0.0,
        long_term_returns: Optional[Union[float, List[float], np.ndarray]] = None,
        adjustments: Optional[Union[float, List[float], np.ndarray]] = None,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        **_: Any,
    ) -> float:
        """Calculate a generic reward score using configured weights.

        The base implementation combines the supplied components using the
        weights from :class:`RewardConfig`.  Additional adjustments can be
        supplied, and optional long-term return information can be blended into
        the final score when enabled in the configuration.  Results are clipped
        to the provided ``clip_range`` to keep the reward bounded.

        Args:
            return_component: Contribution from realised returns.
            risk_component: Contribution from risk (typically a penalty).
            benchmark_component: Contribution from benchmark relative
                performance.  If omitted it defaults to ``0``.
            cost_component: Contribution from transaction or carry costs.
            long_term_returns: Optional collection (or single value) of
                long-term return metrics used when ``use_long_term_rewards`` is
                enabled.
            adjustments: Optional additional adjustments to add to the final
                reward (for example strategy-specific bonuses/penalties).
            clip_range: Lower and upper bound used to clip the final score.

        Returns:
            Reward value after combining all components.
        """
        try:
            benchmark_component = 0.0 if benchmark_component is None else benchmark_component

            reward = (
                self.config.return_weight * return_component
                + self.config.risk_weight * risk_component
                + self.config.benchmark_weight * benchmark_component
                + self.config.cost_weight * cost_component
            )

            if self.config.use_long_term_rewards and long_term_returns is not None:
                if isinstance(long_term_returns, (list, tuple, np.ndarray)):
                    if len(long_term_returns) > 0:
                        long_term_component = float(np.mean(long_term_returns))
                    else:
                        long_term_component = 0.0
                else:
                    long_term_component = float(long_term_returns)

                long_term_component = float(np.clip(long_term_component, clip_range[0], clip_range[1]))
                reward = (reward + long_term_component) / 2.0

            if adjustments is not None:
                if isinstance(adjustments, (list, tuple, np.ndarray)):
                    reward += float(np.sum(adjustments))
                else:
                    reward += float(adjustments)

            reward = float(np.clip(reward, clip_range[0], clip_range[1]))
            return reward
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            return 0.0

class PortfolioRewardCalculator(RewardCalculator):
    """
    Reward calculator for portfolio-level decisions.
    """

    def calculate_reward(
        self,
        previous_value: float,
        current_value: float,
        realized_variance: float,
        running_peak: float,
        transaction_costs: float = 0.0,
    ) -> float:
        """Calculate the risk-adjusted reward for a portfolio decision."""

        if previous_value <= 0:
            return 0.0

        net_current_value = max(0.0, current_value - transaction_costs)
        portfolio_return = (net_current_value - previous_value) / previous_value

        variance_penalty = self.config.lambda_sigma * max(realized_variance, 0.0)

        drawdown_penalty = 0.0
        if running_peak > 0:
            drawdown = max(0.0, (running_peak - net_current_value) / running_peak)
            drawdown_penalty = self.config.lambda_drawdown * drawdown

        reward = portfolio_return - variance_penalty - drawdown_penalty
        reward = float(np.clip(reward, *self.config.reward_clip_range))
        return reward

    def _calculate_turnover(
        self,
        previous_allocation: Dict[str, float],
        current_allocation: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio turnover.

        Args:
            previous_allocation: Previous asset allocation.
            current_allocation: Current asset allocation.

        Returns:
            Turnover value.
        """
        # Get all assets
        all_assets = set(previous_allocation.keys()) | set(current_allocation.keys())

        # Calculate absolute changes
        turnover = 0.0
        for asset in all_assets:
            prev_alloc = previous_allocation.get(asset, 0.0)
            curr_alloc = current_allocation.get(asset, 0.0)
            turnover += abs(curr_alloc - prev_alloc)

        return turnover

class AssetRewardCalculator(RewardCalculator):
    """
    Reward calculator for asset-level decisions.
    """

    def calculate_reward(
        self,
        stock_return: float,
        options_return: float,
        asset_volatility: float,
        stock_percentage: float,
        options_percentage: float,
        transaction_costs: float = 0.0
    ) -> float:
        """
        Calculate reward for an asset allocation decision.

        Args:
            stock_return: Stock return.
            options_return: Options return.
            asset_volatility: Asset volatility.
            stock_percentage: Percentage allocated to stock.
            options_percentage: Percentage allocated to options.
            transaction_costs: Transaction costs.

        Returns:
            Reward value.
        """
        # Calculate combined return
        combined_return = (stock_return * stock_percentage) + (options_return * options_percentage)
        net_return = combined_return - transaction_costs

        variance_penalty = self.config.lambda_sigma * max(asset_volatility, 0.0) ** 2
        reward = net_return - variance_penalty
        reward = float(np.clip(reward, *self.config.reward_clip_range))

        return reward

class HedgingRewardCalculator(RewardCalculator):
    """
    Reward calculator for hedging-level decisions.
    """

    def calculate_reward(
        self,
        stock_return: float,
        options_return: float,
        combined_volatility: float,
        stock_only_volatility: float,
        option_costs: float,
        strategy_type: str
    ) -> float:
        """
        Calculate reward for a hedging strategy decision.

        Args:
            stock_return: Stock return.
            options_return: Options return.
            combined_volatility: Volatility of combined position.
            stock_only_volatility: Volatility of stock-only position.
            option_costs: Option costs.
            strategy_type: Type of hedging strategy.

        Returns:
            Reward value.
        """
        # Calculate combined return and incorporate option costs as a drag
        combined_return = stock_return + options_return - option_costs

        variance_penalty = self.config.lambda_sigma * max(combined_volatility, 0.0) ** 2

        # Estimate drawdown proxy by comparing volatility with stock-only baseline
        drawdown_proxy = 0.0
        if stock_only_volatility > 0:
            volatility_ratio = combined_volatility / stock_only_volatility
            drawdown_proxy = max(0.0, volatility_ratio - 1.0)

        drawdown_penalty = self.config.lambda_drawdown * drawdown_proxy

        reward = combined_return - variance_penalty - drawdown_penalty
        reward = float(np.clip(reward, *self.config.reward_clip_range))

        return reward

    def _estimate_correlation(self, stock_return: float, options_return: float) -> float:
        """
        Estimate correlation between stock and options returns.
        This is a simplified version since we only have point values.

        Args:
            stock_return: Stock return.
            options_return: Options return.

        Returns:
            Estimated correlation.
        """
        # This is a very simplified approach
        # In practice, you'd use a time series of returns
        if stock_return == 0 or options_return == 0:
            return 0.0

        # If both returns have the same sign, positive correlation
        # If opposite signs, negative correlation
        if (stock_return > 0 and options_return > 0) or (stock_return < 0 and options_return < 0):
            return 0.8  # Arbitrary high positive correlation
        else:
            return -0.8  # Arbitrary high negative correlation

class WhatIfCalculator:
    """
    Calculator for what-if comparisons.
    """

    def __init__(self, data_manager: Any):
        """
        Initialize what-if calculator.

        Args:
            data_manager: Data manager for accessing market data.
        """
        self.data_manager = data_manager

    def calculate_portfolio_what_if(
        self,
        actual_allocation: Dict[str, float],
        alternative_allocations: List[Dict[str, float]],
        start_date: date,
        end_date: date,
        initial_portfolio_value: float
    ) -> Dict[str, Any]:
        """
        Calculate what-if scenarios for portfolio allocation.

        Args:
            actual_allocation: Actual asset allocation.
            alternative_allocations: Alternative asset allocations.
            start_date: Start date.
            end_date: End date.
            initial_portfolio_value: Initial portfolio value.

        Returns:
            What-if comparison results.
        """
        # Calculate actual performance
        actual_performance = self._calculate_allocation_performance(
            actual_allocation, start_date, end_date, initial_portfolio_value
        )

        # Calculate alternative performances
        alternative_performances = []
        for i, alt_allocation in enumerate(alternative_allocations):
            alt_performance = self._calculate_allocation_performance(
                alt_allocation, start_date, end_date, initial_portfolio_value
            )
            alternative_performances.append({
                "allocation_id": i,
                "allocation": alt_allocation,
                "performance": alt_performance
            })

        # Find best alternative
        best_alternative = max(
            alternative_performances,
            key=lambda x: x["performance"]["final_value"],
            default=None
        )

        # Calculate regret (difference between best alternative and actual)
        regret = 0.0
        if best_alternative:
            regret = (
                best_alternative["performance"]["final_value"] -
                actual_performance["final_value"]
            ) / initial_portfolio_value

        return {
            "actual_allocation": actual_allocation,
            "actual_performance": actual_performance,
            "alternative_performances": alternative_performances,
            "best_alternative": best_alternative,
            "regret": regret
        }

    def _calculate_allocation_performance(
        self,
        allocation: Dict[str, float],
        start_date: date,
        end_date: date,
        initial_value: float
    ) -> Dict[str, Any]:
        """
        Calculate performance of an allocation.

        Args:
            allocation: Asset allocation.
            start_date: Start date.
            end_date: End date.
            initial_value: Initial value.

        Returns:
            Performance metrics.
        """
        # This is a placeholder implementation
        # In practice, you'd use the data manager to get actual price data
        # and calculate the performance based on that

        # For now, we'll generate random returns
        days = (end_date - start_date).days
        daily_returns = np.random.normal(0.0005, 0.01, days)  # Mean 0.05% daily, 1% std

        # Calculate cumulative return
        cumulative_return = np.prod(1 + daily_returns) - 1

        # Calculate volatility
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized

        # Calculate final value
        final_value = initial_value * (1 + cumulative_return)

        return {
            "cumulative_return": cumulative_return,
            "volatility": volatility,
            "final_value": final_value,
            "sharpe_ratio": cumulative_return / volatility if volatility > 0 else 0
        }

class ReflectionGenerator:
    """
    Generator for reflections on experiences.
    """

    def __init__(
        self,
        llm_client: Any,
        memory_store: ExperienceMemoryStore,
        prompt_templates: PromptTemplates = None
    ):
        """
        Initialize reflection generator.

        Args:
            llm_client: Client for LLM API.
            memory_store: Experience memory store.
            prompt_templates: Prompt templates.
        """
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.prompt_templates = prompt_templates or PromptTemplates()

    def generate_reflection(
        self,
        experience: Experience,
        similar_experiences: List[Experience] = None
    ) -> str:
        """
        Generate a reflection on an experience.

        Args:
            experience: Experience to reflect on.
            similar_experiences: List of similar past experiences.

        Returns:
            Reflection text.
        """
        # Create prompt
        prompt = self.prompt_templates.create_reflection_prompt(
            experience=experience,
            similar_experiences=similar_experiences
        )

        # Generate reflection using LLM
        try:
            reflection = self.llm_client.generate_text(prompt, ReflectionConfig)

            # Clean up reflection (remove any JSON formatting, etc.)
            reflection = reflection.reflection

            # Update experience with reflection
            self.memory_store.add_reflection(experience.id, reflection)

            return reflection

        except Exception as e:
            logger.error(f"Error generating reflection: {str(e)}")
            return "Error generating reflection."

    def _clean_reflection(self, reflection: str) -> str:
        """
        Clean up reflection text.

        Args:
            reflection: Raw reflection text.

        Returns:
            Cleaned reflection text.
        """
        # Remove any JSON or code blocks
        reflection = re.sub(r'```json.*?```', '', reflection, flags=re.DOTALL)
        reflection = re.sub(r'```.*?```', '', reflection, flags=re.DOTALL)

        # Remove any markdown formatting
        reflection = re.sub(r'#+ ', '', reflection)

        # Remove excessive newlines
        reflection = re.sub(r'\n{3,}', '\n\n', reflection)

        # Trim whitespace
        reflection = reflection.strip()

        return reflection

class RewardAndReflectionSystem:
    """
    Combined system for reward calculation and reflection generation.
    """

    def __init__(
        self,
        llm_client: Any,
        data_manager: Any,
        memory_store: ExperienceMemoryStore,
        reward_config: RewardConfig = None,
        prompt_templates: PromptTemplates = None
    ):
        """
        Initialize reward and reflection system.

        Args:
            llm_client: Client for LLM API.
            data_manager: Data manager for accessing market data.
            memory_store: Experience memory store.
            reward_config: Configuration for reward calculation.
            prompt_templates: Prompt templates.
        """
        self.llm_client = llm_client
        self.data_manager = data_manager
        self.memory_store = memory_store
        self.reward_config = reward_config or RewardConfig()
        self.prompt_templates = prompt_templates or PromptTemplates()

        # Initialize calculators
        self.portfolio_calculator = PortfolioRewardCalculator(self.reward_config)
        self.asset_calculator = AssetRewardCalculator(self.reward_config)
        self.hedging_calculator = HedgingRewardCalculator(self.reward_config)
        self.what_if_calculator = WhatIfCalculator(data_manager)

        # Initialize reflection generator
        self.reflection_generator = ReflectionGenerator(
            llm_client=llm_client,
            memory_store=memory_store,
            prompt_templates=prompt_templates
        )

    def process_portfolio_outcome(
        self,
        date_val: date,
        assets: List[str],
        state: Dict[str, Any],
        action: Dict[str, Any],
        previous_value: float,
        current_value: float,
        running_peak: float,
        realized_variance: float,
        transaction_costs: float = 0.0,
        portfolio_return: Optional[float] = None,
        portfolio_volatility: Optional[float] = None,
        benchmark_return: float = 0.0,
        market_regime: str = "neutral",
        confidence: float = 0.0,
        generate_reflection: bool = True
    ) -> Tuple[float, str, str]:
        """
        Process outcome of a portfolio allocation decision.

        Args:
            date_val: Date of the decision.
            assets: List of assets.
            state: State information.
            action: Action taken.
            previous_value: Portfolio value before the action.
            current_value: Portfolio value after the action.
            running_peak: Running peak portfolio value used for drawdown.
            realized_variance: Realised variance estimate for the step.
            transaction_costs: Transaction costs.
            portfolio_return: Optional precomputed return.
            portfolio_volatility: Optional volatility estimate.
            market_regime: Market regime.
            confidence: Confidence in the decision.
            generate_reflection: Whether to generate a reflection.

        Returns:
            Tuple of (reward, experience_id, reflection).
        """
        if portfolio_return is None and previous_value > 0:
            portfolio_return = (current_value - previous_value) / previous_value

        if portfolio_volatility is None:
            portfolio_volatility = float(np.sqrt(max(realized_variance, 0.0)))

        # Calculate reward
        reward = self.portfolio_calculator.calculate_reward(
            previous_value=previous_value,
            current_value=current_value,
            realized_variance=realized_variance,
            running_peak=running_peak,
            transaction_costs=transaction_costs,
        )

        # Create outcome dictionary
        outcome = {
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_volatility,
            "benchmark_return": benchmark_return,
            "transaction_costs": transaction_costs,
            "reward": reward
        }

        # Add what-if comparison if enabled
        if self.reward_config.use_what_if_comparison:
            # This would require alternative allocations and historical data
            # For now, we'll skip this
            pass

        # Add experience to memory
        experience_id = self.memory_store.add_portfolio_experience(
            date_val=date_val,
            assets=assets,
            state=state,
            action=action,
            outcome=outcome,
            market_regime=market_regime,
            confidence=confidence
        )

        # Generate reflection if requested
        reflection = ""
        if generate_reflection:
            # Get experience
            experience = self.memory_store.get_experience_by_id(experience_id)
            if experience:
                # Get similar experiences
                similar_experiences = self.memory_store.get_portfolio_experiences(
                    current_state=state,
                    market_regime=market_regime,
                    assets=assets,
                    num_experiences=3
                )

                # Generate reflection
                reflection = self.reflection_generator.generate_reflection(
                    experience=experience,
                    similar_experiences=similar_experiences
                )

        return reward, experience_id, reflection

    def process_asset_outcome(
        self,
        date_val: date,
        asset: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        stock_return: float,
        options_return: float,
        asset_volatility: float,
        stock_percentage: float,
        options_percentage: float,
        transaction_costs: float = 0.0,
        market_regime: str = "neutral",
        confidence: float = 0.0,
        generate_reflection: bool = True
    ) -> Tuple[float, str, str]:
        """
        Process outcome of an asset allocation decision.

        Args:
            date_val: Date of the decision.
            asset: Asset ticker.
            state: State information.
            action: Action taken.
            stock_return: Stock return.
            options_return: Options return.
            asset_volatility: Asset volatility.
            stock_percentage: Percentage allocated to stock.
            options_percentage: Percentage allocated to options.
            transaction_costs: Transaction costs.
            market_regime: Market regime.
            confidence: Confidence in the decision.
            generate_reflection: Whether to generate a reflection.

        Returns:
            Tuple of (reward, experience_id, reflection).
        """
        # Calculate reward
        reward = self.asset_calculator.calculate_reward(
            stock_return=stock_return,
            options_return=options_return,
            asset_volatility=asset_volatility,
            stock_percentage=stock_percentage,
            options_percentage=options_percentage,
            transaction_costs=transaction_costs
        )

        # Create outcome dictionary
        outcome = {
            "stock_return": stock_return,
            "options_return": options_return,
            "combined_return": (stock_return * stock_percentage) + (options_return * options_percentage),
            "asset_volatility": asset_volatility,
            "transaction_costs": transaction_costs,
            "reward": reward
        }

        # Add experience to memory
        experience_id = self.memory_store.add_asset_experience(
            date_val=date_val,
            asset=asset,
            state=state,
            action=action,
            outcome=outcome,
            market_regime=market_regime,
            confidence=confidence
        )

        # Generate reflection if requested
        reflection = ""
        if generate_reflection:
            # Get experience
            experience = self.memory_store.get_experience_by_id(experience_id)
            if experience:
                # Get similar experiences
                similar_experiences = self.memory_store.get_asset_experiences(
                    asset=asset,
                    current_state=state,
                    market_regime=market_regime,
                    num_experiences=3
                )

                # Generate reflection
                reflection = self.reflection_generator.generate_reflection(
                    experience=experience,
                    similar_experiences=similar_experiences
                )

        return reward, experience_id, reflection

    def process_hedging_outcome(
        self,
        date_val: date,
        asset: str,
        strategy_type: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        stock_return: float,
        options_return: float,
        combined_volatility: float,
        stock_only_volatility: float,
        option_costs: float,
        market_regime: str = "neutral",
        confidence: float = 0.0,
        generate_reflection: bool = True
    ) -> Tuple[float, str, str]:
        """
        Process outcome of a hedging strategy decision.

        Args:
            date_val: Date of the decision.
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            state: State information.
            action: Action taken.
            stock_return: Stock return.
            options_return: Options return.
            combined_volatility: Volatility of combined position.
            stock_only_volatility: Volatility of stock-only position.
            option_costs: Option costs.
            market_regime: Market regime.
            confidence: Confidence in the decision.
            generate_reflection: Whether to generate a reflection.

        Returns:
            Tuple of (reward, experience_id, reflection).
        """
        # Calculate reward
        reward = self.hedging_calculator.calculate_reward(
            stock_return=stock_return,
            options_return=options_return,
            combined_volatility=combined_volatility,
            stock_only_volatility=stock_only_volatility,
            option_costs=option_costs,
            strategy_type=strategy_type
        )

        # Create outcome dictionary
        outcome = {
            "stock_return": stock_return,
            "options_return": options_return,
            "combined_return": stock_return + options_return,
            "combined_volatility": combined_volatility,
            "stock_only_volatility": stock_only_volatility,
            "volatility_reduction": stock_only_volatility - combined_volatility,
            "option_costs": option_costs,
            "reward": reward
        }

        # Add experience to memory
        experience_id = self.memory_store.add_hedging_experience(
            date_val=date_val,
            asset=asset,
            strategy_type=strategy_type,
            state=state,
            action=action,
            outcome=outcome,
            market_regime=market_regime,
            confidence=confidence
        )

        # Generate reflection if requested
        reflection = ""
        if generate_reflection:
            # Get experience
            experience = self.memory_store.get_experience_by_id(experience_id)
            if experience:
                # Get similar experiences
                similar_experiences = self.memory_store.get_hedging_experiences(
                    asset=asset,
                    strategy_type=strategy_type,
                    current_state=state,
                    market_regime=market_regime,
                    num_experiences=3
                )

                # Generate reflection
                reflection = self.reflection_generator.generate_reflection(
                    experience=experience,
                    similar_experiences=similar_experiences
                )

        return reward, experience_id, reflection
