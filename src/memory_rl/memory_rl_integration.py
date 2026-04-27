"""
Integration module for connecting memory-based RL components with the existing system.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Literal, Tuple, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field

from src.utils.option_lookup import convert_hedging_strategy_response

# Import memory-based RL components
from .experience_memory_store import ExperienceMemoryStore, MemoryConfig
from .prompt_templates import PromptTemplates, PromptConfig
from .reward_reflection_system import RewardAndReflectionSystem, RewardConfig
from .hierarchical_memory import HierarchicalMemoryStructure, HierarchicalMemoryConfig
from src.models.simplified_llm_response_models import HedgingStrategyResponse

# Import existing components (placeholders - replace with actual imports)
from src.models.pydantic_models import (
    OptionData, PortfolioAllocation, AssetSplit, HedgingStrategy, 
    MarketRegime, CapitalAllocation, HedgingStrategies, SimulationConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryRLConfig(BaseModel):
    """Configuration for memory-based RL integration."""
    memory_config: HierarchicalMemoryConfig = Field(default_factory=HierarchicalMemoryConfig)
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)
    reward_config: RewardConfig = Field(default_factory=RewardConfig)
    memory_dir: str = "memory"
    enable_reflection: bool = True
    reflection_frequency: int = 5  # Generate reflection every N experiences
    save_frequency: int = 1  # Save memory to disk every N experiences
    
    class Config:
        arbitrary_types_allowed = True

# Extend SimulationConfig to include memory-based RL settings
class ExtendedSimulationConfig(SimulationConfig):
    """Extended simulation configuration with memory-based RL settings."""
    use_memory_rl: bool = False
    memory_rl_config: Optional[MemoryRLConfig] = Field(default_factory=MemoryRLConfig)

class MemoryRLIntegration:
    """
    Integration class for connecting memory-based RL components with the existing system.
    """
    
    def __init__(
        self,
        llm_client: Any,
        data_manager: Any,
        config: MemoryRLConfig = None
    ):
        """
        Initialize the memory-based RL integration.
        
        Args:
            llm_client: Client for LLM API.
            data_manager: Data manager for accessing market data.
            config: Configuration for memory-based RL integration.
        """
        self.llm_client = llm_client
        self.data_manager = data_manager
        self.config = config or MemoryRLConfig()
        
        # Create memory directory if it doesn't exist
        os.makedirs(self.config.memory_dir, exist_ok=True)
        
        # Initialize components
        self.prompt_templates = PromptTemplates(self.config.prompt_config)
        
        self.hierarchical_memory = HierarchicalMemoryStructure(
            config=self.config.memory_config,
            llm_client=llm_client
        )

        self.reward_system = RewardAndReflectionSystem(
            llm_client=llm_client,
            data_manager=data_manager,
            memory_store=self.hierarchical_memory.portfolio_memory,  # Just for initialization
            reward_config=self.config.reward_config,
            prompt_templates=self.prompt_templates
        )
        
        # Load memory from disk if available
        self.hierarchical_memory.load_from_disk()
        
        # Counters for reflection and saving
        self.experience_counter = 0
        
        logger.info("Memory-based RL integration initialized.")
    
    def get_portfolio_allocation(
        self,
        current_date: date,
        portfolio_value: float,
        cash: float,
        assets: List[str],
        market_analysis: Dict[str, MarketRegime],
        current_positions: Dict[str, Dict[str, Any]],
        emergency: float = None
    ) -> Tuple[PortfolioAllocation, Dict[str, Any]]:
        """
        Get portfolio allocation using memory-based RL.
        
        Args:
            current_date: Current date.
            portfolio_value: Current portfolio value.
            cash: Available cash.
            assets: List of assets.
            market_analysis: Market analysis for each asset.
            current_positions: Current positions.
            
        Returns:
            Tuple of (portfolio allocation, decision details).
        """
        # Get relevant experiences and patterns
        current_state = {
            "date": current_date,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "assets": assets,
            "market_regimes": {asset: analysis.model_dump().get("regime", "neutral") for asset, analysis in market_analysis.items()},
            "current_positions": current_positions
        }
        
        # Determine market regime (simplified - use most common regime)
        regimes = [analysis.model_dump().get("regime", "neutral") for analysis in market_analysis.values()]
        market_regime = max(set(regimes), key=regimes.count)
        try:
            experiences, patterns = self.hierarchical_memory.get_portfolio_experiences(
                current_state=current_state,
                market_regime=market_regime,
                assets=assets,
                num_experiences=5,
                include_temporal_patterns=True
            )
        except Exception as e:
            logger.error(f"Error retrieving portfolio experiences: {str(e)}")
            experiences = []
            patterns = []
        
        # Create additional context with temporal patterns
        additional_context = ""
        if patterns:
            additional_context += "Temporal Patterns:\n"
            for i, pattern in enumerate(patterns):
                additional_context += f"Pattern {i+1}:\n{pattern.to_text()}\n"
        
        # Create prompt
        prompt = self.prompt_templates.create_portfolio_prompt(
            current_date=current_date,
            portfolio_value=portfolio_value,
            cash=cash,
            assets=assets,
            market_analysis=market_analysis,
            current_positions=current_positions,
            experiences=experiences,
            emergency=emergency,
            additional_context=additional_context
        )
        
        # Generate allocation using LLM
        try:
            response = self.llm_client.generate_text(prompt, PortfolioAllocation)
            
            # Parse JSON response
            allocation_dict = response.model_dump()
            
            # Create decision details
            decision_details = {
                "source": "memory_rl",
                "confidence": 0.8,  # Placeholder
                "rationale": allocation_dict.get("rationale", ""),
                "experiences_used": [exp.id for exp in experiences],
                "patterns_used": [pattern.id for pattern in patterns]
            }
            
            return response, decision_details
        
        except Exception as e:
            logger.error(f"Error generating portfolio allocation: {str(e)}")
            
            # Fallback to a simple allocation
            allocation = self._create_fallback_portfolio_allocation(assets, portfolio_value, cash)
            
            decision_details = {
                "source": "fallback",
                "confidence": 0.5,
                "rationale": "Error in LLM response, using fallback allocation.",
                "error": str(e)
            }
            
            return allocation, decision_details
    
    def get_asset_allocation(
        self,
        current_date: date,
        asset: str,
        allocation_amount: float,
        market_analysis: Dict[str, Any],
        current_price: float,
        volatility: float,
        hedging_strategies: List[str]
    ) -> Tuple[AssetSplit, Dict[str, Any]]:
        """
        Get asset allocation using memory-based RL.
        
        Args:
            current_date: Current date.
            asset: Asset ticker.
            allocation_amount: Amount allocated to this asset.
            market_analysis: Market analysis for this asset.
            current_price: Current price of the asset.
            volatility: Volatility of the asset.
            
        Returns:
            Tuple of (asset split, decision details).
        """
        market_analysis = market_analysis.model_dump()
        # Get relevant experiences and patterns
        current_state = {
            "date": current_date,
            "asset": asset,
            "allocation_amount": allocation_amount,
            "current_price": current_price,
            "volatility": volatility,
            "market_regime": market_analysis.get("regime", "neutral"),
            "sentiment_score": market_analysis.get("sentiment", 0.0),
            "sentiment_label": market_analysis.get("sentiment_label", "neutral"),
            "outlook_score": market_analysis.get("outlook", 0.0),
            "short_term_outlook": market_analysis.get("short_term_outlook", "stable")
        }
        
        market_regime = market_analysis.get("regime", "neutral")
        
        experiences, patterns = self.hierarchical_memory.get_asset_experiences(
            asset=asset,
            current_state=current_state,
            market_regime=market_regime,
            num_experiences=5,
            include_temporal_patterns=True
        )
        
        # Create additional context with temporal patterns
        additional_context = ""
        if patterns:
            additional_context += "Temporal Patterns:\n"
            for i, pattern in enumerate(patterns):
                additional_context += f"Pattern {i+1}:\n{pattern.to_text()}\n"
        
        # Create prompt
        prompt = self.prompt_templates.create_asset_prompt(
            current_date=current_date,
            asset=asset,
            allocation_amount=allocation_amount,
            market_analysis=market_analysis,
            hedging_strategies=hedging_strategies,
            current_price=current_price,
            volatility=volatility,
            experiences=experiences,
            additional_context=additional_context
        )
        
        # Generate allocation using LLM
        try:
            asset_split = self.llm_client.generate_text(prompt, AssetSplit)
            # Create decision details
            decision_details = {
                "source": "memory_rl",
                "confidence": 0.8,  # Placeholder
                "rationale": asset_split.model_dump().get("rationale", ""),
                "experiences_used": [exp.id for exp in experiences],
                "patterns_used": [pattern.id for pattern in patterns]
            }
            
            return asset_split, decision_details
        
        except Exception as e:
            logger.error(f"Error generating asset allocation: {str(e)}")
            
            # Fallback to a simple allocation
            asset_split = self._create_fallback_asset_split(asset, allocation_amount)
            
            decision_details = {
                "source": "fallback",
                "confidence": 0.5,
                "rationale": "Error in LLM response, using fallback allocation.",
                "error": str(e)
            }
            
            return asset_split, decision_details
    
    def get_hedging_strategy(
        self,
        current_date: date,
        asset: str,
        strategy_type: str,
        stock_position: Dict[str, Any],
        available_options: List[OptionData],
        available_capital: float,
        market_analysis: Dict[str, Any]
    ) -> Tuple[HedgingStrategy, Dict[str, Any]]:
        """
        Get hedging strategy using memory-based RL.
        
        Args:
            current_date: Current date.
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            stock_position: Current stock position.
            available_options: Available options.
            available_capital: Available capital for options.
            market_analysis: Market analysis for this asset.
            
        Returns:
            Tuple of (hedging strategy, decision details).
        """
        # Get relevant experiences and patterns
        current_state = {
            "date": current_date,
            "asset": asset,
            "strategy_type": strategy_type,
            "stock_position": stock_position,
            "available_capital": available_capital,
            "market_regime": market_analysis.get("regime", "neutral"),
            "sentiment_score": market_analysis.get("sentiment", 0.0),
            "sentiment_label": market_analysis.get("sentiment_label", "neutral"),
            "outlook_score": market_analysis.get("outlook", 0.0),
            "short_term_outlook": market_analysis.get("short_term_outlook", "stable")
        }
        
        market_regime = market_analysis.get("regime", "neutral")
        
        experiences, patterns = self.hierarchical_memory.get_hedging_experiences(
            asset=asset,
            strategy_type=strategy_type,
            current_state=current_state,
            market_regime=market_regime,
            num_experiences=5,
            include_temporal_patterns=True
        )
        
        # Create additional context with temporal patterns
        additional_context = ""
        if patterns:
            additional_context += "Temporal Patterns:\n"
            for i, pattern in enumerate(patterns):
                additional_context += f"Pattern {i+1}:\n{pattern.to_text()}\n"
        
        # Create prompt
        prompt = self.prompt_templates.create_hedging_prompt(
            current_date=current_date,
            asset=asset,
            strategy_type=strategy_type,
            stock_position=stock_position,
            available_options=available_options,
            available_capital=available_capital,
            market_analysis=market_analysis,
            experiences=experiences,
            additional_context=additional_context
        )
        
        # Generate strategy using LLM
        try:
            response = self.llm_client.generate_text(prompt, HedgingStrategyResponse)
            
            # Convert to HedgingStrategy
            # hedging_strategy = self._convert_to_hedging_strategy(response.model_dump())
            
            if isinstance(response, HedgingStrategyResponse):
                hedging_strategy = convert_hedging_strategy_response(
                    response=response,
                    available_options=available_options,
                    ticker=asset,
                    available_capital=available_capital,
                    current_date=current_date
                )
            else:
                return HedgingStrategy(asset=asset, strategy_type=strategy_type, rationale="Invalid response type", option_positions=[]), {}
            # Create decision details
            decision_details = {
                "source": "memory_rl",
                "confidence": 0.8,  # Placeholder
                "rationale": response.model_dump().get("rationale", ""),
                "experiences_used": [exp.id for exp in experiences],
                "patterns_used": [pattern.id for pattern in patterns]
            }
            
            return hedging_strategy, decision_details
        
        except Exception as e:
            logger.error(f"Error generating hedging strategy: {str(e)}")
            
            # Fallback to a simple strategy
            # hedging_strategy = self._create_fallback_hedging_strategy(asset, strategy_type, available_options)
            hedging_strategy = None
            decision_details = {
                "source": "fallback",
                "confidence": 0.5,
                "rationale": "Error in LLM response, using fallback strategy.",
                "error": str(e)
            }
            
            return hedging_strategy, decision_details
    
    def update_portfolio_results(
        self,
        date_val: date,
        assets: List[str],
        state: Dict[str, Any],
        action: Dict[str, Any],
        previous_value: float,
        current_value: float,
        running_peak: float,
        realized_variance: float,
        benchmark_return: float,
        transaction_costs: float = 0.0,
        portfolio_return: Optional[float] = None,
        portfolio_volatility: Optional[float] = None,
        market_regime: str = "neutral",
        confidence: float = 0.0
    ) -> None:
        """
        Update portfolio results for memory-based RL learning.
        
        Args:
            date_val: Date of the decision.
            assets: List of assets.
            state: State information.
            action: Action taken.
            previous_value: Portfolio value before the action.
            current_value: Portfolio value after the action.
            running_peak: Running peak portfolio value.
            realized_variance: Realised variance estimate for the step.
            benchmark_return: Benchmark return.
            transaction_costs: Transaction costs.
            portfolio_return: Optional portfolio return if precomputed.
            portfolio_volatility: Optional volatility estimate.
            market_regime: Market regime.
            confidence: Confidence in the decision.
        """
        # Process outcome using reward system
        reward, experience_id, reflection = self.reward_system.process_portfolio_outcome(
            date_val=date_val,
            assets=assets,
            state=state,
            action=action,
            previous_value=previous_value,
            current_value=current_value,
            running_peak=running_peak,
            realized_variance=realized_variance,
            transaction_costs=transaction_costs,
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            benchmark_return=benchmark_return,
            market_regime=market_regime,
            confidence=confidence,
            generate_reflection=self.config.enable_reflection and (self.experience_counter % self.config.reflection_frequency == 0)
        )
        
        # Increment experience counter
        self.experience_counter += 1
        
        # Save memory to disk periodically
        if self.experience_counter % self.config.save_frequency == 0:
            self.hierarchical_memory.save_to_disk()
        
        logger.info(f"Updated portfolio results: reward={reward:.4f}, experience_id={experience_id}")
    
    def update_asset_results(
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
        confidence: float = 0.0
    ) -> None:
        """
        Update asset results for memory-based RL learning.
        
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
        """
        # Process outcome using reward system
        reward, experience_id, reflection = self.reward_system.process_asset_outcome(
            date_val=date_val,
            asset=asset,
            state=state,
            action=action,
            stock_return=stock_return,
            options_return=options_return,
            asset_volatility=asset_volatility,
            stock_percentage=stock_percentage,
            options_percentage=options_percentage,
            transaction_costs=transaction_costs,
            market_regime=market_regime,
            confidence=confidence,
            generate_reflection=self.config.enable_reflection and (self.experience_counter % self.config.reflection_frequency == 0)
        )
        
        # Increment experience counter
        self.experience_counter += 1
        
        # Save memory to disk periodically
        if self.experience_counter % self.config.save_frequency == 0:
            self.hierarchical_memory.save_to_disk()
        
        logger.info(f"Updated asset results for {asset}: reward={reward:.4f}, experience_id={experience_id}")
    
    def update_hedging_results(
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
        confidence: float = 0.0
    ) -> None:
        """
        Update hedging results for memory-based RL learning.
        
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
        """
        # Process outcome using reward system
        reward, experience_id, reflection = self.reward_system.process_hedging_outcome(
            date_val=date_val,
            asset=asset,
            strategy_type=strategy_type,
            state=state,
            action=action,
            stock_return=stock_return,
            options_return=options_return,
            combined_volatility=combined_volatility,
            stock_only_volatility=stock_only_volatility,
            option_costs=option_costs,
            market_regime=market_regime,
            confidence=confidence,
            generate_reflection=self.config.enable_reflection and (self.experience_counter % self.config.reflection_frequency == 0)
        )
        
        # Increment experience counter
        self.experience_counter += 1
        
        # Save memory to disk periodically
        if self.experience_counter % self.config.save_frequency == 0:
            self.hierarchical_memory.save_to_disk()
        
        logger.info(f"Updated hedging results for {asset} {strategy_type}: reward={reward:.4f}, experience_id={experience_id}")
    
    def generate_pattern_descriptions(self) -> None:
        """Generate descriptions for temporal patterns."""
        self.hierarchical_memory.generate_pattern_descriptions()
    
    def generate_cross_level_descriptions(self) -> None:
        """Generate descriptions for cross-level experiences."""
        self.hierarchical_memory.generate_cross_level_descriptions()
    
    def save_memory(self) -> bool:
        """
        Save memory to disk.
        
        Returns:
            True if successful, False otherwise.
        """
        return self.hierarchical_memory.save_to_disk()
    
    def load_memory(self) -> bool:
        """
        Load memory from disk.
        
        Returns:
            True if successful, False otherwise.
        """
        return self.hierarchical_memory.load_from_disk()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory.
        
        Returns:
            Dictionary of memory statistics.
        """
        return self.hierarchical_memory.get_memory_stats()
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON.
            
        Returns:
            Extracted JSON as dictionary.
        """
        # Find JSON between triple backticks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without backticks
            json_match = re.search(r'(\{[\s\S]*\})', text)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("No JSON found in response")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {str(e)}")
            logger.error(f"JSON string: {json_str}")
            raise ValueError(f"Invalid JSON in response: {str(e)}")
    
    def _convert_to_portfolio_allocation(self, allocation_dict: Dict[str, Any], assets: List[str]) -> PortfolioAllocation:
        """
        Convert allocation dictionary to PortfolioAllocation.
        
        Args:
            allocation_dict: Allocation dictionary from LLM.
            assets: List of assets.
            
        Returns:
            PortfolioAllocation object.
        """
        # Extract allocations
        allocations = []
        for alloc in allocation_dict.get("allocations", []):
            allocations.append({
                "asset": alloc.get("asset", ""),
                "amount": alloc.get("amount", 0.0),
                "percentage": alloc.get("percentage", 0.0)
            })
        
        # Create PortfolioAllocation
        return PortfolioAllocation(
            total_capital=allocation_dict.get("total_capital", 0.0),
            allocations=allocations,
            cash_reserve=allocation_dict.get("cash_reserve", 0.0),
            rationale=allocation_dict.get("rationale", "")
        )
    
    def _convert_to_asset_split(self, allocation_dict: Dict[str, Any]) -> AssetSplit:
        """
        Convert allocation dictionary to AssetSplit.
        
        Args:
            allocation_dict: Allocation dictionary from LLM.
            
        Returns:
            AssetSplit object.
        """
        # Extract hedging strategies
        hedging_strategies = {}
        for strategy, percentage in allocation_dict.get("hedging_strategies", {}).items():
            hedging_strategies[strategy] = percentage
        
        # Create AssetSplit
        return AssetSplit(
            asset=allocation_dict.get("asset", ""),
            total_allocation=allocation_dict.get("total_allocation", 0.0),
            stock_percentage=allocation_dict.get("stock_percentage", 0.0),
            options_percentage=allocation_dict.get("options_percentage", 0.0),
            stock_amount=allocation_dict.get("stock_amount", 0.0),
            options_amount=allocation_dict.get("options_amount", 0.0),
            rationale=allocation_dict.get("rationale", ""),
            hedging_strategies=hedging_strategies
        )
    
    def _convert_to_hedging_strategy(self, strategy_dict: Dict[str, Any]) -> HedgingStrategy:
        """
        Convert strategy dictionary to HedgingStrategy.
        
        Args:
            strategy_dict: Strategy dictionary from LLM.
            
        Returns:
            HedgingStrategy object.
        """
        # Extract option positions
        option_positions = []
        for pos in strategy_dict.get("option_contracts", []):
            option_positions.append({
                "option_type": pos.get("option_type", ""),
                "option_id": pos.get("option_id", ""),
                "contracts": pos.get("contracts", 0),
                "action": pos.get("action", ""),
            })
        # Create HedgingStrategy
        return HedgingStrategy(
            asset=strategy_dict.get("asset", ""),
            strategy_type=strategy_dict.get("strategy_type", ""),
            rationale=strategy_dict.get("rationale", ""),
            option_positions=option_positions,
            estimated_cost=strategy_dict.get("estimated_cost", 0.0),
        )
    
    def _create_fallback_portfolio_allocation(self, assets: List[str], portfolio_value: float, cash: float) -> PortfolioAllocation:
        """
        Create a fallback portfolio allocation.
        
        Args:
            assets: List of assets.
            portfolio_value: Current portfolio value.
            cash: Available cash.
            
        Returns:
            PortfolioAllocation object.
        """
        # Equal allocation to all assets
        num_assets = len(assets)
        if num_assets == 0:
            return PortfolioAllocation(
                total_capital=portfolio_value,
                allocations=[],
                cash_reserve=portfolio_value,
                rationale="Fallback allocation: no assets provided."
            )
        
        # Allocate 80% to assets, 20% to cash
        asset_allocation = 0.8 * portfolio_value
        cash_reserve = 0.2 * portfolio_value
        
        # Equal allocation to each asset
        per_asset = asset_allocation / num_assets
        per_asset_percentage = 0.8 / num_assets
        
        allocations = []
        for asset in assets:
            allocations.append({
                "asset": asset,
                "amount": per_asset,
                "percentage": per_asset_percentage
            })
        
        return PortfolioAllocation(
            total_capital=portfolio_value,
            allocations=allocations,
            cash_reserve=cash_reserve,
            rationale="Fallback allocation: equal allocation to all assets with 20% cash reserve."
        )
    
    def _create_fallback_asset_split(self, asset: str, allocation_amount: float) -> AssetSplit:
        """
        Create a fallback asset split.
        
        Args:
            asset: Asset ticker.
            allocation_amount: Amount allocated to this asset.
            
        Returns:
            AssetSplit object.
        """
        # 70% stock, 30% options
        stock_percentage = 0.7
        options_percentage = 0.3
        
        stock_amount = stock_percentage * allocation_amount
        options_amount = options_percentage * allocation_amount
        
        # Equal allocation to all hedging strategies
        hedging_strategies = {
            "collar": 0.33,
            "straddle": 0.33,
            "delta_neutral": 0.34
        }
        
        return AssetSplit(
            asset=asset,
            total_allocation=allocation_amount,
            stock_percentage=stock_percentage,
            options_percentage=options_percentage,
            stock_amount=stock_amount,
            options_amount=options_amount,
            rationale="Fallback allocation: 70% stock, 30% options with equal allocation to hedging strategies.",
            hedging_strategies=hedging_strategies
        )
    
    def _create_fallback_hedging_strategy(self, asset: str, strategy_type: str, available_options: List[Dict[str, Any]]) -> HedgingStrategy:
        """
        Create a fallback hedging strategy.
        
        Args:
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            available_options: Available options.
            
        Returns:
            HedgingStrategy object.
        """
        # Find suitable options based on strategy type
        option_positions = []
        
        if strategy_type == "collar" and len(available_options) >= 2:
            # Find a put and a call
            put_options = [opt for opt in available_options if opt.get("option_type", 1) == 0]
            call_options = [opt for opt in available_options if opt.get("option_type", 1) == 1]
            
            if put_options and call_options:
                # Sort puts by strike (descending) and calls by strike (ascending)
                put_options.sort(key=lambda x: x.get("strike", 0), reverse=True)
                call_options.sort(key=lambda x: x.get("strike", 0))
                
                # Select first put and first call
                put = put_options[0]
                call = call_options[0]
                
                option_positions = [
                    {
                        "option_type": "put",
                        "option_id": put.get("option_id", ""),
                        "strike": put.get("strike", 0.0),
                        "expiration_date": put.get("expire_date", ""),
                        "contracts": 1,
                        "premium_per_contract": put.get("last", 0.0),
                        "action": "buy",
                        "rationale": "Downside protection"
                    },
                    {
                        "option_type": "call",
                        "option_id": call.get("option_id", ""),
                        "strike": call.get("strike", 0.0),
                        "expiration_date": call.get("expire_date", ""),
                        "contracts": 1,
                        "premium_per_contract": call.get("last", 0.0),
                        "action": "sell",
                        "rationale": "Premium collection"
                    }
                ]
        
        elif strategy_type == "straddle" and len(available_options) >= 2:
            # Find a put and a call with similar strikes
            put_options = [opt for opt in available_options if opt.get("option_type", 1) == 0]
            call_options = [opt for opt in available_options if opt.get("option_type", 1) == 1]
            
            if put_options and call_options:
                # Find closest to ATM
                put_options.sort(key=lambda x: abs(x.get("moneyness", 1) - 1))
                call_options.sort(key=lambda x: abs(x.get("moneyness", 1) - 1))
                
                put = put_options[0]
                call = call_options[0]
                
                option_positions = [
                    {
                        "option_type": "put",
                        "option_id": put.get("option_id", ""),
                        "strike": put.get("strike", 0.0),
                        "expiration_date": put.get("expire_date", ""),
                        "contracts": 1,
                        "premium_per_contract": put.get("last", 0.0),
                        "action": "buy",
                        "rationale": "Profit from downside movement"
                    },
                    {
                        "option_type": "call",
                        "option_id": call.get("option_id", ""),
                        "strike": call.get("strike", 0.0),
                        "expiration_date": call.get("expire_date", ""),
                        "contracts": 1,
                        "premium_per_contract": call.get("last", 0.0),
                        "action": "buy",
                        "rationale": "Profit from upside movement"
                    }
                ]
        
        elif strategy_type == "delta_neutral" and available_options:
            # Just pick the first option
            opt = available_options[0]
            
            option_positions = [
                {
                    "option_type": "put" if opt.get("option_type", 1) == 0 else "call",
                    "option_id": opt.get("option_id", ""),
                    "strike": opt.get("strike", 0.0),
                    "expiration_date": opt.get("expire_date", ""),
                    "contracts": 1,
                    "premium_per_contract": opt.get("last", 0.0),
                    "action": "buy",
                    "rationale": "Delta hedge"
                }
            ]
        
        return HedgingStrategy(
            asset=asset,
            strategy_type=strategy_type,
            rationale="Fallback strategy: simple implementation based on available options.",
            option_positions=option_positions,
            estimated_cost=sum(pos.get("premium_per_contract", 0) * pos.get("contracts", 0) for pos in option_positions),
            estimated_protection=0.1,
            confidence=0.5
        )
