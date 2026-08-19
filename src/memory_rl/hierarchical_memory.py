"""
Hierarchical Memory Structure for memory-based reinforcement learning.
Organizes experiences at different levels of the decision-making process.
"""

import os
import json
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union, Literal, Tuple, Any
import logging
from pydantic import BaseModel, Field
import uuid
from collections import defaultdict
import random

from .experience_memory_store import (
    Experience, PortfolioExperience, AssetExperience, HedgingExperience,
    ExperienceMemoryStore, MemoryConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalMemoryConfig(BaseModel):
    """Configuration for hierarchical memory structure."""
    portfolio_memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    asset_memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    hedging_memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    enable_cross_level_retrieval: bool = True
    enable_temporal_patterns: bool = True
    max_temporal_pattern_length: int = 5
    memory_dir: str = "memory"

class TemporalPattern(BaseModel):
    """Representation of a temporal pattern in experiences."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    level: Literal["portfolio", "asset", "hedging"]
    experience_ids: List[str]
    start_date: date
    end_date: date
    market_regime: str
    cumulative_reward: float
    description: Optional[str] = None

    def to_text(self) -> str:
        """Convert temporal pattern to text format for inclusion in prompts."""
        text = f"--- Temporal Pattern from {self.start_date} to {self.end_date} (Market Regime: {self.market_regime}) ---\n"
        text += f"Level: {self.level}\n"
        text += f"Cumulative Reward: {self.cumulative_reward:.4f}\n"

        if self.description:
            text += f"Description: {self.description}\n"

        return text

class CrossLevelExperience(BaseModel):
    """Representation of related experiences across different levels."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: date
    portfolio_experience_id: Optional[str] = None
    asset_experience_ids: Dict[str, str] = {}  # asset -> experience_id
    hedging_experience_ids: Dict[str, Dict[str, str]] = {}  # asset -> strategy -> experience_id
    market_regime: str
    cumulative_reward: float
    description: Optional[str] = None

    def to_text(self) -> str:
        """Convert cross-level experience to text format for inclusion in prompts."""
        text = f"--- Cross-Level Experience from {self.date} (Market Regime: {self.market_regime}) ---\n"
        text += f"Cumulative Reward: {self.cumulative_reward:.4f}\n"

        if self.description:
            text += f"Description: {self.description}\n"

        return text

class HierarchicalMemoryStructure:
    """
    Hierarchical memory structure for organizing experiences at different levels.
    """

    def __init__(
        self,
        config: HierarchicalMemoryConfig = None,
        llm_client: Any = None
    ):
        """
        Initialize hierarchical memory structure.

        Args:
            config: Configuration for hierarchical memory.
            llm_client: Client for LLM API (used for generating descriptions).
        """
        self.config = config or HierarchicalMemoryConfig()
        self.llm_client = llm_client

        # Create memory directory if it doesn't exist
        os.makedirs(self.config.memory_dir, exist_ok=True)

        # Initialize memory stores for each level
        self.portfolio_memory = ExperienceMemoryStore(self.config.portfolio_memory_config)
        self.asset_memory = ExperienceMemoryStore(self.config.asset_memory_config)
        self.hedging_memory = ExperienceMemoryStore(self.config.hedging_memory_config)

        # Initialize temporal patterns
        self.temporal_patterns: Dict[str, List[TemporalPattern]] = {
            "portfolio": [],
            "asset": defaultdict(list),  # asset -> patterns
            "hedging": defaultdict(lambda: defaultdict(list))  # asset -> strategy -> patterns
        }

        # Initialize cross-level experiences
        self.cross_level_experiences: List[CrossLevelExperience] = []
        self.load_from_disk()


    def add_portfolio_experience(
        self,
        date_val: date,
        assets: List[str],
        state: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        market_regime: str = "neutral",
        confidence: float = 0.0,
        reflection: Optional[str] = None,
        tags: List[str] = []
    ) -> str:
        """
        Add a portfolio-level experience.

        Args:
            date_val: Date of the experience.
            assets: List of assets involved.
            state: State information.
            action: Action taken.
            outcome: Outcome of the action.
            market_regime: Market regime at the time.
            confidence: Confidence in the decision.
            reflection: Reflection on the experience.
            tags: Tags for categorizing the experience.

        Returns:
            ID of the added experience.
        """
        # Add to portfolio memory
        experience_id = self.portfolio_memory.add_portfolio_experience(
            date_val=date_val,
            assets=assets,
            state=state,
            action=action,
            outcome=outcome,
            market_regime=market_regime,
            confidence=confidence,
            reflection=reflection,
            tags=tags
        )

        # Update temporal patterns if enabled
        if self.config.enable_temporal_patterns:
            self._update_temporal_patterns("portfolio", None, None, date_val, experience_id, outcome.get("reward", 0.0), market_regime)

        # Update cross-level experiences if enabled
        if self.config.enable_cross_level_retrieval:
            self._update_cross_level_experiences(date_val, portfolio_id=experience_id, market_regime=market_regime)

        return experience_id

    def add_asset_experience(
        self,
        date_val: date,
        asset: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        market_regime: str = "neutral",
        confidence: float = 0.0,
        reflection: Optional[str] = None,
        tags: List[str] = []
    ) -> str:
        """
        Add an asset-level experience.

        Args:
            date_val: Date of the experience.
            asset: Asset ticker.
            state: State information.
            action: Action taken.
            outcome: Outcome of the action.
            market_regime: Market regime at the time.
            confidence: Confidence in the decision.
            reflection: Reflection on the experience.
            tags: Tags for categorizing the experience.

        Returns:
            ID of the added experience.
        """
        # Add to asset memory
        experience_id = self.asset_memory.add_asset_experience(
            date_val=date_val,
            asset=asset,
            state=state,
            action=action,
            outcome=outcome,
            market_regime=market_regime,
            confidence=confidence,
            reflection=reflection,
            tags=tags
        )

        # Update temporal patterns if enabled
        if self.config.enable_temporal_patterns:
            self._update_temporal_patterns("asset", asset, None, date_val, experience_id, outcome.get("reward", 0.0), market_regime)

        # Update cross-level experiences if enabled
        if self.config.enable_cross_level_retrieval:
            self._update_cross_level_experiences(date_val, asset_id={asset: experience_id}, market_regime=market_regime)

        return experience_id

    def add_hedging_experience(
        self,
        date_val: date,
        asset: str,
        strategy_type: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        market_regime: str = "neutral",
        confidence: float = 0.0,
        reflection: Optional[str] = None,
        tags: List[str] = []
    ) -> str:
        """
        Add a hedging-level experience.

        Args:
            date_val: Date of the experience.
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            state: State information.
            action: Action taken.
            outcome: Outcome of the action.
            market_regime: Market regime at the time.
            confidence: Confidence in the decision.
            reflection: Reflection on the experience.
            tags: Tags for categorizing the experience.

        Returns:
            ID of the added experience.
        """
        # Add to hedging memory
        experience_id = self.hedging_memory.add_hedging_experience(
            date_val=date_val,
            asset=asset,
            strategy_type=strategy_type,
            state=state,
            action=action,
            outcome=outcome,
            market_regime=market_regime,
            confidence=confidence,
            reflection=reflection,
            tags=tags
        )

        # Update temporal patterns if enabled
        if self.config.enable_temporal_patterns:
            self._update_temporal_patterns("hedging", asset, strategy_type, date_val, experience_id, outcome.get("reward", 0.0), market_regime)

        # Update cross-level experiences if enabled
        if self.config.enable_cross_level_retrieval:
            hedging_id = {asset: {strategy_type: experience_id}}
            self._update_cross_level_experiences(date_val, hedging_id=hedging_id, market_regime=market_regime)

        return experience_id

    def _update_temporal_patterns(
        self,
        level: Literal["portfolio", "asset", "hedging"],
        asset: Optional[str],
        strategy_type: Optional[str],
        date_val: date,
        experience_id: str,
        reward: float,
        market_regime: str
    ) -> None:
        """
        Update temporal patterns with a new experience.

        Args:
            level: Level of the experience.
            asset: Asset ticker (for asset and hedging levels).
            strategy_type: Type of hedging strategy (for hedging level).
            date_val: Date of the experience.
            experience_id: ID of the experience.
            reward: Reward value.
            market_regime: Market regime.
        """
        # Get the appropriate pattern list
        if level == "portfolio":
            patterns = self.temporal_patterns["portfolio"]
        elif level == "asset":
            patterns = self.temporal_patterns["asset"][asset]
        elif level == "hedging":
            patterns = self.temporal_patterns["hedging"][asset][strategy_type]

        # Find active patterns that match the market regime
        active_patterns = [
            p for p in patterns
            if (date_val - p.end_date).days <= 7  # Active within the last week
            and p.market_regime == market_regime
            and len(p.experience_ids) < self.config.max_temporal_pattern_length
        ]

        # Add to active patterns
        for pattern in active_patterns:
            pattern.experience_ids.append(experience_id)
            pattern.end_date = date_val
            pattern.cumulative_reward += reward

        # Create a new pattern if no active patterns or if all are full
        if not active_patterns or all(len(p.experience_ids) >= self.config.max_temporal_pattern_length for p in active_patterns):
            new_pattern = TemporalPattern(
                level=level,
                experience_ids=[experience_id],
                start_date=date_val,
                end_date=date_val,
                market_regime=market_regime,
                cumulative_reward=reward
            )

            # Add to the appropriate list
            if level == "portfolio":
                self.temporal_patterns["portfolio"].append(new_pattern)
            elif level == "asset":
                self.temporal_patterns["asset"][asset].append(new_pattern)
            elif level == "hedging":
                self.temporal_patterns["hedging"][asset][strategy_type].append(new_pattern)

    def _update_cross_level_experiences(
        self,
        date_val: date,
        portfolio_id: Optional[str] = None,
        asset_id: Optional[Dict[str, str]] = None,
        hedging_id: Optional[Dict[str, Dict[str, str]]] = None,
        market_regime: str = "neutral"
    ) -> None:
        """
        Update cross-level experiences with new experience IDs.

        Args:
            date_val: Date of the experiences.
            portfolio_id: ID of portfolio experience.
            asset_id: Dict mapping asset to experience ID.
            hedging_id: Dict mapping asset to dict mapping strategy to experience ID.
            market_regime: Market regime.
        """
        # Find existing cross-level experience for this date
        existing = next((e for e in self.cross_level_experiences if e.date == date_val), None)

        if existing:
            # Update existing cross-level experience
            if portfolio_id:
                existing.portfolio_experience_id = portfolio_id

            if asset_id:
                for asset, exp_id in asset_id.items():
                    existing.asset_experience_ids[asset] = exp_id

            if hedging_id:
                for asset, strategies in hedging_id.items():
                    if asset not in existing.hedging_experience_ids:
                        existing.hedging_experience_ids[asset] = {}

                    for strategy, exp_id in strategies.items():
                        existing.hedging_experience_ids[asset][strategy] = exp_id

            # Update market regime if not set
            if not existing.market_regime:
                existing.market_regime = market_regime

            # Calculate cumulative reward (placeholder - would need to get actual rewards)
            existing.cumulative_reward = 0.0  # This would be calculated from the actual experiences
        else:
            # Create new cross-level experience
            new_cross_level = CrossLevelExperience(
                date=date_val,
                portfolio_experience_id=portfolio_id,
                asset_experience_ids=asset_id or {},
                hedging_experience_ids=hedging_id or {},
                market_regime=market_regime,
                cumulative_reward=0.0  # This would be calculated from the actual experiences
            )

            self.cross_level_experiences.append(new_cross_level)

    def get_portfolio_experiences(
        self,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        assets: Optional[List[str]] = None,
        num_experiences: int = 5,
        include_temporal_patterns: bool = True
    ) -> Tuple[List[PortfolioExperience], List[TemporalPattern]]:
        """
        Get relevant portfolio experiences and temporal patterns.

        Args:
            current_state: Current state for similarity matching.
            market_regime: Optional market regime to filter by.
            assets: Optional list of assets to filter by.
            num_experiences: Number of experiences to retrieve.
            include_temporal_patterns: Whether to include temporal patterns.

        Returns:
            Tuple of (experiences, temporal patterns).
        """
        # Get experiences from portfolio memory
        experiences = self.portfolio_memory.get_portfolio_experiences(
            current_state=current_state,
            market_regime=market_regime,
            assets=assets,
            num_experiences=num_experiences
        )

        # Get temporal patterns if enabled
        patterns = []
        if include_temporal_patterns and self.config.enable_temporal_patterns:
            # Filter patterns by market regime
            if market_regime:
                patterns = [p for p in self.temporal_patterns["portfolio"] if p.market_regime == market_regime]
            else:
                patterns = self.temporal_patterns["portfolio"]

            # Sort by cumulative reward (descending)
            patterns.sort(key=lambda p: p.cumulative_reward, reverse=True)

            # Limit number of patterns
            patterns = patterns[:3]  # Limit to top 3 patterns

        return experiences, patterns

    def get_portfolio_experiences_random(
        self,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        assets: Optional[List[str]] = None,
        num_experiences: int = 5,
        include_temporal_patterns: bool = True,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[PortfolioExperience], List[TemporalPattern]]:
        experiences = list(self.portfolio_memory.portfolio_experiences)
        if market_regime:
            experiences = [exp for exp in experiences if exp.metadata.market_regime == market_regime]
        if assets:
            experiences = [exp for exp in experiences if any(asset in exp.assets for asset in assets)]
        if not experiences:
            experiences = list(self.portfolio_memory.portfolio_experiences)
        patterns: List[TemporalPattern] = []
        if include_temporal_patterns and self.config.enable_temporal_patterns:
            patterns = list(self.temporal_patterns["portfolio"])
            if market_regime:
                patterns = [p for p in patterns if p.market_regime == market_regime]
        chooser = rng or random
        return chooser.sample(experiences, min(num_experiences, len(experiences))), chooser.sample(patterns, min(3, len(patterns)))

    def get_asset_experiences(
        self,
        asset: str,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        num_experiences: int = 5,
        include_temporal_patterns: bool = True
    ) -> Tuple[List[AssetExperience], List[TemporalPattern]]:
        """
        Get relevant asset experiences and temporal patterns.

        Args:
            asset: Asset ticker.
            current_state: Current state for similarity matching.
            market_regime: Optional market regime to filter by.
            num_experiences: Number of experiences to retrieve.
            include_temporal_patterns: Whether to include temporal patterns.

        Returns:
            Tuple of (experiences, temporal patterns).
        """
        # Get experiences from asset memory
        experiences = self.asset_memory.get_asset_experiences(
            asset=asset,
            current_state=current_state,
            market_regime=market_regime,
            num_experiences=num_experiences
        )

        # Get temporal patterns if enabled
        patterns = []
        if include_temporal_patterns and self.config.enable_temporal_patterns:
            # Check if asset has patterns
            if asset in self.temporal_patterns["asset"]:
                # Filter patterns by market regime
                if market_regime:
                    patterns = [p for p in self.temporal_patterns["asset"][asset] if p.market_regime == market_regime]
                else:
                    patterns = self.temporal_patterns["asset"][asset]

                # Sort by cumulative reward (descending)
                patterns.sort(key=lambda p: p.cumulative_reward, reverse=True)

                # Limit number of patterns
                patterns = patterns[:3]  # Limit to top 3 patterns

        return experiences, patterns

    def get_asset_experiences_random(
        self,
        asset: str,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        num_experiences: int = 5,
        include_temporal_patterns: bool = True,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[AssetExperience], List[TemporalPattern]]:
        experiences = list(self.asset_memory.asset_experiences.get(asset, []))
        if market_regime:
            experiences = [exp for exp in experiences if exp.metadata.market_regime == market_regime]
        if not experiences:
            experiences = list(self.asset_memory.asset_experiences.get(asset, []))
        patterns: List[TemporalPattern] = []
        if include_temporal_patterns and self.config.enable_temporal_patterns:
            patterns = list(self.temporal_patterns["asset"].get(asset, []))
            if market_regime:
                patterns = [p for p in patterns if p.market_regime == market_regime]
        chooser = rng or random
        return chooser.sample(experiences, min(num_experiences, len(experiences))), chooser.sample(patterns, min(3, len(patterns)))

    def get_hedging_experiences(
        self,
        asset: str,
        strategy_type: str,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        num_experiences: int = 5,
        include_temporal_patterns: bool = True
    ) -> Tuple[List[HedgingExperience], List[TemporalPattern]]:
        """
        Get relevant hedging experiences and temporal patterns.

        Args:
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            current_state: Current state for similarity matching.
            market_regime: Optional market regime to filter by.
            num_experiences: Number of experiences to retrieve.
            include_temporal_patterns: Whether to include temporal patterns.

        Returns:
            Tuple of (experiences, temporal patterns).
        """
        # Get experiences from hedging memory
        experiences = self.hedging_memory.get_hedging_experiences(
            asset=asset,
            strategy_type=strategy_type,
            current_state=current_state,
            market_regime=market_regime,
            num_experiences=num_experiences
        )

        # Get temporal patterns if enabled
        patterns = []
        if include_temporal_patterns and self.config.enable_temporal_patterns:
            # Check if asset and strategy have patterns
            if asset in self.temporal_patterns["hedging"] and strategy_type in self.temporal_patterns["hedging"][asset]:
                # Filter patterns by market regime
                if market_regime:
                    patterns = [
                        p for p in self.temporal_patterns["hedging"][asset][strategy_type]
                        if p.market_regime == market_regime
                    ]
                else:
                    patterns = self.temporal_patterns["hedging"][asset][strategy_type]

                # Sort by cumulative reward (descending)
                patterns.sort(key=lambda p: p.cumulative_reward, reverse=True)

                # Limit number of patterns
                patterns = patterns[:3]  # Limit to top 3 patterns

        return experiences, patterns

    def get_hedging_experiences_random(
        self,
        asset: str,
        strategy_type: str,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        num_experiences: int = 5,
        include_temporal_patterns: bool = True,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[HedgingExperience], List[TemporalPattern]]:
        experiences = list(self.hedging_memory.hedging_experiences.get(asset, {}).get(strategy_type, []))
        if market_regime:
            experiences = [exp for exp in experiences if exp.metadata.market_regime == market_regime]
        if not experiences:
            experiences = list(self.hedging_memory.hedging_experiences.get(asset, {}).get(strategy_type, []))
        patterns: List[TemporalPattern] = []
        if include_temporal_patterns and self.config.enable_temporal_patterns:
            patterns = list(self.temporal_patterns["hedging"].get(asset, {}).get(strategy_type, []))
            if market_regime:
                patterns = [p for p in patterns if p.market_regime == market_regime]
        chooser = rng or random
        return chooser.sample(experiences, min(num_experiences, len(experiences))), chooser.sample(patterns, min(3, len(patterns)))

    def get_cross_level_experiences(
        self,
        date_val: date,
        market_regime: Optional[str] = None
    ) -> List[CrossLevelExperience]:
        """
        Get cross-level experiences for a specific date.

        Args:
            date_val: Date to get experiences for.
            market_regime: Optional market regime to filter by.

        Returns:
            List of cross-level experiences.
        """
        if not self.config.enable_cross_level_retrieval:
            return []

        # Filter by date
        experiences = [e for e in self.cross_level_experiences if e.date == date_val]

        # Filter by market regime if specified
        if market_regime:
            experiences = [e for e in experiences if e.market_regime == market_regime]

        return experiences

    def get_experience_by_id(self, experience_id: str, level: Literal["portfolio", "asset", "hedging"]) -> Optional[Experience]:
        """
        Get an experience by ID and level.

        Args:
            experience_id: ID of the experience.
            level: Level of the experience.

        Returns:
            Experience if found, None otherwise.
        """
        if level == "portfolio":
            return self.portfolio_memory.get_experience_by_id(experience_id)
        elif level == "asset":
            return self.asset_memory.get_experience_by_id(experience_id)
        elif level == "hedging":
            return self.hedging_memory.get_experience_by_id(experience_id)

        return None

    def generate_pattern_descriptions(self) -> None:
        """
        Generate descriptions for temporal patterns using LLM.
        """
        if not self.llm_client:
            logger.warning("LLM client not provided, cannot generate pattern descriptions")
            return

        # Generate descriptions for portfolio patterns
        for pattern in self.temporal_patterns["portfolio"]:
            if not pattern.description:
                self._generate_pattern_description(pattern)

        # Generate descriptions for asset patterns
        for asset, patterns in self.temporal_patterns["asset"].items():
            for pattern in patterns:
                if not pattern.description:
                    self._generate_pattern_description(pattern)

        # Generate descriptions for hedging patterns
        for asset, strategies in self.temporal_patterns["hedging"].items():
            for strategy, patterns in strategies.items():
                for pattern in patterns:
                    if not pattern.description:
                        self._generate_pattern_description(pattern)

    def _generate_pattern_description(self, pattern: TemporalPattern) -> None:
        """
        Generate a description for a temporal pattern.

        Args:
            pattern: Temporal pattern to generate description for.
        """
        try:
            # Get experiences in the pattern
            experiences = []
            for exp_id in pattern.experience_ids:
                exp = self.get_experience_by_id(exp_id, pattern.level)
                if exp:
                    experiences.append(exp)

            if not experiences:
                return

            # Create prompt
            prompt = f"""
You are analyzing a sequence of trading decisions and outcomes in a {pattern.level}-level strategy during a {pattern.market_regime} market regime from {pattern.start_date} to {pattern.end_date}.

The cumulative reward for this sequence was {pattern.cumulative_reward:.4f}.

Please provide a brief description (2-3 sentences) of the pattern you observe in these experiences, focusing on what worked well or didn't work well, and any lessons that can be learned.

Experiences:
"""

            for i, exp in enumerate(experiences):
                prompt += f"\nExperience {i+1}:\n{exp.to_text()}\n"

            # Generate description
            description = self.llm_client.generate_text(prompt, max_tokens=100)

            # Clean up description
            description = description.strip()

            # Update pattern
            pattern.description = description

        except Exception as e:
            logger.error(f"Error generating pattern description: {str(e)}")

    def generate_cross_level_descriptions(self) -> None:
        """
        Generate descriptions for cross-level experiences using LLM.
        """
        if not self.llm_client:
            logger.warning("LLM client not provided, cannot generate cross-level descriptions")
            return

        for cross_level in self.cross_level_experiences:
            if not cross_level.description:
                self._generate_cross_level_description(cross_level)

    def _generate_cross_level_description(self, cross_level: CrossLevelExperience) -> None:
        """
        Generate a description for a cross-level experience.

        Args:
            cross_level: Cross-level experience to generate description for.
        """
        try:
            # Get experiences
            experiences = []

            # Portfolio experience
            if cross_level.portfolio_experience_id:
                portfolio_exp = self.portfolio_memory.get_experience_by_id(cross_level.portfolio_experience_id)
                if portfolio_exp:
                    experiences.append(("portfolio", portfolio_exp))

            # Asset experiences
            for asset, exp_id in cross_level.asset_experience_ids.items():
                asset_exp = self.asset_memory.get_experience_by_id(exp_id)
                if asset_exp:
                    experiences.append((f"asset ({asset})", asset_exp))

            # Hedging experiences
            for asset, strategies in cross_level.hedging_experience_ids.items():
                for strategy, exp_id in strategies.items():
                    hedging_exp = self.hedging_memory.get_experience_by_id(exp_id)
                    if hedging_exp:
                        experiences.append((f"hedging ({asset}, {strategy})", hedging_exp))

            if not experiences:
                return

            # Create prompt
            prompt = f"""
You are analyzing a set of trading decisions and outcomes across different levels (portfolio, asset, hedging) on {cross_level.date} during a {cross_level.market_regime} market regime.

Please provide a brief description (2-3 sentences) of how decisions at different levels interacted with each other, focusing on any synergies or conflicts between levels, and any lessons that can be learned.

Experiences:
"""

            for i, (level_desc, exp) in enumerate(experiences):
                prompt += f"\n{level_desc.capitalize()} Experience:\n{exp.to_text()}\n"

            # Generate description
            description = self.llm_client.generate_text(prompt, max_tokens=100)

            # Clean up description
            description = description.strip()

            # Update cross-level experience
            cross_level.description = description

        except Exception as e:
            logger.error(f"Error generating cross-level description: {str(e)}")

    def save_to_disk(self) -> bool:
        """
        Save all memory structures to disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Save memory stores
            self.portfolio_memory.save_to_disk()
            self.asset_memory.save_to_disk()
            self.hedging_memory.save_to_disk()

            # Save temporal patterns
            patterns_path = os.path.join(self.config.memory_dir, "temporal_patterns.json")
            with open(patterns_path, 'w') as f:
                # Convert to serializable format
                patterns_dict = {
                    "portfolio": [p.dict() for p in self.temporal_patterns["portfolio"]],
                    "asset": {
                        asset: [p.dict() for p in patterns]
                        for asset, patterns in self.temporal_patterns["asset"].items()
                    },
                    "hedging": {
                        asset: {
                            strategy: [p.dict() for p in patterns]
                            for strategy, patterns in strategies.items()
                        }
                        for asset, strategies in self.temporal_patterns["hedging"].items()
                    }
                }
                json.dump(patterns_dict, f, default=self._json_serializer)

            # Save cross-level experiences
            cross_level_path = os.path.join(self.config.memory_dir, "cross_level_experiences.json")
            with open(cross_level_path, 'w') as f:
                json.dump([e.dict() for e in self.cross_level_experiences], f, default=self._json_serializer)

            logger.info(f"Saved hierarchical memory to {self.config.memory_dir}")
            return True

        except Exception as e:
            logger.error(f"Error saving hierarchical memory: {str(e)}")
            return False

    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def load_from_disk(self) -> bool:
        """
        Load all memory structures from disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Load memory stores
            self.portfolio_memory.load_from_disk()
            self.asset_memory.load_from_disk()
            self.hedging_memory.load_from_disk()

            # Load temporal patterns
            patterns_path = os.path.join(self.config.memory_dir, "temporal_patterns.json")
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r') as f:
                    patterns_dict = json.load(f)

                    # Reset patterns
                    self.temporal_patterns = {
                        "portfolio": [],
                        "asset": defaultdict(list),
                        "hedging": defaultdict(lambda: defaultdict(list))
                    }

                    # Load portfolio patterns
                    for p_dict in patterns_dict.get("portfolio", []):
                        self.temporal_patterns["portfolio"].append(TemporalPattern(**p_dict))

                    # Load asset patterns
                    for asset, patterns in patterns_dict.get("asset", {}).items():
                        for p_dict in patterns:
                            self.temporal_patterns["asset"][asset].append(TemporalPattern(**p_dict))

                    # Load hedging patterns
                    for asset, strategies in patterns_dict.get("hedging", {}).items():
                        for strategy, patterns in strategies.items():
                            for p_dict in patterns:
                                self.temporal_patterns["hedging"][asset][strategy].append(TemporalPattern(**p_dict))

            # Load cross-level experiences
            cross_level_path = os.path.join(self.config.memory_dir, "cross_level_experiences.json")
            if os.path.exists(cross_level_path):
                with open(cross_level_path, 'r') as f:
                    experiences = json.load(f)

                    # Reset cross-level experiences
                    self.cross_level_experiences = []

                    # Load experiences
                    for e_dict in experiences:
                        self.cross_level_experiences.append(CrossLevelExperience(**e_dict))

            logger.info(f"Loaded hierarchical memory from {self.config.memory_dir}")
            return True

        except Exception as e:
            logger.error(f"Error loading hierarchical memory: {str(e)}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hierarchical memory structure.

        Returns:
            Dictionary of memory statistics.
        """
        # Get stats from memory stores
        portfolio_stats = self.portfolio_memory.get_memory_stats()
        asset_stats = self.asset_memory.get_memory_stats()
        hedging_stats = self.hedging_memory.get_memory_stats()

        # Count temporal patterns
        portfolio_patterns = len(self.temporal_patterns["portfolio"])

        asset_patterns = {}
        for asset, patterns in self.temporal_patterns["asset"].items():
            asset_patterns[asset] = len(patterns)

        hedging_patterns = {}
        for asset, strategies in self.temporal_patterns["hedging"].items():
            hedging_patterns[asset] = {}
            for strategy, patterns in strategies.items():
                hedging_patterns[asset][strategy] = len(patterns)

        # Count cross-level experiences
        cross_level_count = len(self.cross_level_experiences)

        # Return stats
        return {
            "portfolio_memory": portfolio_stats,
            "asset_memory": asset_stats,
            "hedging_memory": hedging_stats,
            "temporal_patterns": {
                "portfolio": portfolio_patterns,
                "asset": asset_patterns,
                "hedging": hedging_patterns
            },
            "cross_level_experiences": cross_level_count
        }
