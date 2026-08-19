"""
Experience Memory Store for memory-based reinforcement learning.
Stores and retrieves experiences for LLM-based decision making without weight updates.
"""

import os
import json
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union, Literal, Tuple, Any
import logging
from pydantic import BaseModel, Field
import uuid
import re
from collections import defaultdict
try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperienceMetadata(BaseModel):
    """Metadata for an experience entry."""
    timestamp: datetime = Field(default_factory=datetime.now)
    market_regime: str = "neutral"
    confidence: float = 0.0
    tags: List[str] = []
    importance: float = 1.0  # Higher values indicate more important experiences

class Experience(BaseModel):
    """Base class for all experience entries."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: date
    level: Literal["portfolio", "asset", "hedging"]
    state: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    reflection: Optional[str] = None
    metadata: Optional[ExperienceMetadata] = None

    def to_text(self) -> str:
        """Convert experience to text format for inclusion in prompts."""
        state_text = self._dict_to_text(self.state, "State")
        action_text = self._dict_to_text(self.action, "Action")
        outcome_text = self._dict_to_text(self.outcome, "Outcome")
        text = ""
        if self.metadata:
            text = f"--- Experience from {self.date} (Market Regime: {self.metadata.market_regime}) ---\n"
        text += state_text + "\n"
        text += action_text + "\n"
        text += outcome_text + "\n"

        if self.reflection:
            text += f"Reflection: {self.reflection}\n"

        return text

    def _dict_to_text(self, d: Dict[str, Any], section_name: str) -> str:
        """Convert a dictionary to a formatted text string."""
        text = f"{section_name}:\n"
        for key, value in d.items():
            # Format key from snake_case to Title Case
            formatted_key = ' '.join(word.capitalize() for word in key.split('_'))

            # Format value based on type
            if isinstance(value, dict):
                text += f"  {formatted_key}:\n"
                for sub_key, sub_value in value.items():
                    sub_formatted_key = ' '.join(word.capitalize() for word in sub_key.split('_'))
                    text += f"    {sub_formatted_key}: {sub_value}\n"
            elif isinstance(value, list):
                text += f"  {formatted_key}: {', '.join(str(item) for item in value)}\n"
            elif isinstance(value, float):
                text += f"  {formatted_key}: {value:.4f}\n"
            else:
                text += f"  {formatted_key}: {value}\n"

        return text

class PortfolioExperience(Experience):
    """Experience entry for portfolio-level decisions."""
    level: Literal["portfolio"] = "portfolio"
    assets: List[str] = []

    def to_text(self) -> str:
        """Convert portfolio experience to text format."""
        base_text = super().to_text()
        assets_text = f"Assets: {', '.join(self.assets)}\n"
        return f"{assets_text}{base_text}"

class AssetExperience(Experience):
    """Experience entry for asset-level decisions."""
    level: Literal["asset"] = "asset"
    asset: str

    def to_text(self) -> str:
        """Convert asset experience to text format."""
        base_text = super().to_text()
        return f"Asset: {self.asset}\n{base_text}"

class HedgingExperience(Experience):
    """Experience entry for hedging-level decisions."""
    level: Literal["hedging"] = "hedging"
    asset: str
    strategy_type: str

    def to_text(self) -> str:
        """Convert hedging experience to text format."""
        base_text = super().to_text()
        return f"Asset: {self.asset}, Strategy: {self.strategy_type}\n{base_text}"

class MemoryConfig(BaseModel):
    """Configuration for the experience memory store."""
    max_portfolio_experiences: int = 1000
    max_asset_experiences: int = 500
    max_hedging_experiences: int = 300
    recency_weight: float = 0.6
    similarity_weight: float = 0.3
    outcome_weight: float = 0.1
    use_embeddings: bool = True
    embedding_dimension: int = 768
    default_experiences_to_retrieve: int = 5
    memory_dir: str = "memory"
    reflection_enabled: bool = True

class ExperienceMemoryStore:
    """
    Memory store for experiences in memory-based reinforcement learning.
    Stores and retrieves experiences for LLM-based decision making.
    """

    def __init__(self, config: MemoryConfig = None):
        """
        Initialize the experience memory store.

        Args:
            config: Configuration for the memory store.
        """
        self.config = config or MemoryConfig()

        # Create memory directory if it doesn't exist
        os.makedirs(self.config.memory_dir, exist_ok=True)

        # Initialize memory stores
        self.portfolio_experiences: List[PortfolioExperience] = []
        self.asset_experiences: Dict[str, List[AssetExperience]] = defaultdict(list)
        self.hedging_experiences: Dict[str, Dict[str, List[HedgingExperience]]] = defaultdict(lambda: defaultdict(list))
        self.embedder = None
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                logger.warning("SentenceTransformer backend unavailable. Falling back to non-embedding retrieval.")

        if self.embedder is None:
            self.config.use_embeddings = False

        # Initialize embedding index if enabled
        self.embedding_index = None
        if self.config.use_embeddings:
            self._initialize_embedding_index()

    def _initialize_embedding_index(self):
        """Initialize the embedding index for similarity search."""
        try:
            if faiss is None:
                raise ImportError("faiss is not installed")
            self.embedding_index = {
                "portfolio": {
                    "index": faiss.IndexFlatL2(self.config.embedding_dimension),
                    "ids": []
                },
                "asset": {
                    "index": faiss.IndexFlatL2(self.config.embedding_dimension),
                    "ids": []
                },
                "hedging": {
                    "index": faiss.IndexFlatL2(self.config.embedding_dimension),
                    "ids": []
                }
            }
            logger.info("Embedding index initialized successfully.")
        except ImportError:
            logger.warning("FAISS not available. Similarity search will use simpler methods.")
            self.config.use_embeddings = False

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
        # Create metadata
        metadata = ExperienceMetadata(
            timestamp=datetime.now(),
            market_regime=market_regime,
            confidence=confidence,
            tags=tags,
            importance=self._calculate_importance(outcome)
        )

        # Create experience
        experience = PortfolioExperience(
            date=date_val,
            assets=assets,
            state=state,
            action=action,
            outcome=outcome,
            reflection=reflection,
            metadata=metadata
        )

        # Add to memory
        self.portfolio_experiences.append(experience)

        # Trim if needed
        if len(self.portfolio_experiences) > self.config.max_portfolio_experiences:
            # Sort by importance and recency
            self.portfolio_experiences.sort(
                key=lambda x: (x.metadata.importance, (datetime.now() - x.metadata.timestamp).total_seconds()),
                reverse=True
            )
            self.portfolio_experiences = self.portfolio_experiences[:self.config.max_portfolio_experiences]

        # Add to embedding index if enabled
        if self.config.use_embeddings and self.embedding_index:
            self._add_to_embedding_index(experience, "portfolio")

        logger.info(f"Added portfolio experience {experience.id} for date {date_val}")
        return experience.id

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
        # Create metadata
        metadata = ExperienceMetadata(
            timestamp=datetime.now(),
            market_regime=market_regime,
            confidence=confidence,
            tags=tags,
            importance=self._calculate_importance(outcome)
        )

        # Create experience
        experience = AssetExperience(
            date=date_val,
            asset=asset,
            state=state,
            action=action,
            outcome=outcome,
            reflection=reflection,
            metadata=metadata
        )

        # Add to memory
        self.asset_experiences[asset].append(experience)

        # Trim if needed
        if len(self.asset_experiences[asset]) > self.config.max_asset_experiences:
            # Sort by importance and recency
            self.asset_experiences[asset].sort(
                key=lambda x: (x.metadata.importance, (datetime.now() - x.metadata.timestamp).total_seconds()),
                reverse=True
            )
            self.asset_experiences[asset] = self.asset_experiences[asset][:self.config.max_asset_experiences]

        # Add to embedding index if enabled
        if self.config.use_embeddings and self.embedding_index:
            self._add_to_embedding_index(experience, "asset")

        logger.info(f"Added asset experience {experience.id} for {asset} on date {date_val}")
        return experience.id

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
        # Create metadata
        metadata = ExperienceMetadata(
            timestamp=datetime.now(),
            market_regime=market_regime,
            confidence=confidence,
            tags=tags,
            importance=self._calculate_importance(outcome)
        )

        # Create experience
        experience = HedgingExperience(
            date=date_val,
            asset=asset,
            strategy_type=strategy_type,
            state=state,
            action=action,
            outcome=outcome,
            reflection=reflection,
            metadata=metadata
        )

        # Add to memory
        self.hedging_experiences[asset][strategy_type].append(experience)

        # Trim if needed
        if len(self.hedging_experiences[asset][strategy_type]) > self.config.max_hedging_experiences:
            # Sort by importance and recency
            self.hedging_experiences[asset][strategy_type].sort(
                key=lambda x: (x.metadata.importance, (datetime.now() - x.metadata.timestamp).total_seconds()),
                reverse=True
            )
            self.hedging_experiences[asset][strategy_type] = self.hedging_experiences[asset][strategy_type][:self.config.max_hedging_experiences]

        # Add to embedding index if enabled
        if self.config.use_embeddings and self.embedding_index:
            self._add_to_embedding_index(experience, "hedging")

        logger.info(f"Added hedging experience {experience.id} for {asset} {strategy_type} on date {date_val}")
        return experience.id

    def _calculate_importance(self, outcome: Dict[str, Any]) -> float:
        """
        Calculate importance score for an experience based on its outcome.

        Args:
            outcome: Outcome dictionary.

        Returns:
            Importance score between 0 and 2 (higher is more important).
        """
        # This is a simple implementation - in practice, you'd use more sophisticated logic
        # based on the specific metrics in your outcome dictionary

        # Check for extreme values (very good or very bad outcomes)
        importance = 1.0  # Default importance

        # Look for return-related keys
        for key in ['return', 'returns', 'profit', 'pnl', 'performance']:
            if key in outcome:
                value = outcome[key]
                if isinstance(value, (int, float)):
                    # Higher importance for extreme returns (positive or negative)
                    importance += min(1.0, abs(value))

        # Look for risk-related keys
        for key in ['risk', 'volatility', 'drawdown', 'loss']:
            if key in outcome:
                value = outcome[key]
                if isinstance(value, (int, float)):
                    # Higher importance for high risk events
                    importance += min(1.0, abs(value))

        # Cap at 2.0
        return min(2.0, importance)

    def _add_to_embedding_index(self, experience: Experience, level: str):
        """
        Add an experience to the embedding index.

        Args:
            experience: Experience to add.
            level: Level of the experience.
        """
        if not self.embedding_index:
            return

        # Generate embedding
        embedding = self._generate_embedding(experience)

        # Add to index
        try:
            if level not in self.embedding_index:
                raise KeyError(f"Embedding index level '{level}' is not initialized")

            vector = np.array([embedding], dtype=np.float32)
            if vector.shape[1] != self.config.embedding_dimension:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {self.config.embedding_dimension}, got {vector.shape[1]}"
                )

            self.embedding_index[level]["index"].add(vector)
            self.embedding_index[level]["ids"].append(experience.id)
        except KeyError as exc:
            logger.warning(f"Skipping embedding update: {exc}")
        except ValueError as exc:
            logger.warning(f"Could not add embedding for experience {experience.id}: {exc}")
        except Exception as exc:
            if faiss is not None and isinstance(exc, faiss.FaissError):
                logger.warning(
                    "FAISS error while adding experience %s to embedding index: %s",
                    experience.id,
                    exc,
                )
                return
            logger.exception(
                "Unexpected error while adding experience %s to embedding index",
                experience.id,
            )
            raise

    def _generate_embedding(self, experience: Experience) -> np.ndarray:
        """
        Generate an embedding for an experience.

        Args:
            experience: Experience to generate embedding for.

        Returns:
            Embedding vector.
        """
        if self.embedder is None:
            return np.zeros(self.config.embedding_dimension, dtype=np.float32)
        try:
            return self.embedder.encode(experience.to_text())
        except Exception as e:
            return np.random.randn(self.config.embedding_dimension).astype(np.float32)

    def get_portfolio_experiences(
        self,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        assets: Optional[List[str]] = None,
        num_experiences: int = None,
        recency_days: Optional[int] = None
    ) -> List[PortfolioExperience]:
        """
        Get relevant portfolio experiences.

        Args:
            current_state: Current state for similarity matching.
            market_regime: Optional market regime to filter by.
            assets: Optional list of assets to filter by.
            num_experiences: Number of experiences to retrieve.
            recency_days: Optional maximum age of experiences in days.

        Returns:
            List of relevant portfolio experiences.
        """
        if not self.portfolio_experiences:
            return []

        num_experiences = num_experiences or self.config.default_experiences_to_retrieve

        # Filter by market regime if specified
        filtered_experiences = self.portfolio_experiences
        if market_regime:
            filtered_experiences = [exp for exp in filtered_experiences if exp.metadata.market_regime == market_regime]

        # Filter by assets if specified
        if assets:
            filtered_experiences = [
                exp for exp in filtered_experiences
                if any(asset in exp.assets for asset in assets)
            ]

        # Filter by recency if specified
        if recency_days:
            cutoff_date = datetime.now() - timedelta(days=recency_days)
            filtered_experiences = [
                exp for exp in filtered_experiences
                if exp.metadata.timestamp >= cutoff_date
            ]

        # If no experiences left after filtering, return original list
        if not filtered_experiences:
            filtered_experiences = self.portfolio_experiences

        # If using embeddings, use similarity search
        if self.config.use_embeddings and self.embedding_index:
            return self._get_similar_experiences_embedding(
                current_state,
                filtered_experiences,
                "portfolio",
                num_experiences
            )

        # Otherwise, use simpler ranking
        return self._rank_experiences(current_state, filtered_experiences, num_experiences)

    def get_asset_experiences(
        self,
        asset: str,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        num_experiences: int = None,
        recency_days: Optional[int] = None
    ) -> List[AssetExperience]:
        """
        Get relevant asset experiences.

        Args:
            asset: Asset ticker.
            current_state: Current state for similarity matching.
            market_regime: Optional market regime to filter by.
            num_experiences: Number of experiences to retrieve.
            recency_days: Optional maximum age of experiences in days.

        Returns:
            List of relevant asset experiences.
        """
        if asset not in self.asset_experiences or not self.asset_experiences[asset]:
            return []

        num_experiences = num_experiences or self.config.default_experiences_to_retrieve

        # Filter by market regime if specified
        filtered_experiences = self.asset_experiences[asset]
        if market_regime:
            filtered_experiences = [exp for exp in filtered_experiences if exp.metadata.market_regime == market_regime]

        # Filter by recency if specified
        if recency_days:
            cutoff_date = datetime.now() - timedelta(days=recency_days)
            filtered_experiences = [
                exp for exp in filtered_experiences
                if exp.metadata.timestamp >= cutoff_date
            ]

        # If no experiences left after filtering, return original list
        if not filtered_experiences:
            filtered_experiences = self.asset_experiences[asset]

        # If using embeddings, use similarity search
        if self.config.use_embeddings and self.embedding_index:
            return self._get_similar_experiences_embedding(
                current_state,
                filtered_experiences,
                "asset",
                num_experiences
            )

        # Otherwise, use simpler ranking
        return self._rank_experiences(current_state, filtered_experiences, num_experiences)

    def get_hedging_experiences(
        self,
        asset: str,
        strategy_type: str,
        current_state: Dict[str, Any],
        market_regime: Optional[str] = None,
        num_experiences: int = None,
        recency_days: Optional[int] = None
    ) -> List[HedgingExperience]:
        """
        Get relevant hedging experiences.

        Args:
            asset: Asset ticker.
            strategy_type: Type of hedging strategy.
            current_state: Current state for similarity matching.
            market_regime: Optional market regime to filter by.
            num_experiences: Number of experiences to retrieve.
            recency_days: Optional maximum age of experiences in days.

        Returns:
            List of relevant hedging experiences.
        """
        if asset not in self.hedging_experiences or strategy_type not in self.hedging_experiences[asset] or not self.hedging_experiences[asset][strategy_type]:
            return []

        num_experiences = num_experiences or self.config.default_experiences_to_retrieve

        # Filter by market regime if specified
        filtered_experiences = self.hedging_experiences[asset][strategy_type]
        if market_regime:
            filtered_experiences = [exp for exp in filtered_experiences if exp.metadata.market_regime == market_regime]

        # Filter by recency if specified
        if recency_days:
            cutoff_date = datetime.now() - timedelta(days=recency_days)
            filtered_experiences = [
                exp for exp in filtered_experiences
                if exp.metadata.timestamp >= cutoff_date
            ]

        # If no experiences left after filtering, return original list
        if not filtered_experiences:
            filtered_experiences = self.hedging_experiences[asset][strategy_type]

        # If using embeddings, use similarity search
        if self.config.use_embeddings and self.embedding_index:
            return self._get_similar_experiences_embedding(
                current_state,
                filtered_experiences,
                "hedging",
                num_experiences
            )

        # Otherwise, use simpler ranking
        return self._rank_experiences(current_state, filtered_experiences, num_experiences)

    def _get_similar_experiences_embedding(
        self,
        current_state: Dict[str, Any],
        experiences: List[Experience],
        level: str,
        num_experiences: int
    ) -> List[Experience]:
        """
        Get similar experiences using embedding-based similarity.

        Args:
            current_state: Current state for similarity matching.
            experiences: List of experiences to search.
            level: Level of the experiences.
            num_experiences: Number of experiences to retrieve.

        Returns:
            List of similar experiences.
        """
        if not self.embedding_index or not experiences:
            return experiences[:num_experiences]

        # Create a temporary experience for embedding generation
        temp_experience = Experience(
            date=date.today(),
            level=level,
            state=current_state,
            action={},
            outcome={}
        )

        # Generate embedding
        query_embedding = self._generate_embedding(temp_experience)

        # Search index
        distances, indices = self.embedding_index[level]["index"].search(
            np.array([query_embedding], dtype=np.float32),
            min(len(experiences), num_experiences * 2)  # Get more than needed for filtering
        )

        # Get experience IDs
        if self.embedding_index[level]["ids"] == []:
            return experiences[:num_experiences]
        experience_ids = [self.embedding_index[level]["ids"][i] for i in indices[0]]

        # Map back to experiences
        id_to_experience = {exp.id: exp for exp in experiences}
        similar_experiences = [id_to_experience[exp_id] for exp_id in experience_ids if exp_id in id_to_experience]

        # Return top N
        return similar_experiences[:num_experiences]

    def _rank_experiences(
        self,
        current_state: Dict[str, Any],
        experiences: List[Experience],
        num_experiences: int
    ) -> List[Experience]:
        """
        Rank experiences by a combination of recency, similarity, and outcome.

        Args:
            current_state: Current state for similarity matching.
            experiences: List of experiences to rank.
            num_experiences: Number of experiences to retrieve.

        Returns:
            List of ranked experiences.
        """
        if not experiences:
            return []

        # Calculate scores
        scores = []
        for exp in experiences:
            # Recency score (newer is better)
            recency_score = 1.0 - min(1.0, (datetime.now() - exp.metadata.timestamp).total_seconds() / (365 * 24 * 60 * 60))

            # Similarity score (placeholder - in practice, you'd use more sophisticated similarity)
            similarity_score = self._calculate_similarity(current_state, exp.state)

            # Outcome score (higher importance is better)
            outcome_score = exp.metadata.importance / 2.0  # Normalize to 0-1

            # Combined score
            combined_score = (
                self.config.recency_weight * recency_score +
                self.config.similarity_weight * similarity_score +
                self.config.outcome_weight * outcome_score
            )

            scores.append((exp, combined_score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top N experiences
        return [exp for exp, _ in scores[:num_experiences]]

    def _calculate_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two states.

        Args:
            state1: First state.
            state2: Second state.

        Returns:
            Similarity score between 0 and 1.
        """
        # This is a simple implementation - in practice, you'd use more sophisticated
        # similarity measures based on the specific structure of your state

        # Find common keys
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0

        # Calculate similarity for each common key
        similarities = []
        for key in common_keys:
            val1 = state1[key]
            val2 = state2[key]

            # Calculate similarity based on type
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
                else:
                    similarities.append(1.0)  # Both are zero
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (exact match only)
                similarities.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, dict) and isinstance(val2, dict):
                # Recursive similarity for nested dictionaries
                similarities.append(self._calculate_similarity(val1, val2))
            else:
                # Default similarity
                similarities.append(0.5)

        # Return average similarity
        return sum(similarities) / len(similarities)

    def add_reflection(self, experience_id: str, reflection: str) -> bool:
        """
        Add or update reflection for an experience.

        Args:
            experience_id: ID of the experience.
            reflection: Reflection text.

        Returns:
            True if successful, False otherwise.
        """
        # Search in portfolio experiences
        for exp in self.portfolio_experiences:
            if exp.id == experience_id:
                exp.reflection = reflection
                return True

        # Search in asset experiences
        for asset_exps in self.asset_experiences.values():
            for exp in asset_exps:
                if exp.id == experience_id:
                    exp.reflection = reflection
                    return True

        # Search in hedging experiences
        for asset_strategies in self.hedging_experiences.values():
            for strategy_exps in asset_strategies.values():
                for exp in strategy_exps:
                    if exp.id == experience_id:
                        exp.reflection = reflection
                        return True

        logger.warning(f"Experience {experience_id} not found for adding reflection")
        return False

    def get_experience_by_id(self, experience_id: str) -> Optional[Experience]:
        """
        Get an experience by ID.

        Args:
            experience_id: ID of the experience.

        Returns:
            Experience if found, None otherwise.
        """
        # Search in portfolio experiences
        for exp in self.portfolio_experiences:
            if exp.id == experience_id:
                return exp

        # Search in asset experiences
        for asset_exps in self.asset_experiences.values():
            for exp in asset_exps:
                if exp.id == experience_id:
                    return exp

        # Search in hedging experiences
        for asset_strategies in self.hedging_experiences.values():
            for strategy_exps in asset_strategies.values():
                for exp in strategy_exps:
                    if exp.id == experience_id:
                        return exp

        return None

    def save_to_disk(self) -> bool:
        """
        Save all experiences to disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Save portfolio experiences
            portfolio_path = os.path.join(self.config.memory_dir, "portfolio_experiences.json")
            if self.portfolio_experiences != []:
                with open(portfolio_path, 'w') as f:
                    json.dump([exp.dict() for exp in self.portfolio_experiences], f, default=self._json_serializer)

            # Save asset experiences
            for asset, experiences in self.asset_experiences.items():
                asset_path = os.path.join(self.config.memory_dir, f"asset_{asset}_experiences.json")
                with open(asset_path, 'w') as f:
                    json.dump([exp.dict() for exp in experiences], f, default=self._json_serializer)

            # Save hedging experiences
            for asset, strategies in self.hedging_experiences.items():
                for strategy, experiences in strategies.items():
                    hedging_path = os.path.join(self.config.memory_dir, f"hedging_{asset}_{strategy}_experiences.json")
                    with open(hedging_path, 'w') as f:
                        json.dump([exp.dict() for exp in experiences], f, default=self._json_serializer)

            logger.info(f"Saved experiences to {self.config.memory_dir}")
            return True

        except Exception as e:
            logger.error(f"Error saving experiences: {str(e)}")
            return False

    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def load_from_disk(self) -> bool:
        """
        Load all experiences from disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Clear existing experiences
            self.portfolio_experiences = []
            self.asset_experiences = defaultdict(list)
            self.hedging_experiences = defaultdict(lambda: defaultdict(list))

            # Load portfolio experiences
            portfolio_path = os.path.join(self.config.memory_dir, "portfolio_experiences.json")
            if os.path.exists(portfolio_path):
                with open(portfolio_path, 'r') as f:
                    experiences = json.load(f)
                    for exp_dict in experiences:
                        self.portfolio_experiences.append(PortfolioExperience(**exp_dict))

            # Load asset experiences
            for file_name in os.listdir(self.config.memory_dir):
                if file_name.startswith("asset_") and file_name.endswith("_experiences.json"):
                    asset = file_name[6:-17]  # Extract asset from file name
                    asset_path = os.path.join(self.config.memory_dir, file_name)
                    with open(asset_path, 'r') as f:
                        experiences = json.load(f)
                        for exp_dict in experiences:
                            self.asset_experiences[asset].append(AssetExperience(**exp_dict))

            # Load hedging experiences
            for file_name in os.listdir(self.config.memory_dir):
                if file_name.startswith("hedging_") and file_name.endswith("_experiences.json"):
                    # Extract asset and strategy from file name
                    parts = file_name[8:-17].split("_")
                    if len(parts) >= 2:
                        asset = parts[0]
                        strategy = "_".join(parts[1:])
                        hedging_path = os.path.join(self.config.memory_dir, file_name)
                        with open(hedging_path, 'r') as f:
                            experiences = json.load(f)
                            for exp_dict in experiences:
                                self.hedging_experiences[asset][strategy].append(HedgingExperience(**exp_dict))

            # Rebuild embedding index if enabled
            if self.config.use_embeddings and self.embedding_index:
                self._rebuild_embedding_index()

            logger.info(f"Loaded experiences from {self.config.memory_dir}")
            return True

        except Exception as e:
            logger.error(f"Error loading experiences: {str(e)}")
            return False

    def _rebuild_embedding_index(self):
        """Rebuild the embedding index from all experiences."""
        if not self.config.use_embeddings or not self.embedding_index:
            return

        # Clear existing indices
        for level in self.embedding_index:
            self.embedding_index[level]["index"].reset()
            self.embedding_index[level]["ids"] = []

        # Add portfolio experiences
        for exp in self.portfolio_experiences:
            self._add_to_embedding_index(exp, "portfolio")

        # Add asset experiences
        for asset_exps in self.asset_experiences.values():
            for exp in asset_exps:
                self._add_to_embedding_index(exp, "asset")

        # Add hedging experiences
        for asset_strategies in self.hedging_experiences.values():
            for strategy_exps in asset_strategies.values():
                for exp in strategy_exps:
                    self._add_to_embedding_index(exp, "hedging")

        logger.info("Rebuilt embedding index")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.

        Returns:
            Dictionary of memory statistics.
        """
        # Count experiences
        portfolio_count = len(self.portfolio_experiences)

        asset_counts = {}
        for asset, exps in self.asset_experiences.items():
            asset_counts[asset] = len(exps)

        hedging_counts = {}
        for asset, strategies in self.hedging_experiences.items():
            hedging_counts[asset] = {}
            for strategy, exps in strategies.items():
                hedging_counts[asset][strategy] = len(exps)

        # Calculate total
        total_count = portfolio_count
        total_count += sum(asset_counts.values())
        total_count += sum(len(exps) for strategies in hedging_counts.values() for exps in strategies.values())

        # Return stats
        return {
            "total_experiences": total_count,
            "portfolio_experiences": portfolio_count,
            "asset_experiences": asset_counts,
            "hedging_experiences": hedging_counts,
            "use_embeddings": self.config.use_embeddings
        }
