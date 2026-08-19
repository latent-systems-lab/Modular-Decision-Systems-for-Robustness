# src/agents/base_agent.py
"""
Abstract base class for hedging or trading agents, each using LLM logic.
Provides a common interface and shared functionality for performance tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, agent_id: str) -> None:
        """
        Initialize the agent with a unique ID and type.
        Also initializes a performance metric (for future RL integration).

        :param agent_id: Unique identifier for the agent.
        """
        self.agent_id = agent_id
        self.performance = 0.0

    @abstractmethod
    def get_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Must be implemented by subclasses.
        Should return a recommended trade action as a dictionary.

        :param state: A dictionary representing the current state (market data, options data, etc.).
        :return: A dictionary with the recommended trade action.
        """
        pass

    def update_performance(self, reward: float) -> None:
        """
        Update the agent's performance metric by adding the provided reward.

        :param reward: The reward value to add.
        """
        self.performance += reward

    def get_performance(self) -> float:
        """
        Retrieve the current performance metric for the agent.

        :return: Current performance as a float.
        """
        return self.performance
