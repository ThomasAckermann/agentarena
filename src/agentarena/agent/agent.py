"""
Base agent interface for AgentArena.
"""

from agentarena.game.action import Action
from agentarena.models.observations import GameObservation


class Agent:
    """
    Base class for all agents in the game.

    This interface defines the methods that all agent implementations
    must provide.
    """

    def __init__(self, name: str = "Agent") -> None:
        """
        Initialize the agent.

        Args:
            name: Agent name for display and logging
        """
        self.name: str = name

    def reset(self) -> None:
        """Called at the start of a stage to reset agent state."""

    def get_action(self, observation: GameObservation) -> Action:
        """
        Determine the next action based on the current game state.

        Args:
            observation: Current game state observation

        Returns:
            Action: The action to take

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Agent subclasses must implement get_action")
