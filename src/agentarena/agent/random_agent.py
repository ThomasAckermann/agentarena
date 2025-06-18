"""
Random agent that takes random actions.
"""

import random

from agentarena.agent.agent import Agent
from agentarena.models.action import Action, Direction
from agentarena.models.observations import GameObservation


class RandomAgent(Agent):
    """
    Agent that selects actions randomly.

    This agent is useful as a baseline or for testing.
    """

    def __init__(self, name: str = "RandomAgent") -> None:
        """
        Initialize the random agent.

        Args:
            name: Agent name for display and logging
        """
        super().__init__(name)

        # All possible directions for random selection
        self.all_directions = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
            Direction.TOP_LEFT,
            Direction.TOP_RIGHT,
            Direction.DOWN_LEFT,
            Direction.DOWN_RIGHT,
        ]

    def get_action(self, observation: GameObservation) -> Action:
        """
        Generate a random action.

        Args:
            observation: Current game state (unused in random agent)

        Returns:
            Action: A randomly selected action
        """
        del observation
        # Random 50% chance of shooting
        is_shooting = random.choice([True, False])

        # Random direction selection
        direction = random.choice(self.all_directions)

        return Action(is_shooting=is_shooting, direction=direction)
