"""
Manual agent controlled by the player's keyboard input.
"""

import pygame

from agentarena.agent.agent import Agent
from agentarena.models.action import Action, Direction
from agentarena.models.observations import GameObservation


class ManualAgent(Agent):
    """
    Human-controlled agent that processes keyboard input.

    This agent translates keyboard presses into game actions.
    """

    def __init__(self, name: str = "Human") -> None:
        """
        Initialize the manual agent.

        Args:
            name: Agent name for display and logging
        """
        super().__init__(name)

    def get_action(self, observation: GameObservation) -> Action:
        """
        Convert keyboard input to game actions.

        Args:
            observation: Current game state (unused in manual agent)

        Returns:
            Action: The player's action based on keyboard input
        """
        keys = pygame.key.get_pressed()
        is_shooting = False
        direction = None

        # Determine direction based on key combinations
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            direction = Direction.TOP_RIGHT
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            direction = Direction.TOP_LEFT
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            direction = Direction.DOWN_RIGHT
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            direction = Direction.DOWN_LEFT
        elif keys[pygame.K_UP]:
            direction = Direction.UP
        elif keys[pygame.K_DOWN]:
            direction = Direction.DOWN
        elif keys[pygame.K_LEFT]:
            direction = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            direction = Direction.RIGHT

        # Check if shooting
        if keys[pygame.K_SPACE]:
            is_shooting = True

        return Action(is_shooting=is_shooting, direction=direction)
