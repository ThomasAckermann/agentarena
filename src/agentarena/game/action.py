"""
Action definitions and utilities for AgentArena.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Union

from pydantic import BaseModel, computed_field


class Direction(str, Enum):
    """
    Enumeration of possible movement directions.
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    DOWN_LEFT = "bottom_left"
    DOWN_RIGHT = "bottom_right"


# Mapping from direction enums to 2D vector components
DIRECTION_VECTORS: dict[Direction, tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
    Direction.TOP_LEFT: (-1, -1),
    Direction.TOP_RIGHT: (1, -1),
    Direction.DOWN_LEFT: (-1, 1),
    Direction.DOWN_RIGHT: (1, 1),
}


class Action(BaseModel):
    """
    Represents an action that an agent can take in the game.

    An action consists of a movement direction and whether
    the agent is shooting.
    """

    is_shooting: bool = False
    direction: Optional[Direction] = None

    @computed_field
    def direction_vector(self) -> Tuple[int, int]:
        """
        Get the 2D vector representation of the direction.

        Returns:
            Tuple[int, int]: (dx, dy) direction vector components
        """
        if self.direction is None:
            return (0, 0)
        return DIRECTION_VECTORS[self.direction]

    @computed_field
    def direction_vector(self) -> tuple[int, int]:
        """
        Get the 2D vector representation of the direction.

        Returns:
            Tuple[int, int]: (dx, dy) direction vector components
        """
        if self.direction is None:
            return (0, 0)
        return DIRECTION_VECTORS[self.direction]

    def get_direction_vector(self) -> tuple[int, int]:
        """
        Get the 2D vector representation of the direction.
        Returns:
            Tuple[int, int]: (dx, dy) direction vector components
        """
        if self.direction is None:
            return (0, 0)
        return DIRECTION_VECTORS[self.direction]
