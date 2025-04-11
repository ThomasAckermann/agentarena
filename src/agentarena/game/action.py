from enum import Enum

from pydantic import BaseModel


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    DOWN_LEFT = "bottom_left"
    DOWN_RIGHT = "bottom_right"


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
    is_shooting: bool = False
    direction: Direction | None = None

    class Config:
        validate_assignment = True

    def get_direction_vector(self) -> tuple[int, int]:
        if self.direction is None:
            return (0, 0)
        return DIRECTION_VECTORS[self.direction]
