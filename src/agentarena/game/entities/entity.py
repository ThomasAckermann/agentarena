import pygame


class Entity:
    def __init__(
        self,
        width: int,
        height: int,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.x = x
        self.y = y

    @property
    def rect(self) -> pygame.Rect:
        """Get the rectangle for collision detection (cast to integer only here)."""
        if self.x is None or self.y is None:
            # Default position for entities without position
            return pygame.Rect(0, 0, self.width, self.height)
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    @property
    def position(self) -> tuple[float, float]:
        """Get the precise floating-point position of the entity."""
        return (self.x if self.x is not None else 0.0, self.y if self.y is not None else 0.0)

    @rect.setter
    def rect(self, value: pygame.Rect) -> None:
        """Set the internal rectangle representation."""
        self._rect = value

    def __str__(self) -> str:
        """Convert entity to string representation."""
        return str(self.__dict__)
