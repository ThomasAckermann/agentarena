import pygame


class Entity:
    def __init__(
        self,
        width: int,
        height: int,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.x = x
        self.y = y

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    @rect.setter
    def rect(self, value: pygame.Rect) -> None:
        """Set the internal rectangle representation."""
        self._rect = value

    def __str__(self) -> str:
        """Convert entity to string representation."""
        return str(self.__dict__)
