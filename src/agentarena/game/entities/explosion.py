from agentarena.game.entities.entity import Entity


class Explosion(Entity):
    """Entity representing an explosion animation effect."""

    def __init__(
        self,
        width: int,
        height: int,
        explosion_type: str,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        """Initialize explosion entity.

        Args:
            width: Width of the explosion sprite
            height: Height of the explosion sprite
            explosion_type: Type of explosion ("player" or "enemy")
            x: X-coordinate position
            y: Y-coordinate position
        """
        super().__init__(width=width, height=height, x=x, y=y)

        self.explosion_type: str = explosion_type
        self.frame: int = 0
        self.max_frames: int = 4  # We have 3 frames for each explosion
        self.frame_duration: int = 1  # How many game ticks to show each frame
        self.current_tick: int = 0
        self.finished: bool = False

    def update(self) -> None:
        """Update the explosion animation state."""
        self.current_tick += 1
        if self.current_tick >= self.frame_duration:
            self.current_tick = 0
            self.frame += 1
            if self.frame >= self.max_frames:
                self.finished = True
