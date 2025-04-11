from agentarena.game.entities.entity import Entity


class Projectile(Entity):
    def __init__(
        self,
        width: int,
        height: int,
        owner: str,
        direction: list[int],
        x: int | None = None,
        y: int | None = None,
        speed: int = 20,
    ) -> None:
        super().__init__(width=width, height=height, x=x, y=y)
        self.direction: list[int] = direction
        self.speed: int = speed
        self.owner: str = owner
