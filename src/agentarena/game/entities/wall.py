from agentarena.game.entities.entity import Entity


class Wall(Entity):
    def __init__(
        self,
        width: int,
        height: int,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        super().__init__(width=width, height=height, x=x, y=y)
