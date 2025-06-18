from agentarena.agent.agent import Agent
from agentarena.game.entities.entity import Entity


class Player(Entity):
    def __init__(
        self,
        orientation: list[int] | None,
        agent: Agent,
        width: int,
        height: int,
        x: int | None = None,
        y: int | None = None,
        speed: int = 100,
        health: int = 3,
        cooldown: int = 0,
        ammunition: int = 3,
        is_reloading: bool = False,
    ) -> None:
        super().__init__(width=width, height=height, x=x, y=y)
        self.orientation: list[int] | None = orientation
        self.health: int = health
        self.cooldown: int = cooldown
        self.ammunition: int = ammunition
        self.agent: Agent = agent
        self.speed: int = speed
        self.is_reloading: bool = is_reloading
