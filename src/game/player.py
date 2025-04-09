from agent.agent import Agent


class Player:
    def __init__(
        self,
        position: list[int] | None,
        orientation: list[int] | None,
        agent: Agent,
    ):
        self.position: list[int] | None = position
        self.orientation: list[int] | None = orientation
        self.health = 3
        self.cooldown = 0
        self.ammunition = 3
        self.reload_time = 1
        self.agent = agent

    def update_ammo(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.cooldown == 0:
            self.ammunition = 0
