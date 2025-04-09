from agent.agent import Agent


class player:
    def __init__(self, position: list[int], orientation: list[int], agent: Agent):
        self.position: list[int] = position
        self.orientation: list[int] = orientation
        self.health = 3
        self.cooldown = 0
        self.ammonition = 3
        self.reload_time = 1
        self.agent = agent
