import random

from agentarena.agent.agent import Agent
from agentarena.game.action import Action, Direction


class RandomAgent(Agent):
    def get_action(self, observation):
        is_shooting = random.choice([True, False])
        direction = random.choice(
            [
                Direction.UP,
                Direction.DOWN,
                Direction.LEFT,
                Direction.RIGHT,
                Direction.TOP_LEFT,
                Direction.TOP_RIGHT,
                Direction.DOWN_LEFT,
                Direction.DOWN_RIGHT,
            ],
        )
        return Action(is_shooting=is_shooting, direction=direction)
