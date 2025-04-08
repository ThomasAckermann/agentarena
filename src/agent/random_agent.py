import random

from agent.agent import Agent


class RandomAgent(Agent):
    def get_action(self, observation):
        return random.choice(["UP", "DOWN", "LEFT", "RIGHT", "SHOOT", "WAIT"])
