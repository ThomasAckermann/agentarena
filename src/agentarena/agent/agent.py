from agentarena.game.action import Action


class Agent:
    def __init__(self, name: str = "Agent"):
        self.name: str = name

    def reset(self) -> None:
        """Called at the start of a stage"""

    def get_action(self, observation) -> Action:
        """
        Outputs the next action based on the current state of the game.
        return: action
        """
        raise NotImplementedError
