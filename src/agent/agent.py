class Agent:
    def __init__(self, name="Agent"):
        self.name = name

    def reset(self):
        """Called at the start of a stage"""
        pass

    def get_action(self, observation):
        """
        Outputs the next action based on the current state of the game.
        return: action ("UP", "DOWN", "LEFT", "RIGHT", "SHOOT")
        """
        raise NotImplementedError
