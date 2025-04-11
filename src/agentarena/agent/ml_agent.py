from agentarena.agent.agent import Agent


class MLAgent(Agent):
    def __init__(self, model=None):
        super().__init__("MLAgent")
        self.model = model

    def get_action(self, observation):
        state_vector = self.encode_observation(observation)
        action = self.model.predict(state_vector)
        return action

    def encode_observation(self, observation):
        pass
