from agents import DummyMLAgent, RuleBasedAgent
from game import Game

# Headless training loop
player_agent = DummyMLAgent()
enemy_agents = [RuleBasedAgent()]

game = Game(player_agent=player_agent, enemy_agents=enemy_agents, headless=True)

episodes = 1000
for ep in range(episodes):
    game.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        game.update()
        steps += 1
        # Optionale Logging/Auswertung hier
