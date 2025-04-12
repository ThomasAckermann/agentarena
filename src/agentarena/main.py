import pygame

# from agent.default_agent import DefaultAgent
from agentarena.agent.manual_agent import ManualAgent
from agentarena.agent.random_agent import RandomAgent
from agentarena.game.game import Game
from agentarena.config import load_config, GameConfig


def main():
    pygame.init()
    config: GameConfig = load_config()

    screen = pygame.display.set_mode((config.display_width, config.display_height))

    clock = pygame.time.Clock()

    # select agent
    player_agent = ManualAgent()
    enemy_agent = RandomAgent(name="Enemy1")

    # initialize game
    game = Game(screen, player_agent, enemy_agent, clock, config)

    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False

        game.update()
        clock.tick(config.fps)


if __name__ == "__main__":
    main()
