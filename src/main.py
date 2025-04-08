import pygame

# from agent.default_agent import DefaultAgent
from agent.manual_agent import ManualAgent
from agent.random_agent import RandomAgent
from game.game import Game

HEADLESS: bool = False

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()

    # select agent
    player_agent = ManualAgent()
    enemy_agent = RandomAgent(name="Enemy1")

    # initialize game
    game = Game(screen, HEADLESS, player_agent, enemy_agent)

    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False

        game.update()
        clock.tick(10)
