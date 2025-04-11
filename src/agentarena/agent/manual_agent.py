import pygame
from agentarena.game.action import Action, Direction

from agentarena.agent.agent import Agent


class ManualAgent(Agent):
    def get_action(self, observation):
        keys = pygame.key.get_pressed()
        is_shooting = False
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            direction = Direction.TOP_RIGHT
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            direction = Direction.TOP_LEFT
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            direction = Direction.DOWN_RIGHT
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            direction = Direction.DOWN_LEFT
        elif keys[pygame.K_UP]:
            direction = Direction.UP
        elif keys[pygame.K_DOWN]:
            direction = Direction.DOWN
        elif keys[pygame.K_LEFT]:
            direction = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            direction = Direction.RIGHT
        else:
            direction = None
        if keys[pygame.K_SPACE]:
            is_shooting = True
        return Action(is_shooting=is_shooting, direction=direction)
