import pygame

from agent.agent import Agent


class ManualAgent(Agent):
    def get_action(self, observation):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return "UP"
        elif keys[pygame.K_DOWN]:
            return "DOWN"
        elif keys[pygame.K_LEFT]:
            return "LEFT"
        elif keys[pygame.K_RIGHT]:
            return "RIGHT"
        elif keys[pygame.K_SPACE]:
            return "SHOOT"
        return None
