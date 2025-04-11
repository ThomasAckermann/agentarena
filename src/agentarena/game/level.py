import math
import random
import pygame

from agentarena.game.entities.wall import Wall
from agentarena.game.entities.player import Player
from agentarena.config import GameConfig


class Level:
    def __init__(
        self,
        player: Player,
        enemies: list[Player],
        grid_size: int,
        config: GameConfig,
    ):
        self.player = player
        self.enemies = enemies
        self.grid_size = grid_size
        self.walls: list[Wall] = []
        self.config: GameConfig = config

    def generate_level(self):
        """Generate a level with walls"""
        self.generate_walls()

    def generate_walls(self):
        """Generate walls randomly while ensuring player and enemies are not trapped"""
        self.walls = []

        wall_count = int(0.2 * math.pow(self.grid_size, 2))
        for _ in range(wall_count):
            x = random.randint(0, self.grid_size - 1) * self.config.block_width
            y = random.randint(0, self.grid_size - 1) * self.config.block_height

            new_wall = Wall(
                x=x,
                y=y,
                width=self.config.block_width,
                height=self.config.block_height,
            )

            # Prevent spawning walls on top of player or enemies
            if new_wall.rect.colliderect(self.player.rect):
                continue
            if any(new_wall.rect.colliderect(enemy.rect) for enemy in self.enemies):
                continue

            self.walls.append(new_wall)

        self.ensure_playable()

    def ensure_playable(self):
        """Ensure player and enemies are not fully surrounded by walls"""

        def clear_adjacent_if_needed(entity):
            adjacents = [
                pygame.Rect(
                    entity.x + dx * self.config.block_width,
                    entity.y + dy * self.config.block_height,
                    self.config.block_width,
                    self.config.block_height,
                )
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            ]
            for adj in adjacents:
                if any(wall.rect.colliderect(adj) for wall in self.walls):
                    continue
                return  # At least one free space found
            # If all are blocked, remove the first overlapping wall
            for wall in self.walls:
                if wall.rect.colliderect(adjacents[0]):
                    self.walls.remove(wall)
                    break

        clear_adjacent_if_needed(self.player)
        for enemy in self.enemies:
            clear_adjacent_if_needed(enemy)
