import random

from game.player import Player


class Level:
    def __init__(self, player: Player, enemies: list[Player], grid_size: int) -> None:
        self.walls: set = set()
        self.player: Player = player
        self.enemies: list[Player] = enemies
        self.grid_size: int = grid_size

    def generate_walls(self):
        num_walls = random.randint(5, 15)
        walls = set()

        for i in range(num_walls):
            walls.add(
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
        self.walls = walls

    def generate_level(self) -> dict:
        self.generate_walls()
