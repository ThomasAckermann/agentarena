import random

from game.player import Player


class Level:

    def __init__(self, player: Player):
        self.walls: set = set()
        self.player: Player = player
        self.enemies: list[Player] = []

    def generate_walls(self):
        # Generate walls
        all_occupied = {tuple(player_start)} | {tuple(e) for e in enemy_starts}
        num_walls = random.randint(5, 15)
        walls = set()

        while len(walls) < num_walls:
            wall = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if wall not in all_occupied:
                walls.add(wall)

        self.walls = walls

    def generate_player_start_positions(self):
        player_start = [
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        ]

        num_enemies = random.randint(1, 3)
        enemy_starts = []
        while len(enemy_starts) < num_enemies:
            pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]
            if pos != player_start and pos not in enemy_starts:
                enemy_starts.append(pos)

        return {"player_start": player_start, "enemy_starts": enemy_starts}
        pass

    def generate_level(self) -> dict:
        self.generate_walls()
        self.generate_player_start_positions()
