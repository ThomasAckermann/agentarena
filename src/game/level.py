import random

from game.player import Player


class Level:
    def __init__(self, player: Player, enemies: list[Player], grid_size: int):
        self.player = player
        self.enemies = enemies
        self.grid_size = grid_size
        self.walls = set()  # Will store tuples of (x, y) coordinates

    def generate_level(self):
        """Generate a level with walls"""
        self.generate_walls()

    def generate_walls(self):
        """Generate walls randomly while ensuring player and enemies are not trapped"""
        # Clear existing walls
        self.walls = set()

        # Add some random walls
        wall_count = int(self.grid_size * self.grid_size * 0.2)  # 20% of grid as walls
        for _ in range(wall_count):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)

            # Make sure we don't place walls on player or enemies
            if (x, y) != tuple(self.player.position) and not any(
                tuple(enemy.position) == (x, y) for enemy in self.enemies
            ):
                # Use a tuple for the wall position
                self.walls.add((x, y))

        # Ensure the level is playable (basic check)
        # In a real game, you might want more sophisticated checks here
        self.ensure_playable()

    def ensure_playable(self):
        """Basic check to ensure level is playable"""
        # For now, just make sure player isn't surrounded by walls
        player_x, player_y = self.player.position

        # Check at least one adjacent cell is free
        adjacent_cells = [
            (player_x + 1, player_y),
            (player_x - 1, player_y),
            (player_x, player_y + 1),
            (player_x, player_y - 1),
        ]

        # If all adjacent cells are walls, remove one
        if all(cell in self.walls for cell in adjacent_cells):
            if adjacent_cells[0] in self.walls:
                self.walls.remove(adjacent_cells[0])

        # Do the same check for enemies
        for enemy in self.enemies:
            enemy_x, enemy_y = enemy.position

            adjacent_cells = [
                (enemy_x + 1, enemy_y),
                (enemy_x - 1, enemy_y),
                (enemy_x, enemy_y + 1),
                (enemy_x, enemy_y - 1),
            ]

            if all(cell in self.walls for cell in adjacent_cells):
                if adjacent_cells[0] in self.walls:
                    self.walls.remove(adjacent_cells[0])
