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
        config: GameConfig,
    ):
        self.player = player
        self.enemies = enemies
        self.walls: list[Wall] = []
        self.config: GameConfig = config

    def generate_level(self) -> None:
        """Generate a level with walls"""
        self.walls = []
        self.add_border_walls()
        self.generate_wall_clusters()
        self.ensure_playable()

    def add_border_walls(self) -> None:
        """Add walls around the border of the screen"""
        # Add debug print to track execution
        print("Adding border walls...")
        wall_count_horizontal: int = (
            int(self.config.display_width / self.config.block_width) + 1
        )
        wall_count_vertical: int = (
            int(self.config.display_height / self.config.block_height) + 1
        )

        # Top border
        for x in range(0, wall_count_horizontal):
            self.walls.append(
                Wall(
                    x=x * self.config.block_width,
                    y=0,
                    width=self.config.block_width,
                    height=self.config.block_height,
                )
            )

        # Bottom border
        for x in range(0, wall_count_horizontal):
            self.walls.append(
                Wall(
                    x=x * self.config.block_width,
                    y=(self.config.display_height - self.config.block_height),
                    width=self.config.block_width,
                    height=self.config.block_height,
                )
            )

        # Left border
        for y in range(1, wall_count_vertical):
            self.walls.append(
                Wall(
                    x=0,
                    y=y * self.config.block_height,
                    width=self.config.block_width,
                    height=self.config.block_height,
                )
            )

        # Right border
        for y in range(1, wall_count_vertical):
            self.walls.append(
                Wall(
                    x=(self.config.display_width - self.config.block_width),
                    y=y * self.config.block_height,
                    width=self.config.block_width,
                    height=self.config.block_height,
                )
            )

        print(f"Border walls added: {len(self.walls)}")

    def generate_wall_clusters(self) -> None:
        """Generate clusters of walls with better placement coverage"""
        print("Generating wall clusters...")

        # Debugging first
        print(
            f"Display dimensions: {self.config.display_width}x{self.config.display_height}"
        )
        print(f"Block dimensions: {self.config.block_width}x{self.config.block_height}")

        # Parameters
        max_attempts = 150
        target_clusters = 10
        clusters_created = 0

        # Simple patterns that are more likely to fit
        patterns = [
            [(0, 0), (1, 0)],  # 2-block horizontal
            [(0, 0), (0, 1)],  # 2-block vertical
            [(0, 0), (1, 0), (2, 0)],  # 3-block horizontal
            [(0, 0), (0, 1), (0, 2)],  # 3-block vertical
            [(0, 0), (1, 0), (0, 1)],  # L-shape
        ]

        # Grid dimensions
        grid_width = self.config.display_width // self.config.block_width
        grid_height = self.config.display_height // self.config.block_height

        # Safety margin
        margin = 2

        # Debug
        print(f"Grid dimensions: {grid_width}x{grid_height}")

        # Entity buffer zone
        buffer_blocks = 2

        def is_position_valid(grid_x, grid_y):
            """Check if a grid position is valid for wall placement"""
            # Convert grid position to pixel position
            x = grid_x * self.config.block_width
            y = grid_y * self.config.block_height

            # Create a test rectangle
            test_rect = pygame.Rect(
                x, y, self.config.block_width, self.config.block_height
            )

            # Check if it's too close to player or enemies
            for entity in [self.player] + self.enemies:
                buffer_rect = pygame.Rect(
                    entity.x - buffer_blocks * self.config.block_width,
                    entity.y - buffer_blocks * self.config.block_height,
                    (2 * buffer_blocks + 1) * self.config.block_width,
                    (2 * buffer_blocks + 1) * self.config.block_height,
                )
                if buffer_rect.colliderect(test_rect):
                    return False

            # Check if it collides with existing walls
            for wall in self.walls:
                if wall.rect.colliderect(test_rect):
                    return False

            return True

        attempt_count = 0
        failed_attempts = 0

        while clusters_created < target_clusters and attempt_count < max_attempts:
            attempt_count += 1

            # Select a random pattern
            pattern = random.choice(patterns)
            pattern_width = max(dx for dx, dy in pattern) + 1
            pattern_height = max(dy for dx, dy in pattern) + 1

            # Find a random position that can fit the entire pattern
            grid_x = random.randint(margin, grid_width - pattern_width - margin)
            grid_y = random.randint(margin, grid_height - pattern_height - margin)

            # Check if the entire pattern can be placed
            valid_placement = True
            for dx, dy in pattern:
                if not is_position_valid(grid_x + dx, grid_y + dy):
                    valid_placement = False
                    break

            if valid_placement:
                # Place the entire cluster
                new_walls = []
                for dx, dy in pattern:
                    x = (grid_x + dx) * self.config.block_width
                    y = (grid_y + dy) * self.config.block_height
                    new_wall = Wall(
                        x=x,
                        y=y,
                        width=self.config.block_width,
                        height=self.config.block_height,
                    )
                    new_walls.append(new_wall)

                # Add all walls from this cluster
                self.walls.extend(new_walls)
                clusters_created += 1
                print(
                    f"Created cluster #{clusters_created} at ({grid_x}, {grid_y}) with {len(new_walls)} walls"
                )
            else:
                failed_attempts += 1
                if failed_attempts % 20 == 0:
                    print(f"Failed {failed_attempts} attempts to place clusters")

        print(
            f"Wall cluster generation complete. Created {clusters_created} clusters after {attempt_count} attempts."
        )

        # If we couldn't create enough clusters, at least add some random walls
        if clusters_created < 5:
            print("Adding some random individual walls...")
            random_walls_added = 0
            for _ in range(50):  # Try to add up to 50 random walls
                grid_x = random.randint(margin, grid_width - margin)
                grid_y = random.randint(margin, grid_height - margin)

                if is_position_valid(grid_x, grid_y):
                    x = grid_x * self.config.block_width
                    y = grid_y * self.config.block_height
                    self.walls.append(
                        Wall(
                            x=x,
                            y=y,
                            width=self.config.block_width,
                            height=self.config.block_height,
                        )
                    )
                    random_walls_added += 1

                    if random_walls_added >= 15:  # Stop after adding 15 walls
                        break

            print(f"Added {random_walls_added} random individual walls")

    def ensure_playable(self) -> None:
        """Ensure player and enemies are not fully surrounded by walls"""
        print("Ensuring level is playable...")

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

            # Check if any adjacent space is free
            has_free_space = False
            for adj in adjacents:
                if not any(wall.rect.colliderect(adj) for wall in self.walls):
                    has_free_space = True
                    break

            # If all are blocked, remove one wall to create an opening
            if not has_free_space:
                print(f"Entity at ({entity.x}, {entity.y}) is trapped, clearing a path")
                for adj, (dx, dy) in zip(adjacents, [(1, 0), (-1, 0), (0, 1), (0, -1)]):
                    for wall in self.walls[:]:  # Use a copy for safe removal
                        if wall.rect.colliderect(adj):
                            self.walls.remove(wall)
                            return  # Only remove one wall

        # Check and fix for player
        clear_adjacent_if_needed(self.player)

        # Check and fix for all enemies
        for enemy in self.enemies:
            clear_adjacent_if_needed(enemy)

        print("Playability check complete")
