"""
Level generation and management for AgentArena.
"""

import random
from pathlib import Path

import pygame

from agentarena.game.entities.player import Player
from agentarena.game.entities.wall import Wall
from agentarena.models.config import GameConfig
from agentarena.models.entities import WallModel

# Constants for level generation
BUFFER_BLOCKS = 2  # Blocks of space around entities
MARGIN_BLOCKS = 2  # Margin from edge of screen for random walls
MIN_CLUSTERS = 10  # Minimum number of wall clusters
MAX_CLUSTERS = 15  # Maximum number of wall clusters
MIN_RANDOM_WALLS = 5  # Minimum random walls to add if not enough clusters
MAX_RANDOM_WALLS = 15  # Maximum random walls to add if not enough clusters


class Level:
    """
    Level manager that handles generation and management of game levels.
    """

    def __init__(
        self,
        player: Player | None,
        enemies: list[Player],
        config: GameConfig,
    ) -> None:
        """
        Initialize the level generator.

        Args:
            player: The player entity
            enemies: List of enemy entities
            config: Game configuration
        """
        self.player = player
        self.enemies = enemies
        self.walls: list[Wall] = []
        self.config: GameConfig = config
        self.wall_models: list[WallModel] = []

        # Precompute grid dimensions for level generation
        self.grid_width = self.config.display_width // self.config.block_width
        self.grid_height = self.config.display_height // self.config.block_height

        # Generate the level
        self.generate_level()

    def generate_level(self) -> None:
        """Generate a complete level with walls and obstacles."""
        print("Generating level...")
        self.walls = []
        self.wall_models = []

        # Generate the level components
        self.add_border_walls()
        self.generate_wall_clusters()
        self.ensure_playable()

        print(f"Level generation complete with {len(self.walls)} walls")

    def add_border_walls(self) -> None:
        """Add walls around the border of the screen."""
        print("Adding border walls...")
        wall_count_horizontal: int = int(self.config.display_width / self.config.block_width) + 1
        wall_count_vertical: int = int(self.config.display_height / self.config.block_height) + 1

        # Top border
        self._add_wall_row(0, 0, wall_count_horizontal)

        # Bottom border
        self._add_wall_row(
            0,
            self.config.display_height - self.config.block_height,
            wall_count_horizontal,
        )

        # Left border
        self._add_wall_column(0, 1, wall_count_vertical - 1)

        # Right border
        self._add_wall_column(
            self.config.display_width - self.config.block_width,
            1,
            wall_count_vertical - 1,
        )

        print(f"Border walls added: {len(self.walls)}")

    def _add_wall_row(self, start_x: int, y: int, count: int) -> None:
        """
        Add a horizontal row of walls.

        Args:
            start_x: Starting x coordinate
            y: Y coordinate for the row
            count: Number of walls to add
        """
        for i in range(count):
            x = start_x + (i * self.config.block_width)
            self._create_wall(x, y)

    def _add_wall_column(self, x: int, start_y: int, count: int) -> None:
        """
        Add a vertical column of walls.

        Args:
            x: X coordinate for the column
            start_y: Starting y coordinate
            count: Number of walls to add
        """
        for i in range(count):
            y = start_y + (i * self.config.block_height)
            self._create_wall(x, y)

    def generate_wall_clusters(self) -> None:
        """Generate clusters of walls with better placement coverage."""
        print("Generating wall clusters...")

        # Debug info
        print(f"Display dimensions: {self.config.display_width}x{self.config.display_height}")
        print(f"Block dimensions: {self.config.block_width}x{self.config.block_height}")
        print(f"Grid dimensions: {self.grid_width}x{self.grid_height}")

        # Parameters for wall generation
        max_attempts = 150
        target_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
        clusters_created = 0

        # Wall patterns for clusters
        patterns = self._get_wall_patterns()

        # Safety margin from screen edges
        margin = MARGIN_BLOCKS

        # Track attempts and failures
        attempt_count = 0
        failed_attempts = 0

        while clusters_created < target_clusters and attempt_count < max_attempts:
            attempt_count += 1

            # Select a random pattern
            pattern = random.choice(patterns)
            pattern_width = max(dx for dx, dy in pattern) + 1
            pattern_height = max(dy for dx, dy in pattern) + 1

            # Find a random position that can fit the entire pattern
            grid_x = random.randint(margin, self.grid_width - pattern_width - margin)
            grid_y = random.randint(margin, self.grid_height - pattern_height - margin)

            # Check if the entire pattern can be placed
            valid_placement = True
            for dx, dy in pattern:
                if not self._is_position_valid(grid_x + dx, grid_y + dy):
                    valid_placement = False
                    break

            if valid_placement:
                # Place the entire cluster
                new_walls = []
                for dx, dy in pattern:
                    x = (grid_x + dx) * self.config.block_width
                    y = (grid_y + dy) * self.config.block_height
                    new_wall = self._create_wall(x, y)
                    new_walls.append(new_wall)

                clusters_created += 1
                print(
                    f"Created cluster #{clusters_created} at ({grid_x}, {grid_y}) with {len(new_walls)} walls",
                )
            else:
                failed_attempts += 1
                if failed_attempts % 20 == 0:
                    print(f"Failed {failed_attempts} attempts to place clusters")

        print(
            f"Wall cluster generation complete. Created {clusters_created} clusters after {attempt_count} attempts.",
        )

        # If we couldn't create enough clusters, add some random walls
        if clusters_created < MIN_CLUSTERS:
            self._add_random_walls()

    def _get_wall_patterns(self) -> list[list[tuple[int, int]]]:
        """
        Get list of wall patterns for cluster generation.

        Returns:
            List of wall patterns (each pattern is a list of dx,dy coordinates)
        """
        return [
            [(0, 0), (1, 0)],  # 2-block horizontal
            [(0, 0), (0, 1)],  # 2-block vertical
            [(0, 0), (1, 0), (2, 0)],  # 3-block horizontal
            [(0, 0), (0, 1), (0, 2)],  # 3-block vertical
            [(0, 0), (1, 0), (0, 1)],  # L-shape
            [(0, 0), (1, 0), (1, 1)],  # Corner shape
            [(0, 0), (1, 0), (2, 0), (2, 1)],  # J-shape
            [(0, 0), (1, 0), (1, 1), (2, 1)],  # Z-shape
            [(0, 0), (0, 1), (1, 1), (1, 2)],  # S-shape
            [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],  # 2x3 rectangle
            [(0, 0), (1, 0), (0, 1), (1, 1)],  # 2x2 square
        ]

    def _add_random_walls(self) -> None:
        """Add random individual walls when cluster generation fails."""
        print("Adding some random individual walls...")
        random_walls_to_add = random.randint(MIN_RANDOM_WALLS, MAX_RANDOM_WALLS)
        random_walls_added = 0
        max_attempts = 50

        margin = MARGIN_BLOCKS
        attempt_count = 0

        while random_walls_added < random_walls_to_add and attempt_count < max_attempts:
            attempt_count += 1
            grid_x = random.randint(margin, self.grid_width - margin)
            grid_y = random.randint(margin, self.grid_height - margin)

            if self._is_position_valid(grid_x, grid_y):
                x = grid_x * self.config.block_width
                y = grid_y * self.config.block_height
                self._create_wall(x, y)
                random_walls_added += 1

        print(f"Added {random_walls_added} random individual walls")

    def _is_position_valid(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if a grid position is valid for wall placement.

        Args:
            grid_x: Grid x-coordinate
            grid_y: Grid y-coordinate

        Returns:
            bool: True if the position is valid, False otherwise
        """
        # Convert grid position to pixel position
        x = grid_x * self.config.block_width
        y = grid_y * self.config.block_height

        # Create a test rectangle
        test_rect = pygame.Rect(x, y, self.config.block_width, self.config.block_height)

        # Check if it's too close to player or enemies
        entities = []
        if self.player is not None:
            entities.append(self.player)
        entities.extend(self.enemies)

        for entity in entities:
            # Skip entities without position
            if entity.x is None or entity.y is None:
                continue

            buffer_rect = pygame.Rect(
                entity.x - BUFFER_BLOCKS * self.config.block_width,
                entity.y - BUFFER_BLOCKS * self.config.block_height,
                (2 * BUFFER_BLOCKS + 1) * self.config.block_width,
                (2 * BUFFER_BLOCKS + 1) * self.config.block_height,
            )
            if buffer_rect.colliderect(test_rect):
                return False

        # Check if it collides with existing walls
        return not any(wall.rect.colliderect(test_rect) for wall in self.walls)

    def ensure_playable(self) -> None:
        """Ensure player and enemies are not fully surrounded by walls."""
        print("Ensuring level is playable...")

        # Check and fix for player
        if self.player is not None:
            self._clear_adjacent_if_needed(self.player)

        # Check and fix for all enemies
        for enemy in self.enemies:
            self._clear_adjacent_if_needed(enemy)

        print("Playability check complete")

    def _clear_adjacent_if_needed(self, entity: Player) -> None:
        """
        Check if an entity is trapped by walls and clear a path if needed.

        Args:
            entity: The entity to check
        """
        # Skip entities without position
        if entity.x is None or entity.y is None:
            return

        # Check all four directions around the entity
        adjacents = [
            (entity.x + self.config.block_width, entity.y, 1, 0),  # Right
            (entity.x - self.config.block_width, entity.y, -1, 0),  # Left
            (entity.x, entity.y + self.config.block_height, 0, 1),  # Down
            (entity.x, entity.y - self.config.block_height, 0, -1),  # Up
        ]

        # Check if any adjacent space is free
        has_free_space = False
        for adj_x, adj_y, _, _ in adjacents:
            adj_rect = pygame.Rect(adj_x, adj_y, self.config.block_width, self.config.block_height)
            if not any(wall.rect.colliderect(adj_rect) for wall in self.walls):
                has_free_space = True
                break

        # If all are blocked, remove one wall to create an opening
        if not has_free_space and self.walls:
            print(f"Entity at ({entity.x}, {entity.y}) is trapped, clearing a path")

            # Try directions in order (prefer clearing in a specific direction)
            for adj_x, adj_y, _, _ in adjacents:
                adj_rect = pygame.Rect(
                    adj_x,
                    adj_y,
                    self.config.block_width,
                    self.config.block_height,
                )

                for wall in self.walls[:]:  # Use a copy for safe removal
                    if wall.rect.colliderect(adj_rect):
                        self.walls.remove(wall)

                        # Also remove from wall models if applicable
                        for wall_model in self.wall_models[:]:
                            if (
                                wall_model.x == wall.x
                                and wall_model.y == wall.y
                                and wall_model.width == wall.width
                                and wall_model.height == wall.height
                            ):
                                self.wall_models.remove(wall_model)

                        return  # Only remove one wall

    def _create_wall(self, x: int, y: int) -> Wall:
        """
        Create a wall at the specified position.

        Args:
            x: X-coordinate
            y: Y-coordinate

        Returns:
            Wall: The created wall entity
        """
        # Create the wall model
        wall_id = f"wall_{len(self.walls)}"
        wall_model = WallModel(
            id=wall_id,
            x=x,
            y=y,
            width=self.config.block_width,
            height=self.config.block_height,
            entity_type="wall",
        )
        self.wall_models.append(wall_model)

        # Create the wall entity
        wall = Wall(
            x=x,
            y=y,
            width=self.config.block_width,
            height=self.config.block_height,
        )
        self.walls.append(wall)

        return wall

    def save_level(self, path: str) -> None:
        """
        Save the current level to a file.

        Args:
            path: Path to save the level to
        """
        # Create a level data structure
        level_data = {
            "walls": [model.model_dump() for model in self.wall_models],
        }

        # Save to file
        Path(path).write_text(str(level_data))

    def load_level(self, path: str) -> None:
        """
        Load a level from a file.

        Args:
            path: Path to load the level from
        """
        import ast

        # Load level data
        level_data = ast.literal_eval(Path(path).read_text())

        # Clear existing walls
        self.walls = []
        self.wall_models = []

        # Recreate walls from data
        for wall_data in level_data["walls"]:
            wall_model = WallModel.model_validate(wall_data)
            self.wall_models.append(wall_model)

            wall = Wall(
                x=wall_model.x,
                y=wall_model.y,
                width=wall_model.width,
                height=wall_model.height,
            )
            self.walls.append(wall)

        print(f"Loaded level with {len(self.walls)} walls")
