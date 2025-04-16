"""
Asset management system for AgentArena.
"""

import pygame

from agentarena.models.action import Direction
from agentarena.models.config import GameConfig

ASSET_PATH: str = "src/agentarena/assets"
PLAYER_SCALE: float = 0.8  # Scale factor for player sprites


class AssetManager:
    """
    Manages game assets including textures, sounds, and other resources.
    """

    def __init__(self, config: GameConfig) -> None:
        """
        Initialize the asset manager.

        Args:
            config: Game configuration
        """
        self.config = config
        self.textures = {}
        self.floor_background = None

        # Precomputed values
        self.bullet_width = int(self.config.block_width / 2)
        self.bullet_height = int(self.config.block_height / 2)
        self.scaled_width = int(self.config.block_width * PLAYER_SCALE)
        self.scaled_height = int(self.config.block_height * PLAYER_SCALE)

    def load_textures(self) -> None:
        """Load and prepare all game textures."""
        # Initialize pygame font module first
        pygame.font.init()

        # Load textures once and cache them
        self.textures = {
            # Load directional player sprites
            "player": {
                Direction.UP: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_top.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.DOWN: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_bottom.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.LEFT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_left.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.RIGHT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_right.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.TOP_LEFT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_top_left.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.TOP_RIGHT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_top_right.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.DOWN_LEFT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_bottom_left.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.DOWN_RIGHT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_bottom_right.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                # Default sprite if no direction (use DOWN as default)
                None: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_bottom.png"),
                    (self.scaled_width, self.scaled_height),
                ),
            },
            # Load directional enemy sprites
            "enemy": {
                Direction.UP: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_top.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.DOWN: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_bottom.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.LEFT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_left.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.RIGHT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_right.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.TOP_LEFT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_top_left.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.TOP_RIGHT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_top_right.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.DOWN_LEFT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_bottom_left.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                Direction.DOWN_RIGHT: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_bottom_right.png"),
                    (self.scaled_width, self.scaled_height),
                ),
                # Default sprite if no direction (use DOWN as default)
                None: pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_bottom.png"),
                    (self.scaled_width, self.scaled_height),
                ),
            },
            # Load different bullet sprites for player and enemy
            "bullet": {
                "player": pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/player/player_bullet.png"),
                    (self.bullet_width, self.bullet_height),
                ),
                "enemy": pygame.transform.scale(
                    pygame.image.load(f"{ASSET_PATH}/enemy/enemy_bullet.png"),
                    (self.bullet_width, self.bullet_height),
                ),
            },
            # Load wall and floor textures
            "wall": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/wall.png"),
                (self.config.block_width, self.config.block_height),
            ),
            "floor": pygame.image.load(f"{ASSET_PATH}/floor.png"),
            # Load explosion textures
            "explosion": {
                "player": [
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/player/player_explosion_1.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/player/player_explosion_2.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/player/player_explosion_3.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/player/player_explosion_4.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                ],
                "enemy": [
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/enemy/enemy_explosion_1.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/enemy/enemy_explosion_2.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/enemy/enemy_explosion_3.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                    pygame.transform.scale(
                        pygame.image.load(f"{ASSET_PATH}/enemy/enemy_explosion_4.png"),
                        (self.scaled_width, self.scaled_height),
                    ),
                ],
            },
        }

        # Create a repeating floor pattern
        self.create_floor_background()

    def create_floor_background(self) -> None:
        """Create a repeating floor pattern surface."""
        # Get floor tile dimensions
        floor_tile = self.textures["floor"]
        tile_width, tile_height = floor_tile.get_size()  # Typically 128x128

        # Create a surface for the entire background
        background_width = self.config.display_width
        background_height = self.config.display_height

        # Create a new surface for the background
        self.floor_background = pygame.Surface((background_width, background_height))

        # Fill the background with repeating floor tiles
        for y in range(0, background_height, tile_height):
            for x in range(0, background_width, tile_width):
                self.floor_background.blit(floor_tile, (x, y))
