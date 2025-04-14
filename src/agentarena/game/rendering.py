"""
Rendering system for the AgentArena game.
"""

import pygame

from agentarena.models.action import Direction, DIRECTION_VECTORS
from agentarena.game.asset_manager import AssetManager
from agentarena.models.config import GameConfig


class RenderingSystem:
    """Handles all visual rendering for the game."""

    def __init__(
        self, screen: pygame.Surface, asset_manager: AssetManager, config: GameConfig
    ) -> None:
        """
        Initialize the rendering system.

        Args:
            screen: The pygame surface to render to
            asset_manager: Asset manager containing textures
            config: Game configuration
        """
        self.screen = screen
        self.asset_manager = asset_manager
        self.config = config

        # Initialize pygame font module
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 20)

        # Pre-render text for performance
        self._initialize_text_cache()

    def _initialize_text_cache(self) -> None:
        """Initialize cached text surfaces for performance."""
        # Pre-render health text options
        self.health_texts = {}
        for health in range(11):  # Assuming max health is 10
            self.health_texts[health] = self.font.render(f"Health: {health}", True, (255, 255, 255))

        # Pre-render ammo texts
        self.ammo_texts = {}
        for ammo in range(11):
            self.ammo_texts[ammo] = self.font.render(f"Ammo: {ammo}", True, (255, 255, 255))

        # Initial score text
        self.score_text = self.font.render(f"Score: 0", True, (255, 255, 255))

    def get_direction_from_vector(self, direction_vector) -> Direction:
        """Convert direction vector to Direction enum value."""
        if direction_vector is None or (direction_vector[0] == 0 and direction_vector[1] == 0):
            return None

        # Convert the direction vector to a Direction enum by finding the closest match
        for direction, vector in DIRECTION_VECTORS.items():
            if vector[0] == direction_vector[0] and vector[1] == direction_vector[1]:
                return direction

        # Default to DOWN if no match found (shouldn't happen with normalized vectors)
        return Direction.DOWN

    def render(self, player, enemies, bullets, explosions, walls, score, game_time) -> None:
        """
        Render all game elements to the screen.

        Args:
            player: The player entity
            enemies: List of enemy entities
            bullets: List of bullet entities
            explosions: List of explosion entities
            walls: List of wall entities
            score: Current game score
            game_time: Current game time
        """
        # Skip rendering if screen is None (headless mode)
        if self.screen is None:
            return

        # Draw the floor background
        self.screen.blit(self.asset_manager.floor_background, (0, 0))

        # Batch rendering by type
        # Walls
        for wall in walls:
            self.screen.blit(self.asset_manager.textures["wall"], wall.rect)

        # Enemies with directional sprites
        for enemy in enemies:
            # Get the current enemy direction
            enemy_direction = None
            if enemy.orientation:
                enemy_direction = self.get_direction_from_vector(enemy.orientation)

            # Use the appropriate directional texture
            enemy_texture = self.asset_manager.textures["enemy"][enemy_direction]
            self.screen.blit(enemy_texture, enemy.rect)

        # Player with directional sprite
        if player is not None:
            # Get the current player direction
            player_direction = None
            if player.orientation:
                player_direction = self.get_direction_from_vector(player.orientation)

            # Use the appropriate directional texture
            player_texture = self.asset_manager.textures["player"][player_direction]
            self.screen.blit(player_texture, player.rect)

        # Bullets with different sprites based on owner
        for bullet in bullets:
            if bullet.x is not None and bullet.y is not None:
                # Determine the bullet type (player or enemy)
                bullet_owner = "player" if bullet.owner == "player" else "enemy"
                bullet_texture = self.asset_manager.textures["bullet"][bullet_owner]

                self.screen.blit(bullet_texture, bullet.rect)

        # Render explosions
        self._render_explosions(explosions)

        # UI elements
        self._render_ui(player, score, game_time)

        # Update display
        pygame.display.flip()

    def _render_explosions(self, explosions) -> None:
        """
        Render explosion animations.

        Args:
            explosions: List of explosion entities
        """
        for explosion in explosions:
            # Make sure we have textures for this explosion type
            if (
                explosion.explosion_type in self.asset_manager.textures["explosion"]
                and len(self.asset_manager.textures["explosion"][explosion.explosion_type]) > 0
            ):
                # Get the correct explosion texture for this frame with bounds checking
                frame_index = min(
                    explosion.frame,
                    len(self.asset_manager.textures["explosion"][explosion.explosion_type]) - 1,
                )
                explosion_texture = self.asset_manager.textures["explosion"][
                    explosion.explosion_type
                ][frame_index]

                # Scale the texture if the explosion is a smaller impact explosion
                if explosion.width < self.asset_manager.scaled_width:
                    # Create a smaller version of the texture for bullet impacts
                    explosion_texture = pygame.transform.scale(
                        explosion_texture,
                        (explosion.width, explosion.height),
                    )

                # Render the explosion (only if we successfully got a texture)
                self.screen.blit(explosion_texture, explosion.rect)

    def _render_ui(self, player, score, game_time) -> None:
        """
        Render UI elements like health, ammo, score.

        Args:
            player: The player entity
            score: Current game score
            game_time: Current game time
        """
        # UI elements with semi-transparent background
        ui_background = pygame.Surface((200, 100))
        ui_background.set_alpha(128)  # Semi-transparent
        ui_background.fill((0, 0, 0))  # Black background
        self.screen.blit(ui_background, (5, 5))

        if player is not None:
            self.screen.blit(self.health_texts[player.health], (10, 10))
            self.screen.blit(self.ammo_texts[player.ammunition], (10, 30))

        # Update and render score
        self.score_text = self.font.render(f"Score: {score}", True, (255, 255, 255))
        self.screen.blit(self.score_text, (10, 50))

        # Add game time display
        time_text = self.font.render(f"Time: {game_time:.1f}s", True, (255, 255, 255))
        self.screen.blit(time_text, (10, 70))
