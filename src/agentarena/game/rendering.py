"""
Rendering system for the AgentArena game.
"""

import pygame

from agentarena.game.asset_manager import AssetManager
from agentarena.models.action import DIRECTION_VECTORS, Direction
from agentarena.models.config import GameConfig


class RenderingSystem:
    def __init__(
        self,
        screen: pygame.Surface,
        asset_manager: AssetManager,
        config: GameConfig,
    ) -> None:
        self.screen = screen
        self.asset_manager = asset_manager
        self.config = config

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 20)

        self._initialize_text_cache()

    def _initialize_text_cache(self) -> None:
        self.health_texts = {}
        for health in range(11):  # Assuming max health is 10
            self.health_texts[health] = self.font.render(f"Health: {health}", True, (255, 255, 255))
        self.ammo_texts = {}
        for ammo in range(11):
            self.ammo_texts[ammo] = self.font.render(f"Ammo: {ammo}", True, (255, 255, 255))
        self.score_text = self.font.render("Score: 0", True, (255, 255, 255))

    def get_direction_from_vector(self, direction_vector) -> Direction:
        if direction_vector is None or (direction_vector[0] == 0 and direction_vector[1] == 0):
            return None

        for direction, vector in DIRECTION_VECTORS.items():
            if vector[0] == direction_vector[0] and vector[1] == direction_vector[1]:
                return direction

        return Direction.DOWN

    def render(self, player, enemies, bullets, explosions, walls, score, game_time) -> None:
        if self.screen is None:
            return
        self.screen.blit(self.asset_manager.floor_background, (0, 0))

        for wall in walls:
            self.screen.blit(self.asset_manager.textures["wall"], wall.rect)

        for enemy in enemies:
            enemy_direction = None
            if enemy.orientation:
                enemy_direction = self.get_direction_from_vector(enemy.orientation)

            enemy_texture = self.asset_manager.textures["enemy"][enemy_direction]
            self.screen.blit(enemy_texture, enemy.rect)
        if player is not None:
            player_direction = None
            if player.orientation:
                player_direction = self.get_direction_from_vector(player.orientation)

            player_texture = self.asset_manager.textures["player"][player_direction]
            self.screen.blit(player_texture, player.rect)

        for bullet in bullets:
            if bullet.x is not None and bullet.y is not None:
                bullet_owner = "player" if bullet.owner == "player" else "enemy"
                bullet_texture = self.asset_manager.textures["bullet"][bullet_owner]

                self.screen.blit(bullet_texture, bullet.rect)

        self._render_explosions(explosions)
        self._render_ui(player, score, game_time)

        pygame.display.flip()

    def _render_explosions(self, explosions) -> None:
        for explosion in explosions:
            if (
                explosion.explosion_type in self.asset_manager.textures["explosion"]
                and len(self.asset_manager.textures["explosion"][explosion.explosion_type]) > 0
            ):
                frame_index = min(
                    explosion.frame,
                    len(self.asset_manager.textures["explosion"][explosion.explosion_type]) - 1,
                )
                explosion_texture = self.asset_manager.textures["explosion"][
                    explosion.explosion_type
                ][frame_index]

                if explosion.width < self.asset_manager.scaled_width:
                    explosion_texture = pygame.transform.scale(
                        explosion_texture,
                        (explosion.width, explosion.height),
                    )

                self.screen.blit(explosion_texture, explosion.rect)

    def _render_ui(self, player, score, game_time) -> None:
        ui_background = pygame.Surface((200, 100))
        ui_background.set_alpha(128)
        ui_background.fill((0, 0, 0))
        self.screen.blit(ui_background, (5, 5))

        if player is not None:
            self.screen.blit(self.health_texts[player.health], (10, 10))
            self.screen.blit(self.ammo_texts[player.ammunition], (10, 30))

        self.score_text = self.font.render(
            f"Score: {score}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(self.score_text, (10, 50))

        time_text = self.font.render(
            f"Time: {game_time:.1f}s",
            True,
            (255, 255, 255),
        )
        self.screen.blit(time_text, (10, 70))
