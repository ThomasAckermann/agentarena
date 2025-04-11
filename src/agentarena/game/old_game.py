import json
import random
from datetime import datetime

import pygame
from agentarena.agent.agent import Agent

from agentarena.game.action import Action
from agentarena.game.entities.player import Player
from agentarena.game.entities.projectile import Projectile
from agentarena.game.level import Level
from agentarena.config import GameConfig


ASSET_PATH: str = "src/agentarena/assets"
LOG_PATH: str = "src/agentarena/data"


class Game:
    def __init__(
        self,
        screen: pygame.Surface,
        player_agent: Agent,
        enemy_agent: Agent,
        clock: pygame.time.Clock,
        config: GameConfig,
        grid_size: int = 15,
        enemy_count: int = 1,
    ) -> None:
        self.screen: pygame.Surface = screen
        self.player_agent: Agent = player_agent
        self.enemy_agent: Agent = enemy_agent
        self.clock = clock
        self.grid_size: int = grid_size
        self.episode_log: list[dict] = []
        self.enemy_count: int = enemy_count
        self.player: Player | None = None
        self.enemies: list[Player] = []
        self.tile_size: int = 40
        self.dt: float = self.clock.tick(60) / 1000.0
        self.config = config
        self.load_textures()
        self.reset()

    def reset(self) -> None:
        self.create_player()
        self.create_enemies()
        self.level: Level = Level(
            self.player, self.enemies, self.grid_size, self.config
        )
        self.level.generate_level()
        self.events: list = []
        self.bullets: list[Projectile] = []
        self.running = True

    def create_player(self):
        player_position = [
            random.randint(0, self.config.display_width - 1),
            random.randint(0, self.config.display_height - 1),
        ]
        player_orientation = [0, 1]

        self.player = Player(
            x=player_position[0],
            y=player_position[1],
            height=self.config.block_height,
            width=self.config.block_width,
            orientation=player_orientation,
            agent=self.player_agent,
        )

    def create_enemies(self) -> None:
        for i in range(self.enemy_count):
            enemy_position = [
                random.randint(0, self.config.display_width - 1),
                random.randint(0, self.config.display_height - 1),
            ]
            enemy_orientation = [0, 1]
            self.enemies.append(
                Player(
                    x=enemy_position[0],
                    y=enemy_position[1],
                    height=self.config.block_height,
                    width=self.config.block_width,
                    orientation=enemy_orientation,
                    agent=self.enemy_agent,
                )
            )

    def get_observation(self, agent_id: str = "player") -> dict:
        if agent_id == "player":
            return {
                "player_x": self.player.x,
                "player_y": self.player.y,
                "player_dir": self.player.orientation,
                "enemies": [tuple((enemy.x, enemy.y)) for enemy in self.enemies],
                "bullets": self.bullets,
                "health": self.player.health,
            }
        else:
            idx = int(agent_id.replace("enemy_", ""))
            enemy = self.enemies[idx]
            return {
                "player_x": enemy.x,
                "player_y": enemy.y,
                "player_dir": enemy.orientation,
                "enemies": [tuple((self.player.x, self.player.y))],
                "bullets": self.bullets,
                "health": enemy.health,
            }

    def update(self) -> None:
        self.events = []
        if self.player.health <= 0:
            print("Game Over! Player defeated.")
            self.save_episode_log()
            self.running = False
            return

        player_observation = self.get_observation("player")
        player_action = self.player.agent.get_action(player_observation)
        self.apply_action("player", self.player, player_action)

        for i, enemy in enumerate(self.enemies):
            enemy_observation = self.get_observation(f"enemy_{i}")
            enemy_action = enemy.agent.get_action(enemy_observation)
            self.apply_action(f"enemy_{i}", enemy, enemy_action)

        self.move_bullets()
        self.check_collisions()
        self.episode_log.append(
            {
                "observation": self.get_observation("player"),
                "action": player_action,
                "events": self.events.copy(),
                "done": not self.running,
            }
        )
        self.render()

    def apply_action(
        self,
        agent_id: str,
        player: Player,
        action: Action,
    ) -> None:
        if player.cooldown > 0:
            player.cooldown -= 1
        if player.cooldown == 0 and player.is_reloading:
            player.is_reloading = False
            player.ammunition = 3

        fps = self.config.fps

        if action.direction is not None:
            dx, dy = action.get_direction_vector()
            player.orientation = [dx, dy]

            old_x = player.x
            old_y = player.y
            dx, dy = action.get_direction_vector()

            player.orientation = [dx, dy]

            # Move in pixels directly
            player.x += self.dt * player.speed * dx
            player.y += self.dt * player.speed * dy

            # Collision check using rects
            player_rect = pygame.Rect(
                player.x,
                player.y,
                player.width,
                player.height,
            )
            for wall in self.level.walls:
                if player_rect.colliderect(wall.rect):
                    player.x = old_x
                    player.y = old_y
                    break

            player.rect = player_rect

        if action.is_shooting is True:
            dx, dy = player.orientation
            if player.ammunition > 0 and player.cooldown == 0:
                self.bullets.append(
                    Projectile(
                        x=player.x,
                        y=player.y,
                        direction=[dx, dy],
                        owner=agent_id,
                        width=int(self.config.block_width / 2),
                        height=int(self.config.block_height / 2),
                    )
                )
                player.ammunition -= 1
                player.cooldown = int(fps * 0.25)

                if player.ammunition == 0:
                    player.cooldown += 1 * fps
                    player.is_reloading = True

    def move_bullets(self):
        new_bullets = []
        bullet_speed = self.config.bullet_speed
        for bullet in self.bullets:
            dx, dy = bullet.direction
            bullet.x += self.dt * dx * bullet_speed
            bullet.y += self.dt * dy * bullet_speed

            if not (0 <= bullet.x < self.grid_size and 0 <= bullet.y < self.grid_size):
                continue

            for wall in self.level.walls:
                if bullet.rect.colliderect(wall.rect):
                    break
            else:
                new_bullets.append(bullet)

        self.bullets = new_bullets

    def check_collisions(self) -> None:
        for bullet in self.bullets[:]:
            if bullet.owner == "player":
                for i, enemy in enumerate(self.enemies):
                    if bullet.rect.colliderect(enemy.rect):
                        self.events.append({"type": "enemy_hit", "enemy_id": i})
                        enemy.health -= 1
                        self.bullets.remove(bullet)
                        if enemy.health <= 0:
                            print(f"Enemy {i} defeated")
                            self.enemies.remove(enemy)
                        break
            else:
                if bullet.rect.colliderect(self.player.rect):
                    self.events.append({"type": "player_hit"})
                    self.player.health -= 1
                    print(f"Player hit! Health: {self.player.health}")
                    self.bullets.remove(bullet)

    def save_episode_log(self):
        filename = f"{LOG_PATH}/episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.episode_log, f)

    def load_textures(self):
        self.textures = {
            "player": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/player.png"),
                (self.tile_size, self.tile_size),
            ),
            "enemy": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/enemy.png"),
                (self.tile_size, self.tile_size),
            ),
            "bullet": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/bullet.png"),
                (self.tile_size // 2, self.tile_size // 2),
            ),
            "wall": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/wall.png"),
                (self.tile_size, self.tile_size),
            ),
        }

    def render(self):
        self.screen.fill((0, 0, 0))  # Clear screen

        for wall in self.level.walls:
            self.screen.blit(self.textures["wall"], wall.rect)

        for enemy in self.enemies:
            self.screen.blit(self.textures["enemy"], enemy.rect)

        self.screen.blit(self.textures["player"], self.player.rect)

        for bullet in self.bullets:
            self.screen.blit(self.textures["bullet"], bullet.rect)

        font = pygame.font.SysFont("Arial", 18)
        health_text = font.render(
            f"Health: {self.player.health}", True, (255, 255, 255)
        )
        self.screen.blit(health_text, (10, 10))

        pygame.display.flip()
