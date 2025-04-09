import json
import random
from datetime import datetime

import pygame
from agent.agent import Agent

from game.level import Level
from game.player import Player


class Game:
    def __init__(
        self,
        screen,
        headless: bool,
        player: Player,
        grid_size: int = 15,
    ) -> None:
        self.screen = screen
        self.headless: bool = headless
        self.player: Player = player
        self.grid_size: int = grid_size
        self.walls: set = set()
        self.episode_log: list[dict] = []
        self.level = Level(self.player)
        self.reset()

    def reset(self) -> None:
        self.level.generate_level()

        self.player_pos = self.level["player_start"]
        self.player_direction = "UP"
        self.player.health = 3
        self.player_shots = 0
        self.player_cooldown = 0
        self.events: int = []

        self.enemies = [
            {"agent": self.enemy_agent, "pos": pos, "dir": "DOWN", "health": 2}
            for pos in self.level["enemy_starts"]
        ]

        self.bullets = []
        self.running = True

    def get_observation(self, agent_id: str = "player") -> dict:
        if agent_id == "player":
            return {
                "player": tuple(self.player_pos),
                "player_dir": self.player_direction,
                "enemies": [tuple(enemy["pos"]) for enemy in self.enemies],
                "bullets": self.bullets,
                "health": self.player_health,
            }
        else:
            idx = int(agent_id.replace("enemy_", ""))
            enemy = self.enemies[idx]
            return {
                "player": enemy["pos"],
                "player_dir": enemy["dir"],
                "enemies": [self.player_pos],
                "bullets": self.bullets,
                "health": enemy["health"],
            }

    def update(self) -> None:
        self.events = []
        if self.player_health <= 0:
            print("Game Over! Player defeated.")
            self.save_episode_log()
            self.running = False
            return

        player_obs = self.get_observation("player")
        player_action = self.player_agent.get_action(player_obs)
        self.apply_action("player", self.player_pos, player_action)

        for i, enemy in enumerate(self.enemies):
            obs = self.get_observation(f"enemy_{i}")
            action = enemy["agent"].get_action(obs)
            self.apply_action(f"enemy_{i}", enemy["pos"], action)

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

    def apply_action(self, agent_type, entity_pos, action: str) -> None:
        direction_attr = "player_direction" if agent_type == "player" else None
        if "enemy_" in agent_type:
            idx = int(agent_type.replace("enemy_", ""))
            direction_attr = self.enemies[idx]["dir"]

        if action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if agent_type == "player":
                self.player_direction = action
            else:
                self.enemies[idx]["dir"] = action

            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[
                action
            ]
            new_x = max(0, min(self.grid_size - 1, entity_pos[0] + dx))
            new_y = max(0, min(self.grid_size - 1, entity_pos[1] + dy))

            if (new_x, new_y) not in self.walls:
                entity_pos[0] = new_x
                entity_pos[1] = new_y

        elif action == "SHOOT":
            direction = (
                self.player_direction
                if agent_type == "player"
                else self.enemies[idx]["dir"]
            )
            if agent_type == "player":
                if self.player_shots < 3 and self.player_cooldown == 0:
                    self.bullets.append(
                        {"pos": list(entity_pos), "dir": direction, "owner": "player"}
                    )
                    self.player_shots += 1
                    if self.player_shots == 3:
                        self.player_cooldown = 5
            else:
                self.bullets.append(
                    {"pos": list(entity_pos), "dir": direction, "owner": agent_type}
                )

    def move_bullets(self):
        new_bullets = []
        for bullet in self.bullets:
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[
                bullet["dir"]
            ]
            new_x = bullet["pos"][0] + dx
            new_y = bullet["pos"][1] + dy

            # Check bounds
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                continue

            # Check wall collision
            if (new_x, new_y) in self.walls:
                continue  # Bullet is destroyed on impact

            # If no collision, move bullet
            bullet["pos"] = [new_x, new_y]
            new_bullets.append(bullet)

        self.bullets = new_bullets

        if self.player_cooldown > 0:
            self.player_cooldown -= 1
        if self.player_cooldown == 0:
            self.player_shots = 0

    def check_collisions(self) -> None:
        for bullet in self.bullets[:]:
            if bullet["owner"] == "player":
                for i, enemy in enumerate(self.enemies):
                    if bullet["pos"] == enemy["pos"]:
                        self.events.append({"type": "enemy_hit", "enemy_id": i})
                        enemy["health"] -= 1
                        self.bullets.remove(bullet)
                        if enemy["health"] <= 0:
                            print(f"Enemy {i} defeated")
                            del self.enemies[i]
                        break
            else:
                if bullet["pos"] == self.player_pos:
                    self.events.append({"type": "player_hit"})
                    self.player_health -= 1
                    print(f"Player hit! Health: {self.player_health}")
                    self.bullets.remove(bullet)

    def load_textures(self) -> None:
        self.tile_size = 40
        self.player_texture = pygame.image.load("assets/player.png").convert_alpha()
        self.enemy_texture = pygame.image.load("assets/enemy.png").convert_alpha()
        self.wall_texture = pygame.image.load("assets/wall.png").convert_alpha()

        self.player_texture = pygame.transform.scale(
            self.player_texture, (self.tile_size, self.tile_size)
        )
        self.enemy_texture = pygame.transform.scale(
            self.enemy_texture, (self.tile_size, self.tile_size)
        )
        self.wall_texture = pygame.transform.scale(
            self.wall_texture, (self.tile_size, self.tile_size)
        )

    def render(self) -> None:
        if self.headless:
            return

        self.screen.fill((0, 0, 0))
        self.load_textures()
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (
                *[x * self.tile_size for x in self.player_pos],
                self.tile_size,
                self.tile_size,
            ),
        )

        for enemy in self.enemies:
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (
                    *[x * self.tile_size for x in enemy["pos"]],
                    self.tile_size,
                    self.tile_size,
                ),
            )

        for bullet in self.bullets:
            pygame.draw.rect(
                self.screen,
                (255, 255, 0),
                (
                    *[x * self.tile_size for x in bullet["pos"]],
                    self.tile_size // 2,
                    self.tile_size // 2,
                ),
            )
        for wall in self.walls:
            rect = pygame.Rect(
                wall[1] * self.tile_size,
                wall[0] * self.tile_size,
                self.tile_size,
                self.tile_size,
            )
            pygame.draw.rect(self.screen, (100, 100, 100), rect)

        font = pygame.font.SysFont(None, 30)
        text = font.render(f"HP: {self.player_health}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def save_episode_log(self):
        filename = f"data/episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.episode_log, f)
