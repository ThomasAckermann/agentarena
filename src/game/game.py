import json
import random
from datetime import datetime

import pygame
from agent.agent import Agent

from game.action import Action
from game.level import Level
from game.player import Player


class Game:
    def __init__(
        self,
        screen,
        headless: bool,
        player_agent: Agent,
        enemy_agent: Agent,
        grid_size: int = 15,
        enemy_count: int = 1,
    ) -> None:
        self.screen = screen
        self.headless: bool = headless
        self.player_agent: Agent = player_agent
        self.enemy_agent: Agent = enemy_agent
        self.grid_size: int = grid_size
        self.episode_log: list[dict] = []
        self.enemy_count: int = enemy_count
        self.player: Player | None = None
        self.enemies: list[Player] = []
        self.tile_size = 40
        self.load_textures()
        self.reset()

    def reset(self) -> None:
        self.create_player()
        self.create_enemies()
        self.level: Level = Level(self.player, self.enemies, self.grid_size)
        self.level.generate_level()
        self.events: int = []
        self.bullets = []
        self.running = True

    def create_player(self):
        player_position = [
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        ]
        player_orientation = [0, 1]

        self.player = Player(
            player_position,
            player_orientation,
            self.player_agent,
        )

    def create_enemies(self) -> None:
        for i in range(self.enemy_count):
            enemy_position = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]
            enemy_orientation = [0, 1]
            self.enemies.append(
                Player(enemy_position, enemy_orientation, self.enemy_agent)
            )

    def get_observation(self, agent_id: str = "player") -> dict:
        if agent_id == "player":
            return {
                "player": tuple(self.player.position),
                "player_dir": self.player.orientation,
                "enemies": [tuple(enemy.position) for enemy in self.enemies],
                "bullets": self.bullets,
                "health": self.player.health,
            }
        else:
            idx = int(agent_id.replace("enemy_", ""))
            enemy = self.enemies[idx]
            return {
                "player": enemy.position,
                "player_dir": enemy.orientation,
                "enemies": [self.player.position],
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
        """Apply an action to the specified player"""

        # Move action
        if action.direction is not None:
            # Get the direction vector
            dx, dy = action.get_direction_vector()

            # Update player orientation based on movement direction
            player.orientation = [dx, dy]

            # Calculate new position with boundary checks
            new_x = max(0, min(self.grid_size - 1, player.position[0] + dx))
            new_y = max(0, min(self.grid_size - 1, player.position[1] + dy))

            # Check for wall collisions
            if (new_x, new_y) not in self.level.walls:
                player.position[0] = new_x
                player.position[1] = new_y

        # Shoot action
        if action.is_shooting is True:
            dx, dy = player.orientation

            # Check if player can shoot (for cooldown/ammo logic)
            if player.ammunition < 3 and player.cooldown == 0:
                self.bullets.append(
                    {
                        "pos": list(player.position),
                        "dir": [dx, dy],
                        "owner": agent_id,
                    }
                )
                player.ammunition += 1
                if player.ammunition == 3:
                    player.cooldown = 5

    def move_bullets(self):
        new_bullets = []
        for bullet in self.bullets:
            # Use the direction vector directly
            dx, dy = bullet["dir"]
            new_x = bullet["pos"][0] + dx
            new_y = bullet["pos"][1] + dy

            # Check bounds
            if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                continue

            # Check wall collision
            if (new_x, new_y) in self.level.walls:
                continue  # Bullet is destroyed on impact

            # If no collision, move bullet
            bullet["pos"] = [new_x, new_y]
            new_bullets.append(bullet)

        self.bullets = new_bullets

        # Update player cooldown and shot count
        if self.player.cooldown > 0:
            self.player.cooldown -= 1
        if self.player.cooldown == 0:
            self.player.ammunition = 0

    def check_collisions(self) -> None:
        for bullet in self.bullets[:]:
            if bullet["owner"] == "player":
                for i, enemy in enumerate(self.enemies):
                    if bullet["pos"] == enemy.position:
                        self.events.append({"type": "enemy_hit", "enemy_id": i})
                        enemy.health -= 1
                        self.bullets.remove(bullet)
                        if enemy.health <= 0:
                            print(f"Enemy {i} defeated")
                            del self.enemies[i]
                        break
            else:
                if bullet["pos"] == self.player.position:
                    self.events.append({"type": "player_hit"})
                    self.player.health -= 1
                    print(f"Player hit! Health: {self.player.health}")
                    self.bullets.remove(bullet)

    def save_episode_log(self):
        filename = f"data/episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.episode_log, f)

    def load_textures(self):
        self.textures = {
            "player": pygame.transform.scale(
                pygame.image.load("assets/player.png"), (self.tile_size, self.tile_size)
            ),
            "enemy": pygame.transform.scale(
                pygame.image.load("assets/enemy.png"), (self.tile_size, self.tile_size)
            ),
            "bullet": pygame.transform.scale(
                pygame.image.load("assets/bullet.png"),
                (self.tile_size // 2, self.tile_size // 2),
            ),
            "wall": pygame.transform.scale(
                pygame.image.load("assets/wall.png"), (self.tile_size, self.tile_size)
            ),
        }

    def render(self) -> None:
        if self.headless:
            return

        self.screen.fill((0, 0, 0))

        # Draw player
        px, py = self.player.position
        self.screen.blit(
            self.textures["player"], (px * self.tile_size, py * self.tile_size)
        )

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy.position
            self.screen.blit(
                self.textures["enemy"], (ex * self.tile_size, ey * self.tile_size)
            )

        # Draw bullets
        for bullet in self.bullets:
            bx, by = bullet["pos"]
            bullet_pos = (
                bx * self.tile_size + self.tile_size // 4,
                by * self.tile_size + self.tile_size // 4,
            )
            self.screen.blit(self.textures["bullet"], bullet_pos)

        # Draw walls
        for wall in self.level.walls:
            wx, wy = wall
            self.screen.blit(
                self.textures["wall"], (wx * self.tile_size, wy * self.tile_size)
            )

        # UI
        font = pygame.font.SysFont(None, 30)
        text = font.render(f"HP: {self.player.health}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
