import random

import pygame
from agent.agent import Agent


class Game:
    def __init__(
        self,
        screen,
        headless: bool,
        player_agent: Agent,
        enemy_agent: Agent,
        grid_size: int = 15,
    ) -> None:
        self.screen = screen
        self.headless = headless
        self.player_agent = player_agent
        self.enemy_agent = enemy_agent
        self.enemy_agents = None
        self.grid_size = grid_size
        self.reset()

    def reset(self) -> None:
        self.level = self.generate_level()
        self.player_pos = self.level["player_start"]
        self.player_direction = "UP"
        self.player_health = 3
        self.player_shots = 0
        self.player_cooldown = 0

        self.enemy_positions = self.level["enemy_starts"]
        self.enemy_directions = ["DOWN" for _ in self.enemy_positions]
        self.enemy_health = [2 for _ in self.enemy_positions]

        self.bullets = []
        self.running = True

    def generate_level(self) -> dict:
        # Generate player position
        player_start = [
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        ]

        # Generate enemies
        num_enemies = random.randint(1, 3)
        self.enemy_agents = [self.enemy_agent for i in range(num_enemies)]
        enemy_starts = []
        while len(enemy_starts) < num_enemies:
            pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]
            if pos != player_start and pos not in enemy_starts:
                enemy_starts.append(pos)

        return {"player_start": player_start, "enemy_starts": enemy_starts}

    def get_observation(self, agent_id: str = "player") -> dict:
        if agent_id == "player":
            return {
                "player": tuple(self.player_pos),
                "player_dir": self.player_direction,
                "enemies": [tuple(pos) for pos in self.enemy_positions],
                "bullets": self.bullets,
                "health": self.player_health,
            }
        else:
            idx = int(agent_id.replace("enemy_", ""))
            return {
                "player": self.enemy_positions[idx],
                "player_dir": self.enemy_directions[idx],
                "enemies": [self.player_pos],
                "bullets": self.bullets,
                "health": self.enemy_health[idx],
            }

    def update(self) -> None:
        if self.player_health <= 0:
            print("Game Over! Player defeated.")
            self.running = False
            return

        player_obs = self.get_observation("player")
        player_action = self.player_agent.get_action(player_obs)

        enemy_actions = []
        for i, agent in enumerate(self.enemy_agents):
            obs = self.get_observation(f"enemy_{i}")
            enemy_actions.append(agent.get_action(obs))

        self.apply_action("player", self.player_pos, player_action)

        for i, action in enumerate(enemy_actions):
            self.apply_action(f"enemy_{i}", self.enemy_positions[i], action)

        self.move_bullets()
        self.check_collisions()
        self.render()

    def apply_action(self, agent_type, entity_pos, action: str) -> None:
        direction_attr = "player_direction" if agent_type == "player" else None
        if "enemy_" in agent_type:
            idx = int(agent_type.replace("enemy_", ""))
            direction_attr = self.enemy_directions

        if action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if direction_attr:
                if isinstance(direction_attr, list):
                    direction_attr[idx] = action
                else:
                    setattr(self, direction_attr, action)

            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[
                action
            ]
            entity_pos[0] = max(0, min(self.grid_size - 1, entity_pos[0] + dx))
            entity_pos[1] = max(0, min(self.grid_size - 1, entity_pos[1] + dy))

        elif action == "SHOOT":
            direction = (
                self.player_direction
                if agent_type == "player"
                else self.enemy_directions[idx]
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
        for bullet in self.bullets:
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[
                bullet["dir"]
            ]
            bullet["pos"][0] += dx
            bullet["pos"][1] += dy

        self.bullets = [
            b
            for b in self.bullets
            if 0 <= b["pos"][0] < self.grid_size and 0 <= b["pos"][1] < self.grid_size
        ]

        if self.player_cooldown > 0:
            self.player_cooldown -= 1
        if self.player_cooldown == 0:
            self.player_shots = 0

    def check_collisions(self):
        for bullet in self.bullets[:]:
            if bullet["owner"] == "player":
                for i, pos in enumerate(self.enemy_positions):
                    if bullet["pos"] == pos:
                        self.enemy_health[i] -= 1
                        self.bullets.remove(bullet)
                        if self.enemy_health[i] <= 0:
                            print(f"Enemy {i} defeated")
                            del self.enemy_positions[i]
                            del self.enemy_directions[i]
                            del self.enemy_health[i]
                        break
            else:
                if bullet["pos"] == self.player_pos:
                    self.player_health -= 1
                    print(f"Player hit! Health: {self.player_health}")
                    self.bullets.remove(bullet)

    def render(self) -> None:
        if self.headless:
            return

        self.screen.fill((0, 0, 0))
        tile_size = 40

        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (*[x * tile_size for x in self.player_pos], tile_size, tile_size),
        )

        for pos in self.enemy_positions:
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (*[x * tile_size for x in pos], tile_size, tile_size),
            )

        for bullet in self.bullets:
            pygame.draw.rect(
                self.screen,
                (255, 255, 0),
                (
                    *[x * tile_size for x in bullet["pos"]],
                    tile_size // 2,
                    tile_size // 2,
                ),
            )

        font = pygame.font.SysFont(None, 30)
        text = font.render(f"HP: {self.player_health}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
