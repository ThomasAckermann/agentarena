import json
import random
from datetime import datetime

import pygame
from pygame.math import Vector2
from pygame import Rect

from agentarena.agent.agent import Agent
from agentarena.game.action import Action
from agentarena.game.entities.player import Player
from agentarena.game.entities.projectile import Projectile
from agentarena.game.level import Level
from agentarena.config import GameConfig

ASSET_PATH: str = "src/agentarena/assets"
LOG_PATH: str = "src/agentarena/data"
PLAYER_SCALE: float = 0.8



class Game:
    def __init__(
        self,
        screen: pygame.Surface,
        player_agent: Agent,
        enemy_agent: Agent,
        clock: pygame.time.Clock,
        config: GameConfig,

    ) -> None:
        self.screen: pygame.Surface = screen
        self.player_agent: Agent = player_agent
        self.enemy_agent: Agent = enemy_agent
        self.clock = clock
        self.episode_log: list[dict] = []
        self.player: Player | None = None
        self.enemies: list[Player] = []
        self.config = config
        self.textures: dict[str, pygame.Surface] = {}
        # Precomputed values
        self.bullet_width = int(self.config.block_width / 2)
        self.bullet_height = int(self.config.block_height / 2)
        self.scaled_width = self.config.block_width * PLAYER_SCALE
        self.scaled_height = self.config.block_height * PLAYER_SCALE
        # Spatial partitioning grid for collision detection
        self.grid_size = 100  # Size of each grid cell
        self.collision_grid = {}
        self.reset()

    def reset(self) -> None:
        self.load_textures()
        self.create_player()
        self.create_enemies()
        self.level: Level = Level(self.player, self.enemies, self.config)
        self.level.generate_level()
        self.setup_collision_grid()  # Initialize spatial partitioning
        self.events: list = []
        self.bullets: list[Projectile] = []
        self.dt: float = 1 / 60  # Fixed time step for predictable physics
        self.running = True
        self.static_map_data = {
            "walls": [{"wall_x": wall.x, "wall_y": wall.y} for wall in self.level.walls]
        }

    def setup_collision_grid(self):
        """Initialize spatial partitioning grid for collision detection"""
        self.collision_grid = {}
        # Add walls to the grid
        for wall in self.level.walls:
            self.add_to_collision_grid(wall, "wall")

    def add_to_collision_grid(self, obj, obj_type):
        """Add an object to the spatial partitioning grid"""
        grid_x1 = max(0, int(obj.x // self.grid_size))
        grid_y1 = max(0, int(obj.y // self.grid_size))
        grid_x2 = max(0, int((obj.x + obj.width) // self.grid_size))
        grid_y2 = max(0, int((obj.y + obj.height) // self.grid_size))

        for gx in range(grid_x1, grid_x2 + 1):
            for gy in range(grid_y1, grid_y2 + 1):
                grid_key = (gx, gy)
                if grid_key not in self.collision_grid:
                    self.collision_grid[grid_key] = []
                self.collision_grid[grid_key].append((obj, obj_type))

    def get_objects_near(self, obj, obj_types=None):
        """Get objects near the specified object based on grid location"""
        grid_x1 = max(0, int(obj.x // self.grid_size))
        grid_y1 = max(0, int(obj.y // self.grid_size))
        grid_x2 = max(0, int((obj.x + obj.width) // self.grid_size))
        grid_y2 = max(0, int((obj.y + obj.height) // self.grid_size))

        nearby_objects = []
        for gx in range(grid_x1, grid_x2 + 1):
            for gy in range(grid_y1, grid_y2 + 1):
                grid_key = (gx, gy)
                if grid_key in self.collision_grid:
                    for grid_obj, grid_obj_type in self.collision_grid[grid_key]:
                        if obj_types is None or grid_obj_type in obj_types:
                            nearby_objects.append((grid_obj, grid_obj_type))

        return nearby_objects


    def create_player(self):
        player_position = [
            random.randint(

                2 * self.config.block_width,
                self.config.display_width - 2 * self.config.block_width,
            ),
            random.randint(
                2 * self.config.block_height,
                self.config.display_height - 2 * self.config.block_height,

            ),
        ]
        player_orientation = [0, 1]

        self.player = Player(
            x=player_position[0],
            y=player_position[1],
            height=self.scaled_height,
            width=self.scaled_width,
            orientation=player_orientation,
            agent=self.player_agent,
            speed=self.config.player_speed,
        )

    def create_enemies(self) -> None:
        for _ in range(self.config.max_enemies):
            enemy_position = [
                random.randint(
                    2 * self.config.block_width,
                    self.config.display_width - 2 * self.config.block_width,
                ),
                random.randint(
                    2 * self.config.block_height,
                    self.config.display_height - 2 * self.config.block_height,
                ),
            ]
            enemy_orientation = [0, 1]
            self.enemies.append(
                Player(
                    x=enemy_position[0],
                    y=enemy_position[1],
                    height=self.scaled_height,
                    width=self.scaled_width,

                    orientation=enemy_orientation,
                    agent=self.enemy_agent,
                    speed=self.config.player_speed,
                )
            )

    def get_observation(self, agent_id: str = "player") -> dict:
        # Create observations more efficiently
        if agent_id == "player":
            return {
                "player": {
                    "x": self.player.x,
                    "y": self.player.y,
                    "orientation": self.player.orientation,
                    "health": self.player.health,
                },
                "enemies": [
                    {
                        "x": enemy.x,
                        "y": enemy.y,
                        "orientation": enemy.orientation,
                        "health": enemy.health,
                    }
                    for enemy in self.enemies
                ],
                "bullets": [
                    {
                        "x": bullet.x,
                        "y": bullet.y,
                        "direction": bullet.direction,
                        "owner": bullet.owner,
                    }
                    for bullet in self.bullets
                ],
            }
        else:
            idx = int(agent_id.replace("enemy_", ""))
            enemy = self.enemies[idx]
            return {
                "player": {
                    "x": enemy.x,
                    "y": enemy.y,
                    "orientation": enemy.orientation,
                    "health": enemy.health,
                },
                "enemies": [
                    {
                        "x": self.player.x,
                        "y": self.player.y,
                        "orientation": self.player.orientation,
                        "health": self.player.health,
                    }
                ],
                "bullets": [
                    {
                        "x": bullet.x,
                        "y": bullet.y,
                        "direction": bullet.direction,
                        "owner": bullet.owner,
                    }
                    for bullet in self.bullets
                ],
            }

    def update(self) -> None:
        # Use proper time delta for movement
        self.dt = self.clock.tick(60) / 1000.0

        self.events = []
        # At the start of episode, store walls once
        if len(self.episode_log) == 0:
            self.episode_log.append({"static": self.static_map_data})
        if self.player.health <= 0:
            print("Game Over! Player defeated.")
            self.save_episode_log()
            self.running = False
            return

        # Batch process player and enemy actions
        player_observation = self.get_observation("player")
        player_action = self.player.agent.get_action(player_observation)
        self.apply_action("player", self.player, player_action)

        # Process enemies in batch
        for i, enemy in enumerate(self.enemies):
            enemy_observation = self.get_observation(f"enemy_{i}")
            enemy_action = enemy.agent.get_action(enemy_observation)
            self.apply_action(f"enemy_{i}", enemy, enemy_action)

        # Update positions of all bullets at once
        self.move_bullets()

        # Update collision grid with current positions
        self.update_entity_positions()

        # Check for collisions efficiently
        self.check_collisions()

        # Log game state
        self.episode_log.append(
            {
                "observation": self.get_observation("player"),
                "action": player_action.dict(),
                "events": self.events.copy(),
                "done": not self.running,
            }
        )

        # Render the frame
        self.render()

    def update_entity_positions(self):
        """Update the grid with current positions of dynamic entities"""
        # We'll rebuild the dynamic part of the grid each frame
        # (Static objects like walls stay in place)
        self.collision_grid = {}
        self.setup_collision_grid()  # Re-add walls

        # Add player to the grid
        self.add_to_collision_grid(self.player, "player")

        # Add enemies to the grid
        for i, enemy in enumerate(self.enemies):
            self.add_to_collision_grid(enemy, f"enemy_{i}")

        # Add bullets to the grid
        for bullet in self.bullets:
            self.add_to_collision_grid(bullet, f"bullet_{bullet.owner}")

    def apply_action(
        self,
        agent_id: str,
        player: Player,
        action: Action,
    ) -> None:
        # Cooldown and reload mechanics
        if player.cooldown > 0:
            player.cooldown -= 1
        if player.cooldown == 0 and player.is_reloading:
            player.is_reloading = False
            player.ammunition = 3

        fps = self.config.fps

        # Movement processing with vector math
        if action.direction is not None:
            dx, dy = action.get_direction_vector()
            player.orientation = [dx, dy]
            old_x, old_y = player.x, player.y
            movement_vector = Vector2(
                dx * self.dt * player.speed, dy * self.dt * player.speed
            )

            # Apply movement
            player.x += movement_vector.x
            player.y += movement_vector.y

            # Check for wall collisions efficiently using the grid
            collision_detected = False
            nearby_objects = self.get_objects_near(player, ["wall"])

            for wall_obj, _ in nearby_objects:
                if player.rect.colliderect(wall_obj.rect):
                    collision_detected = True
                    break

            if collision_detected:
                player.x, player.y = old_x, old_y

        # Shooting mechanics
        if (
            action.is_shooting is True
            and player.ammunition > 0
            and player.cooldown == 0
        ):
            dx, dy = player.orientation

            # Calculate bullet spawn position (center of player)
            center_x = player.x + (player.width - self.bullet_width) / 2
            center_y = player.y + (player.height - self.bullet_height) / 2

            # Offset bullet in shooting direction
            offset = 5
            center_x += dx * offset
            center_y += dy * offset

            # Create new bullet
            self.bullets.append(
                Projectile(
                    x=center_x,
                    y=center_y,
                    direction=[dx, dy],
                    owner=agent_id,
                    width=self.bullet_width,
                    height=self.bullet_height,
                )
            )

            # Update ammunition and cooldown
            player.ammunition -= 1
            player.cooldown = int(fps * 0.25)

            if player.ammunition == 0:
                player.cooldown += 1 * fps
                player.is_reloading = True

    def move_bullets(self):
        # Process all bullets at once using list comprehension for efficiency
        screen_rect = Rect(0, 0, self.config.display_width, self.config.display_height)
        bullet_speed = self.config.bullet_speed

        new_bullets = []
        for bullet in self.bullets:
            dx, dy = bullet.direction
            bullet.x += self.dt * dx * bullet_speed
            bullet.y += self.dt * dy * bullet_speed

            # Skip bullets that are off-screen
            if not screen_rect.collidepoint(bullet.x, bullet.y):
                continue

            # Check for wall collisions efficiently
            collision_detected = False
            nearby_walls = self.get_objects_near(bullet, ["wall"])

            for wall_obj, _ in nearby_walls:
                if bullet.rect.colliderect(wall_obj.rect):
                    collision_detected = True
                    break

            if not collision_detected:
                new_bullets.append(bullet)

        self.bullets = new_bullets

    def check_collisions(self) -> None:
        # More efficient collision detection using spatial partitioning
        bullets_to_remove = set()

        # Process player bullets
        for bullet_idx, bullet in enumerate(self.bullets):
            if bullet_idx in bullets_to_remove:
                continue

            if bullet.owner == "player":
                # Check for enemy hits

                for i, enemy in enumerate(self.enemies):
                    if bullet.rect.colliderect(enemy.rect):
                        self.events.append({"type": "enemy_hit", "enemy_id": i})
                        enemy.health -= 1

                        bullets_to_remove.add(bullet_idx)
                        if enemy.health <= 0:
                            print(f"Enemy {i} defeated")
                            self.enemies.remove(enemy)
                        break
            else:
                # Check for player hits
                if bullet.rect.colliderect(self.player.rect):
                    self.events.append({"type": "player_hit"})
                    self.player.health -= 1
                    print(f"Player hit! Health: {self.player.health}")
                    bullets_to_remove.add(bullet_idx)

        # Remove bullets in reverse order (to avoid index shifting problems)
        for bullet_idx in sorted(bullets_to_remove, reverse=True):
            if bullet_idx < len(self.bullets):  # Safety check
                self.bullets.pop(bullet_idx)

    def save_episode_log(self) -> None:

        filename = f"{LOG_PATH}/episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.episode_log, f)

    def load_textures(self) -> None:
        # Load textures once and cache them
        self.textures = {
            "player": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/player.png"),
                (self.scaled_width, self.scaled_height),
            ),
            "enemy": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/enemy.png"),
                (self.scaled_width, self.scaled_height),
            ),
            "bullet": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/bullet.png"),
                (self.bullet_width, self.bullet_height),
            ),
            "wall": pygame.transform.scale(
                pygame.image.load(f"{ASSET_PATH}/wall.png"),
                (self.config.block_width, self.config.block_height),
            ),
        }

        # Pre-render health text options
        self.font = pygame.font.SysFont("Arial", 20)
        self.health_texts = {}
        for health in range(0, 11):  # Assuming max health is 10
            self.health_texts[health] = self.font.render(
                f"Health: {health}", True, (255, 255, 255)
            )
        self.ammo_texts = {}
        for ammo in range(0, 11):  # Assuming max health is 10
            self.ammo_texts[ammo] = self.font.render(
                f"Ammo: {ammo}", True, (255, 255, 255)
            )

    def render(self):
        # Optimize rendering by doing grouped operations
        self.screen.fill((0, 0, 0))  # Clear screen

        # Batch rendering by type
        # 1. Walls
        for wall in self.level.walls:
            self.screen.blit(self.textures["wall"], wall.rect)

        # 2. Enemies
        for enemy in self.enemies:
            self.screen.blit(self.textures["enemy"], enemy.rect)

        # 3. Player
        self.screen.blit(self.textures["player"], self.player.rect)

        # 4. Bullets
        for bullet in self.bullets:
            self.screen.blit(self.textures["bullet"], bullet.rect)

        # 5. UI elements - use pre-rendered text
        self.screen.blit(self.health_texts[self.player.health], (10, 10))
        self.screen.blit(self.ammo_texts[self.player.ammunition], (10, 30))

        # Update display
        pygame.display.flip()
