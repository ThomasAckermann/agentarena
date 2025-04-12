import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import pygame
from pygame.math import Vector2

from agentarena.agent.agent import Agent
from agentarena.game.action import Action, Direction
from agentarena.game.entities.player import Player
from agentarena.game.entities.projectile import Projectile
from agentarena.game.level import Level
from agentarena.models.config import GameConfig
from agentarena.models.entities import (
    EntityModel,
    PlayerModel,
    ProjectileModel,
    WallModel,
)
from agentarena.models.events import (
    BulletFiredEvent,
    CollisionEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    GameEvent,
    PlayerHitEvent,
)
from agentarena.models.observations import (
    BulletObservation,
    EnemyObservation,
    GameObservation,
    PlayerObservation,
)

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
        self.episode_log: List[Dict] = []
        self.player: Optional[Player] = None
        self.enemies: List[Player] = []
        self.config = config
        self.textures: Dict[str, pygame.Surface] = {}
        self.game_time: float = 0.0
        self.score: int = 0

        # Precomputed values
        self.bullet_width = int(self.config.block_width / 2)
        self.bullet_height = int(self.config.block_height / 2)
        self.scaled_width = int(self.config.block_width * PLAYER_SCALE)
        self.scaled_height = int(self.config.block_height * PLAYER_SCALE)

        # Spatial partitioning grid for collision detection
        self.grid_size = 100  # Size of each grid cell
        self.collision_grid: Dict[Tuple[int, int], List[Tuple[Union[Player, Projectile], str]]] = {}

        # Unique game ID for this session
        self.game_id = str(uuid.uuid4())

        self.reset()

    def reset(self) -> None:
        """Reset the game state for a new episode."""
        self.load_textures()
        self.create_player()
        self.create_enemies()
        self.level: Level = Level(self.player, self.enemies, self.config)
        self.level.generate_level()
        self.setup_collision_grid()  # Initialize spatial partitioning
        self.events: List[GameEvent] = []
        self.bullets: List[Projectile] = []
        self.dt: float = 1 / 60  # Fixed time step for predictable physics
        self.running = True
        self.game_time = 0.0
        self.score = 0

        # Store static map data
        walls_data = [
            WallModel(
                id=f"wall_{i}",
                x=wall.x,
                y=wall.y,
                width=wall.width,
                height=wall.height,
                entity_type="wall",
            ).model_dump()
            for i, wall in enumerate(self.level.walls)
        ]

        self.static_map_data = {"walls": walls_data}

    def setup_collision_grid(self) -> None:
        """Initialize spatial partitioning grid for collision detection."""
        self.collision_grid = {}
        # Add walls to the grid
        for wall in self.level.walls:
            self.add_to_collision_grid(wall, "wall")

    def add_to_collision_grid(self, obj: Union[Player, Projectile], obj_type: str) -> None:
        """Add an object to the spatial partitioning grid."""
        if obj.x is None or obj.y is None:
            return  # Skip objects without position

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

    def get_objects_near(
        self, obj: Union[Player, Projectile], obj_types: Optional[List[str]] = None
    ) -> List[Tuple[Union[Player, Projectile], str]]:
        """Get objects near the specified object based on grid location."""
        if obj.x is None or obj.y is None:
            return []  # Return empty list for objects without position

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

    def create_player(self) -> None:
        """Create the player entity."""
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

        # Create player data model
        player_model = PlayerModel(
            id="player",
            x=player_position[0],
            y=player_position[1],
            width=self.scaled_width,
            height=self.scaled_height,
            orientation=player_orientation,
            health=3,
            cooldown=0,
            ammunition=3,
            is_reloading=False,
            speed=self.config.player_speed,
        )

        # Create player entity using the data model
        self.player = Player(
            orientation=player_orientation,
            agent=self.player_agent,
            width=self.scaled_width,
            height=self.scaled_height,
            x=player_position[0],
            y=player_position[1],
            speed=self.config.player_speed,
        )

    def create_enemies(self) -> None:
        """Create enemy entities."""
        self.enemies = []

        for i in range(self.config.max_enemies):
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

            # Create enemy data model
            enemy_model = PlayerModel(
                id=f"enemy_{i}",
                x=enemy_position[0],
                y=enemy_position[1],
                width=self.scaled_width,
                height=self.scaled_height,
                orientation=enemy_orientation,
                health=2,  # Enemies have less health than player
                cooldown=0,
                ammunition=3,
                is_reloading=False,
                speed=self.config.player_speed,
            )

            # Create enemy entity using the data model
            self.enemies.append(
                Player(
                    orientation=enemy_orientation,
                    agent=self.enemy_agent,
                    width=self.scaled_width,
                    height=self.scaled_height,
                    x=enemy_position[0],
                    y=enemy_position[1],
                    speed=self.config.player_speed,
                ),
            )

    def get_observation(self, agent_id: str = "player") -> GameObservation:
        """
        Get the current game state observation for an agent.

        Args:
            agent_id: ID of the agent requesting the observation

        Returns:
            GameObservation: Structured game state observation
        """
        if agent_id == "player" and self.player is not None:
            # Player observation
            return GameObservation(
                player=PlayerObservation(
                    x=self.player.x if self.player.x is not None else 0,
                    y=self.player.y if self.player.y is not None else 0,
                    orientation=(
                        self.player.orientation if self.player.orientation is not None else [0, 0]
                    ),
                    health=self.player.health,
                    ammunition=self.player.ammunition,
                    cooldown=self.player.cooldown,
                    is_reloading=self.player.is_reloading,
                ),
                enemies=[
                    EnemyObservation(
                        x=enemy.x if enemy.x is not None else 0,
                        y=enemy.y if enemy.y is not None else 0,
                        orientation=enemy.orientation if enemy.orientation is not None else [0, 0],
                        health=enemy.health,
                    )
                    for enemy in self.enemies
                ],
                bullets=[
                    BulletObservation(
                        x=bullet.x if bullet.x is not None else 0,
                        y=bullet.y if bullet.y is not None else 0,
                        direction=bullet.direction,
                        owner=bullet.owner,
                    )
                    for bullet in self.bullets
                ],
                game_time=self.game_time,
                score=self.score,
            )
        else:
            # Enemy observation (reverse perspective)
            idx = int(agent_id.replace("enemy_", ""))
            if idx < len(self.enemies):
                enemy = self.enemies[idx]

                return GameObservation(
                    # From enemy's perspective, it is the "player"
                    player=PlayerObservation(
                        x=enemy.x if enemy.x is not None else 0,
                        y=enemy.y if enemy.y is not None else 0,
                        orientation=enemy.orientation if enemy.orientation is not None else [0, 0],
                        health=enemy.health,
                        ammunition=enemy.ammunition,
                        cooldown=enemy.cooldown,
                        is_reloading=enemy.is_reloading,
                    ),
                    # From enemy's perspective, the player is an "enemy"
                    enemies=(
                        [
                            EnemyObservation(
                                x=(
                                    self.player.x
                                    if self.player is not None and self.player.x is not None
                                    else 0
                                ),
                                y=(
                                    self.player.y
                                    if self.player is not None and self.player.y is not None
                                    else 0
                                ),
                                orientation=(
                                    self.player.orientation
                                    if self.player is not None
                                    and self.player.orientation is not None
                                    else [0, 0]
                                ),
                                health=self.player.health if self.player is not None else 0,
                            )
                        ]
                        if self.player is not None
                        else []
                    ),
                    bullets=[
                        BulletObservation(
                            x=bullet.x if bullet.x is not None else 0,
                            y=bullet.y if bullet.y is not None else 0,
                            direction=bullet.direction,
                            owner=bullet.owner,
                        )
                        for bullet in self.bullets
                    ],
                    game_time=self.game_time,
                    score=self.score,
                )

            # Fallback empty observation
            return GameObservation(
                player=PlayerObservation(
                    x=0,
                    y=0,
                    orientation=[0, 0],
                    health=0,
                ),
                enemies=[],
                bullets=[],
                game_time=self.game_time,
                score=self.score,
            )

    def update(self) -> None:
        """Update game state for the current frame."""
        # Update game time using delta time
        self.dt = self.clock.tick(60) / 1000.0
        self.game_time += self.dt

        # Clear events for this frame
        self.events = []

        # At the start of episode, store walls once
        if len(self.episode_log) == 0:
            self.episode_log.append({"static": self.static_map_data})

        # Check game over condition
        if self.player is not None and self.player.health <= 0:
            print("Game Over! Player defeated.")

            # Create game over event
            self.events.append(
                EntityDestroyedEvent(
                    timestamp=self.game_time,
                    entity_id="player",
                    entity_type="player",
                    position=(
                        self.player.x if self.player.x is not None else 0,
                        self.player.y if self.player.y is not None else 0,
                    ),
                )
            )

            self.save_episode_log()
            self.running = False
            return

        # Process player actions if player exists
        if self.player is not None:
            # Get player observation and action
            player_observation = self.get_observation("player")
            player_action = self.player.agent.get_action(player_observation)
            self.apply_action("player", self.player, player_action)

        # Process enemy actions
        for i, enemy in enumerate(self.enemies):
            enemy_observation = self.get_observation(f"enemy_{i}")
            enemy_action = enemy.agent.get_action(enemy_observation)
            self.apply_action(f"enemy_{i}", enemy, enemy_action)

        # Update positions of all bullets
        self.move_bullets()

        # Update collision grid with current positions
        self.update_entity_positions()

        # Check for collisions efficiently
        self.check_collisions()

        # Log game state - convert events to dictionaries for logging
        event_dicts = [event.model_dump() for event in self.events]

        # Track player action for the log
        player_action_dict = (
            player_action.model_dump()
            if self.player is not None
            else {"is_shooting": False, "direction": None}
        )

        self.episode_log.append(
            {
                "observation": self.get_observation("player").model_dump(),
                "action": player_action_dict,
                "events": event_dicts,
                "done": not self.running,
                "game_time": self.game_time,
            },
        )

        # Render the frame if we have a screen
        if self.screen is not None:
            self.render()

    def update_entity_positions(self) -> None:
        """Update the grid with current positions of dynamic entities."""
        # Rebuild the dynamic part of the grid each frame
        # (Static objects like walls stay in place)
        self.collision_grid = {}
        self.setup_collision_grid()  # Re-add walls

        # Add player to the grid
        if self.player is not None:
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
        """
        Apply an action to a player or enemy.

        Args:
            agent_id: ID of the agent performing the action
            player: Player entity to apply the action to
            action: Action to apply
        """
        # Cooldown and reload mechanics
        if player.cooldown > 0:
            player.cooldown -= 1
        if player.cooldown == 0 and player.is_reloading:
            player.is_reloading = False
            player.ammunition = 3

        # Movement processing with vector math
        if action.direction is not None:
            dx, dy = action.get_direction_vector()
            player.orientation = [dx, dy]

            # Skip movement if player has no position
            if player.x is None or player.y is None:
                return

            old_x, old_y = player.x, player.y
            movement_vector = Vector2(dx * self.dt * player.speed, dy * self.dt * player.speed)

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
        if action.is_shooting is True and player.ammunition > 0 and player.cooldown == 0:
            dx, dy = player.orientation if player.orientation is not None else [0, 0]

            # Skip shooting if player has no position
            if player.x is None or player.y is None:
                return

            # Calculate bullet spawn position (center of player)
            center_x = player.x + (player.width - self.bullet_width) / 2
            center_y = player.y + (player.height - self.bullet_height) / 2

            # Offset bullet in shooting direction
            offset = 5
            center_x += dx * offset
            center_y += dy * offset

            # Create bullet data model
            bullet_model = ProjectileModel(
                id=f"bullet_{agent_id}_{self.game_time:.3f}",
                x=int(center_x),
                y=int(center_y),
                width=self.bullet_width,
                height=self.bullet_height,
                direction=[dx, dy],
                owner=agent_id,
                speed=self.config.bullet_speed,
            )

            # Create new bullet entity
            bullet = Projectile(
                x=int(center_x),
                y=int(center_y),
                width=self.bullet_width,
                height=self.bullet_height,
                direction=[dx, dy],
                owner=agent_id,
                speed=self.config.bullet_speed,
            )

            self.bullets.append(bullet)

            # Create bullet fired event
            self.events.append(
                BulletFiredEvent(
                    timestamp=self.game_time,
                    owner_id=agent_id,
                    direction=(dx, dy),
                    position=(center_x, center_y),
                )
            )

            # Update ammunition and cooldown
            player.ammunition -= 1
            player.cooldown = int(self.config.fps * 0.25)

            if player.ammunition == 0:
                player.cooldown += 1 * self.config.fps
                player.is_reloading = True

    def move_bullets(self) -> None:
        """Update positions of all bullets and handle wall collisions."""
        # Create screen boundary rectangle
        screen_rect = pygame.Rect(0, 0, self.config.display_width, self.config.display_height)

        new_bullets = []
        for bullet in self.bullets:
            # Skip bullets without position
            if bullet.x is None or bullet.y is None:
                continue

            dx, dy = bullet.direction
            bullet.x += self.dt * dx * self.config.bullet_speed
            bullet.y += self.dt * dy * self.config.bullet_speed

            # Skip bullets that are off-screen
            if not screen_rect.collidepoint(bullet.x, bullet.y):
                continue

            # Check for wall collisions efficiently
            collision_detected = False
            nearby_walls = self.get_objects_near(bullet, ["wall"])

            for wall_obj, _ in nearby_walls:
                if bullet.rect.colliderect(wall_obj.rect):
                    collision_detected = True

                    # Create collision event
                    if isinstance(wall_obj, pygame.Rect):
                        wall_position = (wall_obj.x, wall_obj.y)
                    else:
                        wall_position = (
                            wall_obj.x if wall_obj.x is not None else 0,
                            wall_obj.y if wall_obj.y is not None else 0,
                        )

                    self.events.append(
                        CollisionEvent(
                            timestamp=self.game_time,
                            entity1_id=f"bullet_{bullet.owner}",
                            entity2_id="wall",
                            position=(bullet.x, bullet.y),
                        )
                    )
                    break

            if not collision_detected:
                new_bullets.append(bullet)

        self.bullets = new_bullets

    def check_collisions(self) -> None:
        """Check for collisions between bullets and entities."""
        # More efficient collision detection using spatial partitioning
        bullets_to_remove = set()

        # Process player bullets
        for bullet_idx, bullet in enumerate(self.bullets):
            if bullet_idx in bullets_to_remove or bullet.x is None or bullet.y is None:
                continue

            if bullet.owner == "player":
                # Check for enemy hits
                for i, enemy in enumerate(self.enemies):
                    if enemy.rect.colliderect(bullet.rect):
                        # Create enemy hit event
                        self.events.append(
                            EnemyHitEvent(
                                timestamp=self.game_time,
                                enemy_id=i,
                                damage=1,
                                position=(bullet.x, bullet.y),
                            )
                        )

                        # Update score
                        self.score += 10

                        # Reduce enemy health
                        enemy.health -= 1

                        bullets_to_remove.add(bullet_idx)

                        # Check if enemy was destroyed
                        if enemy.health <= 0:
                            print(f"Enemy {i} defeated")

                            # Create destroyed event
                            self.events.append(
                                EntityDestroyedEvent(
                                    timestamp=self.game_time,
                                    entity_id=f"enemy_{i}",
                                    entity_type="enemy",
                                    position=(
                                        enemy.x if enemy.x is not None else 0,
                                        enemy.y if enemy.y is not None else 0,
                                    ),
                                )
                            )

                            # Update score for enemy defeat
                            self.score += 50

                            # Remove the enemy
                            self.enemies.remove(enemy)
                        break
            elif self.player is not None and bullet.rect.colliderect(self.player.rect):
                # Player was hit by enemy bullet
                self.events.append(
                    PlayerHitEvent(
                        timestamp=self.game_time,
                        damage=1,
                        bullet_owner=bullet.owner,
                        position=(bullet.x, bullet.y),
                    )
                )

                self.player.health -= 1
                print(f"Player hit! Health: {self.player.health}")
                bullets_to_remove.add(bullet_idx)

        # Remove bullets in reverse order (to avoid index shifting problems)
        for bullet_idx in sorted(bullets_to_remove, reverse=True):
            if bullet_idx < len(self.bullets):  # Safety check
                self.bullets.pop(bullet_idx)

    def save_episode_log(self) -> None:
        """Save the episode log to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(LOG_PATH)
        log_path.mkdir(exist_ok=True, parents=True)

        filename = log_path / f"episode_{timestamp}.json"

        with filename.open("w") as f:
            json.dump(self.episode_log, f)

    def load_textures(self) -> None:
        """Load textures for rendering."""
        # Initialize pygame font module first
        pygame.font.init()

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
        for health in range(11):  # Assuming max health is 10
            self.health_texts[health] = self.font.render(f"Health: {health}", True, (255, 255, 255))
        self.ammo_texts = {}
        for ammo in range(11):  # Assuming max health is 10
            self.ammo_texts[ammo] = self.font.render(f"Ammo: {ammo}", True, (255, 255, 255))

        # Pre-render score text
        self.score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))

    def render(self) -> None:
        """Render the current game state."""
        # Skip rendering if screen is None (headless mode)
        if self.screen is None:
            return

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
        if self.player is not None:
            self.screen.blit(self.textures["player"], self.player.rect)

        # 4. Bullets
        for bullet in self.bullets:
            if bullet.x is not None and bullet.y is not None:
                self.screen.blit(self.textures["bullet"], bullet.rect)

        # 5. UI elements - use pre-rendered text
        if self.player is not None:
            self.screen.blit(self.health_texts[self.player.health], (10, 10))
            self.screen.blit(self.ammo_texts[self.player.ammunition], (10, 30))

        # Update and render score
        self.score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(self.score_text, (10, 50))

        # Add game time display
        time_text = self.font.render(f"Time: {self.game_time:.1f}s", True, (255, 255, 255))
        self.screen.blit(time_text, (10, 70))

        # Update display
        pygame.display.flip()
