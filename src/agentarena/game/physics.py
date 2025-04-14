"""
Physics system for AgentArena, handling movement, collisions, and spatial partitioning.
"""

import pygame
from pygame.math import Vector2

from agentarena.models.action import Action
from agentarena.models.config import GameConfig
from agentarena.models.events import CollisionEvent


class PhysicsSystem:
    """
    Handles physics, movement, and collision detection for the game.
    """

    def __init__(self, config: GameConfig) -> None:
        """
        Initialize the physics system.

        Args:
            config: Game configuration
        """
        self.config = config

        # Spatial partitioning grid for collision detection
        self.grid_size = 100  # Size of each grid cell
        self.collision_grid = {}
        self.static_collision_grid = {}

    def setup_collision_grid(self, walls) -> None:
        """
        Initialize spatial partitioning grid for collision detection.

        Args:
            walls: List of wall entities
        """
        # Create a fresh collision grid
        self.collision_grid = {}

        # Add walls to the grid (these are static and only need to be added once)
        for wall in walls:
            self.add_to_collision_grid(wall, "wall")

        # Store the static grid separately so we don't need to rebuild it every frame
        self.static_collision_grid = self.collision_grid.copy()

    def add_to_collision_grid(self, obj, obj_type: str) -> None:
        """
        Add an object to the spatial partitioning grid.

        Args:
            obj: The object to add
            obj_type: Type of object (wall, player, enemy, bullet)
        """
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

    def update_entity_positions(self, player, enemies, bullets) -> None:
        """
        Update the grid with current positions of dynamic entities.

        Args:
            player: Player entity
            enemies: List of enemy entities
            bullets: List of bullet entities
        """
        # Start with a copy of the static grid (walls) instead of rebuilding
        self.collision_grid = self.static_collision_grid.copy()

        # Only add dynamic entities to the grid
        # Add player to the grid
        if player is not None:
            self.add_to_collision_grid(player, "player")

        # Add enemies to the grid
        for i, enemy in enumerate(enemies):
            self.add_to_collision_grid(enemy, f"enemy_{i}")

        # Add bullets to the grid
        for bullet in bullets:
            self.add_to_collision_grid(bullet, f"bullet_{bullet.owner}")

    def get_objects_near(self, obj, obj_types=None) -> list:
        """
        Get objects near the specified object based on grid location.

        Args:
            obj: The object to check around
            obj_types: List of object types to filter for

        Returns:
            List of nearby objects of the specified types
        """
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

    def apply_action(
        self, agent_id, entity, action, bullets, events, game_time, object_factory, dt
    ) -> None:
        """
        Apply an action to a player or enemy.

        Args:
            agent_id: ID of the agent performing the action
            entity: Entity to apply the action to
            action: Action to apply
            bullets: List of bullets to append to
            events: List to append events to
            game_time: Current game time
            object_factory: Factory to create new objects
            dt: Delta time for movement calculations
        """
        # Cooldown and reload mechanics
        if entity.cooldown > 0:
            entity.cooldown -= 1
        if entity.cooldown == 0 and entity.is_reloading:
            entity.is_reloading = False
            entity.ammunition = 3

        # Movement processing with vector math
        if action.direction is not None:
            # Set the entity orientation based on the direction
            dx, dy = action.get_direction_vector()
            entity.orientation = [
                dx,
                dy,
            ]  # This will be used for determining which sprite to render

            # Skip movement if entity has no position
            if entity.x is None or entity.y is None:
                return

            old_x, old_y = entity.x, entity.y
            movement_vector = Vector2(dx * dt * entity.speed, dy * dt * entity.speed)

            # Apply movement
            entity.x += movement_vector.x
            entity.y += movement_vector.y

            # Check for wall collisions efficiently using the grid
            collision_detected = False
            nearby_objects = self.get_objects_near(entity, ["wall"])

            for wall_obj, _ in nearby_objects:
                if entity.rect.colliderect(wall_obj.rect):
                    collision_detected = True
                    break

            if collision_detected:
                entity.x, entity.y = old_x, old_y

        # Shooting mechanics
        if action.is_shooting is True and entity.ammunition > 0 and entity.cooldown == 0:
            dx, dy = entity.orientation if entity.orientation is not None else [0, 0]

            # Skip shooting if entity has no position
            if entity.x is None or entity.y is None:
                return

            # Calculate bullet spawn position (center of entity)
            center_x = entity.x + (entity.width - self.config.block_width / 2) / 2
            center_y = entity.y + (entity.height - self.config.block_height / 2) / 2

            # Offset bullet in shooting direction
            offset = 5
            center_x += dx * offset
            center_y += dy * offset

            # Create bullet
            bullet = object_factory.create_bullet(int(center_x), int(center_y), [dx, dy], agent_id)
            bullets.append(bullet)

            # Create bullet fired event
            object_factory.create_bullet_fired_event(
                events, game_time, agent_id, (dx, dy), (center_x, center_y)
            )

            # Update ammunition and cooldown
            entity.ammunition -= 1
            entity.cooldown = int(self.config.fps * 0.25)

            if entity.ammunition == 0:
                entity.cooldown += 1 * self.config.fps
                entity.is_reloading = True

    def move_bullets(self, bullets, walls, config, events, game_time, dt) -> None:
        """
        Update positions of all bullets and handle wall collisions.

        Args:
            bullets: List of bullets
            walls: List of walls
            config: Game configuration
            events: List to append events to
            game_time: Current game time
            dt: Delta time for movement calculations
        """
        # Create screen boundary rectangle
        screen_rect = pygame.Rect(
            0,
            0,
            config.display_width,
            config.display_height,
        )

        new_bullets = []
        for bullet in bullets:
            # Skip bullets without position
            if bullet.x is None or bullet.y is None:
                continue

            # Get direction vector (keep as float for precision)
            dx, dy = bullet.direction

            # Calculate new position with floating point precision
            new_x = bullet.x + dt * dx * config.bullet_speed
            new_y = bullet.y + dt * dy * config.bullet_speed

            # Update bullet position (still as floating point)
            bullet.x = new_x
            bullet.y = new_y

            # Skip bullets that are off-screen (use integer rect for this check)
            if not screen_rect.collidepoint(int(bullet.x), int(bullet.y)):
                continue

            # Check for wall collisions efficiently
            collision_detected = False
            nearby_walls = self.get_objects_near(bullet, ["wall"])

            for wall_obj, _ in nearby_walls:
                if bullet.rect.colliderect(wall_obj.rect):
                    collision_detected = True
                    # Create collision event
                    events.append(
                        CollisionEvent(
                            timestamp=game_time,
                            entity1_id=f"bullet_{bullet.owner}",
                            entity2_id="wall",
                            position=(bullet.x, bullet.y),
                        ),
                    )
                    break

            if not collision_detected:
                new_bullets.append(bullet)

        # Update the bullets list with the bullets that didn't collide
        bullets.clear()
        bullets.extend(new_bullets)

    def check_collisions(
        self,
        player,
        enemies,
        bullets,
        events,
        game_time,
        explosions,
        object_factory,
        score_callback,
    ) -> None:
        """
        Check for collisions between bullets and entities.

        Args:
            player: Player entity
            enemies: List of enemy entities
            bullets: List of bullet entities
            events: List to append events to
            game_time: Current game time
            explosions: List of explosions to append to
            object_factory: Factory to create objects
            score_callback: Callback to update score
        """
        # More efficient collision detection using spatial partitioning
        i = 0
        while i < len(bullets):
            bullet = bullets[i]

            # Skip bullets without position
            if bullet.x is None or bullet.y is None:
                i += 1
                continue

            collision_detected = False

            if bullet.owner == "player":
                # Check for enemy hits
                for enemy_idx, enemy in enumerate(enemies[:]):  # Use a copy for safe iteration
                    if enemy.rect.colliderect(bullet.rect):
                        # Create enemy hit event
                        object_factory.create_enemy_hit_event(
                            events, game_time, enemy_idx, bullet.x, bullet.y
                        )

                        # Create explosion at bullet impact position
                        explosions.append(
                            object_factory.create_explosion(bullet.x, bullet.y, "enemy")
                        )

                        # Update score
                        score_callback(10)

                        # Reduce enemy health
                        enemy.health -= 1
                        collision_detected = True

                        # Check if enemy was destroyed
                        if enemy.health <= 0:
                            print(f"Enemy {enemy_idx} defeated")

                            # Create destroyed event
                            object_factory.create_entity_destroyed_event(
                                events, game_time, f"enemy_{enemy_idx}", "enemy", enemy.x, enemy.y
                            )

                            # Create a larger explosion at enemy position when destroyed
                            if enemy.x is not None and enemy.y is not None:
                                explosions.append(
                                    object_factory.create_explosion(enemy.x, enemy.y, "enemy")
                                )

                            # Update score for enemy defeat
                            score_callback(50)

                            # Remove the enemy
                            enemies.remove(enemy)

                        break

            elif player is not None and bullet.rect.colliderect(player.rect):
                # Player was hit by enemy bullet
                object_factory.create_player_hit_event(
                    events, game_time, bullet.owner, bullet.x, bullet.y
                )

                # Create explosion at bullet impact position
                explosions.append(object_factory.create_explosion(bullet.x, bullet.y, "player"))

                player.health -= 1
                print(f"Player hit! Health: {player.health}")

                # Check if player was destroyed
                if player.health <= 0 and player.x is not None and player.y is not None:
                    # Create a larger explosion at player position when destroyed
                    explosions.append(object_factory.create_explosion(player.x, player.y, "player"))

                collision_detected = True

            # Also check for wall collisions to create explosions
            if not collision_detected:
                nearby_walls = self.get_objects_near(bullet, ["wall"])
                for wall_obj, _ in nearby_walls:
                    if bullet.rect.colliderect(wall_obj.rect):
                        # Create collision event
                        events.append(
                            CollisionEvent(
                                timestamp=game_time,
                                entity1_id=f"bullet_{bullet.owner}",
                                entity2_id="wall",
                                position=(bullet.x, bullet.y),
                            ),
                        )
                        # Create small explosion for wall impact
                        # Determine which type of explosion to use based on bullet owner
                        explosion_type = "player" if bullet.owner == "player" else "enemy"
                        explosions.append(
                            object_factory.create_explosion(bullet.x, bullet.y, explosion_type)
                        )

                        collision_detected = True
                        break

            # Remove bullet if collision detected or move to next bullet
            if collision_detected:
                bullets.pop(i)
            else:
                i += 1
