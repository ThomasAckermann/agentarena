"""
Physics system for AgentArena, handling movement, collisions, and spatial partitioning.
"""

import pygame
from pygame.math import Vector2

from agentarena.models.config import GameConfig
from agentarena.models.events import CollisionEvent


class PhysicsSystem:
    def __init__(self, config: GameConfig) -> None:
        self.config = config

        self.grid_size = 100
        self.collision_grid = {}
        self.static_collision_grid = {}

    def setup_collision_grid(self, walls) -> None:
        self.collision_grid = {}

        for wall in walls:
            self.add_to_collision_grid(wall, "wall")

        self.static_collision_grid = self.collision_grid.copy()

    def add_to_collision_grid(self, obj, obj_type: str) -> None:
        if obj.x is None or obj.y is None:
            return None

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
        self.collision_grid = self.static_collision_grid.copy()
        if player is not None:
            self.add_to_collision_grid(player, "player")

        for i, enemy in enumerate(enemies):
            self.add_to_collision_grid(enemy, f"enemy_{i}")
        for bullet in bullets:
            self.add_to_collision_grid(bullet, f"bullet_{bullet.owner}")

    def get_objects_near(self, obj, obj_types=None) -> list:
        if obj.x is None or obj.y is None:
            return []

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
        self,
        agent_id,
        entity,
        action,
        bullets,
        events,
        game_time,
        object_factory,
        dt,
    ) -> None:
        if entity.cooldown > 0:
            entity.cooldown -= 1
        if entity.cooldown == 0 and entity.is_reloading:
            entity.is_reloading = False
            entity.ammunition = 3

        if action.direction is not None:
            dx, dy = action.get_direction_vector()
            entity.orientation = [
                dx,
                dy,
            ]

            if entity.x is None or entity.y is None:
                return

            old_x, old_y = entity.x, entity.y
            movement_vector = Vector2(dx * dt * entity.speed, dy * dt * entity.speed)

            entity.x += movement_vector.x
            entity.y += movement_vector.y

            collision_detected = False
            nearby_objects = self.get_objects_near(entity, ["wall"])

            for wall_obj, _ in nearby_objects:
                if entity.rect.colliderect(wall_obj.rect):
                    collision_detected = True
                    break

            if collision_detected:
                entity.x, entity.y = old_x, old_y

        if action.is_shooting is True and entity.ammunition > 0 and entity.cooldown == 0:
            dx, dy = entity.orientation if entity.orientation is not None else [0, 0]

            if entity.x is None or entity.y is None:
                return

            center_x = entity.x + (entity.width - self.config.block_width / 2) / 2
            center_y = entity.y + (entity.height - self.config.block_height / 2) / 2

            offset = 5
            center_x += dx * offset
            center_y += dy * offset

            bullet = object_factory.create_bullet(int(center_x), int(center_y), [dx, dy], agent_id)
            bullets.append(bullet)

            object_factory.create_bullet_fired_event(
                events,
                game_time,
                agent_id,
                (dx, dy),
                (center_x, center_y),
            )

            entity.ammunition -= 1
            entity.cooldown = int(self.config.fps * 0.25)

            if entity.ammunition == 0:
                entity.cooldown += 1 * self.config.fps
                entity.is_reloading = True

    def move_bullets(self, bullets, walls, config, events, game_time, dt) -> None:
        screen_rect = pygame.Rect(
            0,
            0,
            config.display_width,
            config.display_height,
        )

        new_bullets = []
        for bullet in bullets:
            if bullet.x is None or bullet.y is None:
                continue

            dx, dy = bullet.direction

            new_x = bullet.x + dt * dx * config.bullet_speed
            new_y = bullet.y + dt * dy * config.bullet_speed

            bullet.x = new_x
            bullet.y = new_y

            if not screen_rect.collidepoint(int(bullet.x), int(bullet.y)):
                continue

            collision_detected = False
            nearby_walls = self.get_objects_near(bullet, ["wall"])

            for wall_obj, _ in nearby_walls:
                if bullet.rect.colliderect(wall_obj.rect):
                    collision_detected = True
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
        i = 0
        while i < len(bullets):
            bullet = bullets[i]

            if bullet.x is None or bullet.y is None:
                i += 1
                continue

            collision_detected = False

            if bullet.owner == "player":
                for enemy_idx, enemy in enumerate(enemies[:]):
                    if enemy.rect.colliderect(bullet.rect):
                        object_factory.create_enemy_hit_event(
                            events,
                            game_time,
                            enemy_idx,
                            bullet.x,
                            bullet.y,
                        )

                        explosions.append(
                            object_factory.create_explosion(bullet.x, bullet.y, "enemy"),
                        )

                        score_callback(10)

                        enemy.health -= 1
                        collision_detected = True

                        if enemy.health <= 0:
                            object_factory.create_entity_destroyed_event(
                                events,
                                game_time,
                                f"enemy_{enemy_idx}",
                                "enemy",
                                enemy.x,
                                enemy.y,
                            )

                            if enemy.x is not None and enemy.y is not None:
                                explosions.append(
                                    object_factory.create_explosion(enemy.x, enemy.y, "enemy"),
                                )

                            score_callback(50)

                            enemies.remove(enemy)

                        break

            elif player is not None and bullet.rect.colliderect(player.rect):
                object_factory.create_player_hit_event(
                    events,
                    game_time,
                    bullet.owner,
                    bullet.x,
                    bullet.y,
                )

                explosions.append(object_factory.create_explosion(bullet.x, bullet.y, "player"))

                player.health -= 1

                if player.health <= 0 and player.x is not None and player.y is not None:
                    explosions.append(object_factory.create_explosion(player.x, player.y, "player"))

                collision_detected = True

            if not collision_detected:
                nearby_walls = self.get_objects_near(bullet, ["wall"])
                for wall_obj, _ in nearby_walls:
                    if bullet.rect.colliderect(wall_obj.rect):
                        events.append(
                            CollisionEvent(
                                timestamp=game_time,
                                entity1_id=f"bullet_{bullet.owner}",
                                entity2_id="wall",
                                position=(bullet.x, bullet.y),
                            ),
                        )
                        explosion_type = "player" if bullet.owner == "player" else "enemy"
                        explosions.append(
                            object_factory.create_explosion(bullet.x, bullet.y, explosion_type),
                        )

                        collision_detected = True
                        break

            if collision_detected:
                bullets.pop(i)
            else:
                i += 1
