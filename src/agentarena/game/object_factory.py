import random

from agentarena.agent.agent import Agent
from agentarena.game.entities.explosion import Explosion
from agentarena.game.entities.player import Player
from agentarena.game.entities.projectile import Projectile
from agentarena.game.entities.wall import Wall
from agentarena.game.level import Level
from agentarena.models.config import GameConfig
from agentarena.models.entities import WallModel
from agentarena.models.events import (
    BulletFiredEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    PlayerHitEvent,
)
from agentarena.models.observations import (
    BulletObservation,
    EnemyObservation,
    GameObservation,
    PlayerObservation,
    WallObservation,
)


class ObjectFactory:
    def __init__(self, config: GameConfig, player_agent: Agent, enemy_agent: Agent) -> None:
        self.config = config
        self.player_agent = player_agent
        self.enemy_agent = enemy_agent

        self.bullet_width = int(self.config.block_width / 2)
        self.bullet_height = int(self.config.block_height / 2)
        self.scaled_width = int(self.config.block_width * 0.8)
        self.scaled_height = int(self.config.block_height * 0.8)
        self.spawn_margin = 3 * self.config.block_width
        self.min_x = self.spawn_margin
        self.max_x = self.config.display_width - self.spawn_margin - self.scaled_width
        self.min_y = self.spawn_margin
        self.max_y = self.config.display_height - self.spawn_margin - self.scaled_height

    def create_player(self) -> Player:
        # Calculate safe spawn area
        margin = 3 * self.config.block_width
        min_x = margin
        max_x = self.config.display_width - margin - self.scaled_width
        min_y = margin
        max_y = self.config.display_height - margin - self.scaled_height

        player_x = random.randint(min_x, max_x)
        player_y = random.randint(min_y, max_y)

        player_orientation = [0, 1]

        return Player(
            orientation=player_orientation,
            agent=self.player_agent,
            width=self.scaled_width,
            height=self.scaled_height,
            x=player_x,
            y=player_y,
            speed=self.config.player_speed,
        )

    def create_enemies(self, count: int) -> list[Player]:
        enemies = []

        # Calculate safe spawn area
        margin = 3 * self.config.block_width
        min_x = margin
        max_x = self.config.display_width - margin - self.scaled_width
        min_y = margin
        max_y = self.config.display_height - margin - self.scaled_height

        for _i in range(count):
            enemy_x = random.randint(min_x, max_x)
            enemy_y = random.randint(min_y, max_y)

            enemy_orientation = [0, 1]

            enemies.append(
                Player(
                    orientation=enemy_orientation,
                    agent=self.enemy_agent,
                    width=self.scaled_width,
                    height=self.scaled_height,
                    x=enemy_x,
                    y=enemy_y,
                    speed=self.config.player_speed,
                ),
            )

        return enemies

    def create_level(self, player: Player, enemies: list[Player]) -> Level:
        return Level(player, enemies, self.config)

    def create_bullet(
        self,
        x: int,
        y: int,
        direction: list[int],
        owner: str,
    ) -> Projectile:
        return Projectile(
            x=x,
            y=y,
            width=self.bullet_width,
            height=self.bullet_height,
            direction=direction,
            owner=owner,
            speed=self.config.bullet_speed,
        )

    def create_explosion(
        self,
        x: float,
        y: float,
        explosion_type: str,
    ) -> Explosion:
        center_x = x - (self.scaled_width / 2)
        center_y = y - (self.scaled_height / 2)

        return Explosion(
            x=center_x,
            y=center_y,
            explosion_type=explosion_type,
            width=self.scaled_width,
            height=self.scaled_height,
        )

    def get_walls_data(self, walls: list[Wall]) -> list[dict]:
        return [
            WallModel(
                id=f"wall_{i}",
                x=wall.x,
                y=wall.y,
                width=wall.width,
                height=wall.height,
                entity_type="wall",
            ).model_dump()
            for i, wall in enumerate(walls)
        ]

    def create_player_observation(
        self,
        player: Player,
        enemies: list[Player],
        bullets: list[Projectile],
        walls: list[Wall],
        game_time: float,
        score: int,
    ) -> GameObservation:
        return GameObservation(
            player=PlayerObservation(
                x=player.x if player.x is not None else 0,
                y=player.y if player.y is not None else 0,
                orientation=(player.orientation if player.orientation is not None else [0, 0]),
                health=player.health,
                ammunition=player.ammunition,
                cooldown=player.cooldown,
                is_reloading=player.is_reloading,
            ),
            enemies=[
                EnemyObservation(
                    x=enemy.x if enemy.x is not None else 0,
                    y=enemy.y if enemy.y is not None else 0,
                    orientation=enemy.orientation if enemy.orientation is not None else [0, 0],
                    health=enemy.health,
                )
                for enemy in enemies
            ],
            bullets=[
                BulletObservation(
                    x=bullet.x if bullet.x is not None else 0,
                    y=bullet.y if bullet.y is not None else 0,
                    direction=bullet.direction,
                    owner=bullet.owner,
                )
                for bullet in bullets
            ],
            walls=[
                WallObservation(
                    x=wall.x if not None else 0,
                    y=wall.y if not None else 0,
                )
                for wall in walls
            ],
            game_time=game_time,
            score=score,
        )

    def create_enemy_observation(
        self,
        agent_id: str,
        player: Player,
        enemies: list[Player],
        bullets: list[Projectile],
        walls: list[Wall],
        game_time: float,
        score: int,
    ) -> GameObservation:
        idx = int(agent_id.replace("enemy_", ""))
        if idx < len(enemies):
            enemy = enemies[idx]

            return GameObservation(
                player=PlayerObservation(
                    x=enemy.x if enemy.x is not None else 0,
                    y=enemy.y if enemy.y is not None else 0,
                    orientation=enemy.orientation if enemy.orientation is not None else [0, 0],
                    health=enemy.health,
                    ammunition=enemy.ammunition,
                    cooldown=enemy.cooldown,
                    is_reloading=enemy.is_reloading,
                ),
                enemies=(
                    [
                        EnemyObservation(
                            x=(player.x if player is not None and player.x is not None else 0),
                            y=(player.y if player is not None and player.y is not None else 0),
                            orientation=(
                                player.orientation
                                if player is not None and player.orientation is not None
                                else [0, 0]
                            ),
                            health=player.health if player is not None else 0,
                        ),
                    ]
                    if player is not None
                    else []
                ),
                bullets=[
                    BulletObservation(
                        x=bullet.x if bullet.x is not None else 0,
                        y=bullet.y if bullet.y is not None else 0,
                        direction=bullet.direction,
                        owner=bullet.owner,
                    )
                    for bullet in bullets
                ],
                walls=[
                    WallObservation(
                        x=wall.x if not None else 0,
                        y=wall.y if not None else 0,
                    )
                    for wall in walls
                ],
                game_time=game_time,
                score=score,
            )

        return GameObservation(
            player=PlayerObservation(
                x=0,
                y=0,
                orientation=[0, 0],
                health=0,
            ),
            enemies=[],
            bullets=[],
            game_time=game_time,
            score=score,
        )

    def create_bullet_fired_event(self, events, game_time, owner_id, direction, position) -> None:
        events.append(
            BulletFiredEvent(
                timestamp=game_time,
                owner_id=owner_id,
                direction=direction,
                position=position,
            ),
        )

    def create_enemy_hit_event(self, events, game_time, enemy_id, x, y) -> None:
        events.append(
            EnemyHitEvent(
                timestamp=game_time,
                enemy_id=enemy_id,
                damage=1,
                position=(x, y),
            ),
        )

    def create_player_hit_event(self, events, game_time, bullet_owner, x, y) -> None:
        events.append(
            PlayerHitEvent(
                timestamp=game_time,
                damage=1,
                bullet_owner=bullet_owner,
                position=(x, y),
            ),
        )

    def create_entity_destroyed_event(
        self,
        events,
        game_time,
        entity_id,
        entity_type,
        x,
        y,
    ) -> None:
        events.append(
            EntityDestroyedEvent(
                timestamp=game_time,
                entity_id=entity_id,
                entity_type=entity_type,
                position=(
                    x if x is not None else 0,
                    y if y is not None else 0,
                ),
            ),
        )
