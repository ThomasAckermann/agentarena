"""
Factory for creating game objects in AgentArena.
"""

import random

from agentarena.agent.agent import Agent
from agentarena.game.entities.explosion import Explosion
from agentarena.game.entities.player import Player
from agentarena.game.entities.projectile import Projectile
from agentarena.game.entities.wall import Wall
from agentarena.game.level import Level
from agentarena.models.config import GameConfig
from agentarena.models.entities import PlayerModel, WallModel
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
)


class ObjectFactory:
    """
    Factory for creating game objects.
    """

    def __init__(self, config: GameConfig, player_agent: Agent, enemy_agent: Agent) -> None:
        """
        Initialize the object factory.

        Args:
            config: Game configuration
            player_agent: Agent for the player
            enemy_agent: Agent for enemies
        """
        self.config = config
        self.player_agent = player_agent
        self.enemy_agent = enemy_agent

        # Precomputed values
        self.bullet_width = int(self.config.block_width / 2)
        self.bullet_height = int(self.config.block_height / 2)
        self.scaled_width = int(self.config.block_width * 0.8)
        self.scaled_height = int(self.config.block_height * 0.8)

    def create_player(self) -> Player:
        """
        Create the player entity.

        Returns:
            Player: The newly created player
        """
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
        return Player(
            orientation=player_orientation,
            agent=self.player_agent,
            width=self.scaled_width,
            height=self.scaled_height,
            x=player_position[0],
            y=player_position[1],
            speed=self.config.player_speed,
        )

    def create_enemies(self, count: int) -> list[Player]:
        """
        Create enemy entities.

        Args:
            count: Number of enemies to create

        Returns:
            List of enemy entities
        """
        enemies = []

        for i in range(count):
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
            enemies.append(
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

        return enemies

    def create_level(self, player: Player, enemies: list[Player]) -> Level:
        """
        Create a game level.

        Args:
            player: Player entity
            enemies: List of enemy entities

        Returns:
            Level: The newly created level
        """
        return Level(player, enemies, self.config)

    def create_bullet(self, x: int, y: int, direction: list[int], owner: str) -> Projectile:
        """
        Create a bullet entity.

        Args:
            x: X position
            y: Y position
            direction: Direction vector [dx, dy]
            owner: Owner ID (player or enemy_X)

        Returns:
            Projectile: The newly created bullet
        """
        return Projectile(
            x=x,
            y=y,
            width=self.bullet_width,
            height=self.bullet_height,
            direction=direction,
            owner=owner,
            speed=self.config.bullet_speed,
        )

    def create_explosion(self, x: float, y: float, explosion_type: str) -> Explosion:
        """
        Create an explosion entity.

        Args:
            x: X position
            y: Y position
            explosion_type: Type of explosion ("player" or "enemy")

        Returns:
            Explosion: The newly created explosion
        """
        # Center the explosion on the entity
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
        """
        Get data models for walls.

        Args:
            walls: List of walls

        Returns:
            List of wall data dictionaries
        """
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
        game_time: float,
        score: int,
    ) -> GameObservation:
        """
        Create a game observation from the player's perspective.

        Args:
            player: Player entity
            enemies: List of enemy entities
            bullets: List of bullet entities
            game_time: Current game time
            score: Current score

        Returns:
            GameObservation: The game observation
        """
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
            game_time=game_time,
            score=score,
        )

    def create_enemy_observation(
        self,
        agent_id: str,
        player: Player,
        enemies: list[Player],
        bullets: list[Projectile],
        game_time: float,
        score: int,
    ) -> GameObservation:
        """
        Create a game observation from an enemy's perspective.

        Args:
            agent_id: ID of the agent requesting the observation
            player: Player entity
            enemies: List of enemy entities
            bullets: List of bullet entities
            game_time: Current game time
            score: Current score

        Returns:
            GameObservation: The game observation
        """
        idx = int(agent_id.replace("enemy_", ""))
        if idx < len(enemies):
            enemy = enemies[idx]

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
                game_time=game_time,
                score=score,
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
            game_time=game_time,
            score=score,
        )

    def create_bullet_fired_event(self, events, game_time, owner_id, direction, position) -> None:
        """
        Create a bullet fired event and add it to the events list.

        Args:
            events: List to append the event to
            game_time: Current game time
            owner_id: ID of the entity that fired the bullet
            direction: Direction vector of the bullet (dx, dy)
            position: Position of the bullet (x, y)
        """
        events.append(
            BulletFiredEvent(
                timestamp=game_time,
                owner_id=owner_id,
                direction=direction,
                position=position,
            ),
        )

    def create_enemy_hit_event(self, events, game_time, enemy_id, x, y) -> None:
        """
        Create an enemy hit event and add it to the events list.

        Args:
            events: List to append the event to
            game_time: Current game time
            enemy_id: ID of the enemy that was hit
            x: X position of the hit
            y: Y position of the hit
        """
        events.append(
            EnemyHitEvent(
                timestamp=game_time,
                enemy_id=enemy_id,
                damage=1,
                position=(x, y),
            ),
        )

    def create_player_hit_event(self, events, game_time, bullet_owner, x, y) -> None:
        """
        Create a player hit event and add it to the events list.

        Args:
            events: List to append the event to
            game_time: Current game time
            bullet_owner: ID of the entity that fired the bullet
            x: X position of the hit
            y: Y position of the hit
        """
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
        """
        Create an entity destroyed event and add it to the events list.

        Args:
            events: List to append the event to
            game_time: Current game time
            entity_id: ID of the entity that was destroyed
            entity_type: Type of the entity that was destroyed
            x: X position of the entity
            y: Y position of the entity
        """
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
