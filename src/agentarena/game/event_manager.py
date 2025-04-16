"""
Event management system for AgentArena.
"""

from agentarena.models.events import (
    BulletFiredEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    PlayerHitEvent,
)


class EventManager:
    """
    Manages the creation and handling of game events.
    """

    def __init__(self) -> None:
        """Initialize the event manager."""

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

    def create_player_destroyed_event(self, events, game_time, player) -> None:
        """
        Create a player destroyed event and add it to the events list.

        Args:
            events: List to append the event to
            game_time: Current game time
            player: Player entity that was destroyed
        """
        events.append(
            EntityDestroyedEvent(
                timestamp=game_time,
                entity_id="player",
                entity_type="player",
                position=(
                    player.x if player.x is not None else 0,
                    player.y if player.y is not None else 0,
                ),
            ),
        )
