"""
Event models for tracking game events.
"""

from typing import Literal, Union

from pydantic import BaseModel, Field


class EventBase(BaseModel):
    """Base class for all game events."""

    timestamp: float = Field(..., description="Game time when event occurred")


class PlayerHitEvent(EventBase):
    """Event for when the player is hit by an enemy bullet."""

    type: Literal["player_hit"] = "player_hit"
    damage: int = Field(1, ge=1, description="Amount of damage dealt")
    bullet_owner: str = Field(
        ...,
        description="ID of entity that fired the bullet",
    )
    position: tuple[float, float] = Field(
        ...,
        description="Position where the hit occurred",
    )


class EnemyHitEvent(EventBase):
    """Event for when an enemy is hit by a player bullet."""

    type: Literal["enemy_hit"] = "enemy_hit"
    enemy_id: int = Field(..., description="ID of the enemy that was hit")
    damage: int = Field(1, ge=1, description="Amount of damage dealt")
    position: tuple[float, float] = Field(
        ...,
        description="Position where the hit occurred",
    )


class EntityDestroyedEvent(EventBase):
    """Event for when any entity is destroyed."""

    type: Literal["entity_destroyed"] = "entity_destroyed"
    entity_id: str = Field(..., description="ID of the destroyed entity")
    entity_type: str = Field(..., description="Type of the destroyed entity")
    position: tuple[float, float] = Field(
        ...,
        description="Position where the entity was destroyed",
    )

    def is_enemy_destroyed(self) -> bool:
        """Check if the destroyed entity was an enemy."""
        return self.entity_type == "enemy"

    def is_player_destroyed(self) -> bool:
        """Check if the destroyed entity was the player."""
        return self.entity_type == "player"


class CollisionEvent(EventBase):
    """Event for when entities collide."""

    type: Literal["collision"] = "collision"
    entity1_id: str = Field(..., description="ID of first entity in collision")
    entity2_id: str = Field(
        ...,
        description="ID of second entity in collision",
    )
    position: tuple[float, float] = Field(
        ...,
        description="Position of the collision",
    )


class BulletFiredEvent(EventBase):
    """Event for when a bullet is fired."""

    type: Literal["bullet_fired"] = "bullet_fired"
    owner_id: str = Field(
        ...,
        description="ID of entity that fired the bullet",
    )
    direction: tuple[float, float] = Field(
        ...,
        description="Direction of the bullet",
    )
    position: tuple[float, float] = Field(
        ...,
        description="Starting position of the bullet",
    )


class GameStateChangedEvent(EventBase):
    """Event for significant game state changes."""

    type: Literal["game_state_changed"] = "game_state_changed"
    previous_state: str = Field(..., description="Previous game state")
    new_state: str = Field(..., description="New game state")
    reason: str | None = Field(None, description="Reason for the state change")


# Union type for all possible events
GameEvent = Union[  # noqa: UP007
    PlayerHitEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    CollisionEvent,
    BulletFiredEvent,
    GameStateChangedEvent,
]
