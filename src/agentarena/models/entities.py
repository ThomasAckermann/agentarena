"""
Entity data models for game objects.
"""

from typing import Literal, Union

import pygame
from pydantic import BaseModel, Field, computed_field


class EntityModel(BaseModel):
    """Base model for any game entity data."""

    id: str = Field(..., description="Unique identifier for the entity")
    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")
    x: int = Field(..., description="X coordinate position")
    y: int = Field(..., description="Y coordinate position")

    @computed_field
    def rect(self) -> pygame.Rect:
        """Get the pygame Rect representation of this entity."""
        return pygame.Rect(self.x, self.y, self.width, self.height)

    @computed_field
    def center(self) -> tuple[int, int]:
        """Get the center point of the entity."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def distance_to(self, other: "EntityModel") -> float:
        """Calculate the distance to another entity (from center to center)."""
        dx = self.center[0] - other.center[0]
        dy = self.center[1] - other.center[1]
        return (dx**2 + dy**2) ** 0.5

    def collides_with(self, other: "EntityModel") -> bool:
        """Check if this entity collides with another entity."""
        return self.rect.colliderect(other.rect)

    class Config:
        # Allow pygame.Rect which isn't directly serializable
        arbitrary_types_allowed = True


class PlayerModel(EntityModel):
    """Data model for player entities."""

    entity_type: Literal["player"] = "player"
    orientation: list[int] = Field(
        default=[0, 1],
        description="Direction vector [dx, dy]",
    )
    health: int = Field(3, ge=0, description="Current health points")
    cooldown: int = Field(
        0,
        ge=0,
        description="Cooldown time before next shot",
    )
    ammunition: int = Field(3, ge=0, description="Remaining ammunition")
    is_reloading: bool = False
    speed: int = Field(
        100,
        gt=0,
        description="Movement speed in pixels per second",
    )

    def can_shoot(self) -> bool:
        """Check if the player can shoot."""
        return self.ammunition > 0 and self.cooldown == 0 and not self.is_reloading


class ProjectileModel(EntityModel):
    """Data model for projectile entities."""

    entity_type: Literal["projectile"] = "projectile"
    direction: list[int] = Field(..., description="Direction vector [dx, dy]")
    speed: int = Field(
        20,
        gt=0,
        description="Movement speed in pixels per second",
    )
    owner: str = Field(
        ...,
        description="ID of entity that fired this projectile",
    )
    damage: int = Field(1, gt=0, description="Damage dealt on hit")

    @computed_field
    def velocity(self) -> tuple[float, float]:
        """Get the velocity vector (direction * speed)."""
        return (self.direction[0] * self.speed, self.direction[1] * self.speed)


class WallModel(EntityModel):
    """Data model for wall entities."""

    entity_type: Literal["wall"] = "wall"
    destructible: bool = Field(
        False,
        description="Whether the wall can be destroyed",
    )
    health: int | None = Field(
        None,
        description="Health points for destructible walls",
    )

    def is_destructible(self) -> bool:
        """Check if the wall is destructible."""
        return self.destructible and self.health is not None and self.health > 0


class PowerUpModel(EntityModel):
    """Data model for power-up entities."""

    entity_type: Literal["powerup"] = "powerup"
    power_type: str = Field(..., description="Type of power-up")
    duration: float | None = Field(
        None,
        description="Duration of effect in seconds",
    )
    power_value: int = Field(1, description="Magnitude of the power-up effect")

    def apply_effect(self, player: PlayerModel) -> PlayerModel:
        """Apply power-up effect to a player."""
        if self.power_type == "health":
            player.health += self.power_value
        elif self.power_type == "ammunition":
            player.ammunition += self.power_value
        elif self.power_type == "speed":
            player.speed += self.power_value

        return player


# Union type for all entity data models
GameEntityModel = Union[PlayerModel, ProjectileModel, WallModel, PowerUpModel]
