"""
Observation models for the game state provided to agents.
"""

import numpy as np
from pydantic import BaseModel, Field, computed_field


class EntityBase(BaseModel):
    """Base model for any entity in the game."""

    x: float
    y: float


class PlayerObservation(EntityBase):
    """Structured representation of player data."""

    orientation: list[int] = Field(
        ...,
        description="Direction vector [dx, dy]",
    )
    health: int = Field(..., description="Current health points")
    ammunition: int = Field(3, description="Remaining ammunition")
    cooldown: int = Field(0, description="Cooldown time before next shot")
    is_reloading: bool = False

    @computed_field
    def position(self) -> tuple[float, float]:
        """Get position as a tuple."""
        return (self.x, self.y)


class EnemyObservation(EntityBase):
    """Structured representation of enemy data."""

    orientation: list[int] = Field(
        ...,
        description="Direction vector [dx, dy]",
    )
    health: int = Field(..., ge=0, description="Current health points")

    @computed_field
    def position(self) -> tuple[float, float]:
        """Get position as a tuple."""
        return (self.x, self.y)


class BulletObservation(EntityBase):
    """Structured representation of bullet data."""

    direction: list[int] = Field(..., description="Direction vector [dx, dy]")
    owner: str = Field(..., description="ID of entity that fired the bullet")

    @computed_field
    def position(self) -> tuple[float, float]:
        """Get position as a tuple."""
        return (self.x, self.y)


class WallObservation(EntityBase):
    """Wall observation data model."""

    @computed_field
    def position(self) -> tuple[float, float]:
        """Get position as a tuple."""
        return (self.x, self.y)


class GameObservation(BaseModel):
    """Complete game observation provided to agents."""

    player: PlayerObservation
    enemies: list[EnemyObservation] = Field(default_factory=list)
    bullets: list[BulletObservation] = Field(default_factory=list)
    game_time: float = Field(0.0, description="Current game time in seconds")
    score: int = Field(0, description="Current game score")
    walls: list[WallObservation] = Field(default_factory=list)

    def nearest_enemy(self) -> tuple[EnemyObservation, float] | None:
        """Get the closest enemy to the player and its distance."""
        if not self.enemies:
            return None

        distances = []
        for enemy in self.enemies:
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            distance = np.sqrt(dx**2 + dy**2)
            distances.append((enemy, distance))

        return min(distances, key=lambda x: x[1])

    def bullets_near_player(
        self,
        radius: float = 100.0,
    ) -> list[BulletObservation]:
        """Get all bullets within a certain radius of the player."""
        near_bullets = []
        for bullet in self.bullets:
            if bullet.owner == "player":
                continue  # Skip player's own bullets

            dx = bullet.x - self.player.x
            dy = bullet.y - self.player.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance <= radius:
                near_bullets.append(bullet)

        return near_bullets

    def is_player_in_danger(self, danger_radius: float = 150.0) -> bool:
        """Check if player is in immediate danger from bullets or enemies."""
        # Check for nearby bullets
        if self.bullets_near_player(danger_radius):
            return True

        # Check for nearby enemies
        return self.nearest_enemy() and self.nearest_enemy()[1] < danger_radius
