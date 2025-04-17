"""
Configuration models for the game and application.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class GameConfig(BaseModel):
    """Game configuration parameters."""

    # Agent types
    player_agent: Literal["manual", "rule_based", "ml"] = Field(
        "manual", description="Type of agent controlling the player"
    )
    enemy_agent: Literal["manual", "rule_based", "ml"] = Field(
        "rule_based", description="Type of agent controlling the enemies"
    )

    # Game parameters
    max_enemies: int = Field(
        1,
        ge=1,
        le=10,
        description="Maximum number of enemies allowed on screen",
    )
    block_width: int = Field(
        40,
        gt=0,
        description="Width of a single block/tile in pixels",
    )
    block_height: int = Field(
        40,
        gt=0,
        description="Height of a single block/tile in pixels",
    )
    display_width: int = Field(
        600,
        gt=0,
        description="Total width of the display window in pixels",
    )
    display_height: int = Field(
        600, gt=0, description="Total height of the display window in pixels"
    )
    fps: int = Field(
        30,
        ge=0,
        le=120,
        description="Fps - affects animation smoothness and game speed",
    )
    headless: bool = Field(
        False,
        description="Whether to run the game without graphical display",
    )
    bullet_speed: int = Field(
        30,
        gt=0,
        description="Speed of bullets in pixels per frame",
    )
    player_speed: int = Field(
        30,
        gt=0,
        description="Speed of player in pixels per frame",
    )

    # Optional ML model path
    ml_model_path: Path | None = Field(
        None, description="Path to the ML model file (if using ML agent)"
    )

    # Additional options
    enable_power_ups: bool = False
    difficulty: Literal["easy", "medium", "hard"] = Field(
        "medium", description="Game difficulty level"
    )

    @field_validator("ml_model_path")
    @classmethod
    def validate_model_path(cls, v: Path | None, info) -> Path | None:
        """Validate that the ML model path exists if specified."""
        # Skip validation if path is None or either agent is not ML
        values = info.data
        if (v is None) or (
            (values.get("player_agent") != "ml") and (values.get("enemy_agent") != "ml")
        ):
            return v

        # Check that the model file exists
        if not v.exists():
            error_msg = f"ML model file not found at: {v.resolve()}"
            raise ValueError(error_msg)

        return v


def load_config(path: str | Path = "config.yaml") -> GameConfig:
    """
    Load game configuration from a YAML file.

    Args:
        path: Path to the config file

    Returns:
        GameConfig: Validated configuration object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValidationError: If the config file contains invalid values
    """
    path = Path(path)
    if not path.exists():
        error_msg = f"Config file not found at: {path.resolve()}"
        raise FileNotFoundError(error_msg)

    with path.open("r") as f:
        raw_data = yaml.safe_load(f)

    # Validate and parse using Pydantic
    return GameConfig.model_validate(raw_data)
