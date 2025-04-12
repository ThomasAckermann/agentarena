from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class GameConfig(BaseModel):
    player_agent: Literal["manual", "rule_based", "ml"] = "manual"
    enemy_agent: Literal["manual", "rule_based", "ml"] = "rule_based"
    max_enemies: int = 1
    block_width: int = 40
    block_height: int = 40
    display_width: int = 600
    display_height: int = 600
    fps: int = 30
    headless: bool = False
    bullet_speed: int = 30
    player_speed: int = 30


def load_config(path: str | Path = "config.yaml") -> GameConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.resolve()}")

    with path.open("r") as f:
        raw_data = yaml.safe_load(f)

    # Validate and parse using Pydantic v2
    config = GameConfig.model_validate(raw_data)
    return config
