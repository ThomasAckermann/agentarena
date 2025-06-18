from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RewardType(Enum):
    BASIC = "basic"  # Simple rewards for hits and avoiding damage
    ADVANCED = "advanced"  # More nuanced reward calculation with more factors
    ENHANCED = "enhanced"  # More nuanced reward calculation with more factors


class Experience(BaseModel):
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class EpisodeResult(BaseModel):
    episode_id: int
    total_reward: float
    episode_length: int
    win: bool
    player_health_remaining: int
    enemies_defeated: int
    accuracy: float
    events: list[dict[str, Any]]


class MLAgentConfig(BaseModel):
    """Configuration for the ML agent"""

    learning_rate: float = Field(0.001, description="Learning rate for optimizer")
    gamma: float = Field(0.99, description="Discount factor for future rewards")
    epsilon: float = Field(1.0, description="Initial exploration rate")
    epsilon_min: float = Field(0.01, description="Minimum exploration rate")
    epsilon_decay: float = Field(0.995, description="Rate at which epsilon decays")
    batch_size: int = Field(64, description="Batch size for training")
    memory_capacity: int = Field(10000, description="Capacity of replay buffer")
    target_update_frequency: int = Field(
        10,
        description="How often to update target network (in episodes)",
    )


class TrainingConfig(BaseModel):
    """
    Configuration for the training process.
    """

    model_name: str = "ml_agent"
    episodes: int = 1000
    render: bool = False
    checkpoint_path: Path | None = None
    reward_type: RewardType
    save_frequency: int = 100
    models_dir: Path = Field(default=Path("ml_models"))
    results_dir: Path = Field(default=Path("results"))
    max_steps_per_episode: int = 1000
    ml_config: MLAgentConfig = Field(default_factory=MLAgentConfig)


class TrainingResults(BaseModel):
    """
    Results from a training run.
    """

    episode_rewards: list[float]
    episode_lengths: list[int]
    epsilons: list[float]
    reward_type: str
    timestamp: str
    episodes_completed: int
    ml_config: dict[str, Any]
    episode_details: list[EpisodeResult]
