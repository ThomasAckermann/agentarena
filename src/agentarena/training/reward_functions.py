"""
Reward functions for reinforcement learning in AgentArena.
"""

from enum import Enum

from agentarena.models.events import (
    BulletFiredEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    GameEvent,
    PlayerHitEvent,
)
from agentarena.models.observations import GameObservation

# Constants for reward calculations
NEARBY_DISTANCE_THRESHOLD = 300.0  # Distance threshold for considering enemies "nearby"
BULLET_DODGE_DISTANCE = 100.0  # Distance threshold for bullet dodging detection


class RewardType(Enum):
    """
    Types of reward functions available for training.
    """

    BASIC = "basic"  # Simple rewards for hits and avoiding damage
    AGGRESSIVE = "aggressive"  # Prioritizes dealing damage over safety
    DEFENSIVE = "defensive"  # Prioritizes staying alive over dealing damage
    ADVANCED = "advanced"  # More nuanced reward calculation with more factors


def calculate_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
    reward_type: RewardType = RewardType.BASIC,
) -> float:
    """
    Calculate rewards based on game events and observations.

    Args:
        events: List of events that occurred during the step
        observation: Current game state observation
        previous_observation: Previous game state observation (optional)
        reward_type: Type of reward function to use

    Returns:
        float: The calculated reward
    """
    if reward_type == RewardType.BASIC:
        return _basic_reward(events, observation, previous_observation)
    elif reward_type == RewardType.AGGRESSIVE:
        return _aggressive_reward(events, observation, previous_observation)
    elif reward_type == RewardType.DEFENSIVE:
        return _defensive_reward(events, observation, previous_observation)
    elif reward_type == RewardType.ADVANCED:
        return _advanced_reward(events, observation, previous_observation)

    # Invalid reward type
    error_msg = f"Unknown reward type: {reward_type}"
    raise ValueError(error_msg)


def _basic_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
) -> float:
    """
    Basic reward function focusing on hits and survival.

    Args:
        events: Game events from the current step
        observation: Current game state
        previous_observation: Previous game state (unused in basic reward)

    Returns:
        float: The calculated reward
    """
    reward = 0.0

    # Reward for hitting enemies
    for event in events:
        if isinstance(event, EnemyHitEvent):
            reward += 1.0
        elif isinstance(event, PlayerHitEvent):
            reward -= 1.0
        elif isinstance(event, EntityDestroyedEvent) and event.is_enemy_destroyed():
            reward += 2.0  # Bonus for destroying an enemy

    # Small penalty for each step to encourage faster completion
    reward -= 0.01

    return reward


def _aggressive_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
) -> float:
    """
    Reward function that encourages aggressive play.

    Args:
        events: Game events from the current step
        observation: Current game state
        previous_observation: Previous game state

    Returns:
        float: The calculated reward
    """
    reward = 0.0

    # Large reward for hitting enemies
    for event in events:
        if isinstance(event, EnemyHitEvent):
            reward += 2.0
        elif isinstance(event, PlayerHitEvent):
            reward -= 0.5  # Smaller penalty for getting hit
        elif isinstance(event, EntityDestroyedEvent) and event.is_enemy_destroyed():
            reward += 5.0  # Large bonus for destroying an enemy
        elif isinstance(event, BulletFiredEvent) and event.owner_id == "player":
            reward += 0.1  # Small reward for shooting

    # Reward for being close to enemies (encouraging engagement)
    if nearest_enemy := observation.nearest_enemy():
        enemy, distance = nearest_enemy
        # More reward for being closer to enemies
        reward += max(0, (500.0 - distance) / 500.0) * 0.1

    # Small penalty for each step to encourage faster completion
    reward -= 0.01

    return reward


def _defensive_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
) -> float:
    """
    Reward function that encourages defensive play.

    Args:
        events: Game events from the current step
        observation: Current game state
        previous_observation: Previous game state

    Returns:
        float: The calculated reward
    """
    reward = 0.0

    # Moderate reward for hitting enemies
    for event in events:
        if isinstance(event, EnemyHitEvent):
            reward += 0.5
        elif isinstance(event, PlayerHitEvent):
            reward -= 2.0  # Larger penalty for getting hit
        elif isinstance(event, EntityDestroyedEvent):
            if event.is_enemy_destroyed():
                reward += 1.0  # Bonus for destroying an enemy
            elif event.is_player_destroyed():
                reward -= 5.0  # Large penalty for dying

    # Reward for staying alive
    reward += 0.02

    # Reward for maintaining distance from enemies
    if (
        previous_observation
        and observation.nearest_enemy()
        and previous_observation.nearest_enemy()
    ):
        prev_distance = previous_observation.nearest_enemy()[1]
        curr_distance = observation.nearest_enemy()[1]

        # If there are enemies nearby and the player moved away from them
        if prev_distance < NEARBY_DISTANCE_THRESHOLD and curr_distance > prev_distance:
            reward += 0.05

    # Reward for avoiding bullets
    if len(observation.bullets_near_player()) == 0:
        reward += 0.05  # Small reward for having no bullets nearby

    return reward


def _advanced_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
) -> float:
    """
    Advanced reward function with multiple components.

    Args:
        events: Game events from the current step
        observation: Current game state
        previous_observation: Previous game state

    Returns:
        float: The calculated reward
    """
    reward = 0.0

    # Event-based rewards
    for event in events:
        if isinstance(event, EnemyHitEvent):
            reward += 1.0
        elif isinstance(event, PlayerHitEvent):
            reward -= 1.0
        elif isinstance(event, EntityDestroyedEvent):
            if event.is_enemy_destroyed():
                reward += 3.0  # Bonus for destroying an enemy
            elif event.is_player_destroyed():
                reward -= 5.0  # Large penalty for dying
        elif isinstance(event, BulletFiredEvent) and event.owner_id == "player":
            reward += 0.05  # Small reward for shooting

    # Small penalty for each step to encourage faster completion
    reward -= 0.01

    # If we have previous observation, we can calculate more rewards
    if previous_observation:
        # Reward for moving toward enemies when health is high
        player_health = observation.player.health

        if (
            player_health > 1
            and observation.nearest_enemy()
            and previous_observation.nearest_enemy()
        ):
            # If health is good, encourage attacking
            prev_distance = previous_observation.nearest_enemy()[1]
            curr_distance = observation.nearest_enemy()[1]

            if prev_distance > curr_distance:
                reward += 0.05  # Reward for closing in
        elif observation.nearest_enemy() and previous_observation.nearest_enemy():
            # If health is low, encourage defensive play
            prev_distance = previous_observation.nearest_enemy()[1]
            curr_distance = observation.nearest_enemy()[1]

            if prev_distance < NEARBY_DISTANCE_THRESHOLD and curr_distance > prev_distance:
                reward += 0.1  # Reward for backing away when low health

        # Reward for dodging bullets
        prev_bullets_near = len(previous_observation.bullets_near_player(BULLET_DODGE_DISTANCE))
        curr_bullets_near = len(observation.bullets_near_player(BULLET_DODGE_DISTANCE))

        if prev_bullets_near > curr_bullets_near and prev_bullets_near > 0:
            reward += 0.2  # Reward for having fewer bullets nearby than before

    # Penalty for being close to too many bullets
    bullet_danger = len(observation.bullets_near_player())
    reward -= 0.05 * bullet_danger

    return reward
