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
import math
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
    ENHANCED = "enhanced"  # More nuanced reward calculation with more factors


def normalize_reward(reward: float) -> float:
    return math.tanh(reward / 70.0)  # Divide by a scaling factor


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
    elif reward_type == RewardType.ENHANCED:
        return normalize_reward(
            _enhanced_reward(events, observation, previous_observation),
        )

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
            reward += 1.0  # Bonus for destroying an enemy

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
                reward += 2.0  # Bonus for destroying an enemy
            elif event.is_player_destroyed():
                reward -= 2.0  # Large penalty for dying
        elif isinstance(event, BulletFiredEvent) and event.owner_id == "player":
            reward -= 0.05  # Small penalty for shooting

    if any(isinstance(event, BulletFiredEvent) for event in events):
        # Get player orientation
        player_orient = observation.player.orientation
        # Check if any enemies are roughly in that direction
        enemies_in_direction = False
        for enemy in observation.enemies:
            # Vector from player to enemy
            to_enemy = [enemy.x - observation.player.x, enemy.y - observation.player.y]
            # Normalize
            mag = (to_enemy[0] ** 2 + to_enemy[1] ** 2) ** 0.5
            if mag > 0:
                to_enemy = [to_enemy[0] / mag, to_enemy[1] / mag]
                # Dot product to check alignment
                dot_product = to_enemy[0] * player_orient[0] + to_enemy[1] * player_orient[1]
                if dot_product > 0.7:  # Enemy is roughly in front
                    enemies_in_direction = True
                    break
        if not enemies_in_direction:
            reward -= 0.5  # Penalty for wasteful shooting
        else:
            reward += 0.5

    # Small penalty for each step to encourage faster completion
    reward -= 0.01

    # If we have previous observation, we can calculate more rewards
    if previous_observation:
        # Reward for moving toward enemies when health is high
        if previous_observation.player.orientation != observation.player.orientation:
            reward += 0.2
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
                reward += 0.5  # Reward for closing in
        elif observation.nearest_enemy() and previous_observation.nearest_enemy():
            # If health is low, encourage defensive play
            prev_distance = previous_observation.nearest_enemy()[1]
            curr_distance = observation.nearest_enemy()[1]

            if prev_distance < NEARBY_DISTANCE_THRESHOLD and curr_distance > prev_distance:
                reward += 0.5  # Reward for backing away when low health

        if previous_observation and observation.player.orientation:
            # Get player positions
            prev_x, prev_y = previous_observation.player.x, previous_observation.player.y
            curr_x, curr_y = observation.player.x, observation.player.y

            # Calculate movement distance
            distance_moved = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5

            # Check if player was trying to move (has orientation) but barely moved
            movement_direction = observation.player.orientation
            movement_intent = sum(abs(x) for x in movement_direction) > 0

            # If player intended to move but moved less than a small threshold
            if movement_intent and distance_moved < 2.0:  # Threshold for detecting wall collision
                reward -= 0.5  # Significant penalty for hitting a wall
                # Uncomment for debugging:
                # print(f"Wall collision detected! Distance moved: {distance_moved:.2f}, Penalty: -0.5")

        # Reward for dodging bullets
        prev_bullets_near = len(previous_observation.bullets_near_player(BULLET_DODGE_DISTANCE))
        curr_bullets_near = len(observation.bullets_near_player(BULLET_DODGE_DISTANCE))

        if prev_bullets_near > curr_bullets_near and prev_bullets_near > 0:
            reward += 1  # Reward for having fewer bullets nearby than before

        # Penalty for being close to too many bullets
        bullet_danger = len(observation.bullets_near_player())
        reward -= 0.1 * bullet_danger

    return reward


def calculate_tactical_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation,
):
    """Reward tactical combat decisions"""
    reward = 0

    # Reward for effective shooting (only shoot when enemies are in line of fire)
    bullet_fired = any(isinstance(event, BulletFiredEvent) for event in events)
    if bullet_fired:
        player_dir = observation.player.orientation
        # Unit vector of player direction
        if player_dir and (player_dir[0] ** 2 + player_dir[1] ** 2) > 0:
            player_dir_norm = [player_dir[0], player_dir[1]]
            dir_mag = (player_dir_norm[0] ** 2 + player_dir_norm[1] ** 2) ** 0.5
            if dir_mag > 0:
                player_dir_norm = [d / dir_mag for d in player_dir_norm]

            # Check if any enemies are in the shooting direction
            enemy_in_sights = False
            for enemy in observation.enemies:
                # Vector from player to enemy
                to_enemy = [enemy.x - observation.player.x, enemy.y - observation.player.y]
                distance = (to_enemy[0] ** 2 + to_enemy[1] ** 2) ** 0.5

                if distance > 0:
                    # Normalize
                    to_enemy = [v / distance for v in to_enemy]
                    # Dot product (cosine of angle)
                    alignment = to_enemy[0] * player_dir_norm[0] + to_enemy[1] * player_dir_norm[1]

                    # If enemy is in front (cos > 0.7 means within ~45 degrees)
                    if alignment > 0.7:
                        enemy_in_sights = True
                        # More reward for closer enemies (more likely to hit)
                        reward += min(1.0, 300 / max(distance, 50))
                        break

            # Penalty for shooting when no enemies are in line of fire
            if not enemy_in_sights:
                reward -= 0.3

    # Reward for dodging bullets
    bullets_near_player = 0
    danger_level = 0
    for bullet in observation.bullets:
        if bullet.owner != "player":  # Only enemy bullets
            # Distance from bullet to player
            distance = (
                (bullet.x - observation.player.x) ** 2 + (bullet.y - observation.player.y) ** 2
            ) ** 0.5

            if distance < 150:  # Close bullet
                bullets_near_player += 1
                # Calculate if bullet is moving toward player
                bullet_dir = bullet.direction
                to_player = [observation.player.x - bullet.x, observation.player.y - bullet.y]

                # Normalize to_player
                to_player_mag = (to_player[0] ** 2 + to_player[1] ** 2) ** 0.5
                if to_player_mag > 0:
                    to_player = [v / to_player_mag for v in to_player]

                # Dot product to determine if bullet is coming toward player
                dot_product = bullet_dir[0] * to_player[0] + bullet_dir[1] * to_player[1]

                # If bullet is heading toward player (dot product > 0)
                if dot_product > 0:
                    danger_level += dot_product * (1 - (distance / 150))

    # If there were bullets nearby in previous state but fewer now, agent dodged successfully
    if previous_observation and hasattr(previous_observation, "bullets_near_player"):
        if bullets_near_player < len(previous_observation.bullets_near_player()):
            reward += 0.3 * (len(previous_observation.bullets_near_player()) - bullets_near_player)

    # Penalty based on danger level
    reward -= danger_level * 0.2

    return reward


def calculate_spatial_awareness_reward(observation, previous_observation):
    """Reward agent for staying aware of its surroundings"""
    reward = 0

    # Reward for maintaining good distance from enemies
    if observation.enemies:
        for enemy in observation.enemies:
            distance = (
                (enemy.x - observation.player.x) ** 2 + (enemy.y - observation.player.y) ** 2
            ) ** 0.5

            # Too close is dangerous
            if distance < 100:
                reward -= 0.1
            # Optimal combat distance
            elif 100 <= distance <= 300:
                reward += 0.05
            # Too far is ineffective
            else:
                reward -= 0.01

    return reward


def calculate_strategic_reward(observation, previous_observation):
    """Reward strategic positioning and planning"""
    reward = 0

    # Reward for maintaining line of sight to multiple enemies
    if observation.enemies:
        visible_enemies = 0
        for enemy in observation.enemies:
            # Simple line of sight check (could be more complex with ray casting)
            visible_enemies += 1

        # Reward scales with number of enemies visible
        reward += 0.1 * visible_enemies

    # Reward for not getting cornered
    # Count walls or screen edges nearby
    boundaries = 0
    screen_margin = 50

    # Check proximity to screen edges
    if observation.player.x < screen_margin:
        boundaries += 1
    if observation.player.x > 1200 - screen_margin:  # Assuming 1200 width
        boundaries += 1
    if observation.player.y < screen_margin:
        boundaries += 1
    if observation.player.y > 900 - screen_margin:  # Assuming 900 height
        boundaries += 1

    # Penalty for being near too many boundaries (cornered)
    if boundaries >= 2:
        reward -= 0.2 * boundaries

    return reward


def calculate_learning_reward(observation, agent):
    """Reward exploration and diverse behaviors"""
    reward = 0

    # Encourage action diversity
    if not hasattr(agent, "action_history"):
        agent.action_history = []

    # Get the current action
    current_action = agent.last_action

    # Calculate action diversity score
    if len(agent.action_history) > 20:
        # Count recent actions
        action_counts = {}
        for a in agent.action_history[-20:]:
            action_counts[a] = action_counts.get(a, 0) + 1

        # Calculate entropy of action distribution
        total = len(agent.action_history[-20:])
        entropy = 0
        for count in action_counts.values():
            prob = count / total
            entropy -= prob * math.log(prob)

        # Normalize to [0, 1] range - maximum entropy for uniform distribution
        # of n actions is log(n)
        max_entropy = math.log(len(action_counts))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy

            # Reward high entropy (diverse actions)
            reward += normalized_entropy * 0.3

            # Even higher reward if current action is rarely used
            current_action_freq = action_counts.get(current_action, 0) / total
            if current_action_freq < 0.1:  # Rarely used action
                reward += 0.5

    # Update action history
    agent.action_history.append(current_action)
    if len(agent.action_history) > 100:
        agent.action_history.pop(0)

    return reward


def _enhanced_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
    agent=None,  # Pass the agent for tracking action history
) -> float:
    """
    Enhanced reward function combining multiple strategies.
    """
    # Base reward from original function
    reward = _basic_reward(events, observation, previous_observation)

    # Add tactical combat rewards
    reward += calculate_tactical_reward(events, observation, previous_observation)

    # Add spatial awareness rewards
    reward += calculate_spatial_awareness_reward(observation, previous_observation)

    # Add strategic planning rewards
    reward += calculate_strategic_reward(observation, previous_observation)

    # Add learning progress rewards (if agent is provided)
    if agent:
        reward += calculate_learning_reward(observation, agent)

    return reward
