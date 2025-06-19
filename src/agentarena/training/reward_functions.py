import math

from agentarena.agent.agent import Agent
from agentarena.models.events import (
    BulletFiredEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    GameEvent,
    PlayerHitEvent,
)
from agentarena.models.observations import GameObservation
from agentarena.models.training import RewardType

NEARBY_DISTANCE_THRESHOLD = 300.0  # Distance threshold for considering enemies "nearby"
BULLET_DODGE_DISTANCE = 100.0  # Distance threshold for bullet dodging detection


def calculate_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
    reward_type: RewardType = RewardType.BASIC,
) -> float:
    if reward_type == RewardType.BASIC:
        return _basic_reward(events, observation, previous_observation)
    if reward_type == RewardType.ADVANCED:
        return _advanced_reward(events, observation, previous_observation)
    if reward_type == RewardType.ENHANCED:
        return _enhanced_reward(
            events,
            observation,
            previous_observation,
        )

    # Invalid reward type
    error_msg = f"Unknown reward type: {reward_type}"
    raise ValueError(error_msg)


def _basic_reward(
    events: list[GameEvent],
    observation: GameObservation,  # noqa: ARG001
    previous_observation: GameObservation | None = None,  # noqa: ARG001
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
            reward += 2.0
        elif isinstance(event, PlayerHitEvent):
            reward -= 1.0
        elif isinstance(event, EntityDestroyedEvent) and event.is_enemy_destroyed():
            reward += 4.0  # Bonus for destroying an enemy

    return reward


def _advanced_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
) -> float:
    reward = 0.0

    # Event-based rewards
    for event in events:
        if isinstance(event, EnemyHitEvent):
            reward += 2.0
        elif isinstance(event, PlayerHitEvent):
            reward -= 1.0
        elif isinstance(event, EntityDestroyedEvent):
            if event.is_enemy_destroyed():
                reward += 5.0  # Bonus for destroying an enemy
            elif event.is_player_destroyed():
                reward -= 5.0  # Large penalty for dying
        elif isinstance(event, BulletFiredEvent) and event.owner_id == "player":
            reward -= 0.01

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

    # If we have previous observation, we can calculate more rewards
    if previous_observation:
        # Reward for moving toward enemies when health is high
        if previous_observation.player.orientation != observation.player.orientation:
            reward += 0.5
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
            if movement_intent and distance_moved < 1.0:  # Threshold for detecting wall collision
                reward -= 1  # Significant penalty for hitting a wall

        # Reward for dodging bullets
        prev_bullets_near = len(previous_observation.bullets_near_player(BULLET_DODGE_DISTANCE))
        curr_bullets_near = len(observation.bullets_near_player(BULLET_DODGE_DISTANCE))

        if prev_bullets_near > curr_bullets_near and prev_bullets_near > 0:
            reward += 1  # Reward for having fewer bullets nearby than before

        # Penalty for being close to too many bullets
        bullet_danger = len(observation.bullets_near_player())
        reward -= 0.5 * bullet_danger

    return math.tanh(reward)


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

                dot_product = bullet_dir[0] * to_player[0] + bullet_dir[1] * to_player[1]
                if dot_product > 0:
                    danger_level += dot_product * (1 - (distance / 150))

    if (
        previous_observation
        and hasattr(
            previous_observation,
            "bullets_near_player",
        )
        and bullets_near_player < len(previous_observation.bullets_near_player())
    ):
        reward += 0.3 * (len(previous_observation.bullets_near_player()) - bullets_near_player)

    # Penalty based on danger level
    reward -= danger_level * 0.2

    return reward


def calculate_strategic_reward(observation, previous_observation):  # noqa: ARG001
    reward = 0

    # Reward for maintaining line of sight to multiple enemies
    if observation.enemies:
        visible_enemies = 0
        for _ in observation.enemies:
            # Simple line of sight check (could be more complex with ray casting)
            visible_enemies += 1

        # Reward scales with number of enemies visible
        reward += 0.1 * visible_enemies

    # Reward for not getting cornered
    # Count walls or screen edges nearby
    boundaries = 0
    screen_margin = 128

    # Check proximity to screen edges
    if observation.player.x < screen_margin:
        boundaries += 1
    if observation.player.x > 1200 - screen_margin:  # Assuming 1200 width
        boundaries += 1
    if observation.player.y < screen_margin:
        boundaries += 1
    if observation.player.y > 900 - screen_margin:  # Assuming 900 height
        boundaries += 1

    reward -= 0.1 * boundaries

    return reward


def calculate_learning_reward(observation, agent):  # noqa: ARG001
    reward = 0

    if not hasattr(agent, "action_history"):
        agent.action_history = []

    current_action = agent.last_action

    if len(agent.action_history) > 20:
        action_counts = {}
        for a in agent.action_history[-20:]:
            action_counts[a] = action_counts.get(a, 0) + 1

        total = len(agent.action_history[-20:])
        entropy = 0
        for count in action_counts.values():
            prob = count / total
            entropy -= prob * math.log(prob)

        max_entropy = math.log(len(action_counts))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy

            reward += normalized_entropy * 0.3

            current_action_freq = action_counts.get(current_action, 0) / total
            if current_action_freq > 0.8:
                reward -= 0.5
    agent.action_history.append(current_action)
    if len(agent.action_history) > 100:
        agent.action_history.pop(0)

    return reward


def calculate_map_control_reward(observation: GameObservation, agent=None):
    reward = 0.0

    # Check if agent has position_history attribute, create if not
    if agent and not hasattr(agent, "position_history"):
        agent.position_history = []
        agent.position_grid = {}
        agent.last_new_area_time = 0

    if agent and hasattr(agent, "position_history"):
        player_pos = (observation.player.x, observation.player.y)

        grid_size = 50
        grid_x = int(player_pos[0] / grid_size)
        grid_y = int(player_pos[1] / grid_size)
        grid_pos = (grid_x, grid_y)

        if grid_pos not in agent.position_grid:
            agent.position_grid[grid_pos] = observation.game_time
            agent.last_new_area_time = observation.game_time

            reward += 0.5

        time_since_new_area = observation.game_time - agent.last_new_area_time
        if time_since_new_area > 10.0:
            reward -= 0.05

        agent.position_history.append(player_pos)
        if len(agent.position_history) > 100:
            agent.position_history.pop(0)

    return reward


def winning_reward(observation: GameObservation) -> float:
    reward = 0
    current_enemy_count = len(observation.enemies)
    if current_enemy_count == 0:
        reward += 20.0
    return reward


def _enhanced_reward(
    events: list[GameEvent],
    observation: GameObservation,
    previous_observation: GameObservation | None = None,
    agent: Agent | None = None,
) -> float:
    reward = _basic_reward(events, observation, previous_observation)
    reward += winning_reward(observation)
    other_reward = 0
    other_reward += calculate_tactical_reward(
        events,
        observation,
        previous_observation,
    )
    other_reward += calculate_strategic_reward(
        observation,
        previous_observation,
    )
    if agent:
        other_reward += calculate_learning_reward(observation, agent)

    return math.tanh((3 * reward + other_reward) / 4)
