"""
Reward functions for reinforcement learning in AgentArena
"""

import numpy as np
from enum import Enum

class RewardType(Enum):
    BASIC = "basic"           # Simple rewards for hits and avoiding damage
    AGGRESSIVE = "aggressive" # Prioritizes dealing damage over safety
    DEFENSIVE = "defensive"   # Prioritizes staying alive over dealing damage
    ADVANCED = "advanced"     # More nuanced reward calculation with more factors


def calculate_reward(events, observation, previous_observation=None, reward_type=RewardType.BASIC):
    """
    Calculate rewards based on game events and observations
    
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
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def _basic_reward(events, observation, previous_observation=None):
    """Basic reward function focusing on hits and survival"""
    reward = 0
    
    # Reward for hitting enemies
    for event in events:
        if event['type'] == 'enemy_hit':
            reward += 1.0
        elif event['type'] == 'player_hit':
            reward -= 1.0
    
    # Small penalty for each step to encourage faster completion
    reward -= 0.01
    
    return reward


def _aggressive_reward(events, observation, previous_observation=None):
    """Reward function that encourages aggressive play"""
    reward = 0
    
    # Large reward for hitting enemies
    for event in events:
        if event['type'] == 'enemy_hit':
            reward += 2.0
        elif event['type'] == 'player_hit':
            reward -= 0.5  # Smaller penalty for getting hit
    
    # Reward for shooting
    if _is_shooting(observation, previous_observation):
        reward += 0.1
    
    # Small penalty for each step to encourage faster completion
    reward -= 0.01
    
    return reward


def _defensive_reward(events, observation, previous_observation=None):
    """Reward function that encourages defensive play"""
    reward = 0
    
    # Moderate reward for hitting enemies
    for event in events:
        if event['type'] == 'enemy_hit':
            reward += 0.5
        elif event['type'] == 'player_hit':
            reward -= 2.0  # Larger penalty for getting hit
    
    # Reward for staying alive
    reward += 0.02
    
    # Reward for maintaining distance from enemies
    if previous_observation:
        prev_distance = _min_enemy_distance(previous_observation)
        curr_distance = _min_enemy_distance(observation)
        
        # If there are enemies nearby and the player moved away from them
        if prev_distance < 300 and curr_distance > prev_distance:
            reward += 0.05
    
    return reward


def _advanced_reward(events, observation, previous_observation=None):
    """Advanced reward function with multiple components"""
    reward = 0
    
    # Event-based rewards
    for event in events:
        if event['type'] == 'enemy_hit':
            reward += 1.0
        elif event['type'] == 'player_hit':
            reward -= 1.0
    
    # Small penalty for each step to encourage faster completion
    reward -= 0.01
    
    # If we have previous observation, we can calculate more rewards
    if previous_observation:
        # Reward for moving toward enemies when health is high
        player_health = observation['player']['health']
        if player_health > 1:  # If health is good, encourage attacking
            prev_distance = _min_enemy_distance(previous_observation)
            curr_distance = _min_enemy_distance(observation)
            
            if prev_distance > curr_distance:
                reward += 0.05  # Reward for closing in
        else:
            # If health is low, encourage defensive play
            prev_distance = _min_enemy_distance(previous_observation)
            curr_distance = _min_enemy_distance(observation)
            
            if prev_distance < 300 and curr_distance > prev_distance:
                reward += 0.1  # Reward for backing away when low health
        
        # Reward for dodging bullets
        if _is_dodging_bullets(observation, previous_observation):
            reward += 0.2
    
    # Penalty for being close to too many bullets
    bullet_danger = _bullet_danger_level(observation)
    reward -= 0.05 * bullet_danger
    
    return reward


# Helper functions for reward calculation

def _min_enemy_distance(observation):
    """Calculate minimum distance to any enemy"""
    player_x = observation['player']['x']
    player_y = observation['player']['y']
    
    min_distance = float('inf')
    for enemy in observation['enemies']:
        dx = enemy['x'] - player_x
        dy = enemy['y'] - player_y
        distance = np.sqrt(dx**2 + dy**2)
        min_distance = min(min_distance, distance)
    
    return min_distance if min_distance != float('inf') else 0


def _is_shooting(observation, previous_observation):
    """Determine if player fired a shot in this step"""
    if not previous_observation:
        return False
    
    # Count bullets in current and previous observation
    current_bullets = len(observation['bullets'])
    previous_bullets = len(previous_observation['bullets'])
    
    # If we have more bullets now, player likely shot
    return current_bullets > previous_bullets


def _is_dodging_bullets(observation, previous_observation):
    """Check if player successfully dodged bullets"""
    if not previous_observation:
        return False
    
    player_x = observation['player']['x']
    player_y = observation['player']['y']
    
    # Look for bullets that were close to player in previous observation
    close_bullets_before = 0
    for bullet in previous_observation['bullets']:
        if bullet['owner'] != 'player':  # Only care about enemy bullets
            dx = bullet['x'] - previous_observation['player']['x']
            dy = bullet['y'] - previous_observation['player']['y']
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < 100:  # Bullet was close
                close_bullets_before += 1
    
    # Look for the same bullets in current observation
    # They should now be farther away if dodged successfully
    dodged_bullets = 0
    for bullet in observation['bullets']:
        if bullet['owner'] != 'player':  # Only care about enemy bullets
            dx = bullet['x'] - player_x
            dy = bullet['y'] - player_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 100:  # Bullet is now farther away
                dodged_bullets += 1
    
    # If we had close bullets before and now they're farther, we dodged
    return close_bullets_before > 0 and dodged_bullets >= close_bullets_before


def _bullet_danger_level(observation):
    """Calculate how dangerous the current bullet situation is"""
    player_x = observation['player']['x']
    player_y = observation['player']['y']
    
    danger_level = 0
    for bullet in observation['bullets']:
        if bullet['owner'] != 'player':  # Only care about enemy bullets
            dx = bullet['x'] - player_x
            dy = bullet['y'] - player_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate if bullet is heading toward player
            bullet_dir_x, bullet_dir_y = bullet['direction']
            
            # Normalize direction to player
            dir_to_player_x = -dx / (distance + 1e-6)  # Avoid division by zero
            dir_to_player_y = -dy / (distance + 1e-6)
            
            # Dot product to see if directions align (bullet heading toward player)
            dot_product = bullet_dir_x * dir_to_player_x + bullet_dir_y * dir_to_player_y
            
            # Higher danger for closer bullets moving toward player
            if dot_product > 0:  # Bullet is moving toward player
                danger = (1.0 / (distance + 1)) * dot_product
                danger_level += danger
    
    return danger_level
