"""
Improved rule-based agent with better pathfinding capabilities.
"""

import math
import random
from collections import deque

from agentarena.agent.agent import Agent
from agentarena.models.action import Action, Direction
from agentarena.models.observations import EnemyObservation, GameObservation


class RuleBasedAgent(Agent):
    """
    Improved rule-based agent with robust pathfinding capabilities.

    Features:
    - A* pathfinding algorithm for navigation around obstacles
    - Dynamic waypoint generation based on map analysis
    - Better wall avoidance and navigation
    - Adaptive behavior based on tactical situation
    """

    def __init__(self, name: str = "RuleBasedAgent") -> None:
        super().__init__(name)

        # Behavioral parameters
        self.danger_radius = 250.0
        self.attack_range = 350.0
        self.min_attack_range = 150.0  # Increased minimum distance
        self.optimal_attack_range = 180.0  # Sweet spot for engagement
        self.max_engagement_range = 350.0  # Maximum effective range
        self.wall_avoidance_distance = 100.0

        # Shooting behavior parameters
        self.shooting_probability = 0.4
        self.aggressive_shooting_distance = 200.0
        self.last_shot_time = 0
        self.shot_cooldown_frames = 15

        # Pathfinding parameters
        self.grid_size = 32  # Smaller grid for more precise pathfinding
        self.max_pathfinding_iterations = 100

        # State tracking
        self.last_position: tuple[float, float] | None = None
        self.stuck_counter = 0
        self.current_path: list[tuple[float, float]] = []
        self.current_path_index = 0
        self.frames_since_last_enemy_sight = 0
        self.frame_count = 0
        self.map_grid = {}  # Cache of walkable/non-walkable areas
        self.last_target_enemy_pos = None

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.last_position = None
        self.stuck_counter = 0
        self.current_path = []
        self.current_path_index = 0
        self.frames_since_last_enemy_sight = 0
        self.frame_count = 0
        self.map_grid = {}
        self.last_target_enemy_pos = None
        self.last_shot_time = 0

    def get_action(self, observation: GameObservation) -> Action:
        """
        Determine the best action with improved pathfinding.
        """
        self.frame_count += 1
        player = observation.player

        # Update position tracking
        current_pos = (player.x, player.y)
        if self.last_position:
            distance_moved = self._calculate_distance(current_pos, self.last_position)
            if distance_moved < 3.0:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
        self.last_position = current_pos

        # Track enemy visibility
        if observation.enemies:
            self.frames_since_last_enemy_sight = 0
        else:
            self.frames_since_last_enemy_sight += 1

        # Update map knowledge
        self._update_map_knowledge(observation)

        # Priority 1: Emergency dodge (immediate danger)
        emergency_action = self._get_emergency_dodge_action(observation)
        if emergency_action:
            return emergency_action

        # Priority 2: Aggressive shooting when enemies are visible and in range
        if observation.enemies:
            shooting_action = self._get_balanced_shooting_action(observation)
            if shooting_action:
                return shooting_action

        # Priority 3: Tactical positioning and engagement
        if observation.enemies:
            tactical_action = self._get_tactical_positioning_action(observation)
            if tactical_action:
                return tactical_action

        # Priority 4: Exploration when no enemies visible
        return self._get_exploration_action(observation)

    def _update_map_knowledge(self, observation: GameObservation) -> None:
        """Update our knowledge of the map layout."""
        # Create a grid representation of walkable/non-walkable areas
        for wall in observation.walls:
            # Mark wall cells as non-walkable
            wall_grid_x = int(wall.x // self.grid_size)
            wall_grid_y = int(wall.y // self.grid_size)

            # Mark a 2x2 area around each wall to account for wall size
            for dx in range(-1, 3):
                for dy in range(-1, 3):
                    grid_pos = (wall_grid_x + dx, wall_grid_y + dy)
                    self.map_grid[grid_pos] = False  # Non-walkable

    def _get_tactical_positioning_action(self, observation: GameObservation) -> Action | None:
        """
        Tactical positioning that maintains optimal engagement distance.
        """
        player = observation.player

        if not observation.enemies:
            return None

        # Find closest enemy
        closest_enemy = min(
            observation.enemies,
            key=lambda e: self._calculate_distance((player.x, player.y), (e.x, e.y)),
        )

        current_pos = (player.x, player.y)
        enemy_pos = (closest_enemy.x, closest_enemy.y)
        distance_to_enemy = self._calculate_distance(current_pos, enemy_pos)

        # Determine desired action based on distance
        if distance_to_enemy < self.min_attack_range:
            # Too close - retreat while maintaining line of sight
            return self._get_tactical_retreat_action(observation, closest_enemy)
        elif distance_to_enemy > self.max_engagement_range:
            # Too far - advance carefully
            return self._get_tactical_advance_action(observation, closest_enemy)
        elif abs(distance_to_enemy - self.optimal_attack_range) > 30:
            # Not in optimal range - adjust position
            return self._get_position_adjustment_action(observation, closest_enemy)
        else:
            # In good range - maintain position with tactical movement
            return self._get_tactical_maintenance_action(observation, closest_enemy)

    def _get_tactical_retreat_action(
        self, observation: GameObservation, enemy: EnemyObservation
    ) -> Action | None:
        """
        Retreat while maintaining ability to shoot and dodge.
        """
        player = observation.player
        current_pos = (player.x, player.y)
        enemy_pos = (enemy.x, enemy.y)

        # Calculate retreat direction (away from enemy)
        retreat_vector = self._normalize_vector(self._get_vector(enemy_pos, current_pos))

        # Add perpendicular movement for unpredictability
        perp_vector = (-retreat_vector[1], retreat_vector[0])

        # Combine retreat with lateral movement
        lateral_factor = random.uniform(-0.4, 0.4)
        combined_direction = self._normalize_vector(
            (
                retreat_vector[0] + lateral_factor * perp_vector[0],
                retreat_vector[1] + lateral_factor * perp_vector[1],
            )
        )

        # Check multiple retreat options
        retreat_options = [
            combined_direction,
            retreat_vector,
            self._normalize_vector(
                (retreat_vector[0] + 0.3 * perp_vector[0], retreat_vector[1] + 0.3 * perp_vector[1])
            ),
            self._normalize_vector(
                (retreat_vector[0] - 0.3 * perp_vector[0], retreat_vector[1] - 0.3 * perp_vector[1])
            ),
        ]

        best_option = None
        best_score = -1

        for option in retreat_options:
            score = self._evaluate_tactical_movement(observation, option, enemy_pos)
            if score > best_score:
                best_score = score
                best_option = option

        if best_option and best_score > 0.2:
            return Action(direction=self._vector_to_direction(best_option), is_shooting=False)

        return None

    def _get_tactical_advance_action(
        self, observation: GameObservation, enemy: EnemyObservation
    ) -> Action | None:
        """
        Advance toward enemy while maintaining tactical advantage.
        """
        player = observation.player
        current_pos = (player.x, player.y)
        enemy_pos = (enemy.x, enemy.y)

        # Check if we have clear line of sight
        has_los = self._has_clear_line_of_sight(observation, current_pos, enemy_pos)

        if has_los:
            # Direct approach with caution
            advance_vector = self._normalize_vector(self._get_vector(current_pos, enemy_pos))

            # Add some lateral movement for unpredictability
            perp_vector = (-advance_vector[1], advance_vector[0])
            lateral_factor = random.uniform(-0.3, 0.3)

            tactical_advance = self._normalize_vector(
                (
                    advance_vector[0] + lateral_factor * perp_vector[0],
                    advance_vector[1] + lateral_factor * perp_vector[1],
                )
            )

            if self._evaluate_tactical_movement(observation, tactical_advance, enemy_pos) > 0.3:
                return Action(
                    direction=self._vector_to_direction(tactical_advance), is_shooting=False
                )

        # No clear line of sight - use pathfinding
        return self._get_pathfinding_action_improved(observation)

    def _get_position_adjustment_action(
        self, observation: GameObservation, enemy: EnemyObservation
    ) -> Action | None:
        """
        Fine-tune position to reach optimal engagement range.
        """
        player = observation.player
        current_pos = (player.x, player.y)
        enemy_pos = (enemy.x, enemy.y)
        distance_to_enemy = self._calculate_distance(current_pos, enemy_pos)

        if distance_to_enemy < self.optimal_attack_range:
            # Move away slightly
            direction = self._normalize_vector(self._get_vector(enemy_pos, current_pos))
        else:
            # Move closer slightly
            direction = self._normalize_vector(self._get_vector(current_pos, enemy_pos))

        # Add perpendicular component for better positioning
        perp_vector = (-direction[1], direction[0])

        # Try to position for better angles
        angle_factor = random.uniform(-0.5, 0.5)
        adjusted_direction = self._normalize_vector(
            (
                direction[0] + angle_factor * perp_vector[0],
                direction[1] + angle_factor * perp_vector[1],
            )
        )

        if self._evaluate_tactical_movement(observation, adjusted_direction, enemy_pos) > 0.3:
            return Action(
                direction=self._vector_to_direction(adjusted_direction), is_shooting=False
            )

        return None

    def _get_tactical_maintenance_action(
        self, observation: GameObservation, enemy: EnemyObservation
    ) -> Action | None:
        """
        Maintain position with tactical movement for dodging.
        """
        player = observation.player

        # Check for immediate threats
        nearby_bullets = observation.bullets_near_player(100)
        if nearby_bullets:
            # Prioritize dodging
            return self._get_dodge_movement_action(observation, nearby_bullets)

        # Maintain position with small movements for unpredictability
        current_pos = (player.x, player.y)
        enemy_pos = (enemy.x, enemy.y)

        # Generate small circular/figure-8 movements around current position
        time_factor = (self.frame_count * 0.1) % (2 * math.pi)

        # Create unpredictable movement pattern
        offset_x = math.sin(time_factor) * 0.3
        offset_y = math.cos(time_factor * 1.3) * 0.3

        # Ensure we don't move too close to enemy
        to_enemy = self._normalize_vector(self._get_vector(current_pos, enemy_pos))
        dot_product = offset_x * to_enemy[0] + offset_y * to_enemy[1]

        if dot_product > 0.5:  # Moving toward enemy
            # Reverse the movement
            offset_x = -offset_x
            offset_y = -offset_y

        movement_direction = self._normalize_vector((offset_x, offset_y))

        if self._evaluate_tactical_movement(observation, movement_direction, enemy_pos) > 0.2:
            return Action(
                direction=self._vector_to_direction(movement_direction), is_shooting=False
            )

        return None

    def _get_dodge_movement_action(
        self, observation: GameObservation, nearby_bullets: list
    ) -> Action | None:
        """
        Specific movement for dodging nearby bullets.
        """
        player = observation.player
        current_pos = (player.x, player.y)

        # Calculate combined threat vector from all nearby bullets
        threat_vector = [0, 0]

        for bullet in nearby_bullets:
            bullet_pos = (bullet.x, bullet.y)
            bullet_dir = bullet.direction

            # Vector from bullet to player
            to_player = self._get_vector(bullet_pos, current_pos)
            distance = self._calculate_distance(bullet_pos, current_pos)

            if distance > 0:
                # Weight by proximity and bullet direction alignment
                alignment = (bullet_dir[0] * to_player[0] + bullet_dir[1] * to_player[1]) / distance
                if alignment > 0:  # Bullet coming toward player
                    weight = (1.0 / max(distance, 10)) * alignment
                    threat_vector[0] += bullet_dir[0] * weight
                    threat_vector[1] += bullet_dir[1] * weight

        if threat_vector[0] != 0 or threat_vector[1] != 0:
            # Move perpendicular to threat
            threat_mag = math.sqrt(threat_vector[0] ** 2 + threat_vector[1] ** 2)
            if threat_mag > 0:
                threat_normalized = [threat_vector[0] / threat_mag, threat_vector[1] / threat_mag]

                # Perpendicular directions
                dodge_options = [
                    (-threat_normalized[1], threat_normalized[0]),  # Left perpendicular
                    (threat_normalized[1], -threat_normalized[0]),  # Right perpendicular
                    (-threat_normalized[0], -threat_normalized[1]),  # Directly away
                ]

                best_dodge = None
                best_score = -1

                for dodge_dir in dodge_options:
                    score = self._evaluate_movement_direction(observation, dodge_dir)
                    if score > best_score:
                        best_score = score
                        best_dodge = dodge_dir

                if best_dodge and best_score > 0.3:
                    return Action(
                        direction=self._vector_to_direction(best_dodge), is_shooting=False
                    )

        return None

    def _evaluate_tactical_movement(
        self,
        observation: GameObservation,
        direction: tuple[float, float],
        enemy_pos: tuple[float, float],
    ) -> float:
        """
        Evaluate movement direction for tactical advantage.
        """
        base_score = self._evaluate_movement_direction(observation, direction)

        if base_score <= 0:
            return base_score

        player = observation.player
        current_pos = (player.x, player.y)
        move_distance = 40
        new_pos = (
            current_pos[0] + direction[0] * move_distance,
            current_pos[1] + direction[1] * move_distance,
        )

        # Check new distance to enemy
        new_distance = self._calculate_distance(new_pos, enemy_pos)

        # Bonus for being in optimal range
        if self.min_attack_range <= new_distance <= self.max_engagement_range:
            range_bonus = (
                1.0 - abs(new_distance - self.optimal_attack_range) / self.optimal_attack_range
            )
            base_score += range_bonus * 0.5

        # Penalty for getting too close
        if new_distance < self.min_attack_range:
            base_score -= (self.min_attack_range - new_distance) / self.min_attack_range

        # Bonus for maintaining line of sight
        if self._has_clear_line_of_sight(observation, new_pos, enemy_pos):
            base_score += 0.3

        # Check if this position gives better cover options
        escape_routes = 0
        test_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for test_dir in test_directions:
            test_pos = (new_pos[0] + test_dir[0] * 50, new_pos[1] + test_dir[1] * 50)
            if self._is_position_safe(observation, test_pos):
                escape_routes += 1

        base_score += escape_routes * 0.1

        return base_score

    def _is_position_safe(self, observation: GameObservation, pos: tuple[float, float]) -> bool:
        """Check if a position is safe (no walls, within bounds)."""
        x, y = pos

        # Check bounds
        if x < 50 or x > 1150 or y < 50 or y > 850:
            return False

        # Check walls
        for wall in observation.walls:
            if wall.x - 20 <= x <= wall.x + 84 and wall.y - 20 <= y <= wall.y + 84:
                return False

        return True

    def _get_pathfinding_action_improved(self, observation: GameObservation) -> Action | None:
        """
        Improved pathfinding that aims for optimal engagement distance, not direct contact.
        """
        player = observation.player

        if not observation.enemies:
            return None

        # Find closest enemy
        closest_enemy = min(
            observation.enemies,
            key=lambda e: self._calculate_distance((player.x, player.y), (e.x, e.y)),
        )

        enemy_pos = (closest_enemy.x, closest_enemy.y)
        current_pos = (player.x, player.y)
        current_distance = self._calculate_distance(current_pos, enemy_pos)

        # Calculate target position (optimal engagement distance from enemy)
        direction_to_enemy = self._normalize_vector(self._get_vector(current_pos, enemy_pos))

        # Target position at optimal range
        target_pos = (
            enemy_pos[0] - direction_to_enemy[0] * self.optimal_attack_range,
            enemy_pos[1] - direction_to_enemy[1] * self.optimal_attack_range,
        )

        # If target position is not safe, try alternative positions around the enemy
        if not self._is_position_safe(observation, target_pos):
            target_pos = self._find_safe_engagement_position(observation, enemy_pos)

        if not target_pos:
            return None

        # Check if we need to recalculate path
        should_recalculate = (
            not self.current_path
            or self.stuck_counter > 10
            or self._calculate_distance(enemy_pos, self.last_target_enemy_pos or (0, 0)) > 50
            or self.current_path_index >= len(self.current_path) - 1
            or self._calculate_distance(current_pos, target_pos) < 40  # Close enough to target
        )

        if should_recalculate:
            self.current_path = self._find_path_astar(current_pos, target_pos, observation)
            self.current_path_index = 0
            self.last_target_enemy_pos = enemy_pos
            self.stuck_counter = 0

        # Follow the current path
        if self.current_path and self.current_path_index < len(self.current_path):
            target_waypoint = self.current_path[self.current_path_index]

            # Check if we've reached the current waypoint
            if self._calculate_distance(current_pos, target_waypoint) < 30:
                self.current_path_index += 1
                if self.current_path_index < len(self.current_path):
                    target_waypoint = self.current_path[self.current_path_index]
                else:
                    # Reached end of path - we should be in good position now
                    return None

            # Move toward current waypoint
            direction_to_waypoint = self._normalize_vector(
                self._get_vector(current_pos, target_waypoint)
            )

            # Check if this direction is safe to move
            if self._evaluate_movement_direction(observation, direction_to_waypoint) > 0.1:
                return Action(
                    direction=self._vector_to_direction(direction_to_waypoint),
                    is_shooting=False,
                )
            else:
                # Path is blocked, recalculate
                self.current_path = []
                return self._get_emergency_movement(observation)

        # If no path found, try direct approach with obstacle avoidance
        return self._get_obstacle_avoidance_movement(observation, target_pos)

    def _find_safe_engagement_position(
        self, observation: GameObservation, enemy_pos: tuple[float, float]
    ) -> tuple[float, float] | None:
        """
        Find a safe position at optimal engagement distance from enemy.
        """
        # Try positions in a circle around the enemy at optimal range
        for angle in range(0, 360, 30):  # Check every 30 degrees
            rad = math.radians(angle)
            test_pos = (
                enemy_pos[0] + math.cos(rad) * self.optimal_attack_range,
                enemy_pos[1] + math.sin(rad) * self.optimal_attack_range,
            )

            if self._is_position_safe(observation, test_pos):
                # Check if we have line of sight from this position
                if self._has_clear_line_of_sight(observation, test_pos, enemy_pos):
                    return test_pos

        # If no position with line of sight found, try at max engagement range
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            test_pos = (
                enemy_pos[0] + math.cos(rad) * self.max_engagement_range,
                enemy_pos[1] + math.sin(rad) * self.max_engagement_range,
            )

            if self._is_position_safe(observation, test_pos):
                return test_pos

        return None

    def _find_path_astar(
        self, start: tuple[float, float], goal: tuple[float, float], observation: GameObservation
    ) -> list[tuple[float, float]]:
        """
        A* pathfinding algorithm to find optimal path around obstacles.
        """
        # Convert positions to grid coordinates
        start_grid = (int(start[0] // self.grid_size), int(start[1] // self.grid_size))
        goal_grid = (int(goal[0] // self.grid_size), int(goal[1] // self.grid_size))

        # A* algorithm implementation
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        iterations = 0
        while open_set and iterations < self.max_pathfinding_iterations:
            iterations += 1

            # Get node with lowest f_score
            current_f, current = min(open_set)
            open_set.remove((current_f, current))

            # Check if we reached the goal
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    # Convert back to world coordinates
                    world_pos = (
                        current[0] * self.grid_size + self.grid_size / 2,
                        current[1] * self.grid_size + self.grid_size / 2,
                    )
                    path.append(world_pos)
                    current = came_from[current]
                path.reverse()
                return path

            # Check all neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check if neighbor is walkable
                if not self._is_grid_position_walkable(neighbor, observation):
                    continue

                # Calculate movement cost (diagonal moves cost more)
                move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)

                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))

        # No path found, return empty path
        return []

    def _heuristic(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
        """Manhattan distance heuristic for A*."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_grid_position_walkable(
        self, grid_pos: tuple[int, int], observation: GameObservation
    ) -> bool:
        """Check if a grid position is walkable."""
        # Check bounds (assuming game area)
        world_x = grid_pos[0] * self.grid_size
        world_y = grid_pos[1] * self.grid_size

        if world_x < 50 or world_x > 1150 or world_y < 50 or world_y > 850:
            return False

        # Check if position is cached
        if grid_pos in self.map_grid:
            return self.map_grid[grid_pos]

        # Check against walls
        for wall in observation.walls:
            if wall.x - 40 <= world_x <= wall.x + 104 and wall.y - 40 <= world_y <= wall.y + 104:
                self.map_grid[grid_pos] = False
                return False

        # Position is walkable
        self.map_grid[grid_pos] = True
        return True

    def _get_obstacle_avoidance_movement(
        self, observation: GameObservation, target_pos: tuple[float, float]
    ) -> Action | None:
        """
        Smart obstacle avoidance when direct pathfinding fails.
        """
        player = observation.player
        current_pos = (player.x, player.y)

        # Try different directions in order of preference
        direct_direction = self._normalize_vector(self._get_vector(current_pos, target_pos))

        # Generate candidate directions
        candidate_directions = []

        # Add direct direction first
        candidate_directions.append(direct_direction)

        # Add perpendicular directions
        perp_left = (-direct_direction[1], direct_direction[0])
        perp_right = (direct_direction[1], -direct_direction[0])
        candidate_directions.extend([perp_left, perp_right])

        # Add diagonal combinations
        for weight in [0.7, 0.5]:
            diag_left = self._normalize_vector(
                (
                    direct_direction[0] + weight * perp_left[0],
                    direct_direction[1] + weight * perp_left[1],
                )
            )
            diag_right = self._normalize_vector(
                (
                    direct_direction[0] + weight * perp_right[0],
                    direct_direction[1] + weight * perp_right[1],
                )
            )
            candidate_directions.extend([diag_left, diag_right])

        # Test each direction and pick the best one
        best_direction = None
        best_score = -1

        for direction in candidate_directions:
            score = self._evaluate_movement_direction(observation, direction)

            # Bonus for directions that get us closer to target
            alignment = direction[0] * direct_direction[0] + direction[1] * direct_direction[1]
            score += alignment * 0.5

            if score > best_score:
                best_score = score
                best_direction = direction

        if best_direction and best_score > 0.1:
            return Action(
                direction=self._vector_to_direction(best_direction),
                is_shooting=False,
            )

        return None

    def _get_emergency_movement(self, observation: GameObservation) -> Action:
        """Emergency movement when stuck or blocked."""
        # Try all cardinal and diagonal directions
        emergency_directions = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),  # Cardinal
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),  # Diagonal
        ]

        random.shuffle(emergency_directions)

        for direction in emergency_directions:
            score = self._evaluate_movement_direction(observation, direction)
            if score > 0.3:
                return Action(
                    direction=self._vector_to_direction(direction),
                    is_shooting=False,
                )

        # If all else fails, try to move away from walls
        return self._get_wall_escape_movement(observation)

    def _get_wall_escape_movement(self, observation: GameObservation) -> Action:
        """Try to escape from being surrounded by walls."""
        player = observation.player

        # Find direction with most space
        best_direction = None
        max_clearance = 0

        test_directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]

        for direction in test_directions:
            clearance = self._calculate_clearance_in_direction(
                observation, (player.x, player.y), direction
            )
            if clearance > max_clearance:
                max_clearance = clearance
                best_direction = direction

        if best_direction:
            return Action(
                direction=self._vector_to_direction(best_direction),
                is_shooting=False,
            )

        # Absolute fallback
        return Action(direction=Direction.DOWN, is_shooting=False)

    def _calculate_clearance_in_direction(
        self,
        observation: GameObservation,
        start_pos: tuple[float, float],
        direction: tuple[float, float],
    ) -> float:
        """Calculate how much clear space is available in a direction."""
        clearance = 0
        step_size = 20
        max_steps = 10

        for step in range(1, max_steps + 1):
            test_pos = (
                start_pos[0] + direction[0] * step_size * step,
                start_pos[1] + direction[1] * step_size * step,
            )

            # Check bounds
            if test_pos[0] < 50 or test_pos[0] > 1150 or test_pos[1] < 50 or test_pos[1] > 850:
                break

            # Check walls
            blocked = False
            for wall in observation.walls:
                if (
                    wall.x - 20 <= test_pos[0] <= wall.x + 84
                    and wall.y - 20 <= test_pos[1] <= wall.y + 84
                ):
                    blocked = True
                    break

            if blocked:
                break

            clearance = step * step_size

        return clearance

    # Keep all the existing utility methods unchanged
    def _get_emergency_dodge_action(self, observation: GameObservation) -> Action | None:
        """Emergency dodging for immediate bullet threats."""
        player = observation.player
        immediate_threats = []

        for bullet in observation.bullets:
            if bullet.owner == "player":
                continue

            bullet_pos = (bullet.x, bullet.y)
            distance_to_bullet = self._calculate_distance((player.x, player.y), bullet_pos)

            if distance_to_bullet < 80:
                bullet_dir = bullet.direction
                to_player = self._normalize_vector((player.x - bullet.x, player.y - bullet.y))
                threat_level = bullet_dir[0] * to_player[0] + bullet_dir[1] * to_player[1]

                if threat_level > 0.5:
                    immediate_threats.append((bullet, threat_level, distance_to_bullet))

        if not immediate_threats:
            return None

        dodge_directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ]

        best_direction = None
        best_score = -1

        for dodge_dir in dodge_directions:
            score = self._evaluate_dodge_direction(observation, dodge_dir)
            if score > best_score:
                best_score = score
                best_direction = dodge_dir

        if best_direction and best_score > 0.2:
            return Action(direction=self._vector_to_direction(best_direction), is_shooting=False)

        return None

    def _get_balanced_shooting_action(self, observation: GameObservation) -> Action | None:
        """Balanced shooting action with distance consideration."""
        if not observation.enemies or observation.player.ammunition == 0:
            return None

        player = observation.player

        if self.frame_count - self.last_shot_time < self.shot_cooldown_frames:
            return None

        best_target = None
        best_score = -1

        for enemy in observation.enemies:
            distance = self._calculate_distance((player.x, player.y), (enemy.x, enemy.y))

            # Only shoot if enemy is in proper engagement range
            if distance < self.min_attack_range or distance > self.max_engagement_range:
                continue

            direction_to_enemy = self._normalize_vector(
                self._get_vector((player.x, player.y), (enemy.x, enemy.y)),
            )

            has_los = self._has_clear_line_of_sight(
                observation,
                (player.x, player.y),
                (enemy.x, enemy.y),
            )

            # Only consider shooting if we have clear line of sight
            if not has_los:
                continue

            score = 0

            # Distance scoring - prefer optimal range
            if abs(distance - self.optimal_attack_range) < 30:
                score += 2.0  # Bonus for optimal range
            else:
                range_factor = (
                    1.0 - abs(distance - self.optimal_attack_range) / self.optimal_attack_range
                )
                score += range_factor

            # Line of sight bonus
            score += 1.5

            # Bonus for stationary or predictable enemies (simplified)
            score += 0.5

            score += random.uniform(-0.1, 0.1)

            if score > best_score:
                best_score = score
                best_target = (enemy, direction_to_enemy, has_los)

        if best_target:
            enemy, direction, has_los = best_target
            shoot_probability = self.shooting_probability

            distance = self._calculate_distance((player.x, player.y), (enemy.x, enemy.y))

            # Increase shooting probability for optimal range
            if abs(distance - self.optimal_attack_range) < 30:
                shoot_probability += 0.4

            if has_los:
                shoot_probability += 0.3

            if random.random() < shoot_probability:
                self.last_shot_time = self.frame_count
                return Action(direction=self._vector_to_direction(direction), is_shooting=True)

            # Don't move toward enemy when shooting - maintain position
            return Action(direction=None, is_shooting=False)

        return None

    def _get_exploration_action(self, observation: GameObservation) -> Action:
        """Exploration when no enemies are visible."""
        player = observation.player

        if self.stuck_counter > 3:
            directions = [
                (0, -1),
                (0, 1),
                (-1, 0),
                (1, 0),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),
            ]
            random.shuffle(directions)

            for direction in directions:
                score = self._evaluate_movement_direction(observation, direction)
                if score > 0.2:
                    return Action(direction=self._vector_to_direction(direction), is_shooting=False)

        # Move toward center with randomness
        center = (600, 450)
        to_center = self._normalize_vector(self._get_vector((player.x, player.y), center))
        random_direction = (random.uniform(-1, 1), random.uniform(-1, 1))
        random_direction = self._normalize_vector(random_direction)

        blend_factor = 0.3
        blended_direction = (
            to_center[0] * (1 - blend_factor) + random_direction[0] * blend_factor,
            to_center[1] * (1 - blend_factor) + random_direction[1] * blend_factor,
        )
        blended_direction = self._normalize_vector(blended_direction)

        return Action(direction=self._vector_to_direction(blended_direction), is_shooting=False)

    # Keep all existing utility methods
    def _evaluate_dodge_direction(
        self, observation: GameObservation, direction: tuple[float, float]
    ) -> float:
        """Evaluate how good a direction is for dodging."""
        player = observation.player
        move_distance = 60
        new_pos = (player.x + direction[0] * move_distance, player.y + direction[1] * move_distance)

        score = 1.0

        if new_pos[0] < 50 or new_pos[0] > 1150 or new_pos[1] < 50 or new_pos[1] > 850:
            score -= 1.0

        for wall in observation.walls:
            if (
                wall.x - 20 <= new_pos[0] <= wall.x + 84
                and wall.y - 20 <= new_pos[1] <= wall.y + 84
            ):
                score -= 0.8

        for bullet in observation.bullets:
            if bullet.owner == "player":
                continue
            old_dist = self._calculate_distance((player.x, player.y), (bullet.x, bullet.y))
            new_dist = self._calculate_distance(new_pos, (bullet.x, bullet.y))
            if new_dist > old_dist:
                score += 0.5

        return max(0.0, score)

    def _evaluate_movement_direction(
        self, observation: GameObservation, direction: tuple[float, float]
    ) -> float:
        """Evaluate how good a movement direction is."""
        player = observation.player
        move_distance = 50
        new_pos = (player.x + direction[0] * move_distance, player.y + direction[1] * move_distance)

        score = 1.0

        if new_pos[0] < 50 or new_pos[0] > 1150 or new_pos[1] < 50 or new_pos[1] > 850:
            score -= 0.8

        for wall in observation.walls:
            if (
                wall.x - 15 <= new_pos[0] <= wall.x + 79
                and wall.y - 15 <= new_pos[1] <= wall.y + 79
            ):
                score -= 0.7

        for bullet in observation.bullets:
            if bullet.owner == "player":
                continue
            bullet_distance = self._calculate_distance(new_pos, (bullet.x, bullet.y))
            if bullet_distance < self.danger_radius:
                score -= 0.4 * (1 - bullet_distance / self.danger_radius)

        return max(0.0, score)

    def _has_clear_line_of_sight(
        self, observation: GameObservation, start: tuple[float, float], end: tuple[float, float]
    ) -> bool:
        """Check line of sight between two points."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1:
            return True

        step_x = dx / distance
        step_y = dy / distance
        step_size = 16
        num_steps = int(distance / step_size)

        for i in range(1, num_steps + 1):
            check_x = start[0] + (step_x * step_size * i)
            check_y = start[1] + (step_y * step_size * i)

            for wall in observation.walls:
                if wall.x - 8 <= check_x <= wall.x + 72 and wall.y - 8 <= check_y <= wall.y + 72:
                    return False

        return True

    def _calculate_distance(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _get_vector(
        self, from_pos: tuple[float, float], to_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Get vector from one position to another."""
        return (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])

    def _normalize_vector(self, vector: tuple[float, float]) -> tuple[float, float]:
        """Normalize vector to unit length."""
        magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if magnitude == 0:
            return (0, 0)
        return (vector[0] / magnitude, vector[1] / magnitude)

    def _vector_to_direction(self, vector: tuple[float, float]) -> Direction:
        """Convert a normalized vector to the closest Direction enum."""
        x, y = vector

        if x == 0 and y == 0:
            return Direction.DOWN

        threshold = 0.5

        if abs(x) < threshold and y < -threshold:
            return Direction.UP
        if abs(x) < threshold and y > threshold:
            return Direction.DOWN
        if x < -threshold and abs(y) < threshold:
            return Direction.LEFT
        if x > threshold and abs(y) < threshold:
            return Direction.RIGHT
        if x < -threshold and y < -threshold:
            return Direction.TOP_LEFT
        if x > threshold and y < -threshold:
            return Direction.TOP_RIGHT
        if x < -threshold and y > threshold:
            return Direction.DOWN_LEFT
        if x > threshold and y > threshold:
            return Direction.DOWN_RIGHT

        return Direction.DOWN
