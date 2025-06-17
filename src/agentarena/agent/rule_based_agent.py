"""
Balanced rule-based agent for demonstration collection and pretraining.
This agent provides more balanced action distribution and improved pathfinding.
"""

import math
import random
from typing import Optional, Tuple, List

from agentarena.agent.agent import Agent
from agentarena.models.action import Action, Direction
from agentarena.models.observations import GameObservation, BulletObservation, EnemyObservation


class RuleBasedAgent(Agent):
    """
    Balanced rule-based agent that provides good demonstrations for ML training.

    Features:
    - Balanced shooting vs movement actions
    - Improved pathfinding around walls
    - More diverse action patterns
    - Strategic positioning and engagement
    """

    def __init__(self, name: str = "RuleBasedAgent") -> None:
        super().__init__(name)

        # Behavioral parameters
        self.danger_radius = 150.0
        self.attack_range = 250.0
        self.min_attack_range = 60.0
        self.wall_avoidance_distance = 80.0

        # Shooting behavior parameters
        self.shooting_probability = 0.4  # Higher base shooting probability
        self.aggressive_shooting_distance = 200.0
        self.last_shot_time = 0
        self.shot_cooldown_frames = 15

        # Pathfinding parameters
        self.waypoint_grid_size = 64  # Smaller grid for better pathfinding
        self.max_pathfinding_attempts = 8
        self.exploration_radius = 120

        # State tracking
        self.last_position: Optional[Tuple[float, float]] = None
        self.stuck_counter = 0
        self.current_waypoint: Optional[Tuple[float, float]] = None
        self.frames_since_last_enemy_sight = 0
        self.exploration_targets = []
        self.frame_count = 0

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.last_position = None
        self.stuck_counter = 0
        self.current_waypoint = None
        self.frames_since_last_enemy_sight = 0
        self.exploration_targets = []
        self.frame_count = 0
        self.last_shot_time = 0

    def get_action(self, observation: GameObservation) -> Action:
        """
        Determine the best action with balanced shooting and movement.
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

        # Priority 1: Emergency dodge (immediate danger)
        emergency_action = self._get_emergency_dodge_action(observation)
        if emergency_action:
            return emergency_action

        # Priority 2: Aggressive shooting when enemies are visible and in range
        if observation.enemies:
            shooting_action = self._get_balanced_shooting_action(observation)
            if shooting_action:
                return shooting_action

        # Priority 3: Strategic movement (positioning, pathfinding)
        movement_action = self._get_strategic_movement_action(observation)
        if movement_action:
            return movement_action

        # Priority 4: Exploration when no enemies visible
        return self._get_exploration_action(observation)

    def _get_emergency_dodge_action(self, observation: GameObservation) -> Optional[Action]:
        """
        Emergency dodging for immediate bullet threats.
        """
        player = observation.player
        immediate_threats = []

        for bullet in observation.bullets:
            if bullet.owner == "player":
                continue

            # Calculate bullet trajectory
            bullet_pos = (bullet.x, bullet.y)
            distance_to_bullet = self._calculate_distance((player.x, player.y), bullet_pos)

            # Only consider very close bullets for emergency dodge
            if distance_to_bullet < 80:
                # Check if bullet is heading toward player
                bullet_dir = bullet.direction
                to_player = self._normalize_vector((player.x - bullet.x, player.y - bullet.y))

                # Dot product to check alignment
                threat_level = bullet_dir[0] * to_player[0] + bullet_dir[1] * to_player[1]

                if threat_level > 0.5:  # Bullet is coming toward us
                    immediate_threats.append((bullet, threat_level, distance_to_bullet))

        if not immediate_threats:
            return None

        # Find best dodge direction
        dodge_directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),  # Cardinal directions first
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),  # Then diagonals
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

    def _get_balanced_shooting_action(self, observation: GameObservation) -> Optional[Action]:
        """
        More aggressive and balanced shooting behavior.
        """
        if not observation.enemies or observation.player.ammunition == 0:
            return None

        player = observation.player

        # Shooting cooldown to prevent spam
        if self.frame_count - self.last_shot_time < self.shot_cooldown_frames:
            return None

        best_target = None
        best_score = -1

        for enemy in observation.enemies:
            distance = self._calculate_distance((player.x, player.y), (enemy.x, enemy.y))

            # More lenient distance requirements
            if distance > self.attack_range * 1.8:
                continue

            # Calculate direction to enemy
            direction_to_enemy = self._normalize_vector(
                self._get_vector((player.x, player.y), (enemy.x, enemy.y))
            )

            # Check line of sight
            has_los = self._has_clear_line_of_sight(
                observation, (player.x, player.y), (enemy.x, enemy.y)
            )

            # Calculate shooting score
            score = 0

            # Distance factor (prefer medium range)
            if distance < self.aggressive_shooting_distance:
                score += 1.0 - (distance / self.aggressive_shooting_distance)
            else:
                score += 0.3  # Still shoot at distant enemies but with lower priority

            # Line of sight bonus
            if has_los:
                score += 1.5
            else:
                score += 0.2  # Still might shoot for suppression

            # Close range urgency
            if distance < self.min_attack_range * 1.5:
                score += 2.0

            # Random factor for variety
            score += random.uniform(-0.2, 0.2)

            if score > best_score:
                best_score = score
                best_target = (enemy, direction_to_enemy, has_los)

        if best_target:
            enemy, direction, has_los = best_target

            # Higher shooting probability
            shoot_probability = self.shooting_probability

            # Adjust shooting probability based on situation
            distance = self._calculate_distance((player.x, player.y), (enemy.x, enemy.y))
            if distance < self.aggressive_shooting_distance:
                shoot_probability += 0.3
            if has_los:
                shoot_probability += 0.4
            if distance < self.min_attack_range * 2:
                shoot_probability += 0.5  # Very likely to shoot when close

            # Random shooting for demonstration variety
            if random.random() < shoot_probability:
                self.last_shot_time = self.frame_count
                return Action(direction=self._vector_to_direction(direction), is_shooting=True)
            else:
                # Move toward enemy for positioning
                return Action(direction=self._vector_to_direction(direction), is_shooting=False)

        return None

    def _get_strategic_movement_action(self, observation: GameObservation) -> Optional[Action]:
        """
        Strategic movement including improved pathfinding.
        """
        player = observation.player

        if not observation.enemies:
            return None

        # Find closest enemy
        closest_enemy = min(
            observation.enemies,
            key=lambda e: self._calculate_distance((player.x, player.y), (e.x, e.y)),
        )

        distance_to_enemy = self._calculate_distance(
            (player.x, player.y), (closest_enemy.x, closest_enemy.y)
        )

        # If too close, try to back away while maintaining sight
        if distance_to_enemy < self.min_attack_range:
            retreat_directions = self._get_retreat_directions(observation, closest_enemy)
            for retreat_dir in retreat_directions:
                if self._evaluate_movement_direction(observation, retreat_dir) > 0.3:
                    return Action(
                        direction=self._vector_to_direction(retreat_dir), is_shooting=False
                    )

        # If enemy is far or behind wall, use pathfinding
        elif distance_to_enemy > self.attack_range or not self._has_clear_line_of_sight(
            observation, (player.x, player.y), (closest_enemy.x, closest_enemy.y)
        ):
            pathfinding_action = self._get_pathfinding_action(observation, closest_enemy)
            if pathfinding_action:
                return pathfinding_action

        # Try flanking maneuvers
        flank_action = self._get_flanking_action(observation, closest_enemy)
        if flank_action:
            return flank_action

        return None

    def _get_pathfinding_action(
        self, observation: GameObservation, target_enemy: EnemyObservation
    ) -> Optional[Action]:
        """
        Improved pathfinding that can navigate around walls effectively.
        """
        player = observation.player
        target_pos = (target_enemy.x, target_enemy.y)

        # If we're stuck, clear current waypoint
        if self.stuck_counter > 5:
            self.current_waypoint = None
            self.stuck_counter = 0

        # If we don't have a waypoint or reached it, find a new one
        if (
            self.current_waypoint is None
            or self._calculate_distance((player.x, player.y), self.current_waypoint) < 40
        ):
            self.current_waypoint = self._find_best_waypoint(observation, target_pos)

        if self.current_waypoint:
            # Move toward current waypoint
            direction_to_waypoint = self._normalize_vector(
                self._get_vector((player.x, player.y), self.current_waypoint)
            )

            if self._evaluate_movement_direction(observation, direction_to_waypoint) > 0.1:
                return Action(
                    direction=self._vector_to_direction(direction_to_waypoint), is_shooting=False
                )
            else:
                # Waypoint is blocked, find a new one
                self.current_waypoint = None

        # Fallback: wall-following behavior
        return self._get_wall_following_action(observation, target_pos)

    def _find_best_waypoint(
        self, observation: GameObservation, target_pos: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Find the best waypoint for pathfinding using A* inspired approach.
        """
        player = observation.player
        player_pos = (player.x, player.y)

        # Generate waypoint candidates
        candidates = []

        # Grid-based waypoints around the area
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue

                waypoint_x = player.x + dx * self.waypoint_grid_size
                waypoint_y = player.y + dy * self.waypoint_grid_size

                # Keep within bounds
                if 50 <= waypoint_x <= 1150 and 50 <= waypoint_y <= 800:
                    candidates.append((waypoint_x, waypoint_y))

        # Add waypoints around walls (corners are good navigation points)
        for wall in observation.walls:
            wall_waypoints = [
                (wall.x - 80, wall.y - 80),  # Top-left corner
                (wall.x + 144, wall.y - 80),  # Top-right corner
                (wall.x - 80, wall.y + 144),  # Bottom-left corner
                (wall.x + 144, wall.y + 144),  # Bottom-right corner
            ]

            for wx, wy in wall_waypoints:
                if 50 <= wx <= 1150 and 50 <= wy <= 800:
                    candidates.append((wx, wy))

        # Score each candidate
        best_waypoint = None
        best_score = -1

        for candidate in candidates:
            score = self._score_waypoint(observation, player_pos, candidate, target_pos)
            if score > best_score:
                best_score = score
                best_waypoint = candidate

        return best_waypoint if best_score > 0.1 else None

    def _score_waypoint(
        self,
        observation: GameObservation,
        player_pos: Tuple[float, float],
        waypoint: Tuple[float, float],
        target_pos: Tuple[float, float],
    ) -> float:
        """
        Score a waypoint for pathfinding quality.
        """
        score = 0.0

        # Distance to target (prefer waypoints that get us closer)
        dist_player_to_target = self._calculate_distance(player_pos, target_pos)
        dist_waypoint_to_target = self._calculate_distance(waypoint, target_pos)

        if dist_waypoint_to_target < dist_player_to_target:
            score += 1.0 - (dist_waypoint_to_target / max(dist_player_to_target, 1))

        # Reachability from player
        if self._is_path_clear(observation, player_pos, waypoint):
            score += 0.8
        else:
            score -= 0.5

        # Line of sight to target from waypoint
        if self._has_clear_line_of_sight(observation, waypoint, target_pos):
            score += 1.2

        # Distance from player (prefer not too close, not too far)
        dist_to_player = self._calculate_distance(player_pos, waypoint)
        if 60 <= dist_to_player <= 200:
            score += 0.5
        elif dist_to_player < 30:
            score -= 0.8  # Too close

        # Avoid being too close to walls
        min_wall_distance = float("inf")
        for wall in observation.walls:
            wall_center = (wall.x + 32, wall.y + 32)
            wall_distance = self._calculate_distance(waypoint, wall_center)
            min_wall_distance = min(min_wall_distance, wall_distance)

        if min_wall_distance > 60:
            score += 0.3
        elif min_wall_distance < 30:
            score -= 0.7

        return score

    def _get_wall_following_action(
        self, observation: GameObservation, target_pos: Tuple[float, float]
    ) -> Optional[Action]:
        """
        Wall-following behavior for when pathfinding fails.
        """
        player = observation.player

        # Try moving perpendicular to walls we're near
        nearby_walls = []
        for wall in observation.walls:
            wall_center = (wall.x + 32, wall.y + 32)
            distance = self._calculate_distance((player.x, player.y), wall_center)
            if distance < 150:
                nearby_walls.append((wall, distance))

        if nearby_walls:
            # Sort by distance
            nearby_walls.sort(key=lambda x: x[1])
            closest_wall = nearby_walls[0][0]

            # Calculate perpendicular directions to the wall
            wall_center = (closest_wall.x + 32, closest_wall.y + 32)
            to_wall = self._normalize_vector(self._get_vector((player.x, player.y), wall_center))

            # Perpendicular directions
            perp_directions = [
                (-to_wall[1], to_wall[0]),  # Left perpendicular
                (to_wall[1], -to_wall[0]),  # Right perpendicular
            ]

            # Choose the perpendicular direction that gets us closer to target
            target_direction = self._normalize_vector(
                self._get_vector((player.x, player.y), target_pos)
            )

            best_direction = None
            best_alignment = -1

            for perp_dir in perp_directions:
                alignment = perp_dir[0] * target_direction[0] + perp_dir[1] * target_direction[1]
                movement_score = self._evaluate_movement_direction(observation, perp_dir)

                combined_score = alignment + movement_score
                if combined_score > best_alignment:
                    best_alignment = combined_score
                    best_direction = perp_dir

            if best_direction and best_alignment > 0:
                return Action(
                    direction=self._vector_to_direction(best_direction), is_shooting=False
                )

        # Fallback: try moving toward target with some randomness
        target_direction = self._normalize_vector(
            self._get_vector((player.x, player.y), target_pos)
        )

        # Add some randomness for variety
        random_offset = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        modified_direction = self._normalize_vector(
            (target_direction[0] + random_offset[0], target_direction[1] + random_offset[1])
        )

        if self._evaluate_movement_direction(observation, modified_direction) > 0.1:
            return Action(
                direction=self._vector_to_direction(modified_direction), is_shooting=False
            )

        return None

    def _get_exploration_action(self, observation: GameObservation) -> Action:
        """
        Exploration when no enemies are visible with more diverse movement.
        """
        player = observation.player

        # If stuck, try all directions
        if self.stuck_counter > 3:
            directions = [
                (0, -1),
                (0, 1),
                (-1, 0),
                (1, 0),  # Cardinal
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),  # Diagonal
            ]

            # Shuffle for randomness
            random.shuffle(directions)

            for direction in directions:
                score = self._evaluate_movement_direction(observation, direction)
                if score > 0.2:
                    return Action(direction=self._vector_to_direction(direction), is_shooting=False)

        # Generate exploration targets if we don't have any
        if not self.exploration_targets:
            self.exploration_targets = self._generate_exploration_targets(observation)

        # Move toward nearest exploration target
        if self.exploration_targets:
            target = min(
                self.exploration_targets,
                key=lambda t: self._calculate_distance((player.x, player.y), t),
            )

            # Remove target if we're close
            if self._calculate_distance((player.x, player.y), target) < 50:
                self.exploration_targets.remove(target)
            else:
                direction = self._normalize_vector(self._get_vector((player.x, player.y), target))

                if self._evaluate_movement_direction(observation, direction) > 0.1:
                    return Action(direction=self._vector_to_direction(direction), is_shooting=False)

        # Random movement with bias toward center
        center = (600, 450)
        to_center = self._normalize_vector(self._get_vector((player.x, player.y), center))

        # Add randomness
        random_direction = (random.uniform(-1, 1), random.uniform(-1, 1))
        random_direction = self._normalize_vector(random_direction)

        # Blend center and random
        blend_factor = 0.3
        blended_direction = (
            to_center[0] * (1 - blend_factor) + random_direction[0] * blend_factor,
            to_center[1] * (1 - blend_factor) + random_direction[1] * blend_factor,
        )
        blended_direction = self._normalize_vector(blended_direction)

        return Action(direction=self._vector_to_direction(blended_direction), is_shooting=False)

    def _generate_exploration_targets(
        self, observation: GameObservation
    ) -> List[Tuple[float, float]]:
        """
        Generate points of interest for exploration.
        """
        targets = []

        # Add corners
        targets.extend([(150, 150), (1050, 150), (150, 750), (1050, 750)])

        # Add points around walls (good vantage points)
        for wall in observation.walls:
            wall_targets = [
                (wall.x - 100, wall.y + 32),
                (wall.x + 164, wall.y + 32),
                (wall.x + 32, wall.y - 100),
                (wall.x + 32, wall.y + 164),
            ]

            for target in wall_targets:
                if 100 <= target[0] <= 1100 and 100 <= target[1] <= 800:
                    targets.append(target)

        # Add some random points
        for _ in range(5):
            random_target = (random.randint(150, 1050), random.randint(150, 750))
            targets.append(random_target)

        return targets

    def _get_retreat_directions(
        self, observation: GameObservation, enemy: EnemyObservation
    ) -> List[Tuple[float, float]]:
        """
        Get good retreat directions when too close to enemy.
        """
        player = observation.player

        # Primary retreat: directly away from enemy
        away_from_enemy = self._normalize_vector(
            self._get_vector((enemy.x, enemy.y), (player.x, player.y))
        )

        # Secondary retreats: perpendicular to enemy
        perpendicular_left = (-away_from_enemy[1], away_from_enemy[0])
        perpendicular_right = (away_from_enemy[1], -away_from_enemy[0])

        # Diagonal retreats
        diagonal_left = self._normalize_vector(
            (away_from_enemy[0] + perpendicular_left[0], away_from_enemy[1] + perpendicular_left[1])
        )
        diagonal_right = self._normalize_vector(
            (
                away_from_enemy[0] + perpendicular_right[0],
                away_from_enemy[1] + perpendicular_right[1],
            )
        )

        return [
            away_from_enemy,
            diagonal_left,
            diagonal_right,
            perpendicular_left,
            perpendicular_right,
        ]

    def _get_flanking_action(
        self, observation: GameObservation, enemy: EnemyObservation
    ) -> Optional[Action]:
        """
        Try to flank the enemy for better positioning.
        """
        player = observation.player

        # Calculate flanking directions
        to_enemy = self._normalize_vector(
            self._get_vector((player.x, player.y), (enemy.x, enemy.y))
        )

        flank_left = (-to_enemy[1], to_enemy[0])
        flank_right = (to_enemy[1], -to_enemy[0])

        # Score both flanking directions
        left_score = self._evaluate_movement_direction(observation, flank_left)
        right_score = self._evaluate_movement_direction(observation, flank_right)

        # Add bonus for flanking directions that maintain good distance
        distance_to_enemy = self._calculate_distance((player.x, player.y), (enemy.x, enemy.y))

        if self.min_attack_range <= distance_to_enemy <= self.attack_range:
            left_score += 0.5
            right_score += 0.5

        # Choose best flanking direction
        if left_score > 0.4 and left_score >= right_score:
            return Action(direction=self._vector_to_direction(flank_left), is_shooting=False)
        elif right_score > 0.4:
            return Action(direction=self._vector_to_direction(flank_right), is_shooting=False)

        return None

    # Utility methods
    def _evaluate_dodge_direction(
        self, observation: GameObservation, direction: Tuple[float, float]
    ) -> float:
        """
        Evaluate how good a direction is for dodging.
        """
        player = observation.player
        move_distance = 60
        new_pos = (player.x + direction[0] * move_distance, player.y + direction[1] * move_distance)

        score = 1.0

        # Penalty for going out of bounds
        if new_pos[0] < 50 or new_pos[0] > 1150 or new_pos[1] < 50 or new_pos[1] > 850:
            score -= 1.0

        # Penalty for hitting walls
        for wall in observation.walls:
            if (
                wall.x - 20 <= new_pos[0] <= wall.x + 84
                and wall.y - 20 <= new_pos[1] <= wall.y + 84
            ):
                score -= 0.8

        # Bonus for moving away from bullets
        for bullet in observation.bullets:
            if bullet.owner == "player":
                continue

            old_dist = self._calculate_distance((player.x, player.y), (bullet.x, bullet.y))
            new_dist = self._calculate_distance(new_pos, (bullet.x, bullet.y))

            if new_dist > old_dist:
                score += 0.5

        return max(0.0, score)

    def _evaluate_movement_direction(
        self, observation: GameObservation, direction: Tuple[float, float]
    ) -> float:
        """
        Evaluate how good a movement direction is.
        """
        player = observation.player
        move_distance = 50
        new_pos = (player.x + direction[0] * move_distance, player.y + direction[1] * move_distance)

        score = 1.0

        # Boundary check
        if new_pos[0] < 50 or new_pos[0] > 1150 or new_pos[1] < 50 or new_pos[1] > 850:
            score -= 0.8

        # Wall collision check
        for wall in observation.walls:
            if (
                wall.x - 15 <= new_pos[0] <= wall.x + 79
                and wall.y - 15 <= new_pos[1] <= wall.y + 79
            ):
                score -= 0.7

        # Bullet avoidance
        for bullet in observation.bullets:
            if bullet.owner == "player":
                continue

            bullet_distance = self._calculate_distance(new_pos, (bullet.x, bullet.y))
            if bullet_distance < self.danger_radius:
                score -= 0.4 * (1 - bullet_distance / self.danger_radius)

        return max(0.0, score)

    def _has_clear_line_of_sight(
        self, observation: GameObservation, start: Tuple[float, float], end: Tuple[float, float]
    ) -> bool:
        """
        Check line of sight between two points.
        """
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

    def _is_path_clear(
        self, observation: GameObservation, start: Tuple[float, float], end: Tuple[float, float]
    ) -> bool:
        """
        Check if path is clear for movement.
        """
        return self._has_clear_line_of_sight(observation, start, end)

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _get_vector(
        self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Get vector from one position to another."""
        return (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])

    def _normalize_vector(self, vector: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize vector to unit length."""
        magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if magnitude == 0:
            return (0, 0)
        return (vector[0] / magnitude, vector[1] / magnitude)

    def _vector_to_direction(self, vector: Tuple[float, float]) -> Direction:
        """Convert a normalized vector to the closest Direction enum."""
        x, y = vector

        # Handle zero vector
        if x == 0 and y == 0:
            return Direction.DOWN  # Default fallback

        # Define thresholds for diagonal vs straight movement
        threshold = 0.5

        # Pure vertical movement
        if abs(x) < threshold and y < -threshold:
            return Direction.UP
        elif abs(x) < threshold and y > threshold:
            return Direction.DOWN
        # Pure horizontal movement
        elif x < -threshold and abs(y) < threshold:
            return Direction.LEFT
        elif x > threshold and abs(y) < threshold:
            return Direction.RIGHT
        # Diagonal movement
        elif x < -threshold and y < -threshold:
            return Direction.TOP_LEFT
        elif x > threshold and y < -threshold:
            return Direction.TOP_RIGHT
        elif x < -threshold and y > threshold:
            return Direction.DOWN_LEFT
        elif x > threshold and y > threshold:
            return Direction.DOWN_RIGHT
        else:
            # Default fallback for edge cases
            return Direction.DOWN

    def _direction_to_vector(self, direction: Direction) -> Tuple[float, float]:
        """Convert Direction enum to normalized vector."""
        direction_vectors = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
            Direction.TOP_LEFT: (-1, -1),
            Direction.TOP_RIGHT: (1, -1),
            Direction.DOWN_LEFT: (-1, 1),
            Direction.DOWN_RIGHT: (1, 1),
        }
        vector = direction_vectors.get(direction, (0, 0))
        # Normalize diagonal vectors
        if vector[0] != 0 and vector[1] != 0:
            magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
            return (vector[0] / magnitude, vector[1] / magnitude)
        return vector
