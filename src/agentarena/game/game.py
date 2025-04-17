"""
Main game class and core game loop for AgentArena.
"""

import uuid
from datetime import datetime
from pathlib import Path

import pygame

from agentarena.agent.agent import Agent
from agentarena.game.asset_manager import AssetManager
from agentarena.game.event_manager import EventManager
from agentarena.game.object_factory import ObjectFactory
from agentarena.game.physics import PhysicsSystem
from agentarena.game.rendering import RenderingSystem
from agentarena.models.config import GameConfig
from agentarena.models.observations import GameObservation

LOG_PATH: str = "src/agentarena/data"


class Game:
    """Main game class that manages the game state and coordinates all systems."""

    def __init__(
        self,
        screen: pygame.Surface,
        player_agent: Agent,
        enemy_agent: Agent,
        clock: pygame.time.Clock,
        config: GameConfig,
    ) -> None:
        """
        Initialize the game.

        Args:
            screen: Pygame surface for rendering
            player_agent: Agent controlling the player
            enemy_agent: Agent controlling the enemies
            clock: Pygame clock for timing
            config: Game configuration
        """
        self.screen: pygame.Surface = screen
        self.player_agent: Agent = player_agent
        self.enemy_agent: Agent = enemy_agent
        self.clock = clock
        self.config = config

        # Game state
        self.running = True
        self.game_time: float = 0.0
        self.score: int = 0
        self.dt: float = 1 / 60  # Fixed time step for predictable physics
        self.episode_log: list[dict] = []

        # Game objects
        self.player = None
        self.enemies = []
        self.bullets = []
        self.explosions = []

        # Unique game ID for this session
        self.game_id = str(uuid.uuid4())

        # Initialize systems
        self.asset_manager = AssetManager(self.config)
        self.object_factory = ObjectFactory(self.config, self.player_agent, self.enemy_agent)
        self.physics_system = PhysicsSystem(self.config)
        self.event_manager = EventManager()

        # The rendering system needs to be initialized last since it depends on assets
        if self.screen is not None:
            self.rendering_system = RenderingSystem(self.screen, self.asset_manager, self.config)
        else:
            self.rendering_system = None

        # Initialize the game
        self.reset()

    def reset(self) -> None:
        """Reset the game state for a new episode."""
        # Reset game state
        self.game_time = 0.0
        self.score = 0
        self.running = True
        self.events = []
        self.bullets = []
        self.explosions = []
        self.episode_log = []

        # Reset the agents
        self.player_agent.reset()
        self.enemy_agent.reset()

        # Load assets
        self.asset_manager.load_textures()

        # Create game objects
        self.player = self.object_factory.create_player()
        self.enemies = self.object_factory.create_enemies(
            self.config.max_enemies,
        )

        # Initialize level
        self.level = self.object_factory.create_level(
            self.player,
            self.enemies,
        )

        # Initialize physics system with new level data
        self.physics_system.setup_collision_grid(self.level.walls)

        # Store static map data for logging
        walls_data = self.object_factory.get_walls_data(self.level.walls)
        self.static_map_data = {"walls": walls_data}

    def get_observation(self, agent_id: str = "player") -> GameObservation:
        """
        Get the current game state observation for an agent.

        Args:
            agent_id: ID of the agent requesting the observation

        Returns:
            GameObservation: Structured game state observation
        """
        if agent_id == "player" and self.player is not None:
            # Player observation
            return self.object_factory.create_player_observation(
                self.player,
                self.enemies,
                self.bullets,
                self.level.walls,
                self.game_time,
                self.score,
            )
        else:
            # Enemy observation (reverse perspective)
            return self.object_factory.create_enemy_observation(
                agent_id,
                self.player,
                self.enemies,
                self.bullets,
                self.level.walls,
                self.game_time,
                self.score,
            )

    def update(self) -> None:
        """Update game state for the current frame."""
        # Update game time using delta time
        self.dt = self.clock.tick(self.config.fps) / 1000.0 if self.config.fps > 0 else 1.0 / 60.0
        self.game_time += self.dt

        # Clear events for this frame
        self.events = []

        # At the start of episode, store walls once
        if len(self.episode_log) == 0:
            self.episode_log.append({"static": self.static_map_data})

        # Check game over condition
        if self.player is not None and self.player.health <= 0:
            self.event_manager.create_player_destroyed_event(
                self.events,
                self.game_time,
                self.player,
            )
            self.save_episode_log()
            self.running = False
            return

        if len(self.enemies) == 0:
            # All enemies defeated - victory!
            self.score += 100  # Big score bonus for winning
            self.save_episode_log()
            self.running = False
            return

        # Process player actions if player exists
        if self.player is not None:
            # Get player observation and action
            player_observation = self.get_observation("player")
            player_action = self.player.agent.get_action(player_observation)
            self.physics_system.apply_action(
                "player",
                self.player,
                player_action,
                self.bullets,
                self.events,
                self.game_time,
                self.object_factory,
                self.dt,
            )

        # Process enemy actions
        for i, enemy in enumerate(self.enemies):
            enemy_observation = self.get_observation(f"enemy_{i}")
            enemy_action = enemy.agent.get_action(enemy_observation)
            self.physics_system.apply_action(
                f"enemy_{i}",
                enemy,
                enemy_action,
                self.bullets,
                self.events,
                self.game_time,
                self.object_factory,
                self.dt,
            )

        # Update positions of all bullets
        self.physics_system.move_bullets(
            self.bullets,
            self.level.walls,
            self.config,
            self.events,
            self.game_time,
            self.dt,
        )

        # Update collision grid with current positions
        self.physics_system.update_entity_positions(self.player, self.enemies, self.bullets)

        # Check for collisions efficiently
        self.physics_system.check_collisions(
            self.player,
            self.enemies,
            self.bullets,
            self.events,
            self.game_time,
            self.explosions,
            self.object_factory,
            self.score_callback,
        )

        # Update explosions
        self._update_explosions()

        # Log game state - convert events to dictionaries for logging
        event_dicts = [event.model_dump() for event in self.events]

        # Track player action for the log
        player_action_dict = (
            player_action.model_dump()
            if self.player is not None
            else {"is_shooting": False, "direction": None}
        )

        self.episode_log.append(
            {
                "observation": self.get_observation("player").model_dump(),
                "action": player_action_dict,
                "events": event_dicts,
                "done": not self.running,
                "game_time": self.game_time,
            },
        )

        # Render the frame if we have a screen
        if self.rendering_system is not None:
            self.rendering_system.render(
                self.player,
                self.enemies,
                self.bullets,
                self.explosions,
                self.level.walls,
                self.score,
                self.game_time,
            )

    def score_callback(self, points: int) -> None:
        """Callback to update the score."""
        self.score += points

    def _update_explosions(self) -> None:
        """Update all active explosions and remove finished ones."""
        i = 0
        while i < len(self.explosions):
            explosion = self.explosions[i]
            explosion.update()

            if explosion.finished:
                self.explosions.pop(i)
            else:
                i += 1

    def save_episode_log(self) -> None:
        """Save the episode log to a JSON file, filtering out non-serializable objects."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(LOG_PATH)
        log_path.mkdir(exist_ok=True, parents=True)

        filename = log_path / f"episode_{timestamp}.json"

        # Create a sanitized copy of the episode log
        sanitized_log = self._sanitize_for_json(self.episode_log)

        with filename.open("w") as f:
            import json

            json.dump(sanitized_log, f)

    def _sanitize_for_json(self, data):
        """Recursively remove pygame.Rect and other non-serializable objects from data structure."""
        if isinstance(data, dict):
            return {
                k: self._sanitize_for_json(v)
                for k, v in data.items()
                if not isinstance(v, pygame.Rect)
            }
        elif isinstance(data, list):
            return [
                self._sanitize_for_json(item) for item in data if not isinstance(item, pygame.Rect)
            ]
        elif isinstance(data, pygame.Rect):
            # Skip Rect objects entirely
            return None
        else:
            return data
