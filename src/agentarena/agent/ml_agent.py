"""
Machine learning agent for AgentArena.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from agentarena.agent.agent import Agent
from agentarena.game.action import Action, Direction
from agentarena.models.observations import GameObservation
from agentarena.models.training import Experience, MLAgentConfig

# Constants for state encoding
MAX_ENEMIES = 3  # Match with config.max_enemies
MAX_BULLETS = 5
ENEMY_FEATURES = 5  # Features per enemy: rel_x, rel_y, distance, angle, health
BULLET_FEATURES = 5  # Features per bullet: rel_x, rel_y, distance, direction_danger, is_enemy
PLAYER_FEATURES = 1  # Health

# Calculate expected state vector size
STATE_SIZE = PLAYER_FEATURES + (MAX_ENEMIES * ENEMY_FEATURES) + (MAX_BULLETS * BULLET_FEATURES)


class ReplayMemory:
    """Experience replay buffer to store and sample transitions"""

    def __init__(self, capacity: int = 10000) -> None:
        """
        Initialize replay memory buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memory: list[Experience] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Save a transition to the replay buffer.

        Args:
            state: Current state representation
            action: Action taken
            reward: Reward received
            next_state: Next state representation
            done: Whether the episode is done
        """
        # Create Experience model
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        # Add to memory with circular buffer logic
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample a batch of transitions randomly.

        Args:
            batch_size: Number of samples to return


        Returns:
            List of Experience objects
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Get the current size of the memory.

        Returns:
            int: Number of experiences in memory
        """
        return len(self.memory)


class DQNModel(nn.Module):
    """Deep Q-Network model architecture"""

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize the neural network.

        Args:
            input_size: Dimension of input state vector
            output_size: Number of possible actions
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor with Q-values
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLAgent(Agent):
    """Agent that uses machine learning to decide actions"""

    def __init__(
        self,
        name: str = "MLAgent",
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        is_training: bool = True,
        config: MLAgentConfig | None = None,
    ) -> None:
        """
        Initialize the ML agent.

        Args:
            name: Agent name for display and logging
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            is_training: Whether the agent is in training mode
            config: Optional MLAgentConfig with hyperparameters
        """
        super().__init__(name)
        self.is_training = is_training

        # Use config if provided, otherwise use default parameters
        if config:
            self.gamma = config.gamma
            self.epsilon = config.epsilon
            self.epsilon_min = config.epsilon_min
            self.epsilon_decay = config.epsilon_decay
            self.learning_rate = config.learning_rate
            self.batch_size = config.batch_size
            self.memory_capacity = config.memory_capacity
        else:
            # Hyperparameters
            self.gamma = gamma  # Discount factor
            self.epsilon = epsilon  # Exploration rate
            self.epsilon_min = epsilon_min  # Minimum exploration rate
            self.epsilon_decay = epsilon_decay  # Decay rate for exploration
            self.learning_rate = learning_rate  # Learning rate
            self.batch_size = 64
            self.memory_capacity = 10000

        # Setup the action space
        self.directions = [
            None,
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
            Direction.TOP_LEFT,
            Direction.TOP_RIGHT,
            Direction.DOWN_LEFT,
            Direction.DOWN_RIGHT,
        ]

        # Two actions: direction (9 options) and shooting (yes/no)
        self.n_actions = len(self.directions) * 2

        # State representation size (fixed size based on constants)
        self.state_size = STATE_SIZE
        self.model = self._initialize_model(self.state_size)

        # Experience replay
        self.memory = ReplayMemory(self.memory_capacity)

        # Track last state and action for learning
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.accumulated_reward = 0.0

    def _initialize_model(self, state_size: int) -> DQNModel:
        """
        Initialize the neural network model.

        Args:
            state_size: Input dimension size

        Returns:
            Initialized DQN model
        """
        model = DQNModel(state_size, self.n_actions)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model

    def reset(self) -> None:
        """Reset the agent state at the beginning of an episode."""
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0.0

        # Decay epsilon
        if self.is_training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def encode_observation(self, observation: GameObservation) -> np.ndarray:
        """
        Convert game observation to a state vector for the neural network.

        Args:
            observation: Current game state

        Returns:
            numpy array: Encoded state vector
        """
        # Extract player information
        player = observation.player
        player_x = player.x
        player_y = player.y
        player_health = player.health

        # Process enemies
        enemy_features = []
        for enemy in observation.enemies[:MAX_ENEMIES]:
            # Calculate relative position to player
            rel_x = enemy.x - player_x
            rel_y = enemy.y - player_y
            distance = np.sqrt(rel_x**2 + rel_y**2)

            # Calculate angle between player and enemy
            angle = np.arctan2(rel_y, rel_x)

            # Add enemy health
            enemy_health = enemy.health

            enemy_features.extend(
                [rel_x, rel_y, distance, angle, enemy_health],
            )

        # Pad if there are fewer enemies than expected
        expected_enemy_features = MAX_ENEMIES * ENEMY_FEATURES

        if len(enemy_features) < expected_enemy_features:
            # Pad with zeros for missing enemies
            padding = [0] * (expected_enemy_features - len(enemy_features))
            enemy_features.extend(padding)

        # Process bullets
        bullet_features = []
        for bullet in observation.bullets[:MAX_BULLETS]:
            # Calculate relative position to player
            rel_x = bullet.x - player_x
            rel_y = bullet.y - player_y
            distance = np.sqrt(rel_x**2 + rel_y**2)

            # Calculate danger level (is this bullet coming toward the player?)
            dx, dy = bullet.direction
            # Dot product between bullet direction and direction to player
            direction_danger = -(rel_x * dx + rel_y * dy) / (distance + 1e-6)

            # Is this an enemy bullet?
            is_enemy = 1 if bullet.owner != "player" else 0

            bullet_features.extend(
                [rel_x, rel_y, distance, direction_danger, is_enemy],
            )

        # Pad bullet features if needed
        expected_bullet_features = MAX_BULLETS * BULLET_FEATURES

        if len(bullet_features) < expected_bullet_features:
            padding = [0] * (expected_bullet_features - len(bullet_features))
            bullet_features.extend(padding)

        # Combine all features
        state = [player_health] + enemy_features + bullet_features

        # Ensure we always have the correct state size
        assert (
            len(state) == self.state_size
        ), f"State size mismatch: got {len(state)}, expected {self.state_size}"

        return np.array(state, dtype=np.float32)

    def _action_to_game_action(self, action_idx: int) -> Action:
        """
        Convert network output (action index) to game Action.

        Args:
            action_idx: Index of the selected action

        Returns:
            Action: Game action object
        """
        # Determine direction and shooting from action index
        direction_idx = action_idx % len(self.directions)
        is_shooting = action_idx >= len(self.directions)

        direction = self.directions[direction_idx]
        return Action(is_shooting=is_shooting, direction=direction)

    def _game_action_to_idx(self, action: Action) -> int:
        """
        Convert game Action to network action index.

        Args:
            action: Game action

        Returns:
            int: Action index for the network
        """
        direction_idx = (
            self.directions.index(action.direction) if action.direction in self.directions else 0
        )
        shooting_offset = len(self.directions) if action.is_shooting else 0
        return direction_idx + shooting_offset

    def get_action(self, observation: GameObservation) -> Action:
        """
        Choose an action based on the current observation.

        Args:
            observation: Current game state

        Returns:
            Action: Selected game action
        """
        # Convert observation to state vector
        state = self.encode_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Store last state for learning
        if self.is_training:
            self.last_state = state

        # Epsilon-greedy action selection
        if self.is_training and random.random() < self.epsilon:
            # Explore: select a random action
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # Exploit: select the action with highest predicted Q-value
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action_idx = q_values.max(1)[1].item()

        # Convert to game action
        action = self._action_to_game_action(action_idx)

        # Store the selected action for learning
        if self.is_training:
            self.last_action = action_idx

        return action

    def learn(self, next_observation: GameObservation, reward: float, done: bool) -> None:
        """
        Update the model based on received reward.

        Args:
            next_observation: New game state after taking action
            reward: Reward received for the action
            done: Whether the episode is done
        """
        if not self.is_training or self.last_state is None or self.last_action is None:
            return

        # Accumulate reward
        self.accumulated_reward += reward

        # Convert next observation to state
        next_state = self.encode_observation(next_observation)

        # Store transition in replay memory
        self.memory.push(self.last_state, self.last_action, reward, next_state, done)

        # Update last state
        self.last_state = next_state

        # Only train if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        experiences = self.memory.sample(self.batch_size)

        # Convert experiences to tensors
        states = torch.FloatTensor(np.array([exp.state for exp in experiences]))
        actions = torch.LongTensor([exp.action for exp in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in experiences]))
        dones = torch.FloatTensor([exp.done for exp in experiences])

        # Compute current Q values
        current_q_values = self.model(states).gather(1, actions)

        # Compute next Q values (for DQN)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]

        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model to
        """
        if self.model is not None:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "state_size": self.state_size,
                    "n_actions": self.n_actions,
                    "epsilon": self.epsilon,
                },
                path,
            )

    def load_model(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.state_size = checkpoint["state_size"]
        self.n_actions = checkpoint["n_actions"]
        self.epsilon = checkpoint["epsilon"]

        # Initialize model with correct dimensions
        self.model = self._initialize_model(self.state_size)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # Set to evaluation mode
