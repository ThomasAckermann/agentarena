"""
Machine learning agent for AgentArena.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from agentarena.agent.agent import Agent
from agentarena.models.action import Action, Direction
from agentarena.models.observations import GameObservation
from agentarena.models.training import Experience, MLAgentConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Updated constants for state encoding
MAX_ENEMIES = 3
MAX_BULLETS = 5
ENEMY_FEATURES = 7
BULLET_FEATURES = 5
PLAYER_FEATURES = 3
MAX_WALLS = 120
WALL_FEATURES_PER_WALL = 2  # x, y
TOTAL_WALL_FEATURES = MAX_WALLS * WALL_FEATURES_PER_WALL

# Calculate expected state vector size
STATE_SIZE = (
    PLAYER_FEATURES
    + (MAX_ENEMIES * ENEMY_FEATURES)
    + (MAX_BULLETS * BULLET_FEATURES)
    + (MAX_WALLS * WALL_FEATURES_PER_WALL)
)


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


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with BatchNorm for improved learning stability.
    Separates state value and action advantage calculations for better learning.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Initialize the Dueling DQN network.

        Args:
            input_size: Size of the input state vector
            output_size: Number of possible actions
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.feature_network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, output_size),
        )

        self.wall_encoder_stream = nn.Sequential(
            nn.Linear(TOTAL_WALL_FEATURES, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
        )

        # Initialize weights using xavier_uniform
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that combines value and advantage streams.

        Args:
            x: Input state tensor

        Returns:
            Q-values for each action
        """
        # Extract features
        features = self.feature_network(x)

        # Calculate state value
        value = self.value_stream(features)

        # Calculate action advantages
        advantages = self.advantage_stream(features)

        # Combine value and advantages using the dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


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
        self.recent_actions = []
        self.shots_fired = 0
        self.enemy_hits = 0

        # Use config if provided, otherwise use default parameters
        if config:
            self.gamma = config.gamma
            self.epsilon = config.epsilon
            self.epsilon_min = config.epsilon_min
            self.epsilon_decay = config.epsilon_decay
            self.learning_rate = config.learning_rate
            self.batch_size = config.batch_size
            self.memory_capacity = config.memory_capacity
            # Add target network update frequency (Default to 10 if not specified in config)
            # Instead of .get(), use getattr with a default value
            self.target_update_frequency = getattr(config, "target_update_frequency", 10)
        else:
            # Hyperparameters
            self.gamma = gamma  # Discount factor
            self.epsilon = epsilon  # Exploration rate
            self.epsilon_min = epsilon_min  # Minimum exploration rate
            self.epsilon_decay = epsilon_decay  # Decay rate for exploration
            self.learning_rate = learning_rate  # Learning rate
            self.batch_size = 64
            self.memory_capacity = 10000
            self.target_update_frequency = 10  # Update target network every 10 episodes

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

        # Initialize both policy and target networks
        self.policy_net, self.target_net = self._initialize_model(self.state_size)

        # Experience replay
        self.memory = ReplayMemory(self.memory_capacity)

        # Track last state and action for learning
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.accumulated_reward = 0.0

        # Track steps and episodes for target network updates
        self.steps_done = 0
        self.episodes_done = 0

    def _initialize_model(self, state_size: int) -> tuple[DuelingDQN, DuelingDQN]:
        """
        Initialize both policy and target networks.

        Args:
            state_size: Input dimension size

        Returns:
            Tuple of (policy_net, target_net)
        """
        # Create policy network
        policy_net = DuelingDQN(state_size, self.n_actions).to(device)

        # Create target network with identical structure
        target_net = DuelingDQN(state_size, self.n_actions).to(device)

        # Initialize target net with same weights
        target_net.load_state_dict(policy_net.state_dict())

        # Set target network to evaluation mode (affects batchnorm/dropout)
        target_net.eval()

        # Setup optimizer only for policy network
        self.optimizer = optim.Adam(policy_net.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.9995,
        )

        return policy_net, target_net

    def reset(self) -> None:
        """Reset the agent state at the beginning of an episode."""
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0.0
        self.shots_fired = 0
        self.enemy_hits = 0

    def encode_observation(self, observation: GameObservation) -> np.ndarray:
        """Enhanced state encoding with walls and absolute position."""
        # Extract player information
        player = observation.player
        player_x = player.x
        player_y = player.y
        player_health = player.health

        # Normalize absolute position to [0,1] range
        norm_x = player_x / 1200.0  # Assuming 1200 is display width
        norm_y = player_y / 900.0  # Assuming 900 is display height

        # Process enemies (existing code)
        enemy_features = []
        for enemy in observation.enemies[:MAX_ENEMIES]:
            # Calculate relative position to player
            rel_x = enemy.x - player_x
            rel_y = enemy.y - player_y
            abs_enemy_x = enemy.x
            abs_enemy_y = enemy.y
            distance = np.sqrt(rel_x**2 + rel_y**2)

            # Normalize relative positions
            rel_x /= 1200.0
            rel_y /= 900.0
            abs_enemy_x /= 1200.0
            abs_enemy_y /= 900.0
            distance /= np.sqrt(1200.0**2 + 900.0**2)

            # Calculate angle between player and enemy
            angle = np.arctan2(rel_y, rel_x) / np.pi  # Normalize to [-1,1]

            # Add enemy health (normalized)
            enemy_health = enemy.health / 3.0  # Assuming max health is 3

            enemy_features.extend(
                [
                    rel_x,
                    rel_y,
                    abs_enemy_x,
                    abs_enemy_y,
                    distance,
                    angle,
                    enemy_health,
                ],
            )

        # Pad enemy features if needed (existing code)
        expected_enemy_features = MAX_ENEMIES * ENEMY_FEATURES
        if len(enemy_features) < expected_enemy_features:
            padding = [0] * (expected_enemy_features - len(enemy_features))
            enemy_features.extend(padding)

        # Process bullets (existing code with normalization)
        bullet_features = []
        for bullet in observation.bullets[:MAX_BULLETS]:
            rel_x = (bullet.x - player_x) / 1200.0
            rel_y = (bullet.y - player_y) / 900.0
            distance = np.sqrt(rel_x**2 + rel_y**2)

            # Direction danger calculation
            dx, dy = bullet.direction
            direction_danger = -(rel_x * dx + rel_y * dy) / (distance + 1e-6)

            is_enemy = 1 if bullet.owner != "player" else 0

            bullet_features.extend(
                [
                    rel_x,
                    rel_y,
                    distance,
                    direction_danger,
                    is_enemy,
                ],
            )

        # Pad bullet features if needed (existing code)
        expected_bullet_features = MAX_BULLETS * BULLET_FEATURES
        if len(bullet_features) < expected_bullet_features:
            padding = [0] * (expected_bullet_features - len(bullet_features))
            bullet_features.extend(padding)

        wall_features = []

        # Process up to a maximum number of walls (e.g., 20)
        max_walls = len(observation.walls)
        for i in range(MAX_WALLS):
            if i < max_walls:
                # Normalize wall coordinates to [0,1] range
                norm_wall_x = observation.walls[i].x / 1200.0
                norm_wall_y = observation.walls[i].y / 900.0

                # Add features for this wall
                wall_features.extend(
                    [
                        norm_wall_x,
                        norm_wall_y,
                    ],
                )
            else:
                wall_features.extend([0.0, 0.0])

        # Combine all features and adjust STATE_SIZE constant accordingly
        state = (
            [player_health / 3.0, norm_x, norm_y] + enemy_features + bullet_features + wall_features
        )

        # Update assertion for debugging
        expected_size = (
            PLAYER_FEATURES
            + (MAX_ENEMIES * ENEMY_FEATURES)
            + (MAX_BULLETS * BULLET_FEATURES)
            + len(wall_features)
        )

        assert (
            len(state) == expected_size
        ), f"State size mismatch: got {len(state)}, expected {expected_size}"

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
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Store last state for learning
        if self.is_training:
            self.last_state = state

        # Set networks to evaluation mode for inference
        training_mode = self.policy_net.training
        self.policy_net.eval()

        # Epsilon-greedy action selection
        if self.is_training and random.random() < self.epsilon:
            # Explore: select a random action
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # Exploit: select the action with highest predicted Q-value
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)  # Use policy_net instead of model
                action_idx = q_values.max(1)[1].item()

        # Convert to game action
        action = self._action_to_game_action(action_idx)
        if action.is_shooting:
            self.shots_fired += 1

        # Store the selected action for learning
        if self.is_training:
            self.last_action = action_idx

        # Every 1000 calls, debug Q-values
        if hasattr(self, "get_action_count"):
            self.get_action_count += 1
        else:
            self.get_action_count = 0

        if self.get_action_count % 1000 == 0:
            with torch.no_grad():
                self.policy_net.eval()  # Ensure in eval mode for debugging
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]

                top_actions = np.argsort(q_values)[-3:][::-1]
                for action_idx in top_actions:
                    action = self._action_to_game_action(action_idx)

        if self.is_training:
            self.recent_actions.append(action_idx)
            if len(self.recent_actions) > 100:
                self.recent_actions.pop(0)

        # Restore original training mode
        self.policy_net.train(training_mode)

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

        if len(self.recent_actions) > 20:  # Need enough history
            # Count action frequencies
            action_counts = {}
            for a in self.recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1

        # Accumulate reward
        self.accumulated_reward += reward

        # Convert next observation to state
        next_state = self.encode_observation(next_observation)

        # Store transition in replay memory
        self.memory.push(self.last_state, self.last_action, reward, next_state, done)

        # Update last state
        self.last_state = next_state

        # Increment step counter
        self.steps_done += 1

        # Only train if we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        experiences = self.memory.sample(self.batch_size)

        # Convert experiences to tensors
        states = torch.FloatTensor(
            np.array([exp.state for exp in experiences]),
        ).to(device)
        actions = (
            torch.LongTensor(
                [exp.action for exp in experiences],
            )
            .unsqueeze(1)
            .to(device)
        )
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(device)
        next_states = torch.FloatTensor(
            np.array([exp.next_state for exp in experiences]),
        ).to(device)
        dones = torch.FloatTensor([exp.done for exp in experiences]).to(device)

        # Compute current Q values using policy network
        current_q_values = self.policy_net(states).gather(1, actions)

        # ----- Double DQN Implementation -----
        # 1. Get actions that would be chosen by policy_net for next states
        with torch.no_grad():
            next_action_indices = self.policy_net(next_states).max(1)[1].unsqueeze(1)

            # 2. Evaluate those actions using target_net
            next_q_values = self.target_net(next_states).gather(1, next_action_indices).squeeze()

        # Compute target Q values (without tanh constraint)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        if done:  # Update at the end of episodes
            self.episodes_done += 1

            # Decay epsilon at the end of each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Update target network periodically
            if self.episodes_done % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model to
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "state_size": self.state_size,
                "n_actions": self.n_actions,
                "epsilon": self.epsilon,
                "episodes_done": self.episodes_done,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.state_size = checkpoint["state_size"]
        self.n_actions = checkpoint["n_actions"]
        self.epsilon = checkpoint["epsilon"]

        # Initialize models
        self.policy_net, self.target_net = self._initialize_model(self.state_size)

        # Check if we're loading an older model without separate networks
        if "policy_net_state_dict" in checkpoint:
            # New format with separate networks
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        else:
            # Old format with single model
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint["model_state_dict"])

        # Load episode counter if available
        if "episodes_done" in checkpoint:
            self.episodes_done = checkpoint["episodes_done"]

        # Set models to evaluation mode for inference
        if not self.is_training:
            self.policy_net.eval()
            self.target_net.eval()

    def load_model_weights_only(self, path: str) -> None:
        """
        Load only the model weights from a checkpoint, not the hyperparameters.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)

        # Initialize models with current hyperparameters
        self.policy_net, self.target_net = self._initialize_model(self.state_size)

        # Check if we're loading an older model without separate networks
        if "policy_net_state_dict" in checkpoint:
            # New format with separate networks
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        else:
            # Old format with single model
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint["model_state_dict"])

        # Important: DON'T load epsilon, gamma, lr, etc.
        # We'll keep the values that were passed to the constructor

        print("Loaded model weights only. Using current hyperparameters:")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - Epsilon: {self.epsilon}")
        print(f"  - Epsilon decay: {self.epsilon_decay}")
