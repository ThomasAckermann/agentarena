import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from agentarena.agent.agent import Agent
from agentarena.game.action import Action, Direction

# Constants for state encoding
MAX_ENEMIES = 3  # Match with config.max_enemies
MAX_BULLETS = 5
ENEMY_FEATURES = 5  # Features per enemy: rel_x, rel_y, distance, angle, health
BULLET_FEATURES = 5  # Features per bullet: rel_x, rel_y, distance, direction_danger, is_enemy
PLAYER_FEATURES = 1  # Health

# Calculate expected state vector size
STATE_SIZE = PLAYER_FEATURES + (MAX_ENEMIES * ENEMY_FEATURES) + (MAX_BULLETS * BULLET_FEATURES)

# Define a named tuple for storing experiences
Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
    """Experience replay buffer to store and sample transitions"""

    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition to the replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions randomly"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNModel(nn.Module):
    """Deep Q-Network model architecture"""

    def __init__(self, input_size, output_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLAgent(Agent):
    """Agent that uses machine learning to decide actions"""

    def __init__(
        self,
        name="MLAgent",
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        is_training=True,
    ):
        super().__init__(name)
        self.is_training = is_training

        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.learning_rate = learning_rate  # Learning rate

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
        self.memory = ReplayMemory()
        self.batch_size = 64

        # Track last state and action for learning
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0

    def _initialize_model(self, state_size):
        """Initialize the neural network model"""
        model = DQNModel(state_size, self.n_actions)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model

    def reset(self):
        """Reset the agent state at the beginning of an episode"""
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0

        # Decay epsilon
        if self.is_training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def encode_observation(self, observation):
        """Convert game observation to a state vector for the neural network"""
        # Extract player information
        player = observation["player"]
        player_x = player["x"]
        player_y = player["y"]
        player_health = player["health"]

        # Process enemies
        enemy_features = []
        for enemy in observation["enemies"][:MAX_ENEMIES]:
            # Calculate relative position to player
            rel_x = enemy["x"] - player_x
            rel_y = enemy["y"] - player_y
            distance = np.sqrt(rel_x**2 + rel_y**2)

            # Calculate angle between player and enemy
            angle = np.arctan2(rel_y, rel_x)

            # Add enemy health
            enemy_health = enemy["health"]

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
        for bullet in observation["bullets"][:MAX_BULLETS]:
            # Calculate relative position to player
            rel_x = bullet["x"] - player_x
            rel_y = bullet["y"] - player_y
            distance = np.sqrt(rel_x**2 + rel_y**2)

            # Calculate danger level (is this bullet coming toward the player?)
            dx, dy = bullet["direction"]
            # Dot product between bullet direction and direction to player
            direction_danger = -(rel_x * dx + rel_y * dy) / (distance + 1e-6)

            # Is this an enemy bullet?
            is_enemy = 1 if bullet["owner"] != "player" else 0

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

    def _action_to_game_action(self, action_idx):
        """Convert network output (action index) to game Action"""
        # Determine direction and shooting from action index
        direction_idx = action_idx % len(self.directions)
        is_shooting = action_idx >= len(self.directions)

        direction = self.directions[direction_idx]
        return Action(is_shooting=is_shooting, direction=direction)

    def _game_action_to_idx(self, action):
        """Convert game Action to network action index"""
        direction_idx = (
            self.directions.index(action.direction) if action.direction in self.directions else 0
        )
        shooting_offset = len(self.directions) if action.is_shooting else 0
        return direction_idx + shooting_offset

    def get_action(self, observation):
        """Choose an action based on the current observation"""
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

    def learn(self, next_observation, reward, done):
        """Update the model based on received reward"""
        if not self.is_training or self.last_state is None:
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
        batch = Experience(*zip(*experiences, strict=False))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done)

        # Compute current Q values
        current_q_values = self.model(state_batch).gather(1, action_batch)

        # Compute next Q values (for DQN)
        with torch.no_grad():
            next_q_values = self.model(next_state_batch).max(1)[0]

        # Compute target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        """Save the model to disk"""
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

    def load_model(self, path):
        """Load the model from disk"""
        checkpoint = torch.load(path)
        self.state_size = checkpoint["state_size"]
        self.n_actions = checkpoint["n_actions"]
        self.epsilon = checkpoint["epsilon"]

        # Initialize model with correct dimensions
        self.model = self._initialize_model(self.state_size)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # Set to evaluation mode
