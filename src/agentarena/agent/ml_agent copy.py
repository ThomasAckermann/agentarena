import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from agentarena.agent.agent import Agent
from agentarena.models.action import Action, Direction
from agentarena.models.observations import GameObservation
from agentarena.models.training import Experience, MLAgentConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_RECENT_ACTIONS: int = 100
MAX_ENEMIES = 3
MAX_BULLETS = 12
ENEMY_FEATURES = 7
BULLET_FEATURES = 5
PLAYER_FEATURES = 3
MAX_WALLS = 120
WALL_FEATURES_PER_WALL = 2  # x, y
TOTAL_WALL_FEATURES = MAX_WALLS * WALL_FEATURES_PER_WALL

STATE_SIZE = (
    PLAYER_FEATURES
    + (MAX_ENEMIES * ENEMY_FEATURES)
    + (MAX_BULLETS * BULLET_FEATURES)
    + (MAX_WALLS * WALL_FEATURES_PER_WALL)
)


class PrioritizedReplay:
    def __init__(self, capacity: int = 50000, winning_bonus: float = 10.0) -> None:
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0
        self.winning_bonus = winning_bonus

    def push(self, state, action, reward, next_state, done):
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        priority = abs(reward) + 1.0

        if done and reward > 0:
            priority *= self.winning_bonus

        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(priority)
        else:
            self.memory[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        probs = np.array(self.priorities) / sum(self.priorities)

        indices = np.random.choice(len(self.memory), batch_size, p=probs)  # noqa: NPY002
        return [self.memory[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.memory)


class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, mode: str = "q_learning") -> None:  # noqa: ARG002
        super().__init__()
        self.mode = mode

        # Feature dimensions
        self.PLAYER_FEATURES = 3
        self.ENEMY_FEATURES = 7
        self.BULLET_FEATURES = 5
        self.WALL_FEATURES_PER_WALL = 2
        self.MAX_ENEMIES = 3
        self.MAX_BULLETS = 12
        self.MAX_WALLS = 120

        # Embedding dimension
        self.embed_dim = 128

        # Feature embedders with proper initialization and normalization
        self.player_embedder = nn.Sequential(
            nn.Linear(self.PLAYER_FEATURES, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.enemy_embedder = nn.Sequential(
            nn.Linear(self.ENEMY_FEATURES, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.bullet_embedder = nn.Sequential(
            nn.Linear(self.BULLET_FEATURES, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.wall_embedder = nn.Sequential(
            nn.Linear(self.WALL_FEATURES_PER_WALL, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        # MultiheadAttention layers for each entity type
        self.enemy_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        self.bullet_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        self.wall_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # Feature aggregation with proper normalization
        self.feature_aggregator = nn.Sequential(
            nn.Linear(self.embed_dim * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
        )

        # Output heads with smaller initial weights
        self.q_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size),
        )

        self.action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Careful initialization to prevent NaN"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def safe_multihead_attention(self, attention_layer, query, key, value, key_padding_mask=None):
        """Safely apply multihead attention with NaN handling"""
        # Ensure inputs are finite
        if not torch.isfinite(query).all():
            query = torch.nan_to_num(query, nan=0.0, posinf=1.0, neginf=-1.0)
        if not torch.isfinite(key).all():
            key = torch.nan_to_num(key, nan=0.0, posinf=1.0, neginf=-1.0)
        if not torch.isfinite(value).all():
            value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)

        try:
            # Apply attention
            attended_output, attention_weights = attention_layer(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=False,  # We don't need weights for efficiency
            )

            # Check for NaN in output
            if not torch.isfinite(attended_output).all():
                print("Warning: NaN in attention output, using fallback")
                # Fallback: just return the query (no attention)
                return query

            return attended_output  # noqa: TRY300

        except Exception as e:
            print(f"Warning: Attention computation failed ({e}), using fallback")
            # Fallback: return query unchanged
            return query

    def forward(self, x: torch.Tensor, mode: str = "q_learning") -> torch.Tensor:
        # Check input for NaN
        if not torch.isfinite(x).all():
            print("Warning: NaN in input, clipping values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size = x.shape[0]

        # Parse input features
        idx = 0
        player_features = x[:, idx : idx + self.PLAYER_FEATURES]
        idx += self.PLAYER_FEATURES

        enemy_features = x[:, idx : idx + (self.MAX_ENEMIES * self.ENEMY_FEATURES)]
        enemy_features = enemy_features.view(batch_size, self.MAX_ENEMIES, self.ENEMY_FEATURES)
        idx += self.MAX_ENEMIES * self.ENEMY_FEATURES

        bullet_features = x[:, idx : idx + (self.MAX_BULLETS * self.BULLET_FEATURES)]
        bullet_features = bullet_features.view(batch_size, self.MAX_BULLETS, self.BULLET_FEATURES)
        idx += self.MAX_BULLETS * self.BULLET_FEATURES

        wall_features = x[:, idx : idx + (self.MAX_WALLS * self.WALL_FEATURES_PER_WALL)]
        wall_features = wall_features.view(batch_size, self.MAX_WALLS, self.WALL_FEATURES_PER_WALL)

        # Embed features
        player_embed = self.player_embedder(player_features).unsqueeze(1)  # [B, 1, embed_dim]
        enemy_embeds = self.enemy_embedder(enemy_features)  # [B, max_enemies, embed_dim]
        bullet_embeds = self.bullet_embedder(bullet_features)  # [B, max_bullets, embed_dim]
        wall_embeds = self.wall_embedder(wall_features)  # [B, max_walls, embed_dim]

        # Create padding masks for MultiheadAttention
        # True values indicate positions that should be ignored
        enemy_padding_mask = torch.sum(torch.abs(enemy_features), dim=-1) < 1e-6  # [B, max_enemies]
        bullet_padding_mask = (
            torch.sum(torch.abs(bullet_features), dim=-1) < 1e-6
        )  # [B, max_bullets]
        wall_padding_mask = torch.sum(torch.abs(wall_features), dim=-1) < 1e-6  # [B, max_walls]

        # Apply multihead attention using player as query and entities as key/value
        enemy_context = self.safe_multihead_attention(
            attention_layer=self.enemy_attention,
            query=player_embed,  # [B, 1, embed_dim]
            key=enemy_embeds,  # [B, max_enemies, embed_dim]
            value=enemy_embeds,  # [B, max_enemies, embed_dim]
            key_padding_mask=enemy_padding_mask,  # [B, max_enemies]
        ).squeeze(1)  # [B, embed_dim]

        bullet_context = self.safe_multihead_attention(
            attention_layer=self.bullet_attention,
            query=player_embed,  # [B, 1, embed_dim]
            key=bullet_embeds,  # [B, max_bullets, embed_dim]
            value=bullet_embeds,  # [B, max_bullets, embed_dim]
            key_padding_mask=bullet_padding_mask,  # [B, max_bullets]
        ).squeeze(1)  # [B, embed_dim]

        wall_context = self.safe_multihead_attention(
            attention_layer=self.wall_attention,
            query=player_embed,  # [B, 1, embed_dim]
            key=wall_embeds,  # [B, max_walls, embed_dim]
            value=wall_embeds,  # [B, max_walls, embed_dim]
            key_padding_mask=wall_padding_mask,  # [B, max_walls]
        ).squeeze(1)  # [B, embed_dim]

        # Combine all contexts
        combined_features = torch.cat(
            [player_embed.squeeze(1), enemy_context, bullet_context, wall_context],
            dim=1,
        )

        # Check for NaN before final processing
        if not torch.isfinite(combined_features).all():
            print("Warning: NaN in combined features")
            combined_features = torch.nan_to_num(
                combined_features,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )

        features = self.feature_aggregator(combined_features)
        # Final NaN check
        if not torch.isfinite(features).all():
            print("Warning: NaN in final features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        output = self.q_head(features) if mode == "q_learning" else self.action_head(features)

        # Final output check
        if not torch.isfinite(output).all():
            print("Warning: NaN in output")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return output

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, mode="q_learning")

    def get_action_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, mode="imitation")


class MLAgent(Agent):
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
        super().__init__(name)
        self.is_training = is_training
        self.recent_actions = []
        self.shots_fired = 0
        self.enemy_hits = 0

        if config:
            self.gamma = config.gamma
            self.epsilon = config.epsilon
            self.epsilon_min = config.epsilon_min
            self.epsilon_decay = config.epsilon_decay
            self.learning_rate = config.learning_rate
            self.batch_size = config.batch_size
            self.memory_capacity = config.memory_capacity
        else:
            self.gamma = gamma  # Discount factor
            self.epsilon = epsilon  # Exploration rate
            self.epsilon_min = epsilon_min  # Minimum exploration rate
            self.epsilon_decay = epsilon_decay  # Decay rate for exploration
            self.learning_rate = learning_rate  # Learning rate
            self.batch_size = 256
            self.memory_capacity = 20000

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
        self.state_size = STATE_SIZE
        self.policy_net = self._initialize_model(self.state_size)
        self.memory = PrioritizedReplay(self.memory_capacity)
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.accumulated_reward = 0.0

        self.steps_done = 0
        self.episodes_done = 0
        self.training_mode = "q_learning"

    def _initialize_model(self, state_size: int) -> PolicyNetwork:
        policy_net = PolicyNetwork(state_size, self.n_actions).to(device)
        policy_net.load_state_dict(policy_net.state_dict())
        policy_net.eval()

        self.optimizer = optim.Adam(
            policy_net.parameters(),
            lr=self.learning_rate,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.9995,
        )

        return policy_net

    def set_training_mode(self, mode: str) -> None:
        if mode not in ["q_learning", "imitation"]:
            msg = f"Invalid training mode: {mode}"
            raise ValueError(msg)
        self.training_mode = mode

    def reset(self) -> None:
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0.0
        self.shots_fired = 0
        self.enemy_hits = 0

    def encode_observation(self, observation: GameObservation) -> np.ndarray:
        player = observation.player
        player_x = player.x
        player_y = player.y
        player_health = player.health

        norm_x = player_x / 1200.0
        norm_y = player_y / 900.0

        enemy_features = []
        for enemy in observation.enemies[:MAX_ENEMIES]:
            rel_x = enemy.x - player_x
            rel_y = enemy.y - player_y
            abs_enemy_x = enemy.x
            abs_enemy_y = enemy.y
            distance = np.sqrt(rel_x**2 + rel_y**2)

            rel_x /= 1200.0
            rel_y /= 900.0
            abs_enemy_x /= 1200.0
            abs_enemy_y /= 900.0
            distance /= np.sqrt(1200.0**2 + 900.0**2)
            angle = np.arctan2(rel_y, rel_x) / np.pi
            enemy_health = enemy.health / 3.0

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

        expected_enemy_features = MAX_ENEMIES * ENEMY_FEATURES
        if len(enemy_features) < expected_enemy_features:
            padding = [0] * (expected_enemy_features - len(enemy_features))
            enemy_features.extend(padding)

        bullet_features = []
        for bullet in observation.bullets[:MAX_BULLETS]:
            rel_x = (bullet.x - player_x) / 1200.0
            rel_y = (bullet.y - player_y) / 900.0
            distance = np.sqrt(rel_x**2 + rel_y**2)

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

        expected_bullet_features = MAX_BULLETS * BULLET_FEATURES
        if len(bullet_features) < expected_bullet_features:
            padding = [0] * (expected_bullet_features - len(bullet_features))
            bullet_features.extend(padding)

        wall_features = []

        max_walls = len(observation.walls)
        for i in range(MAX_WALLS):
            if i < max_walls:
                norm_wall_x = observation.walls[i].x / 1200.0
                norm_wall_y = observation.walls[i].y / 900.0

                wall_features.extend(
                    [
                        norm_wall_x,
                        norm_wall_y,
                    ],
                )
            else:
                wall_features.extend([0.0, 0.0])

        state = [
            player_health / 3.0,
            norm_x,
            norm_y,
            *enemy_features,
            *bullet_features,
            *wall_features,
        ]

        expected_size = (
            PLAYER_FEATURES
            + (MAX_ENEMIES * ENEMY_FEATURES)
            + (MAX_BULLETS * BULLET_FEATURES)
            + len(wall_features)
        )

        assert len(state) == expected_size, (
            f"State size mismatch: got {len(state)}, expected {expected_size}"
        )

        return np.array(state, dtype=np.float32)

    def _action_to_game_action(self, action_idx: int) -> Action:
        direction_idx = action_idx % len(self.directions)
        is_shooting = action_idx >= len(self.directions)

        direction = self.directions[direction_idx]
        return Action(is_shooting=is_shooting, direction=direction)

    def _game_action_to_idx(self, action: Action) -> int:
        direction_idx = (
            self.directions.index(action.direction) if action.direction in self.directions else 0
        )
        shooting_offset = len(self.directions) if action.is_shooting else 0
        return direction_idx + shooting_offset

    def get_action(self, observation: GameObservation) -> Action:
        state = self.encode_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        if self.is_training:
            self.last_state = state

        training_mode = self.policy_net.training
        self.policy_net.eval()

        if self.is_training and random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net.get_q_values(state_tensor)
                action_idx = q_values.max(1)[1].item()

        action = self._action_to_game_action(action_idx)
        if action.is_shooting:
            self.shots_fired += 1

        if self.is_training:
            self.last_action = action_idx

        if hasattr(self, "get_action_count"):
            self.get_action_count += 1
        else:
            self.get_action_count = 0

        if self.get_action_count % 1000 == 0:
            with torch.no_grad():
                self.policy_net.eval()
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]

                top_actions = np.argsort(q_values)[-3:][::-1]
                for action_idx in top_actions:
                    action = self._action_to_game_action(action_idx)

        if self.is_training:
            self.recent_actions.append(action_idx)
            if len(self.recent_actions) > 100:
                self.recent_actions.pop(0)

        self.policy_net.train(training_mode)

        return action

    def learn_from_demonstrations(
        self,
        observation: GameObservation,
        target_action_one_hot: torch.Tensor,
    ) -> float:
        self.policy_net.train()

        state = self.encode_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_logits = self.policy_net.get_action_logits(state_tensor)
        target_indices = torch.argmax(target_action_one_hot.unsqueeze(0), dim=1).to(device)
        loss = F.cross_entropy(action_logits, target_indices)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def learn(self, next_observation: GameObservation, reward: float, done: bool) -> None:
        if not self.is_training or self.last_state is None or self.last_action is None:
            return

        if len(self.recent_actions) > 20:
            action_counts = {}
            for a in self.recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1

        self.accumulated_reward += reward

        next_state = self.encode_observation(next_observation)

        self.memory.push(
            self.last_state,
            self.last_action,
            reward,
            next_state,
            done,
        )

        self.last_state = next_state
        self.steps_done += 1
        if len(self.memory) < self.batch_size:
            return
        experiences = self.memory.sample(self.batch_size)
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

        current_q_values = self.policy_net.get_q_values(states).gather(1, actions)

        with torch.no_grad():
            next_action_indices = self.policy_net.get_q_values(next_states).max(1)[1].unsqueeze(1)

            next_q_values = (
                self.policy_net.get_q_values(next_states).gather(1, next_action_indices).squeeze()
            )

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        if done:
            self.episodes_done += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save_model(self, path: str) -> None:
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "state_size": self.state_size,
                "n_actions": self.n_actions,
                "epsilon": self.epsilon,
                "episodes_done": self.episodes_done,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.state_size = checkpoint["state_size"]
        self.n_actions = checkpoint["n_actions"]
        self.epsilon = checkpoint["epsilon"]

        self.policy_net = self._initialize_model(self.state_size)

        if "policy_net_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        else:
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])

        if "episodes_done" in checkpoint:
            self.episodes_done = checkpoint["episodes_done"]
        if not self.is_training:
            self.policy_net.eval()

    def load_model_weights_only(self, path: str) -> None:
        checkpoint = torch.load(path)

        self.policy_net = self._initialize_model(self.state_size)

        if "policy_net_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        else:
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])

        print("Loaded model weights only. Using current hyperparameters:")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - Epsilon: {self.epsilon}")
        print(f"  - Epsilon decay: {self.epsilon_decay}")
