"""
Demonstration data collection and processing for offline learning.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agentarena.models.action import Action, Direction
from agentarena.models.observations import GameObservation


class ActionEncoder:

    def __init__(self):

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
        self.n_actions = len(self.directions) * 2

    def encode_action(self, action: Action) -> np.ndarray:

        direction_idx = (
            self.directions.index(action.direction) if action.direction in self.directions else 0
        )

        shooting_offset = len(self.directions) if action.is_shooting else 0
        action_idx = direction_idx + shooting_offset


        one_hot = np.zeros(self.n_actions)
        one_hot[action_idx] = 1.0

        return one_hot

    def decode_action(self, one_hot: np.ndarray) -> Action:

        action_idx = np.argmax(one_hot)

        direction_idx = action_idx % len(self.directions)
        is_shooting = action_idx >= len(self.directions)

        direction = self.directions[direction_idx]
        return Action(is_shooting=is_shooting, direction=direction)

    def action_idx_to_one_hot(self, action_idx: int) -> np.ndarray:
        """Convert action index to one-hot vector."""
        one_hot = np.zeros(self.n_actions)
        one_hot[action_idx] = 1.0
        return one_hot


class DemonstrationLogger:
    def __init__(self, demonstrations_dir: str = "demonstrations"):

        self.demonstrations_dir = Path(demonstrations_dir)
        self.demonstrations_dir.mkdir(exist_ok=True)
        self.action_encoder = ActionEncoder()

        self.current_episode = []
        self.episode_active = False

    def start_episode(self) -> None:
        self.current_episode = []
        self.episode_active = True

    def log_step(self, observation: GameObservation, action: Action) -> None:
        if not self.episode_active:
            return

        from agentarena.agent.ml_agent import MLAgent

        temp_agent = MLAgent()
        state_vector = temp_agent.encode_observation(observation)
        action_one_hot = self.action_encoder.encode_action(action)

        step_data = {
            "state": state_vector.tolist(),
            "action": action_one_hot.tolist(),
            "observation": observation.model_dump(),
        }

        self.current_episode.append(step_data)

    def end_episode(self, won: bool = False, score: int = 0) -> str:
        if not self.episode_active or not self.current_episode:
            return ""
        episode_data = {
            "steps": self.current_episode,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_steps": len(self.current_episode),
                "won": won,
                "score": score,
            },
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{timestamp}_{score}pts_{'win' if won else 'loss'}.json"
        filepath = self.demonstrations_dir / filename

        with open(filepath, "w") as f:
            json.dump(episode_data, f, indent=2)

        print(f"Demonstration saved: {filepath} ({len(self.current_episode)} steps)")

        self.episode_active = False
        self.current_episode = []

        return str(filepath)


class DemonstrationDataset(Dataset):
    def __init__(self, demonstrations_dir: str = "demonstrations"):
        self.demonstrations_dir = Path(demonstrations_dir)
        self.action_encoder = ActionEncoder()

        self.states = []
        self.actions = []
        self._load_demonstrations()

    def _load_demonstrations(self) -> None:

        demo_files = list(self.demonstrations_dir.glob("demo_*.json"))

        if not demo_files:
            print(f"Warning: No demonstration files found in {self.demonstrations_dir}")
            return

        total_steps = 0
        successful_episodes = 0

        for demo_file in demo_files:
            try:
                with open(demo_file) as f:

                    episode_data = json.load(f)

                steps = episode_data["steps"]
                metadata = episode_data["metadata"]


                for step in steps:
                    self.states.append(np.array(step["state"], dtype=np.float32))
                    self.actions.append(np.array(step["action"], dtype=np.float32))

                total_steps += len(steps)
                successful_episodes += 1

            except Exception as e:
                print(f"Error loading {demo_file}: {e}")

        print(f"Loaded {total_steps} demonstration steps from {successful_episodes} episodes")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.FloatTensor(self.states[idx])
        action = torch.FloatTensor(self.actions[idx])
        return state, action


def create_demonstration_dataloader(
    demonstrations_dir: str = "demonstrations",
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    dataset = DemonstrationDataset(demonstrations_dir)

    if len(dataset) == 0:
        raise ValueError("No demonstration data found. Please collect demonstrations first.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def analyze_demonstrations(demonstrations_dir: str = "demonstrations") -> None:

    demo_dir = Path(demonstrations_dir)
    demo_files = list(demo_dir.glob("demo_*.json"))

    if not demo_files:
        print("No demonstration files found.")
        return

    total_episodes = len(demo_files)
    total_steps = 0
    winning_episodes = 0
    scores = []

    action_counts = {}

    for demo_file in demo_files:
        with open(demo_file) as f:
            episode_data = json.load(f)

        metadata = episode_data["metadata"]
        steps = episode_data["steps"]

        total_steps += len(steps)
        if metadata.get("won", False):
            winning_episodes += 1
        scores.append(metadata.get("score", 0))

        for step in steps:
            action_vec = np.array(step["action"])
            action_idx = np.argmax(action_vec)
            action_counts[action_idx] = action_counts.get(action_idx, 0) + 1

    print("\n=== Demonstration Analysis ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Average steps per episode: {total_steps / total_episodes:.1f}")
    print(f"Winning episodes: {winning_episodes} ({winning_episodes/total_episodes*100:.1f}%)")
    print(f"Average score: {np.mean(scores):.1f}")
    print(f"Score range: {min(scores)} - {max(scores)}")

    print("\nAction distribution:")
    encoder = ActionEncoder()
    for action_idx, count in sorted(action_counts.items()):
        action = encoder.decode_action(encoder.action_idx_to_one_hot(action_idx))
        percentage = count / total_steps * 100
        print(f"  {action_idx:2d}: {action} - {count:4d} times ({percentage:5.1f}%)")


if __name__ == "__main__":
    print("Testing demonstration data collection...")

    analyze_demonstrations()

    try:
        dataset = DemonstrationDataset()
        print(f"Dataset loaded with {len(dataset)} samples")

        if len(dataset) > 0:
            state, action = dataset[0]
            print(f"Sample state shape: {state.shape}")
            print(f"Sample action shape: {action.shape}")
    except ValueError as e:
        print(f"Dataset loading failed: {e}")
