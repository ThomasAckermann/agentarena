"""
Demonstration data collection and processing for offline learning.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from agentarena.agent.manual_agent import ManualAgent
from agentarena.models.action import Action, Direction
from agentarena.models.observations import GameObservation


class ActionEncoder:
    """Handles encoding and decoding of actions for neural networks."""

    def __init__(self):
        """Initialize the action encoder with the action space."""
        # Define the action space: 9 directions × 2 shooting states = 18 actions
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
        """
        Convert an Action object to a one-hot encoded vector.

        Args:
            action: Action object to encode

        Returns:
            One-hot encoded action vector of shape (18,)
        """
        # Find direction index
        direction_idx = (
            self.directions.index(action.direction) if action.direction in self.directions else 0
        )

        # Calculate action index: direction + shooting offset
        shooting_offset = len(self.directions) if action.is_shooting else 0
        action_idx = direction_idx + shooting_offset

        # Create one-hot vector
        one_hot = np.zeros(self.n_actions)
        one_hot[action_idx] = 1.0

        return one_hot

    def decode_action(self, one_hot: np.ndarray) -> Action:
        """
        Convert a one-hot encoded vector back to an Action object.

        Args:
            one_hot: One-hot encoded action vector

        Returns:
            Action object
        """
        action_idx = np.argmax(one_hot)

        # Determine direction and shooting from action index
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
    """Handles logging of human demonstrations."""

    def __init__(self, demonstrations_dir: str = "demonstrations"):
        """
        Initialize the demonstration logger.

        Args:
            demonstrations_dir: Directory to save demonstration files
        """
        self.demonstrations_dir = Path(demonstrations_dir)
        self.demonstrations_dir.mkdir(exist_ok=True)
        self.action_encoder = ActionEncoder()

        # Current episode data
        self.current_episode = []
        self.episode_active = False

    def start_episode(self) -> None:
        """Start logging a new episode."""
        self.current_episode = []
        self.episode_active = True

    def log_step(self, observation: GameObservation, action: Action) -> None:
        """
        Log a single step in the current episode.

        Args:
            observation: Current game state observation
            action: Action taken by the human player
        """
        if not self.episode_active:
            return

        # Encode the observation (reuse existing ML agent encoding)
        from agentarena.agent.ml_agent import MLAgent

        temp_agent = MLAgent()
        state_vector = temp_agent.encode_observation(observation)

        # Encode the action
        action_one_hot = self.action_encoder.encode_action(action)

        # Store the step
        step_data = {
            "state": state_vector.tolist(),
            "action": action_one_hot.tolist(),
            "observation": observation.model_dump(),  # Keep raw observation for analysis
        }

        self.current_episode.append(step_data)

    def end_episode(self, won: bool = False, score: int = 0) -> str:
        """
        End the current episode and save it to disk.

        Args:
            won: Whether the episode was won
            score: Final score achieved

        Returns:
            Path to the saved episode file
        """
        if not self.episode_active or not self.current_episode:
            return ""

        # Create episode metadata
        episode_data = {
            "steps": self.current_episode,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_steps": len(self.current_episode),
                "won": won,
                "score": score,
            },
        }

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{timestamp}_{score}pts_{'win' if won else 'loss'}.json"
        filepath = self.demonstrations_dir / filename

        with open(filepath, "w") as f:
            json.dump(episode_data, f, indent=2)

        print(f"Demonstration saved: {filepath} ({len(self.current_episode)} steps)")

        # Reset for next episode
        self.episode_active = False
        self.current_episode = []

        return str(filepath)


class DemonstrationDataset(Dataset):
    """PyTorch dataset for loading demonstration data."""

    def __init__(self, demonstrations_dir: str = "demonstrations"):
        """
        Initialize the dataset.

        Args:
            demonstrations_dir: Directory containing demonstration files
        """
        self.demonstrations_dir = Path(demonstrations_dir)
        self.action_encoder = ActionEncoder()

        # Load all demonstration files
        self.states = []
        self.actions = []
        self._load_demonstrations()

    def _load_demonstrations(self) -> None:
        """Load all demonstration files from the directory."""
        demo_files = list(self.demonstrations_dir.glob("demo_*.json"))

        if not demo_files:
            print(f"Warning: No demonstration files found in {self.demonstrations_dir}")
            return

        total_steps = 0
        successful_episodes = 0

        for demo_file in demo_files:
            try:
                with open(demo_file, "r") as f:
                    episode_data = json.load(f)

                steps = episode_data["steps"]
                metadata = episode_data["metadata"]

                # Optionally filter by quality (e.g., only winning episodes)
                # if not metadata.get('won', False):
                #     continue

                for step in steps:
                    self.states.append(np.array(step["state"], dtype=np.float32))
                    self.actions.append(np.array(step["action"], dtype=np.float32))

                total_steps += len(steps)
                successful_episodes += 1

            except Exception as e:
                print(f"Error loading {demo_file}: {e}")

        print(f"Loaded {total_steps} demonstration steps from {successful_episodes} episodes")

    def __len__(self) -> int:
        """Return the number of demonstration steps."""
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single demonstration step.

        Args:
            idx: Index of the step

        Returns:
            Tuple of (state_tensor, action_tensor)
        """
        state = torch.FloatTensor(self.states[idx])
        action = torch.FloatTensor(self.actions[idx])
        return state, action


def create_demonstration_dataloader(
    demonstrations_dir: str = "demonstrations", batch_size: int = 64, shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for demonstration data.

    Args:
        demonstrations_dir: Directory containing demonstration files
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader for the demonstration dataset
    """
    dataset = DemonstrationDataset(demonstrations_dir)

    if len(dataset) == 0:
        raise ValueError("No demonstration data found. Please collect demonstrations first.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Utility functions for data management
def collect_demonstrations_session(game_config_path: str = "config.yaml") -> None:
    """
    Run a demonstration collection session.
    This should be called when the game is run with manual agent.
    """
    print("Starting demonstration collection session...")
    print("Play the game normally. Your gameplay will be recorded for training.")
    print("Press Ctrl+C to stop collecting demonstrations.")

    # This function would be integrated into the main game loop
    # when manual agent is detected


def analyze_demonstrations(demonstrations_dir: str = "demonstrations") -> None:
    """
    Analyze collected demonstrations and print statistics.

    Args:
        demonstrations_dir: Directory containing demonstration files
    """
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
        with open(demo_file, "r") as f:
            episode_data = json.load(f)

        metadata = episode_data["metadata"]
        steps = episode_data["steps"]

        total_steps += len(steps)
        if metadata.get("won", False):
            winning_episodes += 1
        scores.append(metadata.get("score", 0))

        # Count actions
        for step in steps:
            action_vec = np.array(step["action"])
            action_idx = np.argmax(action_vec)
            action_counts[action_idx] = action_counts.get(action_idx, 0) + 1

    print(f"\n=== Demonstration Analysis ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Average steps per episode: {total_steps / total_episodes:.1f}")
    print(f"Winning episodes: {winning_episodes} ({winning_episodes/total_episodes*100:.1f}%)")
    print(f"Average score: {np.mean(scores):.1f}")
    print(f"Score range: {min(scores)} - {max(scores)}")

    print(f"\nAction distribution:")
    encoder = ActionEncoder()
    for action_idx, count in sorted(action_counts.items()):
        action = encoder.decode_action(encoder.action_idx_to_one_hot(action_idx))
        percentage = count / total_steps * 100
        print(f"  {action_idx:2d}: {action} - {count:4d} times ({percentage:5.1f}%)")


if __name__ == "__main__":
    # Example usage
    print("Testing demonstration data collection...")

    # Analyze existing demonstrations
    analyze_demonstrations()

    # Test dataset loading
    try:
        dataset = DemonstrationDataset()
        print(f"Dataset loaded with {len(dataset)} samples")

        if len(dataset) > 0:
            state, action = dataset[0]
            print(f"Sample state shape: {state.shape}")
            print(f"Sample action shape: {action.shape}")
    except ValueError as e:
        print(f"Dataset loading failed: {e}")
