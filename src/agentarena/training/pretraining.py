from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from agentarena.agent.ml_agent import MLAgent
from agentarena.models.training import MLAgentConfig
from agentarena.training.demo_collection import DemonstrationDataset


class ImitationLearner:
    def __init__(
        self,
        ml_config: MLAgentConfig,
        demonstrations_dir: str = "demonstrations",
        device: torch.device | None = None,
    ) -> None:
        self.ml_config = ml_config
        self.demonstrations_dir = demonstrations_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create MLAgent with multi-head network
        self.agent = MLAgent(is_training=True, config=ml_config)

        # Set to imitation learning mode
        self.agent.set_training_mode("imitation")

        print(f"ImitationLearner initialized on {self.device}")

    def pretrain(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.2,
        save_path: str | None = None,
        tensorboard_dir: str | None = None,
        balance_actions: bool = True,
        balance_factor: float = 0.7,
    ) -> float:
        print(f"Starting pre-training for {epochs} epochs...")

        try:
            dataset = DemonstrationDataset(self.demonstrations_dir)
        except Exception as e:
            print(f"Error loading demonstrations: {e}")
            return 0.0

        if len(dataset) == 0:
            print("No demonstration data found. Cannot pre-train.")
            return 0.0

        print(f"Loaded {len(dataset)} demonstration samples")

        # Create balanced data loader if requested
        if balance_actions:
            train_loader, val_loader = self._create_balanced_dataloaders(
                dataset,
                batch_size,
                validation_split,
                balance_factor,
            )
        else:
            # Standard train/val split
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset,
                [train_size, val_size],
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        writer = None
        if tensorboard_dir:
            writer = SummaryWriter(tensorboard_dir)

        best_val_accuracy = 0.0

        for epoch in range(epochs):
            train_loss, train_accuracy = self._train_epoch(train_loader)
            val_loss, val_accuracy = self._validate_epoch(val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.3f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.3f}",
            )

            if writer:
                writer.add_scalar("Imitation/Loss/Train", train_loss, epoch)
                writer.add_scalar("Imitation/Loss/Validation", val_loss, epoch)
                writer.add_scalar("Imitation/Accuracy/Train", train_accuracy, epoch)
                writer.add_scalar("Imitation/Accuracy/Validation", val_accuracy, epoch)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_path:
                    self.agent.save_model(save_path)
                    print(f"New best model saved with validation accuracy: {val_accuracy:.3f}")

        if writer:
            writer.close()

        print(f"Pre-training completed. Best validation accuracy: {best_val_accuracy:.3f}")
        return best_val_accuracy

    def _create_balanced_dataloaders(  # noqa: ANN202
        self,
        dataset,
        batch_size: int,
        validation_split: float,
        balance_factor: float,
    ):
        # Calculate action frequencies
        action_counts = Counter()
        for _, action in dataset:
            action_idx = torch.argmax(action).item()
            action_counts[action_idx] += 1

        print("Action distribution before balancing:")
        total_samples = len(dataset)
        for action_idx, count in sorted(action_counts.items()):
            percentage = count / total_samples * 100
            print(f"  Action {action_idx}: {count:4d} samples ({percentage:5.1f}%)")

        # Create sample weights for balancing
        weights = []
        for _, action in dataset:
            action_idx = torch.argmax(action).item()
            freq = action_counts[action_idx]
            # Apply balance factor (1.0 = perfect balance, 0.0 = no balance)
            weight = (1.0 / freq) ** balance_factor
            weights.append(weight)

        # Split indices for train/val
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create weighted samplers
        train_weights = [weights[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

        # Create data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch using the imitation learning head."""
        self.agent.policy_net.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for _, (states, actions) in enumerate(train_loader):
            states = states.to(self.device)  # noqa: PLW2901
            actions = actions.to(self.device)  # noqa: PLW2901

            # Process entire batch at once
            self.agent.optimizer.zero_grad()

            # Get action logits from imitation head for entire batch
            action_logits = self.agent.policy_net.get_action_logits(states)

            # Convert one-hot to class indices for cross-entropy loss
            target_indices = torch.argmax(actions, dim=1)

            # Calculate cross-entropy loss
            loss = torch.nn.functional.cross_entropy(action_logits, target_indices)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), max_norm=1.0)
            self.agent.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            with torch.no_grad():
                predicted = torch.argmax(action_logits, dim=1)
                correct_predictions += (predicted == target_indices).sum().item()
                total_samples += target_indices.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate for one epoch."""
        self.agent.policy_net.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(self.device)  # noqa: PLW2901
                actions = actions.to(self.device)  # noqa: PLW2901

                # Get action logits
                action_logits = self.agent.policy_net.get_action_logits(states)

                # Calculate loss
                target_indices = torch.argmax(actions, dim=1)
                loss = torch.nn.functional.cross_entropy(action_logits, target_indices)
                total_loss += loss.item()

                # Calculate accuracy
                predicted = torch.argmax(action_logits, dim=1)
                correct_predictions += (predicted == target_indices).sum().item()
                total_samples += target_indices.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy


def pretrain_agent(
    demonstrations_dir: str = "demonstrations",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = "models/pretrained_agent.pt",
    tensorboard_dir: str = "runs/pretraining",
    balance_actions: bool = True,
    balance_factor: float = 0.7,
) -> str:
    ml_config = MLAgentConfig(learning_rate=learning_rate)

    learner = ImitationLearner(ml_config, demonstrations_dir)
    final_accuracy = learner.pretrain(
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        tensorboard_dir=tensorboard_dir,
        balance_actions=balance_actions,
        balance_factor=balance_factor,
    )

    if final_accuracy > 0:
        print(f"Pre-training successful! Model saved to {save_path}")
        print(
            "To use for RL training:"
            f"python -m agentarena.training.train --pretrained-model {save_path}",
        )
        return save_path
    print("Pre-training failed!")
    return ""
