
import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from agentarena.agent.ml_agent import MLAgent, PolicyNetwork
from agentarena.models.training import MLAgentConfig
from agentarena.training.demo_collection import DemonstrationDataset


class ImitationLearner:

    def __init__(
        self,
        ml_config: MLAgentConfig,
        demonstrations_dir: str = "demonstrations",

        device: torch.device | None = None,
    ):
        self.ml_config = ml_config
        self.demonstrations_dir = demonstrations_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from agentarena.agent.ml_agent import STATE_SIZE

        self.state_size = STATE_SIZE
        self.n_actions = 18

        self.policy_net = PolicyNetwork(input_size=self.state_size, output_size=self.n_actions).to(
            self.device,
        )

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.ml_config.learning_rate)

        self.criterion = nn.CrossEntropyLoss()

        print(f"ImitationLearner initialized on {self.device}")
        print(f"Network: {self.state_size} -> {self.n_actions}")

    def pretrain(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.2,
        save_path: str | None = None,
        tensorboard_dir: str | None = None,
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

        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.3f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.3f}",
            )

            if writer:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
                writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if save_path:
                    self._save_model(save_path)
                    print(f"New best model saved with validation accuracy: {val_accuracy:.3f}")
        if writer:
            writer.close()

        print(f"Pre-training completed. Best validation accuracy: {best_val_accuracy:.3f}")
        return best_val_accuracy

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch."""
        self.policy_net.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (states, actions) in enumerate(train_loader):
            states = states.to(self.device)
            actions = actions.to(self.device)

            action_indices = torch.argmax(actions, dim=1)

            self.optimizer.zero_grad()
            outputs = self.policy_net(states)
            loss = self.criterion(outputs, action_indices)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == action_indices).sum().item()
            total_samples += action_indices.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate for one epoch."""
        self.policy_net.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                action_indices = torch.argmax(actions, dim=1)
                outputs = self.policy_net(states)
                loss = self.criterion(outputs, action_indices)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == action_indices).sum().item()
                total_samples += action_indices.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def _save_model(self, save_path: str) -> None:
        """Save the pre-trained model."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "state_size": self.state_size,
                "n_actions": self.n_actions,
                "ml_config": self.ml_config.model_dump(),
            },
            save_path,
        )

    def load_pretrained_into_agent(self, agent: MLAgent) -> None:
        agent.policy_net.load_state_dict(self.policy_net.state_dict())
        print("Pre-trained weights loaded into MLAgent")


def pretrain_agent(
    demonstrations_dir: str = "demonstrations",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = "models/pretrained_agent.pt",
    tensorboard_dir: str = "runs/pretraining",
) -> str:
    ml_config = MLAgentConfig(learning_rate=learning_rate)

    learner = ImitationLearner(ml_config, demonstrations_dir)
    final_accuracy = learner.pretrain(
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        tensorboard_dir=tensorboard_dir,
    )


    if final_accuracy > 0:
        print(f"Pre-training successful! Model saved to {save_path}")
        return save_path
    else:
        print("Pre-training failed!")
        return ""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pre-train ML agent using demonstrations",
    )
    parser.add_argument(
        "--demonstrations-dir",
        default="demonstrations",
        help="Directory containing demonstration files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--save-path",
        default="models/pretrained_agent.pt",
        help="Path to save pre-trained model",

    )
    parser.add_argument(
        "--tensorboard-dir",
        default="runs/pretraining",
        help="TensorBoard log directory",
    )

    args = parser.parse_args()
    Path(args.save_path).parent.mkdir(exist_ok=True)
    pretrain_agent(
        demonstrations_dir=args.demonstrations_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
        tensorboard_dir=args.tensorboard_dir,
    )
