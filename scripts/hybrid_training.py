import argparse  # noqa: INP001
import pickle
import random
from datetime import datetime
from pathlib import Path

import pygame
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from agentarena.agent.ml_agent import MLAgent
from agentarena.agent.random_agent import RandomAgent
from agentarena.agent.rule_based_agent import RuleBasedAgent
from agentarena.agent.rule_based_agent_2 import RuleBasedAgent2
from agentarena.models.config import load_config
from agentarena.models.events import EnemyHitEvent
from agentarena.models.training import EpisodeResult, MLAgentConfig, TrainingConfig
from agentarena.training.demo_collection import DemonstrationDataset
from agentarena.training.reward_functions import RewardType, calculate_reward


class HybridTrainingConfig:
    """Configuration for hybrid training schedule."""

    def __init__(
        self,
        total_episodes: int = 5000,
        initial_offline_ratio: float = 0.8,  # Start with 80% offline, 20% online
        final_offline_ratio: float = 0.2,  # End with 20% offline, 80% online
        phase_length: int = 50,  # Episodes per phase
        offline_batch_size: int = 64,
        offline_epochs_per_episode: int = 1,  # How many epochs to run per offline episode
        demonstrations_dir: str = "demonstrations",
        decay_type: str = "linear",  # "linear", "exponential", or "cosine"
    ) -> None:
        self.total_episodes = total_episodes
        self.initial_offline_ratio = initial_offline_ratio
        self.final_offline_ratio = final_offline_ratio
        self.phase_length = phase_length
        self.offline_batch_size = offline_batch_size
        self.offline_epochs_per_episode = offline_epochs_per_episode
        self.demonstrations_dir = demonstrations_dir
        self.decay_type = decay_type

    def get_offline_ratio(self, episode: int) -> float:
        """Calculate the offline training ratio for the current episode."""
        progress = min(episode / self.total_episodes, 1.0)

        if self.decay_type == "linear":
            ratio = self.initial_offline_ratio + progress * (
                self.final_offline_ratio - self.initial_offline_ratio
            )
        elif self.decay_type == "exponential":
            # Exponential decay from initial to final ratio
            decay_rate = -torch.log(
                torch.tensor(self.final_offline_ratio / self.initial_offline_ratio),
            )
            ratio = self.initial_offline_ratio * torch.exp(-decay_rate * progress).item()
        elif self.decay_type == "cosine":
            # Cosine annealing schedule
            ratio = (
                self.final_offline_ratio
                + 0.5
                * (self.initial_offline_ratio - self.final_offline_ratio)
                * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
            )
        else:
            msg = f"Unknown decay type: {self.decay_type}"
            raise ValueError(msg)

        return max(0.0, min(1.0, ratio))

    def should_do_offline_training(self, episode: int) -> bool:
        """Determine if current episode should use offline training."""
        phase_episode = episode % self.phase_length
        offline_ratio = self.get_offline_ratio(episode)
        offline_episodes_in_phase = int(self.phase_length * offline_ratio)

        return phase_episode < offline_episodes_in_phase


class HybridTrainer:
    """Trainer that alternates between offline and online training."""

    def __init__(
        self,
        training_config: TrainingConfig,
        hybrid_config: HybridTrainingConfig,
        pretrained_model_path: str | None = None,
    ) -> None:
        self.training_config = training_config
        self.hybrid_config = hybrid_config
        self.pretrained_model_path = pretrained_model_path

        # Initialize pygame if rendering
        if training_config.render:
            pygame.init()

        # Load game configuration
        self.game_config = load_config()
        if not training_config.render:
            self.game_config.headless = True
            self.game_config.fps = 0  # Uncapped FPS for headless mode

        # Initialize screen if rendering
        self.screen = None
        if training_config.render:
            self.screen = pygame.display.set_mode(
                (self.game_config.display_width, self.game_config.display_height),
            )
            pygame.display.set_caption("AgentArena Hybrid Training")

        self.clock = pygame.time.Clock()

        # Initialize agents
        self.player_agent = MLAgent(
            is_training=True,
            config=training_config.ml_config,
        )
        self.enemy_agent = RuleBasedAgent2()

        # Load pretrained model if provided
        self._load_pretrained_model()

        # Initialize demonstration dataset for offline training
        self._setup_demonstration_data()

        # Setup tensorboard logging
        self._setup_logging()

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilons = []
        self.episode_details = []
        self.offline_losses = []
        self.best_reward = float("-inf")

        # Hybrid training statistics
        self.offline_episodes = 0
        self.online_episodes = 0

    def _load_pretrained_model(self) -> None:
        """Load pretrained model if provided."""
        if self.pretrained_model_path and Path(self.pretrained_model_path).exists():
            print(f"Loading pre-trained model from {self.pretrained_model_path}")
            try:
                self.player_agent.load_model(self.pretrained_model_path)
                # Reduce initial epsilon since we start with a good policy
                self.player_agent.epsilon = min(self.player_agent.epsilon, 0.3)
                print(
                    f"✅ Pre-trained model loaded! Epsilon reduced to {self.player_agent.epsilon}",
                )
            except Exception as e:
                print(f"❌ Error loading pre-trained model: {e}")
                print("Continuing with random initialization...")

    def _setup_demonstration_data(self) -> None:
        """Setup demonstration dataset for offline training."""
        try:
            self.demo_dataset = DemonstrationDataset(self.hybrid_config.demonstrations_dir)
            if len(self.demo_dataset) > 0:
                self.demo_dataloader = DataLoader(
                    self.demo_dataset,
                    batch_size=self.hybrid_config.offline_batch_size,
                    shuffle=True,
                )
                print(
                    f"✅ Loaded {len(self.demo_dataset)} "
                    "demonstration samples for offline training",
                )
            else:
                print("⚠️  No demonstration data found. Offline training will be skipped.")
                self.demo_dataset = None
                self.demo_dataloader = None
        except Exception as e:
            print(f"❌ Error loading demonstration data: {e}")
            self.demo_dataset = None
            self.demo_dataloader = None

    def _setup_logging(self) -> None:
        """Setup tensorboard logging."""
        self.training_config.models_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup TensorBoard writer
        tensorboard_dir = Path("runs")
        tensorboard_dir.mkdir(exist_ok=True)
        tensorboard_run_dir = (
            tensorboard_dir / f"hybrid_{self.training_config.model_name}_{self.timestamp}"
        )
        self.writer = SummaryWriter(tensorboard_run_dir)
        print(f"TensorBoard logs: {tensorboard_run_dir}")

    def train(self) -> MLAgent | None:
        """Main hybrid training loop."""
        print("🚀 Starting hybrid training...")
        print(f"Initial offline ratio: {self.hybrid_config.initial_offline_ratio:.1%}")
        print(f"Final offline ratio: {self.hybrid_config.final_offline_ratio:.1%}")
        print(f"Phase length: {self.hybrid_config.phase_length} episodes")

        try:
            for episode in range(1, self.training_config.episodes + 1):
                # Determine training mode for this episode
                use_offline = self.hybrid_config.should_do_offline_training(episode)
                current_offline_ratio = self.hybrid_config.get_offline_ratio(episode)

                if use_offline and self.demo_dataloader is not None:
                    # Offline training episode
                    try:
                        self._train_offline_episode(episode)
                        self.offline_episodes += 1
                        training_mode = "Offline"
                    except Exception as e:
                        print(f"Error in offline training episode {episode}: {e}")
                        print("Falling back to online training for this episode")
                        episode_reward, episode_length = self._train_online_episode(episode)
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.online_episodes += 1
                        training_mode = "Online (Fallback)"
                else:
                    # Online training episode
                    try:
                        episode_reward, episode_length = self._train_online_episode(episode)
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.online_episodes += 1
                        training_mode = "Online"
                    except Exception as e:
                        print(f"Error in online training episode {episode}: {e}")
                        continue

                # Log progress
                self._log_episode_progress(episode, training_mode, current_offline_ratio)

                # Save model periodically
                if episode % self.training_config.save_frequency == 0:
                    self._save_checkpoint(episode)

                # Save best model based on online performance
                if self.episode_rewards and training_mode.startswith("Online"):
                    avg_reward = sum(
                        self.episode_rewards[-min(100, len(self.episode_rewards)) :],
                    ) / min(100, len(self.episode_rewards))
                    if avg_reward > self.best_reward:
                        self.best_reward = avg_reward
                        self._save_best_model()

        except KeyboardInterrupt:
            print("Training interrupted by user")

        finally:
            self._cleanup_and_save()

        return self.player_agent

    def _train_offline_episode(self, episode: int) -> None:
        self.player_agent.set_training_mode("imitation")
        self.player_agent.policy_net.train()

        # Get the device from the model
        device = next(self.player_agent.policy_net.parameters()).device

        # Run multiple epochs of demonstration data
        total_episode_loss = 0.0
        total_episode_accuracy = 0.0
        total_batches_processed = 0

        for epoch in range(self.hybrid_config.offline_epochs_per_episode):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            for _batch_idx, (states, actions) in enumerate(self.demo_dataloader):
                states = states.to(device)  # noqa: PLW2901
                actions = actions.to(device)  # noqa: PLW2901

                action_logits = self.player_agent.policy_net.get_action_logits(states)
                target_indices = torch.argmax(actions, dim=1).to(device)

                loss = torch.nn.functional.cross_entropy(action_logits, target_indices)

                self.player_agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.player_agent.policy_net.parameters(),
                    max_norm=1.0,
                )
                self.player_agent.optimizer.step()
                self.player_agent.scheduler.step()

                epoch_loss += loss.item()
                with torch.no_grad():
                    predicted = torch.argmax(action_logits, dim=1)
                    batch_accuracy = (predicted == target_indices).float().mean().item()
                    epoch_accuracy += batch_accuracy

                num_batches += 1
                total_batches_processed += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_epoch_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0

            total_episode_loss += avg_epoch_loss
            total_episode_accuracy += avg_epoch_accuracy

            if self.hybrid_config.offline_epochs_per_episode > 1:
                print(
                    f"    Epoch {epoch + 1}/{self.hybrid_config.offline_epochs_per_episode}"
                    f" - Loss: {avg_epoch_loss:.4f}, Acc: {avg_epoch_accuracy:.3f},"
                    f" - lr: {self.player_agent.learning_rate}",
                )
                self.writer.add_scalar(f"Loss/Offline_Epoch_{epoch + 1}", avg_epoch_loss, episode)
                self.writer.add_scalar(
                    f"Accuracy/Offline_Epoch_{epoch + 1}",
                    avg_epoch_accuracy,
                    episode,
                )

        # Calculate episode averages across all epochs
        avg_episode_loss = total_episode_loss / self.hybrid_config.offline_epochs_per_episode
        avg_episode_accuracy = (
            total_episode_accuracy / self.hybrid_config.offline_epochs_per_episode
        )

        # Store loss for logging
        self.offline_losses.append(avg_episode_loss)

        # Log to tensorboard
        self.writer.add_scalar("Loss/Offline_Episode", avg_episode_loss, episode)
        self.writer.add_scalar("Accuracy/Offline_Episode", avg_episode_accuracy, episode)
        self.writer.add_scalar(
            "Training/Offline_Batches_Per_Episode",
            total_batches_processed,
            episode,
        )
        self.writer.add_scalar(
            "Training/Offline_Epochs_Per_Episode",
            self.hybrid_config.offline_epochs_per_episode,
            episode,
        )

        print(
            f"  Offline Ep {episode}: {self.hybrid_config.offline_epochs_per_episode} epochs, "
            f"{total_batches_processed} batches "
            f"- Avg Loss: {avg_episode_loss:.4f}, Avg Acc: {avg_episode_accuracy:.3f}",
        )

    def _train_online_episode(self, episode: int) -> tuple[float, int]:
        """Train one episode using online reinforcement learning."""
        from agentarena.game.game import Game  # noqa: PLC0415

        self.player_agent.set_training_mode("q_learning")
        self.enemy_agent = random.choice([RuleBasedAgent2(), RuleBasedAgent(), RandomAgent()])

        game = Game(self.screen, self.player_agent, self.enemy_agent, self.clock, self.game_config)
        game.reset()

        episode_reward = 0.0
        step = 0
        enemy_hits = 0
        episode_events = []
        previous_observation = None

        while game.running and step < self.training_config.max_steps_per_episode:
            if self.training_config.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return episode_reward, step

            current_observation = game.get_observation("player")

            game.update()

            next_observation = game.get_observation("player")

            reward = calculate_reward(
                events=game.events,
                observation=next_observation,
                previous_observation=previous_observation,
                reward_type=self.training_config.reward_type,
            )

            for event in game.events:
                if isinstance(event, EnemyHitEvent):
                    enemy_hits += 1

            episode_reward += reward
            episode_events.extend([event.model_dump() for event in game.events])
            self.player_agent.learn(next_observation, reward, not game.running)
            previous_observation = current_observation

            if self.training_config.render:
                self.clock.tick(self.game_config.fps)

            step += 1

        episode_result = EpisodeResult(
            episode_id=episode,
            total_reward=episode_reward,
            episode_length=step,
            win=len(game.enemies) == 0,
            player_health_remaining=game.player.health if game.player else 0,
            enemies_defeated=self.game_config.max_enemies - len(game.enemies),
            accuracy=0.0,  # TODO: Calculate accuracy
            events=episode_events,
        )
        self.episode_details.append(episode_result)

        self.epsilons.append(self.player_agent.epsilon)

        self.writer.add_scalar("Reward/Episode", episode_reward, episode)
        self.writer.add_scalar("Length/Episode", step, episode)
        self.writer.add_scalar("Exploration/Epsilon", self.player_agent.epsilon, episode)
        self.writer.add_scalar("Performance/Enemy_Hits", enemy_hits, episode)

        if len(self.episode_rewards) >= 1:
            window_size = min(len(self.episode_rewards) + 1, 100)
            recent_rewards = [*self.episode_rewards[-(window_size - 1) :], episode_reward]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            self.writer.add_scalar("Reward/Average", avg_reward, episode)

            # Win rate
            recent_episodes = [*self.episode_details[-(window_size - 1) :], episode_result]
            win_rate = sum(1 for ep in recent_episodes if ep.win) / len(recent_episodes)
            self.writer.add_scalar("Performance/Win_Rate", win_rate, episode)

        return episode_reward, step

    def _log_episode_progress(self, episode: int, training_mode: str, offline_ratio: float) -> None:
        """Log progress for the current episode."""
        # Calculate recent performance metrics
        recent_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
        avg_reward = 0.0
        win_rate = 0.0
        avg_offline_loss = 0.0

        if self.episode_rewards:
            window_size = min(len(self.episode_rewards), 100)
            avg_reward = sum(self.episode_rewards[-window_size:]) / window_size

        if self.episode_details:
            window_size = min(len(self.episode_details), 100)
            win_rate = sum(1 for ep in self.episode_details[-window_size:] if ep.win) / window_size

        if self.offline_losses:
            window_size = min(len(self.offline_losses), 20)
            avg_offline_loss = sum(self.offline_losses[-window_size:]) / window_size

        # Log hybrid training metrics
        self.writer.add_scalar("Hybrid/Offline_Ratio", offline_ratio, episode)
        self.writer.add_scalar("Hybrid/Offline_Episodes", self.offline_episodes, episode)
        self.writer.add_scalar("Hybrid/Online_Episodes", self.online_episodes, episode)

        # Print progress
        if training_mode == "Online":
            print(
                f"Episode {episode}/{self.training_config.episodes} [{training_mode}] - "
                f"Reward: {recent_reward:.2f} - Avg: {avg_reward:.2f} - "
                f"Win Rate: {win_rate:.1%} - Epsilon: {self.player_agent.epsilon:.4f} - "
                f"Offline Ratio: {offline_ratio:.1%}",
            )
        else:
            print(
                f"Episode {episode}/{self.training_config.episodes} [{training_mode}] - "
                f"Loss: {avg_offline_loss:.4f} - "
                f"Offline Ratio: {offline_ratio:.1%}",
            )

    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        model_path = (
            self.training_config.models_dir
            / f"hybrid_{self.training_config.model_name}_{self.timestamp}_ep{episode}.pt"
        )
        self.player_agent.save_model(model_path)
        print(f"Checkpoint saved: {model_path}")

        # Save training results
        self._save_training_results(episode)

    def _save_best_model(self) -> None:
        """Save the best performing model."""
        best_model_path = (
            self.training_config.models_dir
            / f"hybrid_{self.training_config.model_name}_{self.timestamp}_best.pt"
        )
        self.player_agent.save_model(best_model_path)
        print(f"New best model saved: {best_model_path} (avg reward: {self.best_reward:.2f})")

    def _save_training_results(self, episode: int) -> None:
        """Save detailed training results."""
        self.training_config.results_dir.mkdir(exist_ok=True)
        results_file = (
            self.training_config.results_dir
            / f"hybrid_{self.training_config.model_name}_{self.timestamp}.pkl"
        )

        # Create extended results with hybrid training info
        results_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "epsilons": self.epsilons,
            "offline_losses": self.offline_losses,
            "reward_type": self.training_config.reward_type.value,
            "timestamp": self.timestamp,
            "episodes_completed": episode,
            "ml_config": self.training_config.ml_config.model_dump(),
            "episode_details": [ep.model_dump() for ep in self.episode_details],
            "hybrid_config": {
                "initial_offline_ratio": self.hybrid_config.initial_offline_ratio,
                "final_offline_ratio": self.hybrid_config.final_offline_ratio,
                "phase_length": self.hybrid_config.phase_length,
                "decay_type": self.hybrid_config.decay_type,
            },
            "offline_episodes": self.offline_episodes,
            "online_episodes": self.online_episodes,
        }

        with results_file.open("wb") as f:
            pickle.dump(results_data, f)

    def _cleanup_and_save(self) -> None:
        """Cleanup and save final results."""
        # Save final model
        final_model_path = (
            self.training_config.models_dir
            / f"hybrid_{self.training_config.model_name}_{self.timestamp}_final.pt"
        )
        self.player_agent.save_model(final_model_path)
        print(f"Final model saved: {final_model_path}")

        # Save final results
        total_episodes = self.offline_episodes + self.online_episodes
        self._save_training_results(total_episodes)

        # Close tensorboard writer
        self.writer.close()

        if self.training_config.render:
            pygame.quit()

        print("Training completed!")
        print(f"Total episodes: {total_episodes}")
        print(f"Offline episodes: {self.offline_episodes}")
        print(f"Online episodes: {self.online_episodes}")

        if self.episode_rewards:
            avg_final_reward = sum(
                self.episode_rewards[-min(100, len(self.episode_rewards)) :],
            ) / min(100, len(self.episode_rewards))
            print(f"Final average reward (last 100 episodes): {avg_final_reward:.2f}")

        if self.offline_losses:
            avg_final_loss = sum(self.offline_losses[-min(20, len(self.offline_losses)) :]) / min(
                20,
                len(self.offline_losses),
            )
            print(f"Final average offline loss (last 20 batches): {avg_final_loss:.4f}")


def hybrid_train(
    training_config: TrainingConfig,
    hybrid_config: HybridTrainingConfig,
    pretrained_model_path: str | None = None,
) -> MLAgent | None:
    """Main function to start hybrid training."""
    trainer = HybridTrainer(training_config, hybrid_config, pretrained_model_path)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid training for AgentArena ML agent")

    # Standard training arguments
    parser.add_argument(
        "--model-name",
        default="hybrid_agent",
        help="Name prefix for the saved model",
    )
    parser.add_argument("--episodes", type=int, default=5000, help="Total number of episodes")
    parser.add_argument("--render", action="store_true", help="Render the game during training")
    parser.add_argument("--pretrained-model", help="Path to pre-trained model file")
    parser.add_argument("--save-freq", type=int, default=100, help="How often to save the model")
    parser.add_argument(
        "--reward-type",
        choices=[r.value for r in RewardType],
        default=RewardType.ADVANCED.value,
        help="Reward function type",
    )

    # ML config arguments
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")

    # Hybrid training specific arguments
    parser.add_argument(
        "--initial-offline-ratio",
        type=float,
        default=0.8,
        help="Initial ratio of offline training (0.0-1.0)",
    )
    parser.add_argument(
        "--final-offline-ratio",
        type=float,
        default=0.2,
        help="Final ratio of offline training (0.0-1.0)",
    )
    parser.add_argument(
        "--phase-length",
        type=int,
        default=50,
        help="Number of episodes per training phase",
    )
    parser.add_argument(
        "--offline-epochs-per-episode",
        type=int,
        default=1,
        help="Number of epochs through demonstration data per offline episode",
    )
    parser.add_argument(
        "--decay-type",
        choices=["linear", "exponential", "cosine"],
        default="linear",
        help="Type of schedule decay",
    )
    parser.add_argument(
        "--demonstrations-dir",
        default="demonstrations",
        help="Directory containing demonstration data",
    )
    parser.add_argument(
        "--offline-batch-size",
        type=int,
        default=64,
        help="Batch size for offline training",
    )

    args = parser.parse_args()

    # Create configurations
    ml_config = MLAgentConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    training_config = TrainingConfig(
        model_name=args.model_name,
        episodes=args.episodes,
        render=args.render,
        reward_type=RewardType(args.reward_type),
        save_frequency=args.save_freq,
        ml_config=ml_config,
    )

    hybrid_config = HybridTrainingConfig(
        total_episodes=args.episodes,
        initial_offline_ratio=args.initial_offline_ratio,
        final_offline_ratio=args.final_offline_ratio,
        phase_length=args.phase_length,
        offline_epochs_per_episode=args.offline_epochs_per_episode,
        decay_type=args.decay_type,
        demonstrations_dir=args.demonstrations_dir,
        offline_batch_size=args.offline_batch_size,
    )

    # Start hybrid training
    print("🔥 Starting hybrid training with the following schedule:")
    print(f"  Initial offline ratio: {hybrid_config.initial_offline_ratio:.1%}")
    print(f"  Final offline ratio: {hybrid_config.final_offline_ratio:.1%}")
    print(f"  Phase length: {hybrid_config.phase_length} episodes")
    print(f"  Decay type: {hybrid_config.decay_type}")
    print(f"  Total episodes: {training_config.episodes}")

    hybrid_train(training_config, hybrid_config, args.pretrained_model)
