#!/usr/bin/env python3
"""
Model vs Model training script for AgentArena.

This script allows training one ML model against another ML model,
creating a competitive training environment.

Usage:
    python scripts/train_model_vs_model.py \
        --student-model models/student.pt \
        --teacher-model models/teacher.pt \
        --episodes 5000 \
        --reward-type advanced
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pygame
import torch
from torch.utils.tensorboard import SummaryWriter

from agentarena.agent.ml_agent import MLAgent
from agentarena.game.game import Game
from agentarena.models.config import load_config
from agentarena.models.events import EnemyHitEvent, PlayerHitEvent
from agentarena.models.training import EpisodeResult, MLAgentConfig, TrainingConfig
from agentarena.training.reward_functions import RewardType, calculate_reward

if TYPE_CHECKING:
    from agentarena.models.observations import GameObservation


class ModelVsModelTrainer:
    """Trainer for model vs model competitive learning."""

    def __init__(
        self,
        training_config: TrainingConfig,
        student_model_path: str | None = None,
        teacher_model_path: str | None = None,
    ) -> None:
        self.training_config = training_config
        self.student_model_path = student_model_path
        self.teacher_model_path = teacher_model_path

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
                (self.game_config.display_width, self.game_config.display_height)
            )
            pygame.display.set_caption(
                f"AgentArena: Model vs Model Training - {training_config.reward_type.value}"
            )

        self.clock = pygame.time.Clock()

        # Initialize agents
        self._setup_agents()

        # Setup logging
        self._setup_logging()

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilons = []
        self.episode_details = []
        self.student_wins = 0
        self.teacher_wins = 0
        self.best_reward = float("-inf")

    def _setup_agents(self) -> None:
        """Initialize student and teacher agents."""
        print("🤖 Setting up agents...")

        # Create student agent (the one being trained)
        self.student_agent = MLAgent(
            name="Student",
            is_training=True,
            config=self.training_config.ml_config,
        )
        self.student_agent.set_training_mode("q_learning")

        # Load student model if provided
        if self.student_model_path and Path(self.student_model_path).exists():
            print(f"📚 Loading student model from {self.student_model_path}")
            try:
                self.student_agent.load_model(self.student_model_path)
                # Reduce initial epsilon since we start with a good policy
                self.student_agent.epsilon = min(self.student_agent.epsilon, 0.3)
                print(f"✅ Student model loaded! Epsilon reduced to {self.student_agent.epsilon}")
            except Exception as e:
                print(f"❌ Error loading student model: {e}")
                print("Continuing with random initialization...")
        else:
            print("🎲 Student agent starting with random initialization")

        # Create teacher agent (opponent, not being trained)
        self.teacher_agent = MLAgent(
            name="Teacher",
            is_training=False,  # Teacher doesn't learn during training
            config=self.training_config.ml_config,
        )
        self.teacher_agent.set_training_mode("q_learning")

        # Load teacher model if provided
        if self.teacher_model_path and Path(self.teacher_model_path).exists():
            print(f"🎓 Loading teacher model from {self.teacher_model_path}")
            try:
                self.teacher_agent.load_model(self.teacher_model_path)
                self.teacher_agent.epsilon = 0.1  # Low exploration for teacher
                print(f"✅ Teacher model loaded! Epsilon set to {self.teacher_agent.epsilon}")
            except Exception as e:
                print(f"❌ Error loading teacher model: {e}")
                print("Teacher will use random initialization...")
        else:
            print("❌ No teacher model path provided or file doesn't exist!")
            raise ValueError("Teacher model is required for model vs model training")

    def _setup_logging(self) -> None:
        """Setup tensorboard logging and model directories."""
        # Prepare model directory
        self.training_config.models_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup TensorBoard writer
        tensorboard_dir = Path("runs")
        tensorboard_dir.mkdir(exist_ok=True)
        tensorboard_run_dir = (
            tensorboard_dir / f"model_vs_model_{self.training_config.model_name}_{self.timestamp}"
        )
        self.writer = SummaryWriter(tensorboard_run_dir)
        print(f"📊 TensorBoard logs: {tensorboard_run_dir}")

    def train(self) -> MLAgent | None:
        """Main training loop for model vs model training."""
        print("🚀 Starting model vs model training...")
        print(f"Student: {self.student_model_path or 'Random Init'}")
        print(f"Teacher: {self.teacher_model_path}")
        print(f"Episodes: {self.training_config.episodes}")
        print(f"Reward Type: {self.training_config.reward_type.value}")
        print("-" * 60)

        try:
            for episode in range(1, self.training_config.episodes + 1):
                # Randomly assign roles (student as player or enemy)
                student_as_player = episode % 2 == 1  # Alternate roles each episode

                if student_as_player:
                    player_agent = self.student_agent
                    enemy_agent = self.teacher_agent
                    training_student = True
                else:
                    player_agent = self.teacher_agent
                    enemy_agent = self.student_agent
                    training_student = True  # We still train student even when it's the "enemy"

                # Create game with current role assignment
                game = Game(self.screen, player_agent, enemy_agent, self.clock, self.game_config)

                # Run episode
                episode_reward, episode_length, student_won = self._run_episode(
                    game, episode, student_as_player, training_student
                )

                # Update statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.epsilons.append(self.student_agent.epsilon)

                if student_won:
                    self.student_wins += 1
                else:
                    self.teacher_wins += 1

                # Log progress
                self._log_episode_progress(
                    episode, episode_reward, episode_length, student_won, student_as_player
                )

                # Save model periodically
                if episode % self.training_config.save_frequency == 0:
                    self._save_checkpoint(episode)

                # Save best model based on recent performance
                self._update_best_model(episode_reward)

        except KeyboardInterrupt:
            print("Training interrupted by user")

        finally:
            self._cleanup_and_save()

        return self.student_agent

    def _run_episode(
        self, game: Game, episode: int, student_as_player: bool, training_student: bool
    ) -> tuple[float, int, bool]:
        """Run a single training episode."""
        game.reset()
        episode_reward = 0.0
        step = 0
        enemy_hits = 0
        player_hits = 0
        episode_events = []
        previous_observation = None

        # Determine which agent to train and get rewards for
        training_agent = self.student_agent if training_student else None

        while game.running and step < self.training_config.max_steps_per_episode:
            # Handle pygame events if rendering
            if self.training_config.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return episode_reward, step, False

            # Get current observation
            current_observation = game.get_observation("player")

            # Game update
            game.update()

            # Get new observation after update
            next_observation = game.get_observation("player")

            # Calculate reward for the student agent
            if training_student:
                # Adjust reward perspective based on role
                if student_as_player:
                    # Student is player, use normal rewards
                    reward = calculate_reward(
                        events=game.events,
                        observation=next_observation,
                        previous_observation=previous_observation,
                        reward_type=self.training_config.reward_type,
                    )
                else:
                    # Student is enemy, flip perspective for rewards
                    reward = self._calculate_enemy_perspective_reward(
                        game.events, next_observation, previous_observation
                    )

                episode_reward += reward

                # Train the student agent
                if student_as_player:
                    # Student is player, train normally
                    self.student_agent.learn(next_observation, reward, not game.running)
                else:
                    # Student is enemy, train with enemy observation
                    enemy_observation = game.get_observation("enemy_0")
                    self.student_agent.learn(enemy_observation, reward, not game.running)

            # Count hits for statistics
            for event in game.events:
                if isinstance(event, PlayerHitEvent):
                    player_hits += 1
                elif isinstance(event, EnemyHitEvent):
                    enemy_hits += 1

            episode_events.extend([event.model_dump() for event in game.events])
            previous_observation = current_observation

            if self.training_config.render:
                self.clock.tick(self.game_config.fps)

            step += 1

        # Determine who won
        if student_as_player:
            student_won = len(game.enemies) == 0  # Student wins if all enemies defeated
        else:
            student_won = game.player and game.player.health <= 0  # Student wins if player dies

        # Create episode result for detailed tracking
        episode_result = EpisodeResult(
            episode_id=episode,
            total_reward=episode_reward,
            episode_length=step,
            win=student_won,
            player_health_remaining=game.player.health if game.player else 0,
            enemies_defeated=self.game_config.max_enemies - len(game.enemies),
            accuracy=0.0,  # TODO: Calculate accuracy
            events=episode_events,
        )
        self.episode_details.append(episode_result)

        return episode_reward, step, student_won

    def _calculate_enemy_perspective_reward(
        self, events, observation, previous_observation
    ) -> float:
        """Calculate reward from enemy perspective (flip player/enemy rewards)."""
        reward = 0.0

        # Flip the rewards - what's good for enemies is bad for player
        for event in events:
            if isinstance(event, EnemyHitEvent):
                reward -= 2.0  # Enemy getting hit is bad
            elif isinstance(event, PlayerHitEvent):
                reward += 2.0  # Player getting hit is good
            # Add more enemy-perspective rewards as needed

        return reward

    def _log_episode_progress(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        student_won: bool,
        student_as_player: bool,
    ) -> None:
        """Log progress for the current episode."""
        # Calculate recent performance metrics
        window_size = min(episode, 100)
        avg_reward = sum(self.episode_rewards[-window_size:]) / window_size

        # Calculate win rates
        recent_episodes = self.episode_details[-window_size:]
        student_win_rate = sum(1 for ep in recent_episodes if ep.win) / len(recent_episodes)

        # Log to tensorboard
        self.writer.add_scalar("Reward/Episode", episode_reward, episode)
        self.writer.add_scalar("Reward/Average", avg_reward, episode)
        self.writer.add_scalar("Length/Episode", episode_length, episode)
        self.writer.add_scalar("Exploration/Epsilon", self.student_agent.epsilon, episode)
        self.writer.add_scalar("Performance/Student_Win_Rate", student_win_rate, episode)
        self.writer.add_scalar("Performance/Total_Student_Wins", self.student_wins, episode)
        self.writer.add_scalar("Performance/Total_Teacher_Wins", self.teacher_wins, episode)

        # Log learning rate
        if hasattr(self.student_agent, "scheduler"):
            current_lr = self.student_agent.scheduler.get_last_lr()[0]
            self.writer.add_scalar("Training/Learning_Rate", current_lr, episode)

        # Print progress
        role = "Player" if student_as_player else "Enemy"
        result = "WON" if student_won else "LOST"

        print(
            f"Episode {episode}/{self.training_config.episodes} [{role}] - "
            f"{result} - Steps: {episode_length} - Reward: {episode_reward:.2f} - "
            f"Avg: {avg_reward:.2f} - Win Rate: {student_win_rate:.1%} - "
            f"Epsilon: {self.student_agent.epsilon:.4f} - "
            f"Student: {self.student_wins} | Teacher: {self.teacher_wins}"
        )

    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        model_path = (
            self.training_config.models_dir
            / f"student_vs_teacher_{self.training_config.model_name}_{self.timestamp}_ep{episode}.pt"
        )
        self.student_agent.save_model(model_path)
        print(f"💾 Checkpoint saved: {model_path}")

        # Save training results
        self._save_training_results(episode)

    def _update_best_model(self, episode_reward: float) -> None:
        """Update best model if current performance is better."""
        # Use recent average for best model detection
        if len(self.episode_rewards) >= 100:
            avg_reward = sum(self.episode_rewards[-100:]) / 100
        else:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            best_model_path = (
                self.training_config.models_dir
                / f"student_vs_teacher_{self.training_config.model_name}_{self.timestamp}_best.pt"
            )
            self.student_agent.save_model(best_model_path)
            print(
                f"🏆 New best model saved: {best_model_path} (avg reward: {self.best_reward:.2f})"
            )

    def _save_training_results(self, episode: int) -> None:
        """Save detailed training results."""
        self.training_config.results_dir.mkdir(exist_ok=True)
        results_file = (
            self.training_config.results_dir
            / f"model_vs_model_{self.training_config.model_name}_{self.timestamp}.pkl"
        )

        results_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "epsilons": self.epsilons,
            "reward_type": self.training_config.reward_type.value,
            "timestamp": self.timestamp,
            "episodes_completed": episode,
            "ml_config": self.training_config.ml_config.model_dump(),
            "episode_details": [ep.model_dump() for ep in self.episode_details],
            "student_wins": self.student_wins,
            "teacher_wins": self.teacher_wins,
            "student_model_path": self.student_model_path,
            "teacher_model_path": self.teacher_model_path,
        }

        with results_file.open("wb") as f:
            pickle.dump(results_data, f)

    def _cleanup_and_save(self) -> None:
        """Cleanup and save final results."""
        # Save final model
        final_model_path = (
            self.training_config.models_dir
            / f"student_vs_teacher_{self.training_config.model_name}_{self.timestamp}_final.pt"
        )
        self.student_agent.save_model(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")

        # Save final results
        total_episodes = len(self.episode_rewards)
        self._save_training_results(total_episodes)

        # Close tensorboard writer
        self.writer.close()

        if self.training_config.render:
            pygame.quit()

        # Print final statistics
        print("\n" + "=" * 60)
        print("🏁 TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Total episodes: {total_episodes}")
        print(
            f"Student wins: {self.student_wins} ({self.student_wins / total_episodes * 100:.1f}%)"
        )
        print(
            f"Teacher wins: {self.teacher_wins} ({self.teacher_wins / total_episodes * 100:.1f}%)"
        )

        if self.episode_rewards:
            avg_final_reward = sum(
                self.episode_rewards[-min(100, len(self.episode_rewards)) :]
            ) / min(100, len(self.episode_rewards))
            print(f"Final average reward (last 100 episodes): {avg_final_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train ML agent against another ML agent")

    # Model paths
    parser.add_argument(
        "--student-model",
        help="Path to student model (the one being trained). Optional - will use random init if not provided",
    )
    parser.add_argument(
        "--teacher-model", required=True, help="Path to teacher model (opponent). Required."
    )

    # Training parameters
    parser.add_argument("--model-name", default="student", help="Name prefix for saved models")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    parser.add_argument("--render", action="store_true", help="Render the game during training")
    parser.add_argument("--save-freq", type=int, default=100, help="How often to save the model")

    # Reward and ML parameters
    parser.add_argument(
        "--reward-type",
        choices=[r.value for r in RewardType],
        default=RewardType.ADVANCED.value,
        help="Reward function type",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")

    args = parser.parse_args()

    # Validate teacher model exists
    if not Path(args.teacher_model).exists():
        print(f"❌ Error: Teacher model file not found: {args.teacher_model}")
        return

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

    # Start training
    print("🤖 Starting Model vs Model Training")
    print(f"Student Model: {args.student_model or 'Random Initialization'}")
    print(f"Teacher Model: {args.teacher_model}")
    print(f"Episodes: {args.episodes}")
    print(f"Reward Type: {args.reward_type}")
    print()

    trainer = ModelVsModelTrainer(
        training_config=training_config,
        student_model_path=args.student_model,
        teacher_model_path=args.teacher_model,
    )

    trainer.train()


if __name__ == "__main__":
    main()
