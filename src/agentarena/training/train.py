"""
Training module for AgentArena ML agents.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pygame
import torch

# Add this import for TensorBoard
from torch.utils.tensorboard import SummaryWriter

from agentarena.agent.ml_agent import MLAgent
from agentarena.agent.random_agent import RandomAgent
from agentarena.models.config import load_config
from agentarena.models.events import PlayerHitEvent, EnemyHitEvent
from agentarena.models.observations import GameObservation
from agentarena.models.training import EpisodeResult, MLAgentConfig, TrainingConfig, TrainingResults
from agentarena.training.reward_functions import RewardType, calculate_reward


def train(
    config: TrainingConfig,
    pretrained_model_path: str = None,
) -> MLAgent | None:
    print(
        f"Starting ML agent training with {config.reward_type.value} reward function...",
    )
    if pretrained_model_path:
        print(f"Using pre-trained model: {pretrained_model_path}")

    if config.render:
        pygame.init()
    game_config = load_config()
    if not config.render:
        game_config.headless = True
    if not config.render:
        game_config.fps = 0  # Uncapped FPS for headless mode

    # Initialize screen if rendering
    screen = None
    if config.render:
        screen = pygame.display.set_mode(
            (game_config.display_width, game_config.display_height),
        )
        pygame.display.set_caption(
            f"AgentArena ML Training - {config.reward_type.value}",
        )

    # Create clock
    clock = pygame.time.Clock()

    # Create ML agent with config
    player_agent = MLAgent(
        is_training=True,
        config=config.ml_config,
    )
    enemy_agent = RandomAgent()
    if pretrained_model_path and Path(pretrained_model_path).exists():
        print(f"Loading pre-trained model from {pretrained_model_path}")
        try:
            checkpoint = torch.load(pretrained_model_path, map_location=torch.device("cpu"))
            if "policy_net_state_dict" in checkpoint:
                # New format with separate networks
                player_agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
                player_agent.target_net.load_state_dict(checkpoint["policy_net_state_dict"])
                print("✅ Pre-trained weights loaded successfully!")

                # Optionally reduce initial epsilon since we start with good policy
                player_agent.epsilon = min(player_agent.epsilon, 0.3)
                print(f"Reduced initial epsilon to {player_agent.epsilon} for pre-trained model")
            else:
                print("⚠️  Warning: Pre-trained model format not recognized")
        except Exception as e:
            print(f"❌ Error loading pre-trained model: {e}")
            print("Continuing with random initialization...")
    elif pretrained_model_path:
        print(f"⚠️  Pre-trained model file not found: {pretrained_model_path}")
        print("Continuing with random initialization...")

    # Load checkpoint if provided
    if config.checkpoint_path and config.checkpoint_path.exists():
        print(f"Loading model from {config.checkpoint_path}")
        player_agent = MLAgent(
            is_training=True,
            config=config.ml_config,  # Use the NEW config parameters
        )

        # Modified loading to only load weights, not hyperparameters
        player_agent.load_model_weights_only(config.checkpoint_path)

        print(
            f"Model loaded with new parameters: LR={config.ml_config.learning_rate}, "
            f"gamma={config.ml_config.gamma}, epsilon={config.ml_config.epsilon}"
        )

    # Import game here to avoid circular imports
    from agentarena.game.game import Game

    # Create game
    game = Game(screen, player_agent, enemy_agent, clock, game_config)

    # Prepare model directory
    config.models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup TensorBoard writer
    tensorboard_dir = Path("runs")
    tensorboard_dir.mkdir(exist_ok=True)
    tensorboard_run_dir = (
        tensorboard_dir / f"{config.model_name}_{timestamp}_{config.reward_type.value}"
    )
    writer = SummaryWriter(tensorboard_run_dir)
    print(f"TensorBoard logs will be saved to {tensorboard_run_dir}")

    # Training statistics
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    epsilons: list[float] = []  # Track epsilon values for visualization
    episode_details: list[EpisodeResult] = []
    best_reward = float("-inf")

    try:
        for episode in range(1, config.episodes + 1):
            # Reset the game
            game.reset()
            episode_reward = 0.0
            step = 0
            enemy_hits = 0

            # Track events for this episode
            episode_events = []

            # Initialize previous observation
            previous_observation: GameObservation | None = None

            # Run the episode
            while game.running and step < config.max_steps_per_episode:
                # Handle pygame events if rendering
                if config.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return None

                # Get the current state
                current_observation = game.get_observation("player")

                # Game update (get action from agent, apply it, etc.)
                game.update()

                # Get the new observation after the update
                next_observation = game.get_observation("player")

                # Calculate reward using the appropriate reward function
                reward = calculate_reward(
                    events=game.events,
                    observation=next_observation,
                    previous_observation=previous_observation,
                    reward_type=config.reward_type,
                )
                player_hits = 0

                for event in game.events:
                    if isinstance(event, PlayerHitEvent):
                        player_hits += 1
                    if isinstance(event, EnemyHitEvent):
                        enemy_hits += 1

                episode_reward += reward

                # Store events for logging
                episode_events.extend([event.model_dump() for event in game.events])

                # Learn from this step
                player_agent.learn(next_observation, reward, not game.running)

                # Update previous observation for next step
                previous_observation = current_observation

                # Limit rendering speed if needed
                if config.render:
                    clock.tick(game.config.fps)

                step += 1

            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

            # Create episode result for detailed tracking
            episode_result = EpisodeResult(
                episode_id=episode,
                total_reward=episode_reward,
                episode_length=step,
                win=len(game.enemies) == 0,  # Win if all enemies defeated
                player_health_remaining=game.player.health if game.player else 0,
                enemies_defeated=game.config.max_enemies - len(game.enemies),
                accuracy=0.0,  # TODO: Calculate accuracy
                events=episode_events,
            )
            episode_details.append(episode_result)

            # Calculate moving average
            window_size = min(episode, 100)
            avg_reward = sum(episode_rewards[-window_size:]) / window_size

            # Track epsilon for visualization
            epsilons.append(player_agent.epsilon)

            # Log metrics to TensorBoard
            writer.add_scalar("Reward/episode", episode_reward, episode)
            writer.add_scalar("Reward/average", avg_reward, episode)
            writer.add_scalar("Length/episode", step, episode)
            writer.add_scalar("Exploration/epsilon", player_agent.epsilon, episode)

            # Log win rate
            win_rate = sum(1 for ep in episode_details[-window_size:] if ep.win) / window_size
            writer.add_scalar("Performance/win_rate", win_rate, episode)
            writer.add_scalar("Performance/enemy_hits", enemy_hits, episode)
            writer.add_scalar("Performance/hits_per_step", enemy_hits / max(1, step), episode)

            # Log histogram of Q-values if available
            if hasattr(player_agent, "model") and step > 0:
                # Sample a batch of states for visualization
                if len(player_agent.memory) > 32:
                    sample = player_agent.memory.sample(32)
                    states = torch.FloatTensor([exp.state for exp in sample])
                    with torch.no_grad():
                        q_values = player_agent.model(states)
                        writer.add_histogram("Q-values", q_values.flatten(), episode)

            # Print progress
            print(
                f"Episode {episode}/{config.episodes} - Steps: {step} "
                f"- Reward: {episode_reward:.2f} "
                f"- Avg Reward: {avg_reward:.2f} "
                f"- Epsilon: {player_agent.epsilon:.4f}",
                f"- Learning Rate: {player_agent.scheduler.get_lr()[0]}",
                f"- Enemy Hits: {enemy_hits} ",  # Added this line
            )

            # Save model periodically
            if episode % config.save_frequency == 0:
                model_path = (
                    config.models_dir
                    / f"{config.model_name}_{timestamp}_{config.reward_type.value}_ep{episode}.pt"
                )
                player_agent.save_model(model_path)
                print(f"Model saved to {model_path}")

                # Also save training results
                _save_training_results(
                    config=config,
                    timestamp=timestamp,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    epsilons=epsilons,
                    episode_details=episode_details,
                    episodes_completed=episode,
                )

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model_path = (
                    config.models_dir
                    / f"{config.model_name}_{timestamp}_{config.reward_type.value}_best.pt"
                )
                player_agent.save_model(best_model_path)
                print(f"New best model saved with avg reward: {best_reward:.2f}")

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        # Save final model
        final_model_path = (
            config.models_dir
            / f"{config.model_name}_{timestamp}_{config.reward_type.value}_final.pt"
        )
        player_agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Save final training results
        _save_training_results(
            config=config,
            timestamp=timestamp,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            epsilons=epsilons,
            episode_details=episode_details,
            episodes_completed=len(episode_rewards),
        )

        # Close TensorBoard writer
        writer.close()
        print("TensorBoard writer closed")

        if config.render:
            pygame.quit()

    return player_agent


def _save_training_results(
    config: TrainingConfig,
    timestamp: str,
    episode_rewards: list[float],
    episode_lengths: list[int],
    epsilons: list[float],
    episode_details: list[EpisodeResult],
    episodes_completed: int,
) -> None:
    """
    Save training results to disk.

    Args:
        config: Training configuration
        timestamp: Timestamp string for filename
        episode_rewards: List of rewards for each episode
        episode_lengths: List of steps for each episode
        epsilons: List of epsilon values used
        episode_details: Detailed episode results
        episodes_completed: Number of episodes completed
    """
    # Create results directory if it doesn't exist
    config.results_dir.mkdir(exist_ok=True)

    # Create results file path
    results_file = (
        config.results_dir / f"{config.model_name}_{timestamp}_{config.reward_type.value}.pkl"
    )

    # Get ML config as a dict - this ensures all fields are properly serialized
    ml_config_dict = config.ml_config.model_dump()

    # Create training results model
    results = TrainingResults(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        epsilons=epsilons,
        reward_type=config.reward_type.value,
        timestamp=timestamp,
        episodes_completed=episodes_completed,
        ml_config=ml_config_dict,
        episode_details=episode_details,
    )

    # Save results to disk
    with results_file.open("wb") as f:
        pickle.dump(results.model_dump(), f)

    print(f"Training results saved to {results_file}")


def evaluate(
    model_path: Path,
    episodes: int = 10,
    render: bool = True,
) -> None:
    """
    Evaluate a trained ML agent.

    Args:
        model_path: Path to the model file
        episodes: Number of episodes to evaluate
        render: Whether to render the game
    """
    print(f"Evaluating ML agent from {model_path}...")

    # Initialize pygame if needed for rendering
    if render:
        pygame.init()

    # Load configuration
    config = load_config()

    # Set to headless mode if not rendering
    if not render:
        config.headless = True

    # Initialize screen if rendering
    screen = None
    if render:
        screen = pygame.display.set_mode(
            (config.display_width, config.display_height),
        )
        pygame.display.set_caption("AgentArena ML Evaluation")

    # Create clock
    clock = pygame.time.Clock()

    # Create agents
    player_agent = MLAgent(is_training=False)
    player_agent.load_model(model_path)
    enemy_agent = RandomAgent()

    # Import game here to avoid circular imports
    from agentarena.game.game import Game

    # Create game
    game = Game(screen, player_agent, enemy_agent, clock, config)

    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    episode_details = []

    try:
        for episode in range(1, episodes + 1):
            # Reset the game
            game.reset()
            episode_reward = 0.0
            step = 0

            # Track events for this episode
            episode_events = []

            # Previous observation for reward calculation
            previous_observation = None

            # Run the episode
            while game.running and step < 1000:  # Step limit to prevent infinite episodes
                # Handle pygame events if rendering
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return

                # Store current observation for reward calculation
                current_observation = game.get_observation("player")

                # Game update (get action from agent, apply it, etc.)
                game.update()

                # Get new observation after update
                next_observation = game.get_observation("player")

                # Calculate reward from events (for statistics only)
                reward = calculate_reward(
                    events=game.events,
                    observation=next_observation,
                    previous_observation=previous_observation,
                )
                episode_reward += reward

                # Store events for logging
                episode_events.extend([event.model_dump() for event in game.events])

                # Update previous observation for next step
                previous_observation = current_observation

                # Limit rendering speed if needed
                if render:
                    clock.tick(config.fps)

                step += 1

            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

            # Create episode result
            episode_result = EpisodeResult(
                episode_id=episode,
                total_reward=episode_reward,
                episode_length=step,
                win=len(game.enemies) == 0,  # Win if all enemies defeated
                player_health_remaining=game.player.health if game.player else 0,
                enemies_defeated=config.max_enemies - len(game.enemies),
                accuracy=0.0,  # TODO: Calculate accuracy
                events=episode_events,
            )
            episode_details.append(episode_result)

            # Print progress
            print(f"Episode {episode}/{episodes} - Steps: {step} - Reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("Evaluation interrupted by user")

    finally:
        if render:
            pygame.quit()

    # Print overall statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
    avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
    win_rate = (
        sum(1 for ep in episode_details if ep.win) / len(episode_details)
        if episode_details
        else 0.0
    )

    print("Evaluation complete:")
    print(f"  - Avg Reward: {avg_reward:.2f}")
    print(f"  - Avg Episode Length: {avg_length:.2f}")
    print(f"  - Win Rate: {win_rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate an ML agent for AgentArena")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Mode to run in",
    )
    parser.add_argument(
        "--pretrained-model", help="Path to pre-trained model file (from demonstration learning)"
    )
    parser.add_argument(
        "--model-name",
        default="ml_agent",
        help="Name prefix for the saved model",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the game during training/evaluation",
    )
    parser.add_argument(
        "--model-path",
        help="Path to the model file (required for evaluation, optional for training)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="How often to save the model during training",
    )
    parser.add_argument(
        "--reward-type",
        choices=[r.value for r in RewardType],
        default=RewardType.ADVANCED.value,
        help="Type of reward function to use during training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the neural network",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Initial exploration rate",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.01,
        help="Minimum exploration rate",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Rate at which epsilon decays",
    )

    args = parser.parse_args()

    # Convert reward type string to enum
    reward_type = RewardType(args.reward_type)

    if args.mode == "train":
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
            checkpoint_path=Path(args.model_path) if args.model_path else None,
            reward_type=reward_type,
            save_frequency=args.save_freq,
            ml_config=ml_config,
        )

        train(config=training_config, pretrained_model_path=args.pretrained_model)
    else:  # evaluate
        if not args.model_path:
            parser.error("--model-path is required for evaluation mode")
        evaluate(model_path=Path(args.model_path), episodes=args.episodes, render=args.render)
