import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import pygame

from agentarena.agent.ml_agent import MLAgent
from agentarena.agent.random_agent import RandomAgent
from agentarena.config import load_config
from agentarena.game.game import Game
from agentarena.training.reward_functions import RewardType, calculate_reward


def train(
    episodes: int = 1000,
    model_save_freq: int = 100,
    render: bool = False,
    checkpoint_path: Path | None = None,
    reward_type: RewardType = RewardType.ADVANCED,
) -> MLAgent:
    """Train the ML agent"""
    print(
        f"Starting ML agent training with {reward_type.value} reward function...",
    )

    # Initialize pygame if needed for rendering
    if render:
        pygame.init()

    # Load configuration
    config = load_config()

    # Set to headless mode if not rendering
    if not render:
        config.headless = True

    # Adjust for faster training
    if not render:
        config.fps = 0  # Uncapped FPS for headless mode

    # Initialize screen if rendering
    screen = None
    if render:
        screen = pygame.display.set_mode(
            (config.display_width, config.display_height),
        )
        pygame.display.set_caption(
            f"AgentArena ML Training - {reward_type.value}",
        )

    # Create clock
    clock = pygame.time.Clock()

    # Create agents
    player_agent = MLAgent(is_training=True)
    enemy_agent = RandomAgent()

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        player_agent.load_model(checkpoint_path)

    # Create game
    game = Game(screen, player_agent, enemy_agent, clock, config)

    # Prepare model directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Training statistics
    episode_rewards = []
    episode_lengths = []
    epsilons = []  # Track epsilon values for visualization
    best_reward = float("-inf")

    # File paths for saving results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"training_results_{timestamp}_{reward_type.value}.pkl"

    try:
        for episode in range(1, episodes + 1):
            # Reset the game
            game.reset()
            episode_reward = 0
            step = 0
            previous_observation = None

            # Run the episode
            while game.running and step < 1000:  # Step limit to prevent infinite episodes
                # Handle pygame events if rendering
                if render:
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
                    game.events,
                    next_observation,
                    previous_observation,
                    reward_type,
                )

                episode_reward += reward

                # Learn from this step
                player_agent.learn(next_observation, reward, not game.running)

                # Update previous observation for next step
                previous_observation = current_observation

                # Limit rendering speed if needed
                if render:
                    clock.tick(config.fps)

                step += 1

            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

            # Calculate moving average
            window_size = min(episode, 100)
            avg_reward = sum(episode_rewards[-window_size:]) / window_size

            # Track epsilon for visualization
            epsilons.append(player_agent.epsilon)

            # Print progress
            print(
                f"Episode {episode}/{episodes} - Steps: {step} "
                "- Reward: {episode_reward:.2f} "
                "- Avg Reward: {avg_reward:.2f} "
                "- Epsilon: {player_agent.epsilon:.4f}",
            )

            # Save model periodically
            if episode % model_save_freq == 0:
                model_path = models_dir / f"ml_agent_{timestamp}_{reward_type.value}_ep{episode}.pt"
                player_agent.save_model(model_path)
                print(f"Model saved to {model_path}")

                # Also save training results
                results = {
                    "episode_rewards": episode_rewards,
                    "episode_lengths": episode_lengths,
                    "epsilons": epsilons,
                    "reward_type": reward_type.value,
                    "timestamp": timestamp,
                    "episodes_completed": episode,
                }
                with open(results_file, "wb") as f:
                    pickle.dump(results, f)
                print(f"Training results saved to {results_file}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model_path = models_dir / f"ml_agent_{timestamp}_{reward_type.value}_best.pt"
                player_agent.save_model(best_model_path)
                print(f"New best model saved with avg reward: {best_reward:.2f}")

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        # Save final model
        final_model_path = models_dir / f"ml_agent_{timestamp}_{reward_type.value}_final.pt"
        player_agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Save final training results
        results = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "epsilons": epsilons,
            "reward_type": reward_type.value,
            "timestamp": timestamp,
            "episodes_completed": len(episode_rewards),
        }
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        print(f"Final training results saved to {results_file}")

        if render:
            pygame.quit()

    return player_agent


def evaluate(
    model_path: Path,
    episodes: int = 10,
    render: bool = True,
) -> None:
    """Evaluate a trained ML agent"""
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

    # Create game
    game = Game(screen, player_agent, enemy_agent, clock, config)

    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []

    try:
        for episode in range(1, episodes + 1):
            # Reset the game
            game.reset()
            episode_reward = 0
            step = 0

            # Run the episode
            while game.running and step < 1000:  # Step limit to prevent infinite episodes
                # Handle pygame events if rendering
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return

                # Game update (get action from agent, apply it, etc.)
                game.update()

                # Calculate reward from events (for statistics only)
                reward = calculate_reward(game.events)
                episode_reward += reward

                # Limit rendering speed if needed
                if render:
                    clock.tick(config.fps)

                step += 1

            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

            # Print progress
            print(f"Episode {episode}/{episodes} - Steps: {step} - Reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("Evaluation interrupted by user")

    finally:
        if render:
            pygame.quit()

    # Print overall statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    print(
        f"Evaluation complete - Avg Reward: {avg_reward:.2f} - Avg Episode Length: {avg_length:.2f}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate an ML agent for AgentArena")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Mode to run in",
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

    args = parser.parse_args()

    # Convert reward type string to enum
    reward_type = RewardType(args.reward_type)

    if args.mode == "train":
        train(
            episodes=args.episodes,
            model_save_freq=args.save_freq,
            render=args.render,
            checkpoint_path=args.model_path,
            reward_type=reward_type,
        )
    else:  # evaluate
        if not args.model_path:
            parser.error("--model-path is required for evaluation mode")
        evaluate(model_path=args.model_path, episodes=args.episodes, render=args.render)
