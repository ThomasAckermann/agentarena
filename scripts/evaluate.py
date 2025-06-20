#!/usr/bin/env python3
"""
Model evaluation script for AgentArena.

Usage:
    python evaluate.py --model-path models/my_model.pt --enemy-agent rule_based --episodes 50
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import pygame

from agentarena.agent.ml_agent import MLAgent
from agentarena.agent.random_agent import RandomAgent
from agentarena.agent.rule_based_agent import RuleBasedAgent
from agentarena.agent.rule_based_agent_2 import RuleBasedAgent2
from agentarena.game.game import Game
from agentarena.models.config import load_config
from agentarena.models.events import (
    BulletFiredEvent,
    EnemyHitEvent,
    EntityDestroyedEvent,
    PlayerHitEvent,
)


class EvaluationMetrics:
    """Container for evaluation metrics."""

    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.total_episodes = 0
        self.episode_lengths = []
        self.player_health_remaining = []
        self.enemies_defeated = []
        self.player_shots_fired = 0
        self.enemy_shots_fired = 0
        self.player_hits_dealt = 0
        self.player_hits_taken = 0
        self.player_accuracy_values = []
        self.survival_times = []
        self.total_score = 0

    def add_episode(self, episode_data: Dict):
        """Add data from a single episode."""
        self.total_episodes += 1

        # Win/Loss tracking
        if episode_data["win"]:
            self.wins += 1
        else:
            self.losses += 1

        # Episode metrics
        self.episode_lengths.append(episode_data["episode_length"])
        self.player_health_remaining.append(episode_data["player_health"])
        self.enemies_defeated.append(episode_data["enemies_defeated"])
        self.survival_times.append(episode_data["survival_time"])
        self.total_score += episode_data["score"]

        # Combat metrics
        self.player_shots_fired += episode_data["player_shots"]
        self.enemy_shots_fired += episode_data["enemy_shots"]
        self.player_hits_dealt += episode_data["player_hits_dealt"]
        self.player_hits_taken += episode_data["player_hits_taken"]

        # Calculate accuracy for this episode
        if episode_data["player_shots"] > 0:
            accuracy = episode_data["player_hits_dealt"] / episode_data["player_shots"]
            self.player_accuracy_values.append(accuracy)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if self.total_episodes == 0:
            return {}

        return {
            "total_episodes": self.total_episodes,
            "win_rate": self.wins / self.total_episodes,
            "avg_episode_length": mean(self.episode_lengths),
            "avg_survival_time": mean(self.survival_times),
            "avg_player_health": mean(self.player_health_remaining),
            "avg_enemies_defeated": mean(self.enemies_defeated),
            "avg_score": self.total_score / self.total_episodes,
            "player_accuracy": mean(self.player_accuracy_values)
            if self.player_accuracy_values
            else 0.0,
            "total_player_shots": self.player_shots_fired,
            "total_enemy_shots": self.enemy_shots_fired,
            "total_player_hits_dealt": self.player_hits_dealt,
            "total_player_hits_taken": self.player_hits_taken,
            "hits_per_episode": self.player_hits_dealt / self.total_episodes,
            "damage_taken_per_episode": self.player_hits_taken / self.total_episodes,
        }


def create_enemy_agent(agent_type: str):
    """Create enemy agent based on type."""
    agent_map = {
        "random": RandomAgent,
        "rule_based": RuleBasedAgent,
        "rule_based_2": RuleBasedAgent2,
    }

    if agent_type not in agent_map:
        raise ValueError(
            f"Unknown enemy agent type: {agent_type}. Available: {list(agent_map.keys())}"
        )

    return agent_map[agent_type]()


def run_episode(game: Game, max_steps: int = 2000) -> Dict:
    """Run a single episode and collect metrics."""
    game.reset()

    episode_data = {
        "win": False,
        "episode_length": 0,
        "player_health": 0,
        "enemies_defeated": 0,
        "survival_time": 0.0,
        "score": 0,
        "player_shots": 0,
        "enemy_shots": 0,
        "player_hits_dealt": 0,
        "player_hits_taken": 0,
    }

    start_time = time.time()
    step = 0

    while game.running and step < max_steps:
        game.update()
        step += 1

        # Count events
        for event in game.events:
            if isinstance(event, BulletFiredEvent):
                if event.owner_id == "player":
                    episode_data["player_shots"] += 1
                else:
                    episode_data["enemy_shots"] += 1
            elif isinstance(event, EnemyHitEvent):
                episode_data["player_hits_dealt"] += 1
            elif isinstance(event, PlayerHitEvent):
                episode_data["player_hits_taken"] += 1
            elif isinstance(event, EntityDestroyedEvent) and event.is_enemy_destroyed():
                episode_data["enemies_defeated"] += 1

    # Final episode data
    episode_data["episode_length"] = step
    episode_data["survival_time"] = time.time() - start_time
    episode_data["score"] = game.score
    episode_data["win"] = len(game.enemies) == 0
    episode_data["player_health"] = game.player.health if game.player else 0
    episode_data["enemies_defeated"] = game.config.max_enemies - len(game.enemies)

    return episode_data


def evaluate_model(
    model_path: Path,
    enemy_agent_type: str,
    episodes: int = 50,
    render: bool = False,
    max_steps_per_episode: int = 2000,
) -> EvaluationMetrics:
    """Evaluate a model against a specific enemy agent."""

    print(f"🎯 Evaluating model: {model_path}")
    print(f"🤖 Against enemy: {enemy_agent_type}")
    print(f"📊 Episodes: {episodes}")
    print(f"🖼️  Render: {render}")
    print("-" * 50)

    # Initialize pygame if rendering
    if render:
        pygame.init()

    # Load game configuration
    config = load_config()
    if not render:
        config.headless = True
        config.fps = 0  # Uncapped FPS for faster evaluation

    # Initialize screen if rendering
    screen = None
    if render:
        screen = pygame.display.set_mode((config.display_width, config.display_height))
        pygame.display.set_caption(f"Evaluating {model_path.name} vs {enemy_agent_type}")

    clock = pygame.time.Clock()

    # Create agents
    player_agent = MLAgent(is_training=False)
    player_agent.load_model(str(model_path))
    enemy_agent = create_enemy_agent(enemy_agent_type)

    # Create game
    game = Game(screen, player_agent, enemy_agent, clock, config)

    # Initialize metrics
    metrics = EvaluationMetrics()

    try:
        for episode in range(1, episodes + 1):
            # Run episode
            episode_data = run_episode(game, max_steps_per_episode)
            metrics.add_episode(episode_data)

            # Print progress every 10 episodes
            if episode % 10 == 0 or episode == episodes:
                current_win_rate = metrics.wins / episode
                print(
                    f"Episode {episode:3d}/{episodes} | "
                    f"Win Rate: {current_win_rate:.1%} | "
                    f"Last: {'WIN' if episode_data['win'] else 'LOSS'}"
                )

            # Handle pygame events if rendering
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Evaluation interrupted by user")
                        return metrics

                clock.tick(config.fps)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")

    finally:
        if render:
            pygame.quit()

    return metrics


def print_evaluation_results(metrics: EvaluationMetrics, model_path: Path, enemy_agent_type: str):
    """Print formatted evaluation results."""
    summary = metrics.get_summary()

    if not summary:
        print("No episodes completed.")
        return

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_path.name}")
    print(f"Enemy: {enemy_agent_type}")
    print(f"Episodes: {summary['total_episodes']}")
    print("-" * 60)

    # Performance metrics
    print("🏆 PERFORMANCE METRICS")
    print(f"  Win Rate:           {summary['win_rate']:.1%}")
    print(f"  Average Score:      {summary['avg_score']:.1f}")
    print(f"  Avg Episode Length: {summary['avg_episode_length']:.1f} steps")
    print(f"  Avg Survival Time:  {summary['avg_survival_time']:.1f}s")
    print()

    # Combat metrics
    print("⚔️  COMBAT METRICS")
    print(f"  Player Accuracy:       {summary['player_accuracy']:.1%}")
    print(f"  Hits per Episode:      {summary['hits_per_episode']:.1f}")
    print(f"  Damage Taken/Episode:  {summary['damage_taken_per_episode']:.1f}")
    print(f"  Avg Enemies Defeated:  {summary['avg_enemies_defeated']:.1f}")
    print(f"  Avg Health Remaining:  {summary['avg_player_health']:.1f}")
    print()

    # Detailed combat stats
    print("📊 DETAILED STATS")
    print(f"  Total Player Shots:  {summary['total_player_shots']}")
    print(f"  Total Enemy Shots:   {summary['total_enemy_shots']}")
    print(f"  Total Hits Dealt:    {summary['total_player_hits_dealt']}")
    print(f"  Total Hits Taken:    {summary['total_player_hits_taken']}")
    print("=" * 60)


def save_results_json(
    metrics: EvaluationMetrics, model_path: Path, enemy_agent_type: str, output_path: Path
):
    """Save results to JSON file for comparison."""
    summary = metrics.get_summary()

    results = {
        "model_path": str(model_path),
        "model_name": model_path.name,
        "enemy_agent": enemy_agent_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": summary,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"📁 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AgentArena ML models")
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to the model file to evaluate"
    )
    parser.add_argument(
        "--enemy-agent",
        choices=["random", "rule_based", "rule_based_2"],
        default="rule_based",
        help="Type of enemy agent to test against",
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of episodes to run for evaluation"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game during evaluation (slower)"
    )
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    # Validate model path
    if not args.model_path.exists():
        print(f"❌ Model file not found: {args.model_path}")
        return

    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model_path,
        enemy_agent_type=args.enemy_agent,
        episodes=args.episodes,
        render=args.render,
        max_steps_per_episode=args.max_steps,
    )

    # Print results
    print_evaluation_results(metrics, args.model_path, args.enemy_agent)

    # Save to JSON if requested
    if args.output:
        save_results_json(metrics, args.model_path, args.enemy_agent, args.output)


if __name__ == "__main__":
    main()
