#!/usr/bin/env python3
"""
Comprehensive visualization script for AgentArena training results.
Reads pickle files from the results directory and creates various graphs
for analysis and publication.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
import seaborn as sns

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
warnings.filterwarnings("ignore")


class TrainingResultsVisualizer:
    """Visualizer for training results from AgentArena experiments."""

    def __init__(self, output_dir: str = "evaluation_plots"):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def load_results(self, results_files: List[Path]) -> None:
        """Load training results from pickle files.

        Args:
            results_files: List of paths to pickle files containing training results
        """
        for file_path in results_files:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                # Extract experiment name from filename
                experiment_name = file_path.stem

                # Handle both dict and TrainingResults object formats
                if hasattr(data, "model_dump"):
                    data = data.model_dump()

                self.results[experiment_name] = data
                print(
                    f"✅ Loaded {experiment_name}: {len(data.get('episode_rewards', []))} episodes"
                )

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

    def smooth_data(self, data: List[float], window_size: int = 50) -> np.ndarray:
        """Apply smoothing filter to noisy data.

        Args:
            data: Raw data points
            window_size: Size of smoothing window

        Returns:
            Smoothed data array
        """
        if len(data) < window_size:
            window_size = max(1, len(data) // 4)

        return uniform_filter1d(data, size=window_size, mode="nearest")

    def calculate_moving_average(self, data: List[float], window: int = 100) -> np.ndarray:
        """Calculate moving average of data.

        Args:
            data: Input data
            window: Window size for moving average

        Returns:
            Moving average array
        """
        if len(data) < window:
            window = len(data)

        return np.convolve(data, np.ones(window) / window, mode="valid")

    def plot_training_curves(self, save: bool = True) -> None:
        """Create comprehensive training curves comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Training Performance Comparison", fontsize=16, fontweight="bold")

        # Color palette for different experiments
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))

        for idx, (exp_name, data) in enumerate(self.results.items()):
            color = colors[idx]

            # Extract data
            episode_rewards = data.get("episode_rewards", [])
            episode_lengths = data.get("episode_lengths", [])
            epsilons = data.get("epsilons", [])

            if not episode_rewards:
                continue

            episodes = range(1, len(episode_rewards) + 1)

            # 1. Episode Rewards (Raw and Smoothed)
            axes[0, 0].plot(episodes, episode_rewards, alpha=0.3, color=color, linewidth=0.5)
            smoothed_rewards = self.smooth_data(episode_rewards)
            axes[0, 0].plot(episodes, smoothed_rewards, label=exp_name, color=color, linewidth=2)

            # 2. Moving Average Rewards
            if len(episode_rewards) > 100:
                moving_avg = self.calculate_moving_average(episode_rewards, 100)
                moving_episodes = range(100, len(episode_rewards) + 1)
                axes[0, 1].plot(
                    moving_episodes, moving_avg, label=exp_name, color=color, linewidth=2
                )

            # 3. Episode Lengths
            if episode_lengths:
                smoothed_lengths = self.smooth_data(episode_lengths)
                axes[1, 0].plot(
                    episodes, smoothed_lengths, label=exp_name, color=color, linewidth=2
                )

            # 4. Exploration Rate (Epsilon)
            if epsilons:
                axes[1, 1].plot(episodes, epsilons, label=exp_name, color=color, linewidth=2)

        # Configure subplots
        axes[0, 0].set_title("Episode Rewards (Smoothed)", fontweight="bold")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Moving Average Rewards (100 episodes)", fontweight="bold")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Average Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Episode Lengths (Smoothed)", fontweight="bold")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Exploration Rate (Epsilon)", fontweight="bold")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Epsilon")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
            plt.savefig(self.output_dir / "training_curves.pdf", bbox_inches="tight")

        plt.show()

    def plot_win_rate_analysis(self, save: bool = True) -> None:
        """Analyze win rates from episode details."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Win Rate Analysis", fontsize=16, fontweight="bold")

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))

        for idx, (exp_name, data) in enumerate(self.results.items()):
            color = colors[idx]
            episode_details = data.get("episode_details", [])

            if not episode_details:
                continue

            # Calculate win rate over time with moving window
            wins = [1 if ep.get("win", False) else 0 for ep in episode_details]

            if len(wins) > 50:
                window_size = 100
                win_rates = []
                episodes = []

                for i in range(window_size, len(wins) + 1):
                    window_wins = wins[i - window_size : i]
                    win_rate = sum(window_wins) / len(window_wins)
                    win_rates.append(win_rate)
                    episodes.append(i)

                axes[0].plot(episodes, win_rates, label=exp_name, color=color, linewidth=2)

        # Win rate over time
        axes[0].set_title("Win Rate Over Time (100-episode window)", fontweight="bold")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Win Rate")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Final win rate comparison (bar plot)
        final_win_rates = []
        experiment_names = []

        for exp_name, data in self.results.items():
            episode_details = data.get("episode_details", [])
            if episode_details:
                # Calculate win rate for last 500 episodes or all if fewer
                last_episodes = (
                    episode_details[-500:] if len(episode_details) > 500 else episode_details
                )
                win_rate = sum(1 for ep in last_episodes if ep.get("win", False)) / len(
                    last_episodes
                )
                final_win_rates.append(win_rate)
                experiment_names.append(exp_name)

        if final_win_rates:
            bars = axes[1].bar(
                experiment_names, final_win_rates, color=colors[: len(final_win_rates)]
            )
            axes[1].set_title("Final Win Rate Comparison\n(Last 500 episodes)", fontweight="bold")
            axes[1].set_ylabel("Win Rate")
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, rate in zip(bars, final_win_rates):
                height = bar.get_height()
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{rate:.2%}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "win_rate_analysis.png", dpi=300, bbox_inches="tight")
            plt.savefig(self.output_dir / "win_rate_analysis.pdf", bbox_inches="tight")

        plt.show()

    def plot_performance_distribution(self, save: bool = True) -> None:
        """Create distribution plots for performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Performance Distribution Analysis", fontsize=16, fontweight="bold")

        # Collect data for all experiments
        all_rewards = {}
        all_lengths = {}
        all_survival_times = {}
        all_enemies_defeated = {}

        for exp_name, data in self.results.items():
            episode_rewards = data.get("episode_rewards", [])
            episode_details = data.get("episode_details", [])

            if episode_rewards:
                # Take last 500 episodes for final performance
                final_rewards = (
                    episode_rewards[-500:] if len(episode_rewards) > 500 else episode_rewards
                )
                all_rewards[exp_name] = final_rewards

            if episode_details:
                final_details = (
                    episode_details[-500:] if len(episode_details) > 500 else episode_details
                )
                all_lengths[exp_name] = [ep.get("episode_length", 0) for ep in final_details]
                all_enemies_defeated[exp_name] = [
                    ep.get("enemies_defeated", 0) for ep in final_details
                ]

        # 1. Reward Distribution
        if all_rewards:
            reward_data = []
            reward_labels = []
            for exp_name, rewards in all_rewards.items():
                reward_data.append(rewards)
                reward_labels.append(exp_name)

            axes[0, 0].boxplot(reward_data, labels=reward_labels)
            axes[0, 0].set_title("Reward Distribution", fontweight="bold")
            axes[0, 0].set_ylabel("Episode Reward")
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Episode Length Distribution
        if all_lengths:
            length_data = []
            length_labels = []
            for exp_name, lengths in all_lengths.items():
                length_data.append(lengths)
                length_labels.append(exp_name)

            axes[0, 1].boxplot(length_data, labels=length_labels)
            axes[0, 1].set_title("Episode Length Distribution", fontweight="bold")
            axes[0, 1].set_ylabel("Steps per Episode")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Reward Histogram Comparison
        if all_rewards:
            for exp_name, rewards in all_rewards.items():
                axes[1, 0].hist(rewards, alpha=0.6, label=exp_name, bins=30, density=True)

            axes[1, 0].set_title("Reward Distribution Histogram", fontweight="bold")
            axes[1, 0].set_xlabel("Episode Reward")
            axes[1, 0].set_ylabel("Density")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Enemies Defeated Distribution
        if all_enemies_defeated:
            defeated_data = []
            defeated_labels = []
            for exp_name, defeated in all_enemies_defeated.items():
                defeated_data.append(defeated)
                defeated_labels.append(exp_name)

            axes[1, 1].boxplot(defeated_data, labels=defeated_labels)
            axes[1, 1].set_title("Enemies Defeated Distribution", fontweight="bold")
            axes[1, 1].set_ylabel("Enemies Defeated per Episode")
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / "performance_distribution.png", dpi=300, bbox_inches="tight"
            )
            plt.savefig(self.output_dir / "performance_distribution.pdf", bbox_inches="tight")

        plt.show()

    def plot_learning_stability(self, save: bool = True) -> None:
        """Analyze learning stability and convergence."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Learning Stability Analysis", fontsize=16, fontweight="bold")

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))

        for idx, (exp_name, data) in enumerate(self.results.items()):
            color = colors[idx]
            episode_rewards = data.get("episode_rewards", [])

            if len(episode_rewards) < 200:
                continue

            episodes = np.array(range(len(episode_rewards)))
            rewards = np.array(episode_rewards)

            # 1. Rolling Standard Deviation
            window_size = 100
            rolling_std = []
            rolling_episodes = []

            for i in range(window_size, len(rewards)):
                window_rewards = rewards[i - window_size : i]
                rolling_std.append(np.std(window_rewards))
                rolling_episodes.append(i)

            axes[0, 0].plot(rolling_episodes, rolling_std, label=exp_name, color=color, linewidth=2)

            # 2. Cumulative Mean
            cumulative_mean = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            axes[0, 1].plot(episodes, cumulative_mean, label=exp_name, color=color, linewidth=2)

            # 3. Variance over time (sliding window)
            rolling_var = []
            for i in range(window_size, len(rewards)):
                window_rewards = rewards[i - window_size : i]
                rolling_var.append(np.var(window_rewards))

            axes[1, 0].plot(rolling_episodes, rolling_var, label=exp_name, color=color, linewidth=2)

            # 4. Learning progress (trend analysis)
            if len(rewards) > 500:
                # Split into segments and calculate mean for each
                segment_size = len(rewards) // 10
                segment_means = []
                segment_episodes = []

                for i in range(0, len(rewards), segment_size):
                    segment = rewards[i : i + segment_size]
                    if len(segment) > segment_size // 2:  # Only if segment has enough data
                        segment_means.append(np.mean(segment))
                        segment_episodes.append(i + segment_size // 2)

                axes[1, 1].plot(
                    segment_episodes,
                    segment_means,
                    "o-",
                    label=exp_name,
                    color=color,
                    linewidth=2,
                    markersize=6,
                )

        # Configure subplots
        axes[0, 0].set_title("Rolling Standard Deviation (100 episodes)", fontweight="bold")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Standard Deviation")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Cumulative Mean Reward", fontweight="bold")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Cumulative Mean")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Rolling Variance (100 episodes)", fontweight="bold")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Variance")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Learning Progress (Segmented Means)", fontweight="bold")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Segment Mean Reward")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "learning_stability.png", dpi=300, bbox_inches="tight")
            plt.savefig(self.output_dir / "learning_stability.pdf", bbox_inches="tight")

        plt.show()

    def generate_summary_statistics(self, save: bool = True) -> None:
        """Generate and display summary statistics table."""
        stats_data = []

        for exp_name, data in self.results.items():
            episode_rewards = data.get("episode_rewards", [])
            episode_details = data.get("episode_details", [])

            if not episode_rewards:
                continue

            # Take last 500 episodes for final performance
            final_rewards = (
                episode_rewards[-500:] if len(episode_rewards) > 500 else episode_rewards
            )
            final_details = (
                episode_details[-500:] if len(episode_details) > 500 else episode_details
            )

            # Calculate statistics
            stats = {
                "Experiment": exp_name,
                "Total Episodes": len(episode_rewards),
                "Mean Reward": np.mean(final_rewards),
                "Std Reward": np.std(final_rewards),
                "Max Reward": np.max(final_rewards),
                "Min Reward": np.min(final_rewards),
            }

            if final_details:
                wins = sum(1 for ep in final_details if ep.get("win", False))
                stats["Win Rate"] = wins / len(final_details)
                stats["Avg Episode Length"] = np.mean(
                    [ep.get("episode_length", 0) for ep in final_details]
                )
                stats["Avg Enemies Defeated"] = np.mean(
                    [ep.get("enemies_defeated", 0) for ep in final_details]
                )

            stats_data.append(stats)

        if stats_data:
            df = pd.DataFrame(stats_data)

            # Display table
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS (Last 500 episodes)")
            print("=" * 80)
            print(df.to_string(index=False, float_format="%.3f"))
            print("=" * 80)

            # Save to CSV
            if save:
                df.to_csv(self.output_dir / "summary_statistics.csv", index=False)
                print(
                    f"\n📊 Summary statistics saved to {self.output_dir / 'summary_statistics.csv'}"
                )

    def create_all_plots(self, save: bool = True) -> None:
        """Generate all visualization plots."""
        if not self.results:
            print("❌ No results loaded. Please load results first.")
            return

        print("🎨 Generating training curves...")
        self.plot_training_curves(save)

        print("🎨 Generating win rate analysis...")
        self.plot_win_rate_analysis(save)

        print("🎨 Generating performance distribution...")
        self.plot_performance_distribution(save)

        print("🎨 Generating learning stability analysis...")
        self.plot_learning_stability(save)

        print("📊 Generating summary statistics...")
        self.generate_summary_statistics(save)

        if save:
            print(f"\n✅ All plots saved to {self.output_dir}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize AgentArena training results")
    parser.add_argument("results_files", nargs="+", help="Paths to result pickle files")
    parser.add_argument("--output-dir", default="evaluation_plots", help="Directory to save plots")
    parser.add_argument("--no-save", action="store_true", help="Do not save plots, only display")

    args = parser.parse_args()

    # Convert string paths to Path objects
    result_paths = [Path(f) for f in args.results_files]

    # Check if files exist
    for path in result_paths:
        if not path.exists():
            print(f"❌ File not found: {path}")
            return

    # Create visualizer and load results
    visualizer = TrainingResultsVisualizer(args.output_dir)
    visualizer.load_results(result_paths)

    # Generate all plots
    visualizer.create_all_plots(save=not args.no_save)


if __name__ == "__main__":
    main()
