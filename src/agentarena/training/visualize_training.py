"""
Visualize training results for the ML agent
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(results_file):
    """Plot training metrics from saved results"""
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    # Extract data
    episodes = np.arange(1, len(results["episode_rewards"]) + 1)
    rewards = results["episode_rewards"]
    lengths = results["episode_lengths"]
    epsilons = results.get("epsilons", [])

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("ML Agent Training Progress", fontsize=16)

    # Plot episode rewards
    axes[0].plot(episodes, rewards, "b-")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Total Reward per Episode")
    axes[0].grid(True)

    # Plot smoothed rewards (moving average)
    window_size = min(100, len(rewards))
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
    axes[0].plot(
        episodes[window_size - 1 :],
        smoothed_rewards,
        "r-",
        linewidth=2,
        label=f"{window_size}-episode moving average",
    )
    axes[0].legend()

    # Plot episode lengths
    axes[1].plot(episodes, lengths, "g-")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Steps per Episode")
    axes[1].grid(True)

    # Plot epsilon values if available
    if epsilons:
        axes[2].plot(episodes, epsilons, "m-")
        axes[2].set_ylabel("Epsilon")
        axes[2].set_title("Exploration Rate (Epsilon)")
    else:
        axes[2].set_visible(False)

    axes[2].set_xlabel("Episode")
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(results_file.with_suffix(".png"))
    plt.show()


def plot_comparison(results_files, labels=None):
    """Compare training results from multiple runs"""
    if labels is None:
        labels = [f.stem for f in results_files]

    # Ensure we have a label for each file
    if len(labels) != len(results_files):
        labels = [f"Run {i + 1}" for i in range(len(results_files))]

    # Load all results
    all_results = []
    for results_file in results_files:
        with open(results_file, "rb") as f:
            all_results.append(pickle.load(f))

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("ML Agent Training Comparison", fontsize=16)

    # Plot smoothed rewards for each run
    window_size = 50
    for i, results in enumerate(all_results):
        episodes = np.arange(1, len(results["episode_rewards"]) + 1)
        rewards = results["episode_rewards"]

        if len(rewards) > window_size:
            smoothed_rewards = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            axes[0].plot(
                episodes[window_size - 1 :],
                smoothed_rewards,
                linewidth=2,
                label=labels[i],
            )
        else:
            axes[0].plot(episodes, rewards, linewidth=2, label=labels[i])

    axes[0].set_ylabel("Episode Reward (Moving Average)")
    axes[0].set_title(f"Reward Comparison ({window_size}-episode moving average)")
    axes[0].grid(True)
    axes[0].legend()

    # Plot episode lengths
    for i, results in enumerate(all_results):
        episodes = np.arange(1, len(results["episode_lengths"]) + 1)
        lengths = results["episode_lengths"]

        if len(lengths) > window_size:
            smoothed_lengths = np.convolve(
                lengths,
                np.ones(window_size) / window_size,
                mode="valid",
            )
            axes[1].plot(
                episodes[window_size - 1 :],
                smoothed_lengths,
                linewidth=2,
                label=labels[i],
            )
        else:
            axes[1].plot(episodes, lengths, linewidth=2, label=labels[i])

    axes[1].set_ylabel("Episode Length (Moving Average)")
    axes[1].set_xlabel("Episode")
    axes[1].set_title(f"Episode Length Comparison ({window_size}-episode moving average)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("training_comparison.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ML agent training results")
    parser.add_argument("--files", nargs="+", required=True, help="Path to result files")
    parser.add_argument("--labels", nargs="+", help="Labels for each file (for comparison)")
    parser.add_argument("--compare", action="store_true", help="Compare multiple result files")

    args = parser.parse_args()

    if args.compare and len(args.files) > 1:
        files = [Path(f) for f in args.files]
        plot_comparison(files, args.labels)
    else:
        for file in args.files:
            plot_training_results(Path(file))
