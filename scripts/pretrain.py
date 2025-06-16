#!/usr/bin/env python3
"""Script to pre-train agent on demonstrations."""

from agentarena.training.pretraining import pretrain_agent
import argparse


def main():
    parser = argparse.ArgumentParser(description="Pre-train agent on demonstrations")
    parser.add_argument("--demos-dir", default="demonstrations", help="Demonstrations directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--output", default="models/pretrained_agent.pt", help="Output model path")

    args = parser.parse_args()

    model_path = pretrain_agent(
        demonstrations_dir=args.demos_dir, epochs=args.epochs, save_path=args.output
    )

    if model_path:
        print(f"\n✅ Pre-training complete! Use with:")
        print(f"python -m agentarena.training.train --pretrained-model {model_path}")


if __name__ == "__main__":
    main()
