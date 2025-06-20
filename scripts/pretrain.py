#!/usr/bin/env python3
"""Script to pre-train agent on demonstrations using multi-head network."""

import argparse
from datetime import datetime
from pathlib import Path

from agentarena.training.pretraining import pretrain_agent


def main():
    parser = argparse.ArgumentParser(description="Pre-train agent on demonstrations")
    parser.add_argument("--demos-dir", default="demonstrations", help="Demonstrations directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for training",
    )
    parser.add_argument("--output", default="models/pretrained_agent.pt", help="Output model path")
    parser.add_argument(
        "--tensorboard-dir",
        default=f"runs/pretraining{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--balance-actions",
        action="store_true",
        default=True,
        help="Balance action distribution in training data",
    )
    parser.add_argument(
        "--no-balance-actions",
        dest="balance_actions",
        action="store_false",
        help="Disable action balancing",
    )
    parser.add_argument(
        "--balance-factor",
        type=float,
        default=0.7,
        help="How much to balance actions (1.0=perfect balance, 0.0=no balance)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting pre-training with multi-head network...")
    print(f"📁 Demonstrations directory: {args.demos_dir}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"⚖️  Action balancing: {'Enabled' if args.balance_actions else 'Disabled'}")
    if args.balance_actions:
        print(f"🔧 Balance factor: {args.balance_factor}")
    print(f"💾 Output: {args.output}")
    print()

    model_path = pretrain_agent(
        demonstrations_dir=args.demos_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.output,
        tensorboard_dir=args.tensorboard_dir,
        balance_actions=args.balance_actions,
        balance_factor=args.balance_factor,
    )

    if model_path:
        print("\n✅ Pre-training complete!")
        print(f"📁 Model saved to: {model_path}")
        print("\n🎮 Next steps:")
        print("1. Use for RL training:")
        print(f"   python -m agentarena.training.train --pretrained-model {model_path}")
        print("\n2. Or evaluate the pre-trained model:")
        print(f"   python -m agentarena.main --player ml --ml-model {model_path}")
        print("\n3. View training progress:")
        print(f"   tensorboard --logdir {args.tensorboard_dir}")
    else:
        print("\n❌ Pre-training failed!")
        print("Make sure you have demonstration data in the demonstrations directory.")
        print("To collect demonstrations:")
        print("   python -m agentarena.main --player manual")


if __name__ == "__main__":
    main()
