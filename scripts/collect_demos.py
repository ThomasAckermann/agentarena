#!/usr/bin/env python3
"""Script to collect human demonstrations."""

import argparse

from agentarena.training.demo_collection import analyze_demonstrations


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze demonstrations")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing demonstrations")
    parser.add_argument("--demos-dir", default="demonstrations", help="Demonstrations directory")

    args = parser.parse_args()

    if args.analyze:
        analyze_demonstrations(args.demos_dir)
    else:
        print("To collect demonstrations:")
        print("1. Run: python -m agentarena.main --player manual")
        print("2. Play the game normally - your actions will be recorded")
        print("3. Run: python scripts/collect_demos.py --analyze")


if __name__ == "__main__":
    main()
