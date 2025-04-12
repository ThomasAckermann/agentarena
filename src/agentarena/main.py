import argparse
from pathlib import Path

import pygame

from agentarena.agent.manual_agent import ManualAgent
from agentarena.agent.ml_agent import MLAgent
from agentarena.agent.random_agent import RandomAgent
from agentarena.config import load_config
from agentarena.game.game import Game


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="AgentArena - A 2D shooting game with configurable AI agents",
    )
    parser.add_argument(
        "--player",
        choices=["manual", "random", "ml"],
        default="manual",
        help="Type of agent to control the player",
    )
    parser.add_argument(
        "--enemy",
        choices=["random", "ml"],
        default="random",
        help="Type of agent to control the enemies",
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        help="Path to ML model file (required if using ML agent)",
    )

    args = parser.parse_args()

    # Initialize pygame
    pygame.init()

    # Load configuration
    config = load_config()

    # Create display
    screen = pygame.display.set_mode((config.display_width, config.display_height))
    pygame.display.set_caption("AgentArena")

    # Create clock
    clock = pygame.time.Clock()

    # Create player agent based on arguments
    if args.player == "manual":
        player_agent = ManualAgent()
    elif args.player == "random":
        player_agent = RandomAgent(name="RandomPlayer")
    elif args.player == "ml":
        if not args.ml_model:
            print("Error: --ml-model is required when using ML agent")
            return

        model_path = Path(args.ml_model)
        if not model_path.exists():
            print(f"Error: Model file {args.ml_model} not found")
            return

        player_agent = MLAgent(name="MLPlayer", is_training=False)
        player_agent.load_model(args.ml_model)

    # Create enemy agent based on arguments
    if args.enemy == "random":
        enemy_agent = RandomAgent(name="RandomEnemy")
    elif args.enemy == "ml":
        if not args.ml_model:
            print("Error: --ml-model is required when using ML agent")
            return

        model_path = Path(args.ml_model)
        if not model_path.exists():
            print(f"Error: Model file {args.ml_model} not found")
            return

        enemy_agent = MLAgent(name="MLEnemy", is_training=False)
        enemy_agent.load_model(args.ml_model)

    # Initialize game
    game = Game(screen, player_agent, enemy_agent, clock, config)

    # Main game loop
    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False

        game.update()
        clock.tick(config.fps)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
