import argparse
from argparse import Namespace
from pathlib import Path

import pygame

from agentarena.agent.manual_agent import ManualAgent
from agentarena.agent.ml_agent import MLAgent
from agentarena.agent.random_agent import RandomAgent
from agentarena.agent.rule_based_agent import RuleBasedAgent
from agentarena.agent.rule_based_agent_2 import RuleBasedAgent2
from agentarena.game.game import Game
from agentarena.models.config import GameConfig, load_config


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="AgentArena - A 2D shooting game with configurable AI agents",
    )
    parser.add_argument(
        "--player",
        choices=["manual", "random", "ml", "rule_based", "rule_based_2"],
        default="manual",
        help="Type of agent to control the player",
    )
    parser.add_argument(
        "--enemy",
        choices=["random", "ml", "rule_based", "rule_based_2"],
        default="random",
        help="Type of agent to control the enemies",
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        help="Path to ML model file (required if using ML agent)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    pygame.init()
    config = load_config()
    screen = pygame.display.set_mode(
        (config.display_width, config.display_height),
    )
    pygame.display.set_caption("AgentArena")

    clock = pygame.time.Clock()

    if args.player == "manual":
        player_agent = ManualAgent()
    elif args.player == "random":
        player_agent = RandomAgent(name="RandomPlayer")
    elif args.player == "rule_based":
        player_agent = RuleBasedAgent(name="RuleBased")
    elif args.player == "rule_based_2":
        player_agent = RuleBasedAgent2(name="RuleBased2")
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
    elif args.enemy == "rule_based":
        enemy_agent = RuleBasedAgent(name="RuleBasedEnemy")
    elif args.enemy == "rule_based_2":
        enemy_agent = RuleBasedAgent2(name="RuleBasedEnemy2")
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

    game = Game(screen, player_agent, enemy_agent, clock, config)

    game_loop(game, clock, config)

    pygame.quit()


def game_loop(game: Game, clock: pygame.time.Clock, config: GameConfig) -> None:
    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False

        game.update()
        clock.tick(config.fps)


if __name__ == "__main__":
    main()
