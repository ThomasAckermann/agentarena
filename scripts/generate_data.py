import os  # noqa: INP001
import random
import subprocess
import sys
from pathlib import Path


def run_data_collection(num_runs: int = 10, agent_type: str = "rule_based"):
    """Run the main function multiple times to collect demonstration data."""

    print(f"Starting {num_runs} data collection runs with {agent_type} agent...")

    for i in range(1, num_runs + 1):
        print(f"\nRun {i}/{num_runs}")

        try:
            # Set environment to handle Unicode properly
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            enemy_agent_str = random.choice(["rule_based_2", "rule_based"])

            # Run the main function
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentarena.main",
                    "--player",
                    agent_type,
                    "--enemy",
                    enemy_agent_str,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            print(f"Run {i} completed successfully")

        except subprocess.CalledProcessError as e:
            print(f"Run {i} failed: {e}")
            print(f"Error output: {e.stderr}")
        except KeyboardInterrupt:
            print(f"\nInterrupted after {i - 1} runs")
            break

    demo_dir = Path("demonstrations")
    if demo_dir.exists():
        demo_files = list(demo_dir.glob("demo_*.json"))
        print(f"Total demonstration files: {len(demo_files)}")
    else:
        print("No demonstrations directory found")


if __name__ == "__main__":
    run_data_collection(num_runs=100, agent_type="rule_based")
