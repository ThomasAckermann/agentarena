#!/usr/bin/env python
"""
Verify that the AgentArena ML implementation is correctly installed.
Run this script after setting up your environment to check for any issues.
"""

import importlib
import os


def check_module(module_name):
    """Try to import a module and check if it exists."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.exists(file_path)


def print_status(name, status, message=None):
    """Print status with appropriate formatting."""
    if status:
        print(f"✅ {name}")
    else:
        print(f"❌ {name}")
        if message:
            print(f"   └─ {message}")


def main():
    print("\n=============================================")
    print("AgentArena ML Implementation Verification")
    print("=============================================\n")

    # Check required modules
    required_modules = [
        "agentarena",
        "agentarena.agent",
        "agentarena.game",
        "agentarena.training",
        "torch",
        "numpy",
        "matplotlib",
        "pygame",
    ]

    print("Checking required Python modules:")
    all_modules_present = True
    for module in required_modules:
        module_exists = check_module(module)
        print_status(module, module_exists)
        if not module_exists:
            all_modules_present = False

    # Check if specific implementation files exist
    required_files = [
        "src/agentarena/agent/ml_agent.py",
        "src/agentarena/training/reward_functions.py",
        "src/agentarena/training/train.py",
        "src/agentarena/training/visualize_training.py",
    ]

    print("\nChecking required implementation files:")
    all_files_present = True
    for file_path in required_files:
        file_exists = check_file_exists(file_path)
        print_status(file_path, file_exists)
        if not file_exists:
            all_files_present = False

    # Check if directories exist
    required_dirs = ["models", "results"]

    print("\nChecking required directories:")
    all_dirs_present = True
    for dir_path in required_dirs:
        dir_exists = os.path.isdir(dir_path)
        print_status(dir_path, dir_exists)
        if not dir_exists:
            all_dirs_present = False
            print_status(
                dir_path,
                False,
                "You can create this directory with: mkdir -p " + dir_path,
            )

    # Overall status
    print("\n=============================================")
    if all_modules_present and all_files_present and all_dirs_present:
        print("✅ All checks passed! Your installation looks good.")
        print(
            "You can start training with: "
            "python -m agentarena.training.train"
            " --episodes 100 --render",
        )
    else:
        print("❌ Some checks failed. Please address the issues above.")
    print("=============================================\n")


if __name__ == "__main__":
    main()
