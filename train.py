"""
Training script for Chef's Hat RL agent (Room-based API).

Uses the async Room API (chefshatgym / ChefsHatGYM source): Room + connect_player + room.run().
Variant (STUDENT_ID % 7) selects algorithm and options from config.VARIANTS.
Run: python train.py [--matches 500] [--variant 0] [--seed 42]
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime

# Ensure project root is on path for config and agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _setup_imports():
    """Resolve Room and RandomAgent from pip package or from ChefsHatGYM repo src."""
    try:
        from chefshatgym.rooms.room import Room
        from chefshatgym.agents.random_agent import RandomAgent
        return Room, RandomAgent
    except ImportError:
        pass
    try:
        from ChefsHatGym.rooms.room import Room
        from ChefsHatGym.agents.random_agent import RandomAgent
        return Room, RandomAgent
    except ImportError:
        pass
    # Fallback: assume running next to ChefsHatGYM repo with src/
    src = os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src")
    if os.path.exists(src):
        sys.path.insert(0, src)
    try:
        from rooms.room import Room
        from agents.random_agent import RandomAgent
        return Room, RandomAgent
    except ImportError as e:
        raise ImportError(
            "Could not import Chef's Hat Gym. Install with: pip install chefshatgym\n"
            "Or clone https://github.com/pablovin/ChefsHatGYM and add src/ to path"
        ) from e


def run_training(
    matches: int = 500,
    variant: int = 0,
    seed: int = 42,
    output_dir: str = "outputs",
    model_save_path: str = None,
    config_overrides: dict = None,
):
    """Create Room with 3 random + 1 RL agent (variant from config), run async room.run(), write training_stats.txt."""
    from config import set_all_seeds, get_variant_config, VARIANTS, DQN_CONFIG, A2C_CONFIG
    from agents.rl_agent import RLAgent

    set_all_seeds(seed)
    Room, RandomAgent = _setup_imports()

    cfg = get_variant_config(variant)
    overrides = config_overrides or {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"train_v{variant}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    model_path = model_save_path or os.path.join(run_dir, "model", "rl_model.pt")

    # Room API: max_matches, output_folder; connect 4 players then asyncio.run(room.run())
    room = Room(
        run_remote_room=False,
        room_name=f"Train_V{variant}",
        max_matches=matches,
        output_folder=run_dir,
        save_game_dataset=True,
        save_logs_game=False,
        save_logs_room=False,
    )

    opponents = [
        RandomAgent(name=f"Random{i}", log_directory=room.room_dir, verbose_log=False)
        for i in range(3)
    ]
    for a in opponents:
        room.connect_player(a)

    rl_config = {"training": True, "model_path": model_path, "log_directory": room.room_dir}
    if cfg.get("algorithm") == "a2c":
        rl_config.update(A2C_CONFIG)
        rl_config["algorithm"] = "a2c"
    else:
        rl_config.update(DQN_CONFIG)
        rl_config["double_dqn"] = cfg.get("double_dqn", True)
        rl_config["dueling"] = cfg.get("dueling", True)
        rl_config["reward_shaping"] = cfg.get("reward_shaping", "minimal")
        rl_config["prioritized_replay"] = cfg.get("prioritized_replay", False)
        rl_config["algorithm"] = "dqn"
    rl_config.update(overrides)

    agent = RLAgent(name="RLAgent", verbose_log=False, **rl_config)
    room.connect_player(agent)

    asyncio.run(room.run())

    # Save training stats
    stats_path = os.path.join(run_dir, "training_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Variant: {variant}\n")
        f.write(f"Matches: {matches}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Win rate: {agent.win_count / max(1, agent.match_count) * 100:.1f}%\n")
        f.write(f"Avg position: {sum(agent.positions) / max(1, len(agent.positions)):.2f}\n")

    return room, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=500)
    parser.add_argument("--variant", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    from config import VARIANT
    variant = args.variant if args.variant is not None else VARIANT
    run_training(
        matches=args.matches,
        variant=variant,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
