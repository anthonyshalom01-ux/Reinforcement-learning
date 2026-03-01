"""
Evaluation script for Chef's Hat RL agent (Room-based API).

Loads a trained model, runs N matches against 3 random opponents, computes
win rate, average position, performance score; can plot learning curves.
Run: python evaluate.py --model <path_to_rl_model.pt> [--matches 100]
"""

import os
import sys
import asyncio
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _setup_imports():
    """Resolve Room and RandomAgent (same logic as train.py for Room API)."""
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
    src = os.path.join(os.path.dirname(__file__), "..", "ChefsHatGYM", "src")
    if os.path.exists(src):
        sys.path.insert(0, src)
    from rooms.room import Room
    from agents.random_agent import RandomAgent
    return Room, RandomAgent


def evaluate_agent(
    model_path: str,
    matches: int = 100,
    seed: int = 123,
    output_dir: str = "outputs_eval",
):
    """Create room with 3 random + 1 RL agent (training=False), load model, run matches; return metrics and write evaluation_report.txt."""
    from config import set_all_seeds, DQN_CONFIG
    from agents.rl_agent import RLAgent

    set_all_seeds(seed)
    Room, RandomAgent = _setup_imports()

    ts = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    eval_dir = os.path.join(output_dir, ts)
    os.makedirs(eval_dir, exist_ok=True)

    room = Room(
        run_remote_room=False,
        room_name="Eval",
        max_matches=matches,
        output_folder=eval_dir,
        save_game_dataset=True,
        save_logs_game=False,
        save_logs_room=False,
    )

    for i in range(3):
        room.connect_player(
            RandomAgent(name=f"Random{i}", log_directory=room.room_dir, verbose_log=False)
        )

    rl_config = {
        "training": False,
        "model_path": model_path,
        "log_directory": room.room_dir,
        "algorithm": "dqn",
        **DQN_CONFIG,
    }

    agent = RLAgent(name="RLAgent", verbose_log=False, **rl_config)
    if os.path.exists(model_path):
        import torch
        agent.policy_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    room.connect_player(agent)

    asyncio.run(room.run())

    # Metrics
    positions = agent.positions
    wins = sum(1 for p in positions if p == 1)
    win_rate = wins / len(positions) * 100 if positions else 0
    avg_pos = sum(positions) / len(positions) if positions else 0
    perf_score = _performance_score(room) if hasattr(room, "final_scores") else 0

    metrics = {
        "win_rate": win_rate,
        "avg_position": avg_pos,
        "wins": wins,
        "matches": len(positions),
        "performance_score": perf_score,
    }

    report_path = os.path.join(eval_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Matches: {matches}\n")
        f.write(f"Win rate: {win_rate:.1f}%\n")
        f.write(f"Avg position: {avg_pos:.2f}\n")
        f.write(f"Performance score: {perf_score:.4f}\n")

    return metrics, agent, room


def _performance_score(room):
    """Chef's Hat performance score: ((points*10)/rounds)/matches; approximated here from final scores."""
    if not hasattr(room, "final_scores") or not room.final_scores:
        return 0
    # Approximate: use final score, assume rounds ~ matches*5
    scores = room.final_scores
    total = sum(scores.values())
    matches = getattr(room, "max_matches", 3)
    return (total * 10) / (matches * 5) / matches if matches else 0


def plot_learning_curves(agent, save_path: str):
    """Plot win rate and loss over matches."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    if not agent.positions:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    x = range(1, len(agent.positions) + 1)
    wins_cum = np.cumsum([1 if p == 1 else 0 for p in agent.positions])
    win_rate_curve = wins_cum / np.maximum(np.arange(1, len(agent.positions) + 1), 1) * 100
    ax1.plot(x, win_rate_curve, label="Win rate (%)")
    ax1.set_xlabel("Match")
    ax1.set_ylabel("Win rate (%)")
    ax1.set_title("Learning Curve: Win Rate")
    ax1.legend()
    if agent.loss_history:
        ax2.plot(agent.loss_history, label="Loss", alpha=0.7)
        ax2.set_xlabel("Update step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Need torch for load
try:
    import torch
except ImportError:
    torch = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default="outputs_eval")
    args = parser.parse_args()
    if not torch:
        print("PyTorch required for model loading. pip install torch")
        return
    evaluate_agent(
        model_path=args.model,
        matches=args.matches,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
