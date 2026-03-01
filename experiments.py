"""
Experiment runner for Chef's Hat RL: hyperparameter and variant sweeps.

Runs train.run_training with different config_overrides (lr, gamma, epsilon_decay, variant)
and saves results to experiments/experiment_summary.json.
Run: python experiments.py [--quick]   # --quick uses 50 matches per exp
"""

import os
import sys
import json
import itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_experiment(exp_name, config_overrides, matches=200):
    """Run one training run with given overrides; return dict with exp_name, win_rate, avg_position, config."""
    from train import run_training
    from config import SEED

    overrides = dict(config_overrides)
    variant = overrides.pop("variant", 0)  # passed to run_training; rest go to rl_config

    out_dir = os.path.join("experiments", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model", "rl_model.pt")
    room, agent = run_training(
        matches=matches,
        variant=variant,
        seed=SEED,
        output_dir=out_dir,
        model_save_path=model_path,
        config_overrides=overrides,
    )

    win_rate = agent.win_count / max(1, agent.match_count) * 100
    avg_pos = sum(agent.positions) / max(1, len(agent.positions))
    return {
        "exp_name": exp_name,
        "win_rate": win_rate,
        "avg_position": avg_pos,
        "matches": agent.match_count,
        "config": config_overrides,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 50 matches per exp for testing")
    args = parser.parse_args()
    matches = 50 if args.quick else 200

    results_dir = "experiments"
    os.makedirs(results_dir, exist_ok=True)
    all_results = []

    # 1. Learning rate sweep
    for lr in [1e-4, 3e-4, 1e-3]:
        res = run_experiment(
            f"hp_lr_{lr}",
            {"lr": lr},
            matches=matches,
        )
        all_results.append(res)
        print(f"LR {lr}: Win rate {res['win_rate']:.1f}%")

    # 2. Discount factor (gamma) sweep
    for gamma in [0.95, 0.99, 0.999]:
        res = run_experiment(
            f"hp_gamma_{gamma}",
            {"gamma": gamma},
            matches=matches,
        )
        all_results.append(res)
        print(f"Gamma {gamma}: Win rate {res['win_rate']:.1f}%")

    # 3. Exploration: epsilon decay
    for decay in [0.995, 0.998, 0.9995]:
        res = run_experiment(
            f"exp_eps_decay_{decay}",
            {"epsilon_decay": decay},
            matches=matches,
        )
        all_results.append(res)
        print(f"Epsilon decay {decay}: Win rate {res['win_rate']:.1f}%")

    # 4. Variant comparison: baseline DQN (0), Double DQN (1), dense reward (2)
    for v in [0, 1, 2]:
        res = run_experiment(
            f"variant_{v}",
            {"variant": v},  # variant is passed to run_training, not rl_config
            matches=matches,
        )
        all_results.append(res)
        print(f"Variant {v}: Win rate {res['win_rate']:.1f}%")

    # Save summary
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
