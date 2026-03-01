"""
Chef's Hat RL Agent - Live Demonstration

Runs a small number of matches with clean console output (no library print spam).
  python demo.py [--matches 5] [--model path/to/model.pt]
  python demo.py --train-then-demo   # train 100 matches, then demo 5 with saved model
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["GYM_SILENCE_WARNINGS"] = "1"

# NumPy 2.x: gym uses np.bool8, which was removed
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
from ChefsHatGym.env import ChefsHatEnv

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# -----------------------------------------------------------------------------
# Quiet Random Agent: same behaviour as AgentRandon but no print(possibleActions)
# -----------------------------------------------------------------------------
class QuietRandomAgent(ChefsHatPlayer):
    """Random agent implementing ChefsHatPlayer; chooses uniformly over valid actions only."""
    suffix = "RANDOM"

    def __init__(self, name, log_directory="", use_sufix=True):
        super().__init__(self.suffix, name, "", False, False, log_directory, use_sufix)
        self.reward = None

    def get_action(self, observations):
        """Pick a random valid action (observation[28:228] is the 200-dim action mask)."""
        possible = np.array(observations[28:228])
        valid = np.where(possible > 0.5)[0]
        if len(valid) == 0:
            valid = [199]
        a_idx = int(np.random.choice(valid))
        action = np.zeros(200, dtype=np.float32)
        action[a_idx] = 1.0
        return action.tolist()

    def get_exhanged_cards(self, cards, amount):
        return sorted(cards)[-amount:]

    def update_exchange_cards(self, cards_sent, cards_received):
        pass

    def do_special_action(self, info, specialAction):
        return True

    def observe_special_action(self, action_type, player):
        pass

    def get_reward(self, info):
        pass

    def update_end_match(self, envInfo):
        pass

    def update_action_others(self, envInfo):
        pass

    def update_my_action(self, envInfo):
        pass

    def update_start_match(self, cards, players, starting_player):
        pass

    def update_game_over(self):
        pass


# -----------------------------------------------------------------------------
# Demo RL Agent: evaluation-only DQN (no training), optional model load, optional print
# -----------------------------------------------------------------------------
class DemoRLAgent(ChefsHatPlayer):
    """RL agent for demos: no exploration (epsilon=0), optional loaded model; can print each match result if not silent."""
    suffix = "DQN"

    def __init__(self, name, log_directory="", model_path=None, use_sufix=True, silent=False):
        super().__init__(self.suffix, name, "", False, False, log_directory, use_sufix)
        self.model_path = model_path
        self.silent = silent
        self.state_size = 28
        self.action_size = 200
        self.epsilon = 0.0  # No exploration in demo
        self.positions = []
        self.win_count = 0
        self.match_count = 0
        self.last_state = None
        self.last_action = None
        self.last_mask = None
        self.demo_mode = True

        if TORCH_AVAILABLE:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.state_size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.action_size),
            )
            if model_path and os.path.exists(model_path):
                self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.net.eval()
        else:
            self.net = None

    def _obs_to_state(self, obs):
        """State = first 28 dims (board 11 + hand 17)."""
        return np.array(obs[:28], dtype=np.float32)

    def _get_mask(self, obs):
        """Action mask = dims 28–228 (200 binary)."""
        return np.array(obs[28:228], dtype=np.float32)

    def get_action(self, observations):
        obs = np.array(observations)
        state = self._obs_to_state(obs)
        mask = self._get_mask(obs)
        valid = np.where(mask > 0.5)[0]
        if len(valid) == 0:
            valid = [199]

        if self.net is not None:
            with torch.no_grad():
                q = self.net(torch.from_numpy(state).float().unsqueeze(0))
                q = q.numpy().flatten()
                q[mask < 0.5] = -1e9
                a_idx = int(np.argmax(q))
        else:
            a_idx = int(np.random.choice(valid))

        action = np.zeros(200, dtype=np.float32)
        action[a_idx] = 1.0
        return action.tolist()

    def update_start_match(self, cards, players, starting_player):
        self.last_state = None

    def get_exhanged_cards(self, cards, amount):
        return sorted(cards)[-amount:]

    def update_exchange_cards(self, cards_sent, cards_received):
        pass

    def do_special_action(self, info, specialAction):
        return True

    def observe_special_action(self, action_type, player):
        pass

    def get_reward(self, info):
        pass

    def update_end_match(self, envInfo):
        """Record place (from Match_Score; we are player index 3), update counts; optionally print result."""
        scores = envInfo.get("Match_Score", [])
        self.match_count += 1
        place = 4
        if scores and len(scores) > 3:
            our_points = scores[3]  # we are 4th player added
            place = 4 - our_points if our_points >= 0 else 4
        self.positions.append(place)
        if place == 1:
            self.win_count += 1
        if self.demo_mode and not self.silent:
            medal = ["1st", "2nd", "3rd", "4th"][place - 1]
            print(f"  Match {self.match_count}: {medal} place (RL Agent)", file=sys.__stdout__, flush=True)

    def update_action_others(self, envInfo):
        pass

    def update_my_action(self, envInfo):
        pass

    def update_game_over(self):
        pass


def run_demo(matches=5, model_path=None, output_dir="demo_output", quiet_libs=True):
    """Run one game of `matches` with 1 RL + 3 quiet random agents; print results to stdout unless quiet_libs is False."""
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    os.makedirs(output_dir, exist_ok=True)

    # Suppress library chatter if requested (AgentRandon prints; we use QuietRandomAgent)
    fout = io.StringIO() if quiet_libs else sys.stdout
    ferr = io.StringIO() if quiet_libs else sys.stderr

    print("\n" + "=" * 60, file=sys.__stdout__)
    print("  CHEF'S HAT RL AGENT - DEMONSTRATION", file=sys.__stdout__)
    print("=" * 60, file=sys.__stdout__)
    print(f"\n  Setup: 1 RL Agent vs 3 Random opponents", file=sys.__stdout__)
    print(f"  Matches: {matches}", file=sys.__stdout__)
    print(f"  Model: {model_path or 'Untrained (random init)'}", file=sys.__stdout__)
    print("\n" + "-" * 60, file=sys.__stdout__)
    print("  Match results:\n", file=sys.__stdout__)
    sys.stdout.flush()

    # Redirect library stdout/stderr so only our prints (to sys.__stdout__) appear
    with contextlib.redirect_stdout(fout), contextlib.redirect_stderr(ferr):
        room = ChefsHatRoomLocal(
            room_name="Demo",
            game_type=ChefsHatEnv.GAMETYPE["MATCHES"],
            stop_criteria=matches,
            max_rounds=-1,
            save_dataset=False,
            verbose_console=False,
            verbose_log=False,
            game_verbose_console=False,
            game_verbose_log=False,
            log_directory=output_dir,
        )
        room.add_player(QuietRandomAgent("R1", log_directory=output_dir))
        room.add_player(QuietRandomAgent("R2", log_directory=output_dir))
        room.add_player(QuietRandomAgent("R3", log_directory=output_dir))
        rl = DemoRLAgent("RL", log_directory=output_dir, model_path=model_path)
        room.add_player(rl)

        room.start_new_game()

    print("\n" + "=" * 60, file=sys.__stdout__)
    print("  DEMONSTRATION COMPLETE", file=sys.__stdout__)
    print("=" * 60, file=sys.__stdout__)
    win_rate = rl.win_count / max(1, rl.match_count) * 100
    avg_pos = np.mean(rl.positions) if rl.positions else 0
    print(f"\n  RL Agent summary:", file=sys.__stdout__)
    print(f"    - Win rate: {win_rate:.1f}%", file=sys.__stdout__)
    print(f"    - Avg position: {avg_pos:.2f}", file=sys.__stdout__)
    print(f"    - Matches played: {rl.match_count}", file=sys.__stdout__)
    print("\n", file=sys.__stdout__)

    return room, rl


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Chef's Hat RL Agent Demonstration")
    p.add_argument("--matches", type=int, default=5, help="Number of matches to play")
    p.add_argument("--model", type=str, default=None, help="Path to trained model (.pt)")
    p.add_argument("--output", type=str, default="demo_output")
    p.add_argument("--verbose", action="store_true", help="Show library output")
    p.add_argument("--train-then-demo", action="store_true",
                   help="Train 100 matches first, then run demo with saved model")
    args = p.parse_args()

    if args.train_then_demo:
        # Train then run a short demo with the saved policy
        print("\n[1/2] Training RL agent (100 matches)...")
        from train_gym import run_training
        out_dir = os.path.join(args.output, "train")
        room, agent = run_training(matches=100, seed=42, output_dir=out_dir)
        model_path = os.path.join(args.output, "trained_model.pt")
        if TORCH_AVAILABLE and hasattr(agent, "net") and agent.net is not None:
            os.makedirs(args.output, exist_ok=True)
            torch.save(agent.net.state_dict(), model_path)
        else:
            model_path = None
        print("\n[2/2] Running demonstration with trained model...")
        run_demo(matches=5, model_path=model_path, output_dir=args.output, quiet_libs=not args.verbose)
    else:
        run_demo(
            matches=args.matches,
            model_path=args.model,
            output_dir=args.output,
            quiet_libs=not args.verbose,
        )
