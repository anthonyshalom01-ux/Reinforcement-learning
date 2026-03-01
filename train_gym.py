"""
Training script using the installed ChefsHatGym package (pip: chefshatgym).

Uses ChefsHatRoomLocal and ChefsHatPlayer: 1 DQN agent vs 3 random agents.
Observation: 228 dims (board 11 + hand 17 + possible_actions 200). Action: 200-dim one-hot.
Run: python train_gym.py --matches 100 --seed 42
"""

# NumPy 2.x compatibility: gym's env_checker uses np.bool8, removed in NumPy 2
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
from ChefsHatGym.env import ChefsHatEnv

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seeds(seed):
    """Set random seeds for reproducibility (Python, NumPy, and PyTorch if available)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


class DQNPlayer(ChefsHatPlayer):
    """
    DQN agent implementing the ChefsHatPlayer interface.
    State: first 28 dims of observation (board 11 + hand 17). Action: 200-dim one-hot.
    Uses replay buffer, target network, soft updates, and epsilon-greedy exploration.
    """

    suffix = "DQN"

    def __init__(self, name, log_directory="", training=True, verbose_console=False, verbose_log=False, use_sufix=True):
        super().__init__(self.suffix, name, "", verbose_console, verbose_log, log_directory, use_sufix)
        self.training = training
        self.log_directory = log_directory
        self.this_log_folder = os.path.join(os.path.abspath(log_directory), self.name)
        if log_directory:
            os.makedirs(self.this_log_folder, exist_ok=True)
        self.log = lambda m: None
        self.saveModelIn = self.this_log_folder

        self.state_size = 28
        self.action_size = 200
        self.gamma = 0.99
        self.epsilon = 1.0 if training else 0.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.batch_size = 128
        self.memory = []
        self.memory_size = 10000
        self.positions = []
        self.win_count = 0
        self.match_count = 0
        self.last_state = None
        self.last_action = None
        self.last_mask = None
        self.episode = []

        if TORCH_AVAILABLE:
            self.net = nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_size),
            )
            self.target = nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_size),
            )
            self.target.load_state_dict(self.net.state_dict())
            self.optimizer = Adam(self.net.parameters(), lr=1e-4)
        else:
            self.net = self.target = self.optimizer = None

    def _obs_to_state(self, obs):
        """Extract state vector from full 228-dim observation. obs layout: [0:11] board, [11:28] hand, [28:228] possible actions."""
        return np.array(obs[:28], dtype=np.float32)

    def _get_mask(self, obs):
        """Extract action mask: 200-dim binary, 1 = valid action."""
        return np.array(obs[28:228], dtype=np.float32)

    def _store(self, s, m, a, r, s2, m2, done):
        """Append transition (s, mask, a, r, s', mask', done) to replay buffer; drop oldest if over capacity."""
        self.memory.append((s.copy(), m.copy(), a, r, s2.copy(), m2.copy(), done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def _update(self):
        """One DQN update: sample batch, compute TD targets with target network (masked), MSE loss, soft target update."""
        if not TORCH_AVAILABLE or len(self.memory) < self.batch_size:
            return
        idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        states = torch.from_numpy(np.array([b[0] for b in batch])).float()
        masks = torch.from_numpy(np.array([b[1] for b in batch])).float()
        actions = torch.from_numpy(np.array([b[2] for b in batch])).long()
        rewards = torch.from_numpy(np.array([b[3] for b in batch], dtype=np.float32)).float()
        next_s = torch.from_numpy(np.array([b[4] for b in batch])).float()
        next_m = torch.from_numpy(np.array([b[5] for b in batch])).float()
        dones = torch.from_numpy(np.array([b[6] for b in batch])).float()

        with torch.no_grad():
            next_q = self.target(next_s)
            next_q[next_m == 0] = -float("inf")  # action masking
            targets = rewards + self.gamma * next_q.max(dim=-1)[0] * (1 - dones)
        q = self.net(states)
        q_a = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(q_a, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Soft update: target = 0.01*policy + 0.99*target
        for p, tp in zip(self.net.parameters(), self.target.parameters()):
            tp.data.copy_(0.01 * p.data + 0.99 * tp.data)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, observations):
        """Select action: epsilon-greedy with action masking; store transition and run _update when training."""
        obs = np.array(observations)
        state = self._obs_to_state(obs)
        mask = self._get_mask(obs)
        valid = np.where(mask > 0.5)[0]
        if len(valid) == 0:
            valid = [199]  # fallback to pass

        if self.training and np.random.rand() < self.epsilon:
            a_idx = int(np.random.choice(valid))
        elif TORCH_AVAILABLE:
            with torch.no_grad():
                q = self.net(torch.from_numpy(state).float().unsqueeze(0))
                q = q.numpy().flatten()
                q[mask < 0.5] = -1e9  # mask invalid actions
                a_idx = int(np.argmax(q))
        else:
            a_idx = int(np.random.choice(valid))

        action = np.zeros(200, dtype=np.float32)
        action[a_idx] = 1.0  # ChefsHatGym expects one-hot over 200 actions

        if self.last_state is not None and self.training:
            r = -0.02
            self._store(self.last_state, self.last_mask, self.last_action, r, state, mask, False)
            self._update()

        self.last_state = state
        self.last_action = a_idx
        self.last_mask = mask
        return action.tolist()

    def update_start_match(self, cards, players, starting_player):
        self.episode = []
        self.last_state = None

    def get_exhanged_cards(self, cards, amount):
        """Return highest `amount` cards for role-based exchange (Chef's Hat rule)."""
        return sorted(cards)[-amount:]

    def update_exchange_cards(self, cards_sent, cards_received):
        pass

    def do_special_action(self, info, specialAction):
        """Accept special action (e.g. Food Fight / Dinner Served) when offered."""
        return True

    def observe_special_action(self, action_type, player):
        pass

    def get_reward(self, info):
        """Required by ChefsHatPlayer; reward is applied in update_end_match."""
        pass

    def update_end_match(self, envInfo):
        """Called when a match ends: compute place from Match_Score, store terminal transition, update network."""
        scores = envInfo.get("Match_Score", [])
        self.match_count += 1
        place = 4
        if scores:
            # Match_Score: 3=1st, 2=2nd, 1=3rd, 0=4th. We are player index 3 (added last).
            our_points = scores[3] if len(scores) > 3 else 0
            place = 4 - our_points if our_points >= 0 else 4
        self.positions.append(place)
        if place == 1:
            self.win_count += 1
        reward = 3.0 if place == 1 else -0.02
        if self.last_state is not None and self.training:
            self._store(self.last_state, self.last_mask, self.last_action, reward,
                        self.last_state, self.last_mask, True)
            self._update()
        self.last_state = None

    def update_action_others(self, envInfo):
        pass

    def update_my_action(self, envInfo):
        pass

    def update_game_over(self):
        pass


def run_training(matches=100, seed=42, output_dir="outputs"):
    """Create room with 3 random + 1 DQN player, run one full game (matches), return room and DQN agent."""
    set_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)
    room = ChefsHatRoomLocal(
        room_name="Train",
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
    room.add_player(AgentRandon("R1", log_directory=output_dir, verbose_console=False, verbose_log=False))
    room.add_player(AgentRandon("R2", log_directory=output_dir, verbose_console=False, verbose_log=False))
    room.add_player(AgentRandon("R3", log_directory=output_dir, verbose_console=False, verbose_log=False))
    rl_agent = DQNPlayer("RL", log_directory=output_dir, training=True)
    room.add_player(rl_agent)

    print("Starting training...")
    room.start_new_game()
    print("Training complete.")
    win_rate = rl_agent.win_count / max(1, rl_agent.match_count) * 100
    avg_pos = np.mean(rl_agent.positions) if rl_agent.positions else 0
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg position: {avg_pos:.2f}")
    return room, rl_agent


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="outputs")
    args = p.parse_args()
    run_training(matches=args.matches, seed=args.seed, output_dir=args.output)
