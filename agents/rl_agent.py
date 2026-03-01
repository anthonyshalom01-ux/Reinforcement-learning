"""
RL Agent for Chef's Hat Gym (Room-based / callback API).

Used by train.py with the async Room: implements update_* and request_* callbacks.
- Supports DQN, Double DQN, Dueling DQN (algorithm="dqn"), and A2C (algorithm="a2c").
- State: 28 dims (hand 17 + board 11), normalised by 13. Action: 200 discrete, with masking.
- Reward: match-end placement reward plus optional shaping (pass penalty, etc.).
"""

import os
import logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Adam = None


def _get_logger(name, log_dir, verbose=False):
    """Create a logger that optionally writes to a file in log_dir and/or console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False
    return logger


# -----------------------------------------------------------------------------
# Neural network architectures
# -----------------------------------------------------------------------------

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: shared backbone, then value V(s) and advantage A(s,a); Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)).
    Reduces overestimation by separating state value from action-dependent advantage.
    """

    def __init__(self, state_size, action_size, hidden_sizes=(256, 128, 64)):
        super().__init__()
        layers = []
        prev = state_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.shared = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)
        self.advantage = nn.Linear(prev, action_size)

    def forward(self, x):
        feat = self.shared(x)
        v = self.value(feat)
        adv = self.advantage(feat)
        q = v + (adv - adv.mean(dim=-1, keepdim=True))
        return q


class ActorCriticNet(nn.Module):
    """
    Actor-Critic: shared layers, then policy head (logits over actions) and value head.
    forward() accepts optional mask to zero out invalid action logits before softmax.
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        feat = self.shared(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits, value


# -----------------------------------------------------------------------------
# RLAgent: implements Chef's Hat Room agent interface (update_* / request_*)
# -----------------------------------------------------------------------------

class RLAgent:
    """
    Single agent that the Room calls via update_game_start, request_action, update_match_over, etc.
    Observation comes as dict with hand, board, possible_actions; we encode state and apply action masking.
    """

    def __init__(
        self,
        name="RLAgent",
        log_directory="",
        verbose_log=False,
        training=True,
        model_path=None,
        state_size=28,
        action_size=200,
        gamma=0.99,
        lr=1e-4,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        batch_size=256,
        memory_size=20000,
        algorithm="dqn",
        double_dqn=True,
        dueling=True,
        reward_shaping="minimal",
        prioritized_replay=False,
        target_tau=0.01,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        self.name = name
        self.log_directory = log_directory
        self.verbose_log = verbose_log
        self.training = training
        self.model_path = model_path
        self.log_dir = os.path.join(os.path.abspath(log_directory), "agents", name) if log_directory else ""
        self.log = _get_logger(f"RL_{name}", self.log_dir, verbose_log).info if verbose_log else (lambda m: None)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.reward_shaping = reward_shaping
        self.prioritized_replay = prioritized_replay
        self.target_tau = target_tau

        self.epsilon = epsilon if training else 0.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = []
        self.memory_size = memory_size

        self.all_actions = None
        self.last_state = None
        self.last_action = None
        self.last_mask = None
        self.episode = []
        self.positions = []
        self.rewards_history = []
        self.loss_history = []
        self.win_count = 0
        self.match_count = 0

        # Build networks
        self._build_networks(lr)
        self.train(training)

    def _build_networks(self, lr):
        if self.algorithm == "a2c":
            hidden = getattr(self, "hidden_size", 128)
            self.policy_net = ActorCriticNet(
                self.state_size, self.action_size,
                hidden_size=hidden
            )
            self.target_net = None
            self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        else:
            self.policy_net = DuelingQNetwork(
                self.state_size, self.action_size
            )
            self.target_net = DuelingQNetwork(
                self.state_size, self.action_size
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = Adam(self.policy_net.parameters(), lr=lr)

    def _obs_to_state(self, obs):
        """Convert observation dict to state vector."""
        hand = np.array(obs["hand"], dtype=np.float32).flatten() / 13.0
        board = np.array(obs["board"], dtype=np.float32).flatten() / 13.0
        # Ensure correct size (hand 17, board 11)
        hand = np.pad(hand, (0, max(0, 17 - len(hand))))[:17]
        board = np.pad(board, (0, max(0, 11 - len(board))))[:11]
        return np.concatenate([hand, board]).astype(np.float32)

    def _get_action_mask(self, possible_actions):
        """Create binary mask for valid actions."""
        mask = np.zeros(self.action_size, dtype=np.float32)
        if self.all_actions is None:
            return mask
        for a in possible_actions:
            try:
                idx = self.all_actions.index(a)
                mask[idx] = 1.0
            except (ValueError, TypeError):
                pass
        return mask

    def _get_valid_indices(self, possible_actions):
        """Return list of valid action indices."""
        if self.all_actions is None:
            return []
        return [self.all_actions.index(a) for a in possible_actions if a in self.all_actions]

    def _act(self, state, mask, valid_indices):
        """Epsilon-greedy (DQN) or sample from policy (A2C); invalid actions masked to -inf before argmax/sample."""
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)

        if self.training and np.random.rand() < self.epsilon:
            if valid_indices:
                return int(np.random.choice(valid_indices))
            return 199  # pass

        with torch.no_grad():
            if self.algorithm == "a2c":
                logits, _ = self.policy_net(state_t, mask_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                return int(dist.sample().item())
            else:
                q = self.policy_net(state_t)
                q_masked = q.clone()
                q_masked[mask_t == 0] = -float("inf")
                if torch.all(torch.isinf(q_masked)):
                    return int(np.random.choice(valid_indices)) if valid_indices else 199
                return int(torch.argmax(q_masked, dim=-1).item())

    def _store_transition(self, s, mask, a, r, s_next, mask_next, done):
        """Store transition in replay buffer."""
        self.memory.append((s.copy(), mask.copy(), a, r, s_next.copy(), mask_next.copy(), done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def _shaped_reward(self, action_str, place=None):
        """Dense reward shaping."""
        r = 0.0
        if self.reward_shaping == "minimal":
            r -= 0.02
            if str(action_str).lower() == "pass":
                r -= 1.0
        elif self.reward_shaping == "dense":
            r -= 0.01
            if str(action_str).lower() == "pass":
                r -= 0.5
            if place is not None:
                if place == 1:
                    r += 3.0
                elif place == 2:
                    r += 1.0
                elif place == 3:
                    r -= 0.5
                else:
                    r -= 1.0
        return r

    def _dqn_update(self):
        """Sample batch from replay buffer; TD target with target net (Double DQN: policy picks action, target evaluates); MSE loss; soft target update; epsilon decay."""
        if self.target_net is None or len(self.memory) < self.batch_size:
            return
        idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        states = torch.from_numpy(np.array([b[0] for b in batch])).float()
        masks = torch.from_numpy(np.array([b[1] for b in batch])).float()
        actions = torch.from_numpy(np.array([b[2] for b in batch])).long()
        rewards = torch.from_numpy(np.array([b[3] for b in batch], dtype=np.float32)).float()
        next_states = torch.from_numpy(np.array([b[4] for b in batch])).float()
        next_masks = torch.from_numpy(np.array([b[5] for b in batch])).float()
        dones = torch.from_numpy(np.array([b[6] for b in batch])).float()

        with torch.no_grad():
            if self.double_dqn:
                next_q = self.policy_net(next_states)
                next_q_masked = next_q.clone()
                next_q_masked[next_masks == 0] = -float("inf")
                next_acts = next_q_masked.argmax(dim=-1)
                target_q = self.target_net(next_states)
                target_next = target_q.gather(1, next_acts.unsqueeze(-1)).squeeze(-1)
            else:
                target_q = self.target_net(next_states)
                target_q[next_masks == 0] = -float("inf")
                target_next = target_q.max(dim=-1)[0]
            targets = rewards + self.gamma * target_next * (1 - dones)

        q = self.policy_net(states)
        q_a = q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(q_a, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

        # Soft update target
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.target_tau * p.data + (1 - self.target_tau) * tp.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _a2c_update(self):
        """Compute returns along episode, then actor loss (policy gradient with advantage) + critic loss (MSE to returns)."""
        if len(self.episode) < 2:
            return
        states = torch.from_numpy(np.array([e[0] for e in self.episode])).float()
        masks = torch.from_numpy(np.array([e[1] for e in self.episode])).float()
        actions = torch.from_numpy(np.array([e[2] for e in self.episode])).long()
        rewards = np.array([e[3] for e in self.episode])
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.from_numpy(np.array(returns)).float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits, values = self.policy_net(states, masks)
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob_a = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        advantage = returns - values.squeeze(-1)
        actor_loss = -(log_prob_a * advantage.detach()).mean()
        critic_loss = F.mse_loss(values.squeeze(-1), returns)
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

    # --- Chef's Hat Agent Interface ---

    def update_game_start(self, payload):
        self.log(f"Game start: {list(payload.keys())}")
        if "actions" in payload:
            self.all_actions = list(payload["actions"].values())

    def update_game_over(self, payload):
        """Save policy weights to model_path when game ends (if training)."""
        if self.training and self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.policy_net.state_dict(), self.model_path)

    def update_new_hand(self, payload):
        self.episode = []
        self.last_state = None
        self.last_action = None
        self.last_mask = None

    def update_new_roles(self, payload):
        pass

    def update_food_fight(self, payload):
        pass

    def update_dinner_served(self, payload):
        pass

    def update_hand_after_exchange(self, payload):
        pass

    def update_start_match(self, payload):
        pass

    def update_match_over(self, payload):
        """Compute place from finishing_order, assign terminal reward, store final transition and run one update (DQN or A2C)."""
        finishing_order = payload.get("finishing_order", [])
        player_name = self.name
        try:
            place = finishing_order.index(player_name) + 1
        except ValueError:
            place = 4
        self.positions.append(place)
        self.match_count += 1
        if place == 1:
            self.win_count += 1
        reward = 3.0 if place == 1 else (0.0 if place == 2 else (-0.5 if place == 3 else -1.0))
        if self.reward_shaping == "minimal":
            reward = 3.0 if place == 1 else -0.02
        self.rewards_history.append(reward)

        if self.last_state is not None and self.last_action is not None and self.last_mask is not None and self.training:
            final_transition = (
                self.last_state, self.last_mask, self.last_action,
                reward, self.last_state, self.last_mask, True
            )
            self.episode.append(final_transition)
            if self.algorithm == "a2c":
                self._a2c_update()
            else:
                self._store_transition(*final_transition)
                self._dqn_update()
            self.episode = []
        self.last_state = None
        self.last_action = None
        self.last_mask = None

    def update_player_action(self, payload):
        pass

    def update_pizza_declared(self, payload):
        pass

    def request_cards_to_exchange(self, payload):
        """Return n highest cards for role-based exchange (e.g. Chef receives from Dishwasher)."""
        hand = payload["hand"]
        n = payload["n"]
        return sorted(hand)[-n:]

    def request_special_action(self, payload):
        """Accept special action (Food Fight / Dinner Served) when offered."""
        return True

    def request_action(self, observation):
        """Encode state, build mask, optionally down-weight pass; select action; store transition and update if training."""
        state = self._obs_to_state(observation)
        possible_actions = list(observation.get("possible_actions", []))
        mask = self._get_action_mask(possible_actions)
        valid_indices = self._get_valid_indices(possible_actions)
        if not valid_indices:
            valid_indices = [199]
            mask[199] = 1.0

        # Prefer non-pass when multiple actions available (set pass mask to 0 so Q-net rarely chooses it)
        pass_idx = None
        if self.all_actions:
            for i, a in enumerate(self.all_actions):
                if str(a).lower() == "pass":
                    pass_idx = i
                    break
        mask_eff = mask.copy()
        if pass_idx is not None and len(valid_indices) > 1 and pass_idx in valid_indices:
            valid_indices_nopass = [i for i in valid_indices if i != pass_idx]
            if valid_indices_nopass:
                mask_eff[pass_idx] = 0
                valid_indices = valid_indices_nopass

        action_index = self._act(state, mask_eff, valid_indices)
        action_str = self.all_actions[action_index] if self.all_actions else "pass"

        shaped = self._shaped_reward(action_str)
        if self.last_state is not None and self.last_action is not None and self.last_mask is not None and self.training:
            self.episode.append((
                self.last_state, self.last_mask, self.last_action,
                shaped, state, mask, False
            ))
            self._store_transition(
                self.last_state, self.last_mask, self.last_action,
                shaped, state, mask, False
            )
            if self.algorithm != "a2c" and len(self.memory) >= self.batch_size:
                self._dqn_update()

        self.last_state = state
        self.last_action = action_index
        self.last_mask = mask
        return action_index
