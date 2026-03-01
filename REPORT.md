# RL Agent for Chef's Hat Gym: Design, Implementation, and Critical Evaluation

**University Reinforcement Learning Assignment**  
**Variant:** *(Replace with: your_student_id % 7 → Variant 0–6)*

---

## 1. Chef's Hat Environment Dynamics and Challenges

### 1.1 Environment Overview

Chef's Hat Gym is a multi-agent, turn-based card game environment implementing the physical Chef's Hat board game (Barros et al., 2021). Four players compete to empty their hands by playing cards onto a shared "pizza" (playing field). Each game consists of multiple matches; roles (Chef, Sous-Chef, Waiter, Dishwasher) and card exchanges vary between matches.

### 1.2 Key Challenges

| Challenge | Description | RL Implications |
|-----------|-------------|-----------------|
| **Multi-agent** | 3 opponents act sequentially; their policies are non-stationary during learning | Credit assignment, opponent modelling, self-play |
| **Delayed rewards** | Match outcome (1st–4th place) only at match end; game outcome after all matches | Sparse reward, need for reward shaping or long-horizon credit assignment |
| **Stochasticity** | Card dealing, role assignment, starting player, opponent behaviour | High variance; importance of exploration and robust value estimates |
| **Large action space** | 200 discrete actions (CQJ format: Card, Quantity, Joker) | Action masking essential; invalid actions must be excluded |
| **Partial observability** | Own hand visible; opponents' hands hidden; board state shared | POMDP nature; history or belief state may help |
| **Variable-length episodes** | Match length depends on play; game = multiple matches | Episodic and game-level returns; truncation handling |

### 1.3 Game Mechanics (Relevant for RL)

- **Cards:** 1–11 (ingredients), 12 (joker). Values normalized by 13 in common setups.
- **Observation:** `hand` (padded to 17), `board` (padded to 11), `possible_actions` (list of CQJ strings).
- **Actions:** 200 indexed actions from `get_high_level_actions()`; only `possible_actions` subset is valid per turn.
- **Scoring:** 1st → 3 pts, 2nd → 2 pts, 3rd → 1 pt, 4th → 0 pt. Performance score: `((points×10)/rounds)/matches`.

---

## 2. State Representation and Action Handling

### 2.1 Proposed State Representation

**State vector (28 dims):** Concatenation of normalized hand and board:

```python
hand_norm = np.array(hand, dtype=np.float32).flatten() / 13.0   # 17 dims
board_norm = np.array(board, dtype=np.float32).flatten() / 13.0  # 11 dims
state = np.concatenate([hand_norm, board_norm])  # shape (28,)
```

**Justification:**
- Bounded [0, 1] for stable neural network inputs.
- Preserves card identity (1–11) and joker (12/13).
- Minimal and sufficient for turn-level decisions (what to play, given board).
- Optional extensions: role embedding, match index, round count for richer context.

### 2.2 Action Handling and Masking

**Action space:** 200 discrete indices mapping to CQJ strings (e.g. `C1;Q1;J0`, `pass`).

**Masking strategy:** Only actions in `possible_actions` are valid. Mask invalid actions by setting Q-values to `-inf` before argmax:

```python
valid_indices = [all_actions.index(a) for a in possible_actions]
mask = np.zeros(200, dtype=np.float32)
mask[valid_indices] = 1.0
q_values_masked = np.where(mask, q_values, -np.inf)
action = np.argmax(q_values_masked)
```

**Preference for non-pass:** When multiple valid actions exist, down-weight or exclude `pass` to encourage proactive play; fall back to `pass` only when it is the sole option.

---

## 3. Recommended Algorithm and Implementation

### 3.1 Algorithm Choice

| Variant (ID mod 7) | Algorithm | Rationale |
|-------------------|-----------|------------|
| 0 | DQN | Baseline; handles discrete actions and masking |
| 1 | Double DQN | Reduces overestimation; better stability |
| 2 | Dueling DQN | Separates value and advantage; large action space |
| 3 | DQN + Dense Shaping | Addresses sparse rewards |
| 4 | Actor-Critic (A2C) | On-policy; suitable for episodic games |
| 5 | DQN + Prioritized Replay | Focus on informative transitions |
| 6 | DQN + Curriculum Opponents | Gradual difficulty increase |

**Default recommendation:** **Double Dueling DQN** (Variants 1+2) with action masking and light reward shaping—good balance of stability and performance for Chef's Hat.

### 3.2 Implementation Architecture

The Chef's Hat API is callback-based (Room → Agent), not standard `env.step()`. The RL agent is implemented as a Chef's Hat `BaseAgent` that:

1. Receives observations via `request_action(observation)`.
2. Encodes state, applies action mask, selects action.
3. Stores transitions; receives match-end reward via `update_match_over`.
4. Performs TD updates (e.g. mini-batch replay) when sufficient data available.

---

## 4. Training Loop and Reproducibility

### 4.1 Training Loop Structure

```
for game in range(num_games):
    room.run()  # async; agents receive request_action, update_* callbacks
    # Our RL agent internally:
    #   - Collects (s, a, r, s', done) per turn
    #   - At match end: assigns final reward, pushes to replay buffer
    #   - Samples minibatch, computes TD target, updates network
```

### 4.2 Reproducibility Setup

```python
def set_seeds(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Config: `config.yaml` or `config.py` with `SEED`, `VARIANT`, hyperparameters.

---

## 5. Evaluation Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Win rate** | % of matches placed 1st | Primary success indicator |
| **Avg. position** | Mean finishing rank (1–4) | Overall placement quality |
| **Performance score** | `((points×10)/rounds)/matches` | Incorporates efficiency |
| **Learning curves** | Metric vs. matches/games | Training progress |
| **Invalid action rate** | % of turns with invalid action attempts | Masking effectiveness |

---

## 6. Experimental Design

### 6.1 Hyperparameter Search

- Learning rate: `[1e-4, 3e-4, 1e-3]`
- Gamma: `[0.95, 0.99, 0.999]`
- Epsilon decay: `[0.995, 0.999, 0.9995]`
- Batch size: `[128, 256, 512]`

### 6.2 Exploration Analysis

- Plot epsilon (or equivalent) vs. matches.
- Compare win rate under different decay schedules.

### 6.3 Opponent Behaviour

- **Random opponents:** Baseline.
- **Mixed:** 2× Random + 1× Heuristic (e.g. lowest-card-first).
- **Self-play:** Train against past checkpoints.

---

## 7. Critical Analysis

### 7.1 Limitations

- **Sparse rewards:** Match-end signal only; long delays.
- **Non-stationarity:** Opponent policies change (e.g. if training other agents).
- **Credit assignment:** Linking early actions to final rank is difficult.
- **Sample efficiency:** Many games needed for stable learning.

### 7.2 Instability Sources

- High variance in returns from stochastic dealing and opponents.
- Overestimation in Q-learning (mitigated by Double DQN).
- Catastrophic forgetting if replay buffer too small.

### 7.3 Mitigations

- Reward shaping (e.g. small negative for pass, positive for finishing).
- Action masking to avoid invalid actions.
- Target network and soft updates.
- Sufficient exploration (ε-greedy or similar) early in training.
- Multiple seeds for statistical reporting.

---

## Appendix A: Code Examples

### Training

```python
from train import run_training
from config import get_variant_config

VARIANT = 12345 % 7  # Replace 12345 with your student ID
room, agent = run_training(
    matches=500,
    variant=VARIANT,
    seed=42,
    output_dir="outputs",
)
print(f"Win rate: {agent.win_count / agent.match_count * 100:.1f}%")
```

### Evaluation

```python
python evaluate.py --model outputs/train_v0_20250217_120000/model/rl_model.pt --matches 100
```

### Experiments (Hyperparameters)

```python
python experiments.py                    # Full experiments (200 matches each)
python experiments.py --quick           # Quick test (50 matches each)
```

### Reproducibility

```python
# config.py
STUDENT_ID = 12345  # Your ID
VARIANT = STUDENT_ID % 7
SEED = 42
```

---

## References

- Barros, P., et al. (2021). It's Food Fight! Designing the Chef's Hat Card Game. HRI Companion.
- Barros, P., & Sciutti, A. (2021). Learning from learners: Adapting RL agents to be competitive. ICPR.
