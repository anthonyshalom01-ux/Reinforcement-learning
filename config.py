"""
Configuration for Chef's Hat RL experiments.

Variant is determined by: STUDENT_ID % 7 (0-6). Each variant selects algorithm
and options (e.g. Double DQN, reward shaping) used by train.py and agents.rl_agent.
"""

# =============================================================================
# VARIANT CONFIGURATION - Set your student ID to get your assigned variant
# =============================================================================
STUDENT_ID = 0  # Replace with your student ID
VARIANT = STUDENT_ID % 7  # Variant 0-6

# Variant definitions: algorithm and flags (double_dqn, dueling, reward_shaping, etc.)
VARIANTS = {
    0: {"algorithm": "dqn", "double_dqn": False, "dueling": True, "reward_shaping": "minimal"},
    1: {"algorithm": "dqn", "double_dqn": True, "dueling": True, "reward_shaping": "minimal"},
    2: {"algorithm": "dqn", "double_dqn": True, "dueling": True, "reward_shaping": "dense"},
    3: {"algorithm": "dqn", "double_dqn": False, "dueling": True, "reward_shaping": "dense"},
    4: {"algorithm": "a2c", "reward_shaping": "dense"},
    5: {"algorithm": "dqn", "double_dqn": True, "dueling": True, "prioritized_replay": True},
    6: {"algorithm": "dqn", "double_dqn": True, "dueling": True, "curriculum_opponents": True},
}

# Reproducibility: used by set_all_seeds()
SEED = 42

# Training defaults (used when not overridden by CLI or experiments)
TRAIN_MATCHES = 500
EVAL_MATCHES = 100
NUM_RANDOM_OPPONENTS = 3

# DQN hyperparameters (state_size 28 = hand 17 + board 11; action_size 200)
DQN_CONFIG = {
    "state_size": 28,
    "action_size": 200,
    "gamma": 0.99,
    "lr": 1e-4,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.998,
    "batch_size": 256,
    "memory_size": 20000,
    "target_update_tau": 0.01,
    "hidden_layers": [256, 128, 64],
}

# A2C hyperparameters (Variant 4)
A2C_CONFIG = {
    "state_size": 28,
    "action_size": 200,
    "gamma": 0.99,
    "lr_actor": 3e-4,
    "lr_critic": 1e-3,
    "hidden_size": 128,
    "entropy_coef": 0.01,
}


def get_variant_config(variant=None):
    """Return the config dict for variant 0-6; if variant is None, use VARIANT from this module."""
    v = variant if variant is not None else VARIANT
    return VARIANTS.get(v, VARIANTS[0])


def set_all_seeds(seed: int = SEED):
    """Set random seeds for Python, NumPy, and PyTorch (if available) for reproducible runs."""
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    random.seed(seed)
    np.random.seed(seed)
