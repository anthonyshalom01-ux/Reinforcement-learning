<<<<<<< HEAD
# Chef's Hat RL Agent — University Assignment

Reinforcement learning agent for the Chef's Hat Gym multi-agent card game environment.

## Setup

```bash
pip install -r requirements.txt
```

## Web Interface

```bash
python app.py
```

Then open **http://localhost:5000** in your browser. Use the interface to run demonstrations or training.

## Configuration

Edit `config.py`:
- **STUDENT_ID**: Your student ID → variant = `STUDENT_ID % 7` (0–6)
- **SEED**: Random seed for reproducibility

## Training

```bash
python train.py --matches 500 --variant 0 --seed 42
```

## Demonstration

```bash
python demo.py --matches 5              # Quick demo (5 matches, untrained)
python demo.py --model outputs/.../trained_model.pt  # Demo with trained model
python demo.py --train-then-demo       # Train 100 matches, then demo
```

## Evaluation

```bash
python evaluate.py --model <path_to_model.pt> --matches 100
```

## Experiments

```bash
python experiments.py          # Full runs
python experiments.py --quick # Quick test (fewer matches)
```

## Project Structure

```
├── config.py        # Variant and hyperparameter config
├── train.py         # Training loop
├── evaluate.py      # Evaluation and metrics
├── experiments.py   # Hyperparameter / ablation experiments
├── agents/
│   └── rl_agent.py  # DQN/A2C agent with action masking
├── REPORT.md        # Full academic report
└── requirements.txt
```

## Variants (ID mod 7)

| 0 | DQN baseline | 4 | A2C (Actor-Critic) |
| 1 | Double DQN | 5 | Prioritized replay |
| 2 | Dense reward shaping | 6 | Curriculum opponents |
| 3 | DQN + dense shaping | | |

See `REPORT.md` for full design, implementation details, and critical analysis.
=======
# Reinforcement-learning
>>>>>>> 295538e693116b987d9589f4eef7a65caa781f85
