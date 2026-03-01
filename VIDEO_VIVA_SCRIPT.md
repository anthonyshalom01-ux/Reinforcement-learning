# Video Viva Script (3–5 Minutes)

**Assigned variant:** Defined in `config.py` as `VARIANT = STUDENT_ID % 7`.  
**Current setting:** `STUDENT_ID = 0` → **Variant 0**.  
**Design used:** DQN baseline, with dueling architecture and minimal reward shaping (see VARIANTS[0] in config.py).

Time cues are approximate. If your student ID gives a different variant, update the "My Assigned Variant" section and the design description using the table at the end.

---

## [0:00–0:45] 1. Chef's Hat Gym Environment

"Hello. This video presents my reinforcement learning assignment using Chef's Hat Gym.

**Chef's Hat Gym** is a multi-agent, turn-based card game environment. Four players compete to empty their hands by playing cards onto a shared playing area—the 'pizza.' Each **game** is made of several **matches**. Between matches, roles change—Chef, Sous-Chef, Waiter, Dishwasher—and cards are exchanged according to those roles.

From an RL perspective, the environment has several important properties:

- **Multi-agent**: our agent plays against three opponents, so the environment is non-stationary.
- **Delayed rewards**: we only get a clear outcome at the end of each match—first place gets 3 points, second 2, third 1, fourth 0—so rewards are sparse.
- **Stochasticity**: card dealing, who starts, and opponent behaviour are all random.
- **Large action space**: there are 200 discrete actions in a 'CQJ' format—card, quantity, joker—and only a subset is valid each turn, so **action masking** is essential.

The observation we use is 28 dimensions: normalized hand (17) plus board (11), and we mask invalid actions before the agent chooses."

---

## [0:45–1:30] 2. My Assigned Variant and Design Choices

"My assigned variant is **Variant 0** (from student ID mod 7; set in `config.py` as STUDENT_ID = 0).

For **Variant 0**, the design is: **DQN baseline with dueling architecture and minimal reward shaping**. So we use standard DQN (not Double DQN), a dueling Q-network that separates value and advantage, and light reward shaping.

My main design choices were:

- **State representation**: 28-dimensional vector—hand (17) and board (11) normalized by 13—so the network gets bounded, stable inputs.
- **Action handling**: binary mask over 200 actions; invalid actions get Q-values set to minus infinity so the agent never selects them. I also bias the agent to avoid 'pass' when other moves are legal, to encourage more active play.
- **Algorithm**: DQN implemented in PyTorch—policy network plus target network with soft updates (tau = 0.01), replay buffer, and epsilon-greedy exploration. The Q-network is dueling (shared backbone, then value and advantage heads).
- **Reward**: match-end placement reward (3 for first down to 0 for fourth), plus **minimal** shaping—a small penalty per step (e.g. -0.02) and an extra penalty for passing—to give learning signal before the match ends.

Reproducibility is handled by fixing seeds in `config.py` for Python, NumPy, and PyTorch."

---

## [1:30–2:15] 3. Demonstrate the Trained Agent

"I'll demonstrate the trained agent playing. **[Switch to screen.]**

In the repo you can run the demo in two ways: from the command line with `python demo.py --matches 5` or using the web interface with `python app.py` and opening the browser.

**[Either run `python demo.py --matches 5` in terminal, or show the web UI and click 'Run Demo'.]**

Here we have one RL agent against three random opponents. Each line is the result of one match—first, second, third, or fourth place. At the end we get a summary: win rate and average position.

**[If you have a trained model:] I'll run a short demo with a trained model: `python demo.py --model [path] --matches 5`. You can see the agent consistently achieving better positions than in the untrained case.**

The same logic runs in the web interface: the 'Demonstration' section runs a demo and the 'Output' panel shows match-by-match results and the summary."

---

## [2:15–3:00] 4. Key Experimental Results

"For evaluation I used:

- **Win rate**—percentage of matches where the agent came first.
- **Average position**—mean finishing place over matches.
- **Training curves**—how win rate or loss changes over matches or updates.

**[If you have results:] After training for [e.g. 100 or 500] matches, the agent reached about [X]% win rate and an average position of about [Y]. Learning curves in the repo show that performance improves with more matches and that loss decreases over updates.**

I also ran experiments from `experiments.py`: learning rate, gamma, and epsilon decay sweeps, and comparison across variants. Results are saved in `experiments/experiment_summary.json`. **[If you have a plot or table, mention it.]** In general, Double DQN and some reward shaping gave more stable learning than plain DQN, and too high a learning rate or too fast epsilon decay hurt performance."

---

## [3:00–4:00] 5. Limitations, Challenges, and Improvements

"Several limitations and challenges are worth noting.

- **Sparse rewards**: most of the reward comes only at match end, so credit assignment is hard. I addressed this with reward shaping; alternatives would be eligibility traces or longer-horizon value estimates.
- **Non-stationarity**: opponents’ behaviour changes, especially if they were learning too. The current setup uses fixed random opponents; a natural improvement is self-play or training against a mix of opponents.
- **Instability**: high variance in returns and Q-estimates can cause unstable training. I used a target network and soft updates; for Variant 0 we use standard DQN (Double DQN is available in other variants). Prioritized replay or better tuning could help further.
- **Sample efficiency**: the game needs many matches to learn. Possible improvements include better exploration (e.g. intrinsic motivation), more efficient replay (prioritized or model-based), and curriculum learning—e.g. variant 6—starting against weaker opponents.

The codebase supports multiple variants via `config.py`, so switching algorithm or reward shaping for ablation is straightforward."

---

## [4:00–4:30] 6. Closing

"The repository contains the full implementation: training and evaluation scripts, the RL agent with action masking, the web interface and demo, configuration for variants and seeds, and a written report in REPORT.md. I am happy to explain or justify any part of the implementation, experiments, or results in more detail. Thank you."

---

## Checklist Before Recording

- [ ] Set `STUDENT_ID` in `config.py` and note your variant (STUDENT_ID % 7).
- [ ] Run a short training (e.g. `python train_gym.py --matches 100`) so you have a model to show.
- [ ] Test demo: `python demo.py --matches 5` and/or `python app.py` → Run Demo.
- [ ] If you have experiment results or plots, have the file or figure ready to show.
- [ ] Keep the script next to you; speak clearly and at a steady pace (aim for ~150 words per minute for 4–5 minutes).

---

## Quick Reference: Variant → What to Say

| Variant | Algorithm / focus (matches config.VARIANTS) |
|--------|---------------------------------------------|
| **0**  | **Assigned here:** DQN baseline, dueling, minimal shaping |
| 1 | Double DQN, dueling, minimal shaping |
| 2 | Double DQN, dueling, dense reward shaping |
| 3 | DQN, dueling, dense shaping |
| 4 | A2C (Actor-Critic), dense shaping |
| 5 | DQN + prioritized replay |
| 6 | DQN + curriculum opponents |

**If your STUDENT_ID gives a different variant:** Change Section 2 to that variant and use the row above (e.g. "For Variant 1, the design is: Double DQN with dueling and minimal reward shaping").
