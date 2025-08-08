# Multi-Agent Prisoner's Dilemma Study

**CS 406 Independent Study**  
Eebbaa Felema | Instructor: Alexander Guyer | Summer 2025

## Research Question

How does memory window size affect strategy development in multi-agent reinforcement learning?

## Experiments

Training DQN agents on Iterated Prisoner's Dilemma with different observation windows:
- Memory window 1: Agents see only the last round
- Memory window 5: Agents see last 5 rounds  
- Memory window 10: Agents see last 10 rounds

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

**Quick test (5 minutes):**
```bash
python train_dqn.py --test
```

**Single experiment:**
```bash
python train_dqn.py --memory 5 --timesteps 500000
```

**All experiments:**
```bash
python run_all_experiments.py
```

**Analyze results:**
```bash
python analyze.py
```

## Technical Stack

- Stable Baselines3 (DQN implementation)
- PettingZoo (multi-agent environment framework)
- SuperSuit (environment wrappers)
- Custom Prisoner's Dilemma environment

## Files

- `environments/prisoners_dilemma.py` - Game environment
- `train_dqn.py` - Training script
- `run_all_experiments.py` - Batch runner
- `analyze.py` - Results analysis and plotting# prisoners-dilemma-dqn
