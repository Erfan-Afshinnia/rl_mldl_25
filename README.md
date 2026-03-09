# Robust Reinforcement Learning for Sim-to-Sim Transfer in MuJoCo Hopper

This project studies **robust robotic control with Reinforcement Learning (RL)** under **dynamics mismatch** using the **MuJoCo Hopper** environment.  
The main goal is to understand how well RL policies trained in one simulated domain transfer to another domain with different physical dynamics, and whether **domain randomization** can improve robustness.

The project includes:
- custom Hopper environments for **source**, **target**, and randomized domains
- implementations of **REINFORCE** and **Actor-Critic**
- stronger baselines using **PPO** and **SAC** from Stable-Baselines3
- experiments on **sim-to-sim transfer**
- **Uniform Domain Randomization (UDR)** and **Extended Domain Randomization (extDR)** ablations

---

## Project Overview

A major challenge in robotics is the **reality gap**: a policy may perform well in simulation but fail when the dynamics change.  
To study this in a controlled way, this project creates a **source domain** and a **target domain** in MuJoCo Hopper.

The key dynamics mismatch is:

- the **target domain torso mass is 30% higher** than the source domain

This allows systematic evaluation of how RL methods generalize when the environment changes. The project then studies whether training with randomized dynamics can improve transfer robustness. 

---

## Main Contributions

- Implemented **REINFORCE** with optional constant baseline
- Implemented **Actor-Critic** with a learned value function
- Trained **PPO** and **SAC** baselines using Stable-Baselines3
- Built custom Hopper domains:
  - `CustomHopper-source-v0`
  - `CustomHopper-target-v0`
  - `CustomHopper-udr-v0`
  - `CustomHopper-massdr-v0`
  - `CustomHopper-frictiondr-v0`
  - `CustomHopper-dampingdr-v0`
  - `CustomHopper-extdr-v0`
- Evaluated **zero-shot transfer** from source to target
- Studied the effect of:
  - UDR randomization ranges
  - friction-only randomization
  - damping-only randomization
  - full extended DR with action noise

---

## Repository Structure

```bash
.
├── agent.py                     # REINFORCE and Actor-Critic implementation
├── train.py                     # Training for REINFORCE / Actor-Critic
├── train_sb3.py                 # PPO / SAC training with Stable-Baselines3
├── test.py                      # Test a trained custom policy
├── test_random_policy.py        # Explore the Hopper environment with random actions
├── env/
│   ├── __init__.py
│   ├── custom_hopper.py         # Custom Hopper domains and domain randomization
│   └── mujoco_env.py
├── hparam_search/
│   ├── hparam_results_partial.csv
│   └── hparam_summary_partial.csv
├── GIF/
│   └── hopper_udr_best.gif
├── Bar Plots/
│   ├── ppo_baselines_barplot.png
│   └── sac_robustness_barplot.png
├── colab_starting_code.ipynb
├── requirements.txt
└── README.md
