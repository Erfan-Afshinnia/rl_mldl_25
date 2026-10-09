# Robust Reinforcement Learning for Sim-to-Sim Transfer in MuJoCo Hopper

A reinforcement learning study of **robust control under dynamics mismatch** using a custom MuJoCo Hopper environment.

The project investigates how policies trained in one simulated domain transfer to another domain with different physical dynamics, and whether **domain randomization** improves transfer robustness. It combines from-scratch policy-gradient implementations in PyTorch with PPO and SAC baselines from Stable-Baselines3.

## Research Question

How does a change in simulated dynamics affect reinforcement learning performance, and can domain randomization improve zero-shot transfer between environments?

The custom environment defines source and target domains with different torso masses:

- **Source domain:** the torso mass is scaled to 70% of the target-domain value.
- **Target domain:** uses the original torso mass from the MuJoCo model.

This controlled dynamics mismatch provides the basis for the sim-to-sim transfer experiments.

## Methods

### From-Scratch Reinforcement Learning

Implemented in PyTorch:

- **REINFORCE:** Monte Carlo discounted returns, optional constant baseline, and normalized advantages.
- **Actor-Critic:** a learned state-value network, TD(0) targets, and advantage-based policy updates.
- **Gaussian policy:** continuous-action distribution with a learned per-action standard deviation.

### Stable-Baselines3 Baselines

The repository also implements training scripts for:

- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**

The training script supports configurable training and evaluation environments and compares policy performance on the selected training and test domains.

## Custom Domain-Randomization Environments

The custom Hopper environment supports the following variants:

| Environment | Configuration |
|---|---|
| `CustomHopper-source-v0` | Source dynamics |
| `CustomHopper-target-v0` | Target dynamics |
| `CustomHopper-udr-v0` | Uniform randomization of non-torso body masses |
| `CustomHopper-massdr-v0` | Mass-only ablation |
| `CustomHopper-frictiondr-v0` | Friction-only randomization |
| `CustomHopper-dampingdr-v0` | Damping-only randomization |
| `CustomHopper-extdr-v0` | Mass, damping, friction, and action noise |

The extended randomization condition combines mass randomization with multiplicative damping and friction changes, together with Gaussian action noise. Separate ablation environments allow individual parameter groups to be studied.

## Experimental Design

The project investigates:

- Source-to-source performance
- Source-to-target zero-shot transfer
- Target-to-target reference performance
- Transfer robustness under domain randomization
- Mass, friction, and damping ablations
- SAC hyperparameter sensitivity across multiple random seeds

### SAC Hyperparameter Search

The available results cover a **partial search** of four SAC configurations, with two seeds per configuration. Each run used 50,000 training steps and 10 evaluation episodes.

The best-performing tested configuration by mean return was:

| Parameter | Value |
|---|---:|
| Learning rate | `0.0001` |
| Batch size | `256` |
| Mean return across seeds | `339.03` |
| Standard deviation across seeds | `28.35` |

These results describe the configurations tested so far; they should not be interpreted as an exhaustive search or a guarantee of performance in other conditions. The result tables are in `hparam_search/`.

## Results and Visualizations

### PPO Baseline Comparison

![PPO baseline comparison](Bar%20Plots/ppo_baselines_barplot.png)

### SAC Robustness Comparison

![SAC robustness comparison](Bar%20Plots/sac_robustness_barplot.png)

### Simulation Demonstration

The repository includes an animation of a trained Hopper policy in a randomized environment.

![UDR Hopper simulation](GIF/hopper_udr_best.gif)

## Repository Structure

```text
rl_mldl_25/
├── agent.py
├── train.py
├── train_sb3.py
├── test.py
├── test_random_policy.py
├── env/
│   ├── __init__.py
│   ├── custom_hopper.py
│   ├── mujoco_env.py
│   └── assets/
│       └── hopper.xml
├── hparam_search/
│   ├── hparam_results_partial.csv
│   └── hparam_summary_partial.csv
├── Bar Plots/
│   ├── ppo_baselines_barplot.png
│   └── sac_robustness_barplot.png
├── GIF/
│   └── hopper_udr_best.gif
├── colab_starting_code.ipynb
├── requirements.txt
└── README.md
```

## Running the Training Scripts

The scripts expose the following command-line interfaces. They require a compatible environment for the repository's legacy Gym and MuJoCo dependencies.

**REINFORCE**

```bash
python train.py --algo reinforce --env CustomHopper-source-v0
```

**Actor-Critic**

```bash
python train.py --algo actor_critic --env CustomHopper-source-v0
```

**PPO**

```bash
python train_sb3.py --algo ppo --train-env CustomHopper-source-v0 --test-env CustomHopper-target-v0 --total-timesteps 200000
```

**SAC**

```bash
python train_sb3.py --algo sac --train-env CustomHopper-source-v0 --test-env CustomHopper-target-v0 --total-timesteps 200000
```

**Evaluate a custom PyTorch policy**

```bash
python test.py --model model.mdl --episodes 10
```

## Compatibility and Limitations

- The experiments concern **sim-to-sim transfer**, not real-world or sim-to-real deployment.
- The environment uses the legacy OpenAI Gym API and `mujoco-py`.
- Installing these legacy dependencies may require a compatible Python version and native MuJoCo dependencies. The current dependency setup still needs to be validated on a clean environment.
- The hyperparameter-search results are partial and use two random seeds per configuration.

## Tech Stack

**Python, PyTorch, NumPy, OpenAI Gym, MuJoCo, mujoco-py, Stable-Baselines3, PPO, SAC, REINFORCE, Actor-Critic, Domain Randomization**

## Author

**Erfan Afshinnia**