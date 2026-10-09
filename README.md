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

- **REINFORCE:** Monte Carlo discounted returns, an optional constant baseline, and normalized advantages.
- **Actor-Critic:** a learned state-value network, TD(0) targets, and advantage-based policy updates.
- **Gaussian policy:** continuous-action distribution with a learned per-action standard deviation.

### Stable-Baselines3 Baselines

The repository also supports:

- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**

The Stable-Baselines3 training script supports configurable training and evaluation environments and evaluates policies on the selected training and test domains.

## Custom Domain-Randomization Environments

The environment uses Gymnasium and the official MuJoCo Python bindings.

| Environment | Configuration |
|---|---|
| `CustomHopper-v0` | Default target dynamics |
| `CustomHopper-source-v0` | Source dynamics |
| `CustomHopper-target-v0` | Target dynamics |
| `CustomHopper-udr-v0` | Uniform randomization of non-torso body masses |
| `CustomHopper-massdr-v0` | Mass-only ablation |
| `CustomHopper-frictiondr-v0` | Friction-only randomization |
| `CustomHopper-dampingdr-v0` | Damping-only randomization |
| `CustomHopper-extdr-v0` | Mass, damping, friction, and action noise |

The extended randomization condition combines mass randomization with multiplicative damping and friction changes and Gaussian action noise. Separate ablation environments allow individual parameter groups to be studied.

The observation contains 11 values: five position coordinates and six velocity coordinates. The action space contains three continuous controls bounded between -1 and 1. Registered environments have a maximum episode length of 500 steps.

## Installation

The working environment used for smoke tests was **Python 3.11.9 on Windows**. The pins in `requirements.txt` reflect the package versions tested in that environment.

From the repository root, open PowerShell and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To verify the installed dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

The project uses Gymnasium rather than legacy Gym and the official `mujoco` package rather than `mujoco-py`. The repository's Hopper XML has been converted to a format accepted by the tested MuJoCo version.

## Running the Training Scripts

Run commands from the repository root using the Python interpreter in `.venv`.

### REINFORCE

```powershell
.\.venv\Scripts\python.exe train.py --algo reinforce --env CustomHopper-source-v0
```

### Actor-Critic

```powershell
.\.venv\Scripts\python.exe train.py --algo actor_critic --env CustomHopper-source-v0
```

Both custom algorithms save their final policy state dictionary to `model.mdl`.

### PPO

```powershell
.\.venv\Scripts\python.exe train_sb3.py --algo ppo --train-env CustomHopper-source-v0 --test-env CustomHopper-target-v0 --total-timesteps 200000
```

### SAC

```powershell
.\.venv\Scripts\python.exe train_sb3.py --algo sac --train-env CustomHopper-source-v0 --test-env CustomHopper-target-v0 --total-timesteps 200000
```

The Stable-Baselines3 script saves the trained policy as a ZIP file and evaluates it on the training and test environments. TensorBoard logs are written to `ppo_hopper_tb/` or `sac_hopper_tb/`.

For shorter training runs, use `--total-timesteps` to reduce the number of steps. The default device is CPU; `--device cuda` can be used with a compatible CUDA-enabled PyTorch installation.

### Evaluate a Custom PyTorch Policy

```powershell
.\.venv\Scripts\python.exe test.py --model model.mdl --episodes 10 --env CustomHopper-source-v0
```

Replace the environment ID with `CustomHopper-target-v0` to evaluate in the target domain.

### Test a Random Policy

Run a few episodes without rendering:

```powershell
.\.venv\Scripts\python.exe test_random_policy.py --episodes 5 --seed 42
```

Add `--render` to request a human-rendered simulation.

## Experimental Design

The project investigates:

- Source-to-source performance.
- Source-to-target zero-shot transfer.
- Target-to-target reference performance.
- Transfer robustness under domain randomization.
- Mass, friction, and damping ablations.
- SAC hyperparameter sensitivity across multiple random seeds.

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
├── requirements.txt
└── README.md
```

## Development and Tests

Install the runtime and development dependencies in your virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the automated environment tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests check the MuJoCo XML, all eight registered environment variants, the source-to-target torso-mass ratio, and reproducibility of extended domain randomization.
## Compatibility and Limitations

- The experiments concern **sim-to-sim transfer**, not real-world deployment or sim-to-real transfer.
- The migrated environment uses Gymnasium's reset and step interfaces and the official MuJoCo Python bindings.
- The converted XML's key physical properties were compared with the legacy model, and short simulation rollouts were close but not identical. Long-term trajectory equivalence has not been established.
- Short smoke tests verified that the environment variants, custom training algorithms, PPO, SAC, and policy evaluation execute with the migrated setup. These checks do not establish convergence or benchmark-quality performance.
- The hyperparameter-search results are partial and use two random seeds per configuration.

## Tech Stack

**Python, PyTorch, NumPy, Gymnasium, MuJoCo, Stable-Baselines3, PPO, SAC, REINFORCE, Actor-Critic, Domain Randomization**

## Author

**Erfan Afshinnia**