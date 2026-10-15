# Experiment Summary

This file records the experiments completed so far for Project 4: Reinforcement Learning. Reported standard deviations are those printed by the corresponding evaluation procedure unless explicitly described otherwise.

## Task 1 — Hopper environment

Observation space: continuous, 11 values.
Action space: continuous, 3 actuator commands, each bounded in [-1, 1].
Model dimensions: nq = 6, nv = 6, nu = 3.

Body masses after resetting each environment:

| Body | Source mass | Target mass |
|---|---:|---:|
| World | 0.000000 | 0.000000 |
| Torso | 2.565634 | 3.665191 |
| Thigh | 4.057891 | 4.057891 |
| Leg | 2.781357 | 2.781357 |
| Foot | 5.315575 | 5.315575 |

The source torso mass is 70% of the target torso mass; the other three physical link masses are unchanged.

## Tasks 2 and 3 — REINFORCE and Actor-Critic

Each run used 1,000 training episodes. First/last return and length values are the mean over the first/last 100 episodes. Training time measures the training loop. These are training metrics, not separate deterministic policy evaluations.

| Algorithm | Seed | First-100 mean return | Last-100 mean return | First-100 mean length | Last-100 mean length | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE, no baseline | 42 | -1.300 | 3.560 | 9.80 | 7.00 | 9.88 |
| REINFORCE, no baseline | 7 | 3.432 | 11.193 | 7.15 | 20.08 | 18.89 |
| REINFORCE, constant baseline (-3.2156) | 42 | -1.120 | 24.180 | 9.30 | 21.90 | 17.50 |
| REINFORCE, constant baseline (-3.2156) | 7 | 2.724 | 2.802 | 6.53 | 8.58 | 12.52 |
| Actor-Critic | 42 | 0.529 | 9.065 | 9.33 | 12.73 | 11.90 |
| Actor-Critic | 7 | 2.320 | 7.627 | 6.73 | 10.78 | 16.12 |

The constant baseline improved the seed-42 run but did not consistently improve the seed-7 run. Two seeds are preliminary evidence only and do not establish statistical superiority.

Per-episode CSVs:
- `reinforce_without_baseline_seed42.csv`
- `reinforce_with_baseline_seed42.csv`
- `actor_critic_seed42.csv`
- `reinforce_without_baseline_seed7.csv`
- `reinforce_with_baseline_seed7.csv`
- `actor_critic_seed7.csv`

All six files are expected in this directory.

## Task 4 — PPO baseline

Training: 200,000 timesteps, seed 42, Stable-Baselines3 defaults. Evaluation: 50 episodes on each environment.

| Evaluation | Mean reward | Standard deviation |
|---|---:|---:|
| Source → source | 1449.86 | 129.42 |
| Source → target | 1518.28 | 108.58 |

## Task 5 — Standard SAC source/target baseline

Training: 200,000 timesteps per policy, using the existing script defaults. Evaluation: 50 episodes per configuration.

| Evaluation | Mean reward | Standard deviation |
|---|---:|---:|
| Source → source | 1004.63 | 33.39 |
| Source → target | 687.15 | 2.10 |
| Target → target | 1282.19 | 13.95 |

These values are the recorded results from the original default-configuration SAC runs.

## SAC hyperparameter search

The recorded partial search contains four configurations, each with two seeds, 50,000 training steps, and 10 evaluation episodes per run. The standard deviation below is across the two seed-level mean returns.

| Configuration | Learning rate | Batch size | Mean return across seeds | Standard deviation across seeds |
|---|---:|---:|---:|---:|
| A | 0.0003 | 256 | 279.73 | 29.71 |
| B | 0.0001 | 256 | 339.03 | 28.35 |
| C | 0.0003 | 512 | 278.18 | 11.42 |
| D | 0.0001 | 512 | 314.35 | 28.53 |

Configuration B was best among these four at 50,000 steps. This was a partial search, not an exhaustive one. Raw records are in `hparam_search/hparam_results_partial.csv` and `hparam_search/hparam_summary_partial.csv`.

## Tuned SAC validation

Configuration B was retrained for 200,000 timesteps with seed 42. Evaluation: 50 episodes per environment.

| Evaluation | Mean reward | Standard deviation |
|---|---:|---:|
| Source → source | 674.43 | 77.23 |
| Source → target | 681.77 | 117.48 |

This single run did not outperform the original default-configuration SAC means at 200,000 timesteps. Results from the 50,000-step search and this longer run should be interpreted separately.

Per-run record: `sac_tuned_validation.csv`.

## Task 6 — Uniform Domain Randomization (UDR)

SAC UDR policy trained for 200,000 timesteps. The original UDR setup keeps the torso at source mass and randomizes the other three link masses. Evaluations used 50 episodes.

| Evaluation | Mean reward | Standard deviation |
|---|---:|---:|
| UDR → UDR | 1237.47 | 244.05 |
| UDR → source | 1184.82 | 179.01 |
| UDR → target | 906.33 | 30.80 |

The UDR → target mean was approximately 31.9% higher than the standard SAC source → target mean (906.33 versus 687.15). UDR evaluation variability was substantial, particularly on the randomized domain.

### Additional UDR distribution-width screening (15 October 2026)

To investigate the UDR mass-range hyperparameter, three additional SAC configurations were screened using non-torso mass ranges of +/-10%, +/-30%, and +/-50% around the corresponding source masses. Each configuration used seed 42, 50,000 training timesteps, 10 deterministic evaluation episodes on `CustomHopper-target-v0`, learning rate 0.0003, batch size 256, and CPU training. Other SAC settings were buffer size 1,000,000, learning starts 100, tau 0.005, gamma 0.99, train frequency 1, and gradient steps 1. Evaluation reset seeds were 1000-1009.

| Non-torso mass range | Mean target reward | Reward SD | Mean episode length | Terminations | Truncations | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| +/-10% | 142.38 | 221.82 | 55.00 | 10/10 | 0/10 | 873.25 |
| +/-30% | 382.45 | 39.05 | 130.50 | 10/10 | 0/10 | 860.19 |
| +/-50% | 354.57 | 1.89 | 131.50 | 10/10 | 0/10 | 834.76 |

The +/-30% configuration achieved the highest mean target reward in this screening. The +/-50% configuration produced a slightly longer mean episode, while the +/-10% result was highly variable. All 30 evaluation episodes terminated before the 500-step time limit. This is a preliminary single-seed comparison based on 10 evaluation episodes per configuration; it does not establish a statistically reliable ranking and should not be treated as a replacement for the main 200,000-timestep UDR experiment.

The original UDR model and its 200,000-timestep evaluation results remain unchanged. Screening artifacts are stored in `results/udr_distribution_screen_20261015/`, with aggregate results in `summary.csv`, per-episode records in `episodes.csv`, and separately saved models for each range width.

## Project extension — posture/height-aware reward and torso-mass randomization

The `transferdr-heightposture` SAC policy was trained for 200,000 timesteps and evaluated on the original standard-reward source and target environments. This extension uses torso-mass randomization during training plus posture/height-aware rewards.

| Evaluation | Mean original-environment reward | Standard deviation | Mean length | Time-limit truncations | Early terminations |
|---|---:|---:|---:|---:|---:|
| Source, 50 episodes | 1404.53 | 26.52 | 500.00 | 50/50 | 0/50 |
| Target, 50 episodes | 1519.24 | 34.23 | 499.64 | 49/50 | 1/50 |

Across both domains, 99 of 100 episodes reached the 500-step time limit. This is evidence of sustained locomotion in the tested conditions, not a guarantee that the policy never falls.

Per-episode evaluation record: `extension_heightposture_eval50.csv`.

## Interpretation and limitations

- Policy-gradient training metrics and Stable-Baselines3 evaluation rewards are different measurements and should not be compared as if they were produced by the same evaluation protocol.
- Results from two training seeds are preliminary; the PPO, original SAC, tuned SAC, and UDR benchmark summaries above are single recorded training configurations unless otherwise stated.
- Hyperparameter selection at 50,000 steps did not guarantee the best result at 200,000 steps.
- The final report must describe methods, hyperparameters, evaluation protocols, and limitations honestly. This summary is not itself the final report.

## Corrected from-scratch policy-gradient experiments

The original Gaussian action policy could sample outside the Hopper's [-1, 1] actuator bounds. The policy was corrected to use tanh-squashed Gaussian actions with the corresponding log-probability correction. The effective initial standard deviation is 0.5. The earlier REINFORCE and Actor-Critic results above are historical results from the pre-correction implementation and should not be used as the final algorithm comparison.

Each corrected run used 1,000 episodes. First/last values are means across the first/last 100 training episodes. Times refer to the training loop.

| Algorithm | Seed | First-100 mean return | Last-100 mean return | First-100 mean length | Last-100 mean length | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE, no baseline | 42 | 103.318 | 188.623 | 54.30 | 93.34 | 115.19 |
| REINFORCE, no baseline | 7 | 100.570 | 216.315 | 68.88 | 141.24 | 137.72 |
| REINFORCE, baseline 84.8936 | 42 | 128.936 | 231.754 | 77.58 | 113.98 | 139.27 |
| REINFORCE, baseline 84.8936 | 7 | 77.304 | 258.847 | 48.30 | 157.23 | 131.41 |
| Actor-Critic | 42 | 36.324 | 89.214 | 26.87 | 46.24 | 98.88 |
| Actor-Critic | 7 | 11.255 | 92.410 | 18.26 | 62.83 | 90.68 |

Averaged across the two seeds:

| Algorithm | First-100 mean return | Last-100 mean return | Last-100 mean length | Mean training time (s) |
|---|---:|---:|---:|---:|
| REINFORCE, no baseline | 101.944 | 202.469 | 117.29 | 126.46 |
| REINFORCE, baseline 84.8936 | 103.120 | 245.301 | 135.61 | 135.34 |
| Actor-Critic | 23.790 | 90.812 | 54.54 | 94.78 |

The constant-baseline REINFORCE run had the highest last-100 training return on average in these two seeds, while Actor-Critic had the shortest mean training time. These are preliminary two-seed findings, not proof of universal superiority. The baseline value was estimated heuristically from discounted returns in the corrected no-baseline seed-42 run.

Corrected per-episode files:
- `reinforce_without_baseline_squashed_seed42.csv`
- `reinforce_with_baseline_squashed_seed42.csv`
- `actor_critic_squashed_seed42.csv`
- `reinforce_without_baseline_squashed_seed7.csv`
- `reinforce_with_baseline_squashed_seed7.csv`
- `actor_critic_squashed_seed7.csv`

The corrected action policy passed direct checks: zero out-of-bounds actions in 10,000 samples, finite sampled log-probabilities, effective initial standard deviation 0.5, and bounded deterministic actions. The current pytest suite passed 25 tests in the latest verified run.

## Corrected policy-gradient evaluation: 50 episodes per domain

The following policies were trained with the corrected tanh-squashed Gaussian action policy, then evaluated deterministically for 50 episodes on each standard environment. Rewards are mean episode return ± standard deviation across the 50 episodes.

| Algorithm | Training seed | Source mean reward | Target mean reward | Source mean length | Target mean length | Episodes terminated early (source/target) |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE, no baseline | 42 | 210.53 ± 0.96 | 211.34 ± 1.40 | 97.80 | 98.58 | 50/50; 50/50 |
| REINFORCE, no baseline | 7 | 173.18 ± 2.64 | 171.01 ± 2.69 | 106.62 | 104.52 | 50/50; 50/50 |
| REINFORCE, constant baseline (84.8936) | 42 | 221.95 ± 1.34 | 222.93 ± 1.23 | 99.60 | 100.66 | 50/50; 50/50 |
| REINFORCE, constant baseline (84.8936) | 7 | 155.38 ± 10.40 | 157.77 ± 9.81 | 99.96 | 102.24 | 50/50; 50/50 |
| Actor-Critic | 42 | 172.44 ± 4.96 | 185.67 ± 3.03 | 77.78 | 84.16 | 50/50; 50/50 |
| Actor-Critic | 7 | 111.03 ± 1.69 | 111.48 ± 1.90 | 82.46 | 82.92 | 50/50; 50/50 |

None of these 600 evaluation episodes reached the 500-step time limit. The constant baseline improved the observed seed-42 mean reward but did not improve the seed-7 result. These two seeds provide limited evidence and do not establish universal algorithm rankings.

Evaluation CSV files:
- `reinforce_without_baseline_squashed_seed42_eval50.csv`
- `reinforce_without_baseline_squashed_seed7_eval50.csv`
- `reinforce_with_baseline_squashed_seed42_eval50.csv`
- `reinforce_with_baseline_squashed_seed7_eval50.csv`
- `actor_critic_squashed_seed42_eval50.csv`
- `actor_critic_squashed_seed7_eval50.csv`

## Regenerated SB3 evaluations — 50 episodes per environment

These evaluations were generated after training by loading the saved model files; no policy was retrained for this evaluation batch. Each environment was evaluated for 50 deterministic episodes with reset seeds 1000–1049. The reported standard deviation is the population standard deviation across episode returns. Per-episode records are stored under `results/sb3_eval/`.

### PPO source policy

Model: `ppo_hopper_source.zip`

| Evaluation | Mean reward | Standard deviation | Mean episode length |
|---|---:|---:|---:|
| Source → source | 1466.50 | 119.41 | 475.50 |
| Source → target | 1524.49 | 106.45 | 476.18 |

### SAC source and target policies

The source SAC backup and the current `sac_hopper_source.zip` are distinct saved artifacts and produce different results. They are reported separately rather than treated as the same policy.

| Saved model | Evaluation | Mean reward | Standard deviation | Mean episode length |
|---|---|---:|---:|---:|
| `sac_hopper_source_default_backup.zip` | Source → source | 1007.55 | 32.30 | 293.76 |
| `sac_hopper_source_default_backup.zip` | Source → target | 686.63 | 2.60 | 211.98 |
| `sac_hopper_source.zip` | Source → source | 663.29 | 66.86 | 221.34 |
| `sac_hopper_source.zip` | Source → target | 689.41 | 113.01 | 223.58 |
| `sac_hopper_target_default_backup.zip` | Target → source | 1178.21 | 19.84 | 500.00 |
| `sac_hopper_target_default_backup.zip` | Target → target | 1284.76 | 15.93 | 500.00 |

The default-backup source results closely reproduce the previously recorded default SAC baseline. The current `sac_hopper_source.zip` results differ substantially on source evaluation; model identity must therefore be preserved when comparing runs.

### Uniform Domain Randomization (UDR)

Model: `sac_hopper_udr.zip`

| Evaluation | Mean reward | Standard deviation | Mean episode length |
|---|---:|---:|---:|
| UDR → UDR | 1169.20 | 229.45 | 382.34 |
| UDR → source | 1208.60 | 177.04 | 382.42 |
| UDR → target | 901.44 | 25.59 | 277.70 |

The UDR and source evaluations show substantial variability. These results do not establish that UDR improves performance uniformly across domains.

### Project extension: torso-mass randomization and posture/height-aware reward

Model: `sac_hopper_transferdr-heightposture.zip`. Evaluation was performed on the standard-reward source and target environments, not on the modified training reward.

| Evaluation | Mean reward | Standard deviation | Mean episode length | Time-limit truncations | Early terminations |
|---|---:|---:|---:|---:|---:|
| Extension → source | 1402.16 | 24.56 | 500.00 | 50/50 | 0/50 |
| Extension → target | 1519.18 | 30.64 | 498.72 | 49/50 | 1/50 |

In total, 99 of 100 episodes reached the 500-step time limit. This indicates sustained locomotion in these evaluated conditions, not guaranteed success in all conditions.

### Evaluation CSV inventory

The 13 CSVs in `results/sb3_eval/` each contain 50 episode records:

- `ppo_hopper_source__CustomHopper-source-v0__eval50.csv`
- `ppo_hopper_source__CustomHopper-target-v0__eval50.csv`
- `sac_hopper_source_default_backup__CustomHopper-source-v0__eval50.csv`
- `sac_hopper_source_default_backup__CustomHopper-target-v0__eval50.csv`
- `sac_hopper_source__CustomHopper-source-v0__eval50.csv`
- `sac_hopper_source__CustomHopper-target-v0__eval50.csv`
- `sac_hopper_target_default_backup__CustomHopper-source-v0__eval50.csv`
- `sac_hopper_target_default_backup__CustomHopper-target-v0__eval50.csv`
- `sac_hopper_udr__CustomHopper-udr-v0__eval50.csv`
- `sac_hopper_udr__CustomHopper-source-v0__eval50.csv`
- `sac_hopper_udr__CustomHopper-target-v0__eval50.csv`
- `sac_hopper_transferdr-heightposture__CustomHopper-source-v0__eval50.csv`
- `sac_hopper_transferdr-heightposture__CustomHopper-target-v0__eval50.csv`

These regenerated results are a separate evaluation batch from the historical values recorded in earlier sections. Differences between batches should not be silently merged or attributed to random seeds alone.
