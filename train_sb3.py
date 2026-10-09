"""Train and evaluate PPO or SAC on the custom Hopper environments."""

import argparse

import gymnasium as gym
from env import custom_hopper  # Registers the custom environments.
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


def make_vec_env(env_id: str, n_envs: int = 1, seed: int = 0):
    """Create vectorized Gymnasium environments with reproducible seeds."""

    def make_one(rank):
        def _init():
            env = gym.make(env_id)
            env.action_space.seed(seed + rank)
            return env
        return _init

    vec_env = DummyVecEnv([make_one(rank) for rank in range(n_envs)])
    vec_env.seed(seed)
    return vec_env


def train_ppo(
    train_env_id: str,
    total_timesteps: int = 200_000,
    device: str = "cpu",
    n_envs: int = 1,
    seed: int = 42,
):
    print(f"Creating PPO training environment: {train_env_id}")
    vec_env = make_vec_env(train_env_id, n_envs=n_envs, seed=seed)

    print("Observation space:", vec_env.observation_space)
    print("Action space:", vec_env.action_space)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./ppo_hopper_tb/",
    )

    try:
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps)
        print("PPO training finished.")
    except Exception:
        vec_env.close()
        raise

    return model


def train_sac(
    train_env_id: str,
    total_timesteps: int = 200_000,
    device: str = "cpu",
    n_envs: int = 1,
    seed: int = 42,
):
    print(f"Creating SAC training environment: {train_env_id}")
    vec_env = make_vec_env(train_env_id, n_envs=n_envs, seed=seed)

    print("Observation space:", vec_env.observation_space)
    print("Action space:", vec_env.action_space)

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./sac_hopper_tb/",
    )

    try:
        print(f"Starting SAC training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps)
        print("SAC training finished.")
    except Exception:
        vec_env.close()
        raise

    return model


def evaluate_on_env(model, env_id: str, n_episodes: int = 10):
    """Evaluate a trained model on a standard Gymnasium environment."""
    print(f"\nEvaluating policy on environment: {env_id}")
    eval_env = gym.make(env_id)

    try:
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_episodes,
            render=False,
            deterministic=True,
        )

        print(f"Evaluation over {n_episodes} episodes on {env_id}:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward
    finally:
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PPO or SAC on Custom Hopper."
    )
    parser.add_argument(
        "--algo", type=str, default="ppo",
        choices=("ppo", "sac"),
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--train-env", type=str, default="CustomHopper-source-v0",
        help="Environment ID for training.",
    )
    parser.add_argument(
        "--test-env", type=str, default="CustomHopper-target-v0",
        help="Environment ID for transfer evaluation.",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=200_000,
        help="Total number of training timesteps.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Training device, such as cpu or cuda.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be greater than zero.")

    trainers = {"ppo": train_ppo, "sac": train_sac}
    trainer = trainers[args.algo]

    model = trainer(
        train_env_id=args.train_env,
        total_timesteps=args.total_timesteps,
        device=args.device,
        n_envs=1,
        seed=args.seed,
    )

    suffix = (
        args.train_env.replace("CustomHopper-", "").replace("-v0", "")
        or "default"
    )
    save_name = f"{args.algo}_hopper_{suffix}"
    print(f"\nSaving model to: {save_name}.zip")
    model.save(save_name)

    try:
        evaluate_on_env(model, args.train_env, n_episodes=10)
        evaluate_on_env(model, args.test_env, n_episodes=10)
    finally:
        if model.get_env() is not None:
            model.get_env().close()


if __name__ == "__main__":
    main()