"""Reset/step against an owned game and independent profile; never auto-attach."""
from dataclasses import replace
from train.dqn.config import DoubleDQNConfig
from train.runtime import TrainingRuntimeConfig
from train.sampler import native_environment
from pvz_gym.env_file import resolve_game_paths
from pvz_gym.runtime.save_template import prepare_training_save

def main():
    paths = resolve_game_paths()
    prepare_training_save()
    config = DoubleDQNConfig()
    config = replace(config, env=replace(config.env, level_id=4),
                     runtime=TrainingRuntimeConfig(game_executable=str(paths.executable), num_envs=1))
    env, instance = native_environment(config, config.runtime, 0, None, None)
    try:
        observation, info = env.reset(seed=42)
        observation, reward, terminated, truncated, info = env.step(0)
        print("reward", reward, "terminated", terminated, "truncated", truncated)
    finally:
        try:
            env.close()
        finally:
            if instance is not None:
                instance.close()

if __name__ == "__main__":
    main()
