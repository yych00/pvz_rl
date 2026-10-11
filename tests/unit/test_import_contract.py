"""Import-path compatibility contracts for the reorganized Python layers."""

import unittest


class ImportContractTest(unittest.TestCase):
    def test_canonical_eval_and_legacy_runner_share_objects(self):
        from eval.episode_runner import EpisodeRunner as CanonicalEpisodeRunner
        from eval.random_runner import RandomAgentConfig as CanonicalRandomConfig
        from pvz_gym.runner import EpisodeRunner as LegacyEpisodeRunner
        from pvz_gym.random_runner import RandomAgentConfig as LegacyRandomConfig

        self.assertIs(CanonicalEpisodeRunner, LegacyEpisodeRunner)
        self.assertIs(CanonicalRandomConfig, LegacyRandomConfig)

    def test_reward_compatibility_paths_share_functions(self):
        from pvz_gym.reward import calculate_reward as old_battle
        from pvz_gym.rewards.battle import calculate_reward as new_battle
        from pvz_gym.seed_reward import deck_outcome_reward as old_seed
        from pvz_gym.rewards.seed_selection import deck_outcome_reward as new_seed

        self.assertIs(old_battle, new_battle)
        self.assertIs(old_seed, new_seed)

    def test_curriculum_package_exports_reset_config(self):
        from pvz_gym.curriculum import ResetConfig

        self.assertEqual(ResetConfig(level_id=4).level_id, 4)


if __name__ == "__main__":
    unittest.main()
