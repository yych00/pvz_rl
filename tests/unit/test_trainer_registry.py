import unittest

from train.dqn.config import DoubleDQNConfig
from train.dqn.exploration import EXPLORATION_MODE_LABELS, parse_action_family_weights, parse_exploration_ratio
from train.registry import all_trainer_specs, get_trainer_spec, trainer_key_from_label


class TrainerRegistryTest(unittest.TestCase):
    def test_registry_only_exposes_implemented_trainers(self):
        specs = all_trainer_specs()
        self.assertEqual(
            [spec.key for spec in specs],
            ["random_legal", "double_dqn"],
        )
        self.assertTrue(get_trainer_spec("random_legal").available)
        self.assertTrue(get_trainer_spec("double_dqn").available)
        with self.assertRaises(KeyError):
            get_trainer_spec("ppo")

    def test_double_dqn_schema_exposes_algorithm_only_fields(self):
        spec = get_trainer_spec("double_dqn")
        keys = {parameter.key for parameter in spec.parameters}
        self.assertIn("gamma", keys)
        self.assertIn("replay_capacity", keys)
        self.assertIn("device", keys)
        self.assertIn("autosave_every", keys)
        self.assertIn("checkpoint_every", keys)
        self.assertIn("exploration_type_weights", keys)
        self.assertIn("exploration_mode", keys)
        self.assertIn("exploration_ratio", keys)
        self.assertNotIn("mower_penalty", keys)
        self.assertNotIn("level_id", keys)
        self.assertEqual(trainer_key_from_label("Double DQN"), "double_dqn")

    def test_checkpoint_defaults_distinguish_latest_from_milestones(self):
        spec = get_trainer_spec("double_dqn")
        params = {parameter.key: parameter for parameter in spec.parameters}
        config = DoubleDQNConfig()
        self.assertEqual(params["exploration_mode"].default, EXPLORATION_MODE_LABELS[config.exploration_mode])
        self.assertEqual(params["exploration_mode"].choices, tuple(EXPLORATION_MODE_LABELS.values()))
        self.assertEqual(parse_exploration_ratio(params["exploration_ratio"].default), config.exploration_ratio)
        self.assertEqual(params["autosave_every"].default, config.autosave_every)
        self.assertEqual(params["checkpoint_every"].default, config.checkpoint_every)
        self.assertEqual(params["gamma"].default, config.gamma)
        self.assertEqual(params["n_step"].default, config.n_step)
        self.assertEqual(params["training_architecture"].default, config.training_architecture)
        self.assertEqual(params["training_architecture"].choices, ("actor_learner", "legacy"))
        self.assertEqual(params["policy_sync_interval"].default, config.policy_sync_interval)
        self.assertEqual(params["max_pending_updates"].default, config.max_pending_updates)
        self.assertEqual(params["auto_train_every"].default, config.auto_train_every)
        self.assertEqual(params["auto_train_high_watermark"].default, config.auto_train_high_watermark)
        self.assertEqual(params["auto_train_low_watermark"].default, config.auto_train_low_watermark)
        self.assertEqual(params["train_every"].default, config.train_every)
        self.assertEqual(config.n_step, 3)
        self.assertEqual(
            parse_action_family_weights(params["exploration_type_weights"].default),
            config.exploration_type_weights,
        )
        self.assertNotIn("gamma_reference_ticks", params)
        self.assertIn("100 ticks", params["gamma"].label)
        self.assertLess(config.autosave_every, config.checkpoint_every)


if __name__ == "__main__":
    unittest.main()
