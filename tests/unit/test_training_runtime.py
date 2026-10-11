import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from pvz_gym.runtime import RuntimeCapabilityError, RuntimeConfig
from train.common import EnvConfig
from train.dqn.config import DoubleDQNConfig
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.run_state import load_run_config
from train.runtime import TrainingRuntimeConfig, require_serial_training


class TrainingRuntimeTest(unittest.TestCase):
    def test_initialization_failure_reclaims_owned_game_even_when_client_close_fails(self):
        from train.sampler import native_environment
        runtime = TrainingRuntimeConfig(game_executable="private.exe", instance_root="private-instances")
        client = Mock()
        client.is_game_on.side_effect = RuntimeError("initialization failed")
        client.close.side_effect = RuntimeError("client close failed")
        with patch("pvz_gym.runtime.instances.GameInstance") as factory, \
             patch("pvz_gym.client.PvZClient", return_value=client):
            factory.return_value.start.return_value = 123
            with self.assertRaisesRegex(RuntimeError, "client close failed"):
                native_environment(DoubleDQNConfig(), runtime, 0, None, None)
            factory.return_value.close.assert_called_once()

    def test_runtime_options_are_shared_by_algorithm_configs(self):
        runtime = TrainingRuntimeConfig(num_envs=4, inference_batch_size=4)
        for cls in (DoubleDQNConfig,):
            with self.subTest(cls=cls):
                config = cls(runtime=asdict(runtime))
                self.assertEqual(config.runtime, runtime)
                restored = cls(**json.loads(json.dumps(asdict(config))))
                self.assertEqual(restored.env, config.env)
                self.assertEqual(restored.runtime, config.runtime)
                self.assertEqual(config.env.runtime, RuntimeConfig())
                self.assertEqual(cls().runtime.num_envs, 1)
        self.assertEqual(DoubleDQNConfig().env.simulation_mode, "headless")

    def test_run_config_round_trip_restores_nested_objects_and_legacy_defaults(self):
        config = DoubleDQNConfig(env=EnvConfig(simulation_mode="turbo", runtime=RuntimeConfig(executor="resident")),
                                 runtime=TrainingRuntimeConfig(num_envs=4, asynchronous=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "config.json")
            payload = asdict(config)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_run_config(tmp), config)
            payload.pop("runtime")
            payload["env"].pop("runtime")
            payload["env"].pop("turbo")
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = load_run_config(tmp)
            self.assertEqual(restored.runtime, TrainingRuntimeConfig())
            self.assertEqual(restored.env.runtime, RuntimeConfig())
            payload.pop("training_architecture")
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_run_config(tmp).training_architecture, "legacy")

    def test_missing_multi_instance_binding_rejected_before_files_or_game_connection(self):
        configs = (
            replace(DoubleDQNConfig(), runtime=TrainingRuntimeConfig(num_envs=2)),
            replace(DoubleDQNConfig(), runtime=TrainingRuntimeConfig(num_envs=4, asynchronous=True)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for i, config in enumerate(configs):
                with self.subTest(config=config):
                    run_dir = Path(tmp, str(i))
                    controller = DoubleDQNTrainerController(config, run_dir=run_dir)
                    with patch("train.dqn.base_controller.PvZClient") as create:
                        with self.assertRaisesRegex(ValueError, "distinct"):
                            controller.run()
                        create.assert_not_called()
                    self.assertFalse(run_dir.exists())

    def test_serial_guard_and_configuration_validation(self):
        require_serial_training(TrainingRuntimeConfig())
        for kwargs in ({"num_envs": True}, {"queue_capacity": 0},
                       {"inference_batch_size": -1}, {"asynchronous": 1},
                       {"inference_max_wait_ms": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingRuntimeConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
