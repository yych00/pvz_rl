import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn import agent as agent_module
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBatch


class _FailingEvalEnv:
    def __init__(self):
        self.reset_calls = 0

    def reset(self, *, seed=None):
        self.reset_calls += 1
        raise RuntimeError("synthetic eval reset failure")


class _UnusedAgent:
    def select_action(self, *args, **kwargs):
        raise AssertionError("agent should not be called")


class DoubleDQNSafetyTest(unittest.TestCase):
    def test_manual_save_is_rejected_while_training_is_active(self):
        controller = DoubleDQNTrainerController(DoubleDQNConfig(reward_scheme="staged"))
        controller._state = "RUNNING"
        with self.assertRaisesRegex(RuntimeError, "manual checkpoint save is disabled"):
            controller.save_checkpoint("manual.pt")

    def test_latest_checkpoint_copy_is_atomic_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = run_dir / "source.pt"
            source.write_bytes(b"checkpoint-bytes")
            controller = DoubleDQNTrainerController(DoubleDQNConfig(reward_scheme="staged"), run_dir=run_dir)

            destination = controller._sync_latest_checkpoint(source)

            self.assertEqual(destination.read_bytes(), b"checkpoint-bytes")
            self.assertFalse((run_dir / "latest.pt.tmp").exists())

    def test_eval_failure_returns_incomplete_record_instead_of_raising(self):
        events = []
        controller = DoubleDQNTrainerController(
            DoubleDQNConfig(reward_scheme="staged"),
            on_event=lambda event, payload: events.append((event, payload)),
            eval_seed_count=2,
        )
        controller._agent = _UnusedAgent()
        env = _FailingEvalEnv()

        record = controller._evaluate_current_agent(env, 3, env_step=100)

        self.assertFalse(record["complete"])
        self.assertFalse(record["stopped"])
        self.assertIn("synthetic eval reset failure", record["error"])
        self.assertEqual(env.reset_calls, 1)
        self.assertTrue(any(event == "evaluation_error" for event, _ in events))

    def test_eval_respects_pause_stop_gate_before_reset(self):
        controller = DoubleDQNTrainerController(DoubleDQNConfig(reward_scheme="staged"), eval_seed_count=1)
        controller._agent = _UnusedAgent()
        controller._wait_if_paused = lambda: False
        env = _FailingEvalEnv()

        record = controller._evaluate_current_agent(env, 3, env_step=100)

        self.assertTrue(record["stopped"])
        self.assertFalse(record["complete"])
        self.assertEqual(env.reset_calls, 0)

    @unittest.skipIf(agent_module.torch is None, "PyTorch is required for the DQN smoke test")
    def test_torch_learn_save_load_round_trip_is_atomic(self):
        config = DoubleDQNConfig(reward_scheme="staged",
            credit_assignment="one_step",
            replay_capacity=16,
            batch_size=4,
            hidden_sizes=(16, 8),
            target_update_interval=2,
            device="cpu",
        )
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config, device="cpu")
        observations = {
            name: np.stack([value.copy() for _ in range(4)])
            for name, value in empty_observation().items()
        }
        next_observations = {
            name: value.copy() for name, value in observations.items()
        }
        next_masks = np.zeros((4, ACTION_SPACE_SIZE), dtype=np.bool_)
        next_masks[0:2, 0:2] = True
        next_masks[3, 0] = True
        batch = ReplayBatch(
            observations=observations,
            actions=np.array([0, 1, 0, 1], dtype=np.int64),
            rewards=np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32),
            next_observations=next_observations,
            dones=np.array([False, False, True, False], dtype=np.bool_),
            next_action_masks=next_masks,
            ticks_advanced=np.full(4, 150, dtype=np.int64),
        )

        loss = agent.learn(batch)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(agent.update_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            agent.save(path, env_step=4, episode=1, training_summary={"wins": 1})
            self.assertTrue(path.is_file())
            self.assertFalse(Path(str(path) + ".tmp").exists())

            restored, restored_config, payload = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(restored.update_count, 1)
            self.assertEqual(payload["env_step"], 4)
            self.assertEqual(payload["episode"], 1)
            self.assertEqual(payload["training_summary"], {"wins": 1})
            self.assertEqual(restored_config.hidden_sizes, (16, 8))
            action = restored.select_action(
                empty_observation(),
                np.concatenate(([True], np.zeros(ACTION_SPACE_SIZE - 1, dtype=np.bool_))),
                env_step=4,
                explore=False,
            )
            self.assertEqual(action, 0)

    def test_set_exploration_settings_updates_controller_and_agent_config(self):
        controller = DoubleDQNTrainerController(DoubleDQNConfig(reward_scheme="staged", exploration_mode="epsilon_random"))
        fake_agent = unittest.mock.Mock()
        fake_agent.config = controller.config
        controller._agent = fake_agent
        events = []
        controller.on_event = lambda event, data: events.append((event, data))

        controller.set_exploration_settings(
            mode="q_guided_neighbor",
            ratio=(80, 20),
            type_weights=(1.0, 2.0, 0.1, 0.5, 0.5),
        )
        self.assertEqual(controller.config.exploration_mode, "q_guided_neighbor")
        self.assertEqual(controller.config.exploration_ratio, (80, 20))
        self.assertEqual(controller.config.exploration_type_weights, (1.0, 2.0, 0.1, 0.5, 0.5))
        self.assertEqual(fake_agent.config.exploration_mode, "q_guided_neighbor")
        self.assertEqual(fake_agent.config.exploration_ratio, (80, 20))
        self.assertEqual(fake_agent.config.exploration_type_weights, (1.0, 2.0, 0.1, 0.5, 0.5))
        self.assertTrue(any(e[0] == "exploration_settings_updated" for e in events))


if __name__ == "__main__":
    unittest.main()
