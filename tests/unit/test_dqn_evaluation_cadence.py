import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pvz_gym.observation import empty_observation
from train.dqn.config import DoubleDQNConfig
from train.dqn.controller import DoubleDQNTrainerController


class SingleStepEnv:
    def __init__(self, *args, **kwargs):
        self.action_space = type("ActionSpace", (), {"n": 5})()
        self.mask = np.array([True, False, False, False, False], dtype=np.bool_)

    def reset(self, **kwargs):
        return empty_observation(), {"action_mask": self.mask}

    def step(self, action):
        return empty_observation(), 100.0, True, False, {
            "action_mask": self.mask,
            "result": "WON",
            "action_result": {"success": True},
            "ticks_advanced": 100,
            "episode_ticks": 100,
            "events": {},
        }

    def close(self):
        pass


class RecordingAgent:
    hidden_sizes = (16, 8)
    credit_assignment = "one_step"

    def __init__(self):
        self.actions = []

    def select_action(self, observation, mask, *, env_step, explore):
        self.actions.append((env_step, explore))
        return 0

    def epsilon(self, env_step):
        return 0.5

    def save(self, path, **kwargs):
        path = Path(path)
        path.write_bytes(b"test-model")
        return path


class DQNEvaluationCadenceTest(unittest.TestCase):
    def _run(self, run_dir, *, total_steps, **evaluation_kwargs):
        eval_mode = evaluation_kwargs.pop("evaluation_mode", "per_level")
        config = DoubleDQNConfig(
            training_architecture="legacy",
            evaluation_mode=eval_mode,
            total_steps=total_steps, hidden_sizes=(16, 8), replay_capacity=512,
            batch_size=2, learning_starts=10000, train_every=10000,
            checkpoint_every=0, autosave_every=0, device="cpu", reward_scheme="balanced",
            replay_scheme="uniform",
        )
        events = []
        controller = DoubleDQNTrainerController(
            config, run_dir=run_dir,
            on_event=lambda event, data: events.append((event, data)),
            **evaluation_kwargs,
        )
        agent = RecordingAgent()
        with (
            patch("train.dqn.base_controller.PvZEnv", SingleStepEnv),
            patch("train.dqn.base_controller.PvZClient"),
            patch.object(controller, "_build_agent", return_value=(agent, 0, 0, {})),
        ):
            summary = controller.run()
        return controller, agent, summary, events

    def test_optional_overlay_and_timing_failures_do_not_abort_cleanup(self):
        from unittest.mock import Mock
        overlay = Mock()
        overlay.is_enabled.side_effect = RuntimeError("overlay unavailable")
        closed = Mock()
        original_run = DoubleDQNTrainerController.run
        def continuous_run(controller):
            from dataclasses import replace
            controller.config = replace(controller.config, env=replace(controller.config.env, simulation_mode="continuous"))
            return original_run(controller)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("workbench.gui.q_heatmap_overlay.get_q_heatmap_overlay", return_value=overlay) as lookup,
            patch("train.dqn.base_controller.TimingRecorder.save", side_effect=OSError("timing unavailable")),
            patch.object(SingleStepEnv, "close", closed),
            patch.object(DoubleDQNTrainerController, "run", continuous_run),
            self.assertLogs("train.dqn.base_controller", level="DEBUG") as logs,
        ):
            _, _, summary, events = self._run(Path(tmp), total_steps=1,
                fast_eval_every_episodes=0, full_eval_every_episodes=0)
            self.assertEqual(summary.env_steps, 1)
            closed.assert_called_once()
            # Controllers publish bounded events and never instantiate a renderer.
            lookup.assert_not_called()
            self.assertTrue(any("Could not save timing" in line for line in logs.output))
            self.assertEqual([data["effective_scheme"] for event, data in events
                              if event == "replay_ready"], ["uniform"])

    def test_default_runs_four_games_per_level_at_episode_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller, agent, summary, events = self._run(root, total_steps=401)
            starts = [data for event, data in events if event == "evaluation_start"]
            self.assertEqual([(data["episode"], data["requested_episodes"]) for data in starts],
                             [(400, 4)])
            self.assertEqual([step for step, explore in agent.actions if not explore], [400] * 4)
            self.assertEqual(sum(explore for _, explore in agent.actions), 401)
            self.assertEqual((summary.env_steps, summary.episodes, summary.wins), (401, 401, 401))
            self.assertEqual(len(controller._replay), 401)
            records = [json.loads(line) for line in (root / "evaluations.jsonl").read_text().splitlines()]
            self.assertEqual([(record["episode"], record["kind"], record["episodes"]) for record in records],
                             [(400, "full", 4)])
            best = json.loads((root / "best.json").read_text())
            self.assertEqual((best["episode"], best["requested_episodes"]), (400, 4))
            self.assertTrue((root / "best.pt").exists())
            self.assertTrue((root / "final.pt").exists())

    def test_disabling_both_cadences_runs_no_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, agent, summary, events = self._run(
                root, total_steps=400, fast_eval_every_episodes=0, full_eval_every_episodes=0,
            )
            self.assertFalse(any(event.startswith("evaluation_") for event, _ in events))
            self.assertTrue(all(explore for _, explore in agent.actions))
            self.assertEqual(summary.episodes, 400)
            self.assertFalse((root / "best.pt").exists())
            self.assertTrue((root / "final.pt").exists())

    def test_evaluation_mode_off_runs_no_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, agent, summary, events = self._run(
                root, total_steps=401, evaluation_mode="off",
            )
            self.assertFalse(any(event.startswith("evaluation_") for event, _ in events))
            self.assertTrue(all(explore for _, explore in agent.actions))
            self.assertEqual(summary.episodes, 401)
            self.assertFalse((root / "best.pt").exists())
            self.assertTrue((root / "final.pt").exists())

    def test_custom_fast_and_full_cadences_trigger_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, summary, _ = self._run(
                root, total_steps=6, fast_eval_every_episodes=2, full_eval_every_episodes=3,
                fast_eval_seed_count=1, full_eval_seed_count=2,
            )
            records = [json.loads(line) for line in (root / "evaluations.jsonl").read_text().splitlines()]
            self.assertEqual([(record["episode"], record["kind"], record["episodes"]) for record in records],
                             [(2, "fast", 1), (3, "full", 2), (4, "fast", 1), (6, "full", 2)])
            self.assertEqual(summary.eval_runs, 2)
            self.assertEqual(json.loads((root / "best.json").read_text())["episode"], 3)

    def test_explicit_legacy_protocol_overrides_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, summary, _ = self._run(
                root, total_steps=5, eval_every_episodes=2, eval_seed_count=3,
            )
            records = [json.loads(line) for line in (root / "evaluations.jsonl").read_text().splitlines()]
            self.assertEqual([(record["episode"], record["episodes"]) for record in records],
                             [(2, 3), (4, 3)])
            self.assertEqual(summary.env_steps, 5)
            self.assertEqual(summary.eval_runs, 2)


if __name__ == "__main__":
    unittest.main()
