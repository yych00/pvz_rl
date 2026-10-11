import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from train.dqn.run_state import (
    load_episode_summary,
    load_run_config,
    load_run_seed_types,
    resolve_resume_checkpoint,
)


class DqnRunStateTest(unittest.TestCase):
    def test_episode_summary_recovers_old_checkpoint_metrics_up_to_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            records = [
                {
                    "episode": 1,
                    "env_step": 10,
                    "result": "WON",
                    "training_episode_reward": 1.5,
                    "episode_steps": 10,
                    "epsilon": 0.8,
                    "invalid_actions": 1,
                    "mowers_used": 0,
                    "loss": 0.3,
                },
                {
                    "episode": 2,
                    "env_step": 30,
                    "result": "LOST",
                    "training_episode_reward": -0.5,
                    "episode_steps": 20,
                    "epsilon": 0.7,
                    "invalid_actions": 2,
                    "mowers_used": 1,
                    "loss": 0.2,
                },
                {"episode": 3, "env_step": 40, "result": "WON"},
            ]
            (run_dir / "episodes.jsonl").write_text(
                "\n".join(json.dumps(item) for item in records),
                encoding="utf-8",
            )
            summary = load_episode_summary(run_dir, up_to_episode=2)
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual((summary["wins"], summary["losses"]), (1, 1))
            self.assertAlmostEqual(summary["win_rate"], 0.5)
            self.assertAlmostEqual(summary["mean_training_reward"], 0.5)
            self.assertEqual(summary["invalid_actions"], 3)
            self.assertEqual(summary["mowers_used"], 1)

    def test_load_run_config_accepts_old_layout_and_restores_deck(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "env": {
                            "level_id": 4,
                            "playthrough": 1,
                            "ticks_per_step": 30,
                            "simulation_mode": "headless",
                        },
                        "total_steps": 250000,
                        "checkpoint_every": 25000,
                        "seed": 7,
                        "boss_projectiles": True,
                        "seed_types": [0, 1, 2],
                    }
                ),
                encoding="utf-8",
            )
            config = load_run_config(run_dir)
            self.assertEqual(config.credit_assignment, "one_step")
            self.assertEqual(config.env.level_id, 4)
            self.assertEqual(config.env.simulation_mode, "headless")
            self.assertEqual(config.autosave_every, 10000)
            self.assertEqual(config.checkpoint_every, 25000)
            self.assertEqual(load_run_seed_types(run_dir), (0, 1, 2))

    def test_load_run_config_preserves_current_mode_and_rejects_retired_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = run_dir / "config.json"
            path.write_text(json.dumps({"credit_assignment": "one_step", "boss_projectiles": True}), encoding="utf-8")
            self.assertEqual(load_run_config(run_dir).credit_assignment, "one_step")
            for retired in ({"credit_assignment": "event_heads"}, {"event_credit_weight": 0.3}):
                with self.subTest(retired=retired):
                    path.write_text(json.dumps(dict(retired, boss_projectiles=True)), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "credit_assignment|event_credit_weight"):
                        load_run_config(run_dir)

    def test_load_run_config_rejects_incompatible_projectile_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for payload in ({}, {"boss_projectiles": False}):
                with self.subTest(payload=payload):
                    (run_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Boss projectile input is required"):
                        load_run_config(run_dir)

    def test_newest_checkpoint_wins_over_stale_stopped_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            stopped = run_dir / "stopped.pt"
            latest = run_dir / "latest.pt"
            stopped.write_bytes(b"old-stop")
            latest.write_bytes(b"new-latest")
            now = time.time_ns()
            os.utime(stopped, ns=(now - 2_000_000, now - 2_000_000))
            os.utime(latest, ns=(now, now))
            self.assertEqual(resolve_resume_checkpoint(run_dir), latest)

    def test_stopped_wins_exact_timestamp_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            stopped = run_dir / "stopped.pt"
            latest = run_dir / "latest.pt"
            stopped.write_bytes(b"stop")
            latest.write_bytes(b"latest")
            stamp = time.time_ns()
            os.utime(stopped, ns=(stamp, stamp))
            os.utime(latest, ns=(stamp, stamp))
            self.assertEqual(resolve_resume_checkpoint(run_dir), stopped)


if __name__ == "__main__":
    unittest.main()
