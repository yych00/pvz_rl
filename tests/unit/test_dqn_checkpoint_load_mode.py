import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.checkpoint_contract import declare_checkpoint
from train.dqn.optimized_controller import (
    CHECKPOINT_LOAD_RESUME,
    CHECKPOINT_LOAD_WEIGHTS_ONLY,
    CHECKPOINT_LOAD_WEIGHTS_REPLAY,
    DoubleDQNTrainerController,
)
from train.dqn.replay import ReplayBuffer
from train.dqn.run_state import load_run_config
from train.dqn.run_state import resolve_resume_checkpoint

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Double DQN checkpoint tests require PyTorch")
class DqnCheckpointLoadModeTest(unittest.TestCase):
    def _source_checkpoint(self, directory: Path) -> Path:
        config = DoubleDQNConfig(reward_scheme="staged",
            credit_assignment="one_step",
            total_steps=700_000,
            replay_capacity=8,
            batch_size=2,
            learning_starts=0,
            device="cpu",
        )
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config, device="cpu")
        agent.update_count = 17
        with torch.no_grad():
            first_parameter = next(agent.online.parameters())
            first_parameter.fill_(0.125)
        agent.target.load_state_dict(agent.online.state_dict())

        checkpoint = directory / "final.pt"
        agent.save(checkpoint, env_step=600_000, episode=700)

        replay = ReplayBuffer(capacity=8, action_dim=ACTION_SPACE_SIZE, seed=4)
        observation = empty_observation()
        next_observation = empty_observation()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[0] = True
        replay.add(observation, 0, 0.0, next_observation, False, mask, ticks_advanced=150)
        replay.save(directory / "replay.npz", env_step=600_000, episode=700)
        # A full checkpoint is published only after its matching replay exists.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["replay_file"] = "replay.npz"
        torch.save(declare_checkpoint(payload, replay.checkpoint_payload()), checkpoint)
        return checkpoint

    def test_weights_only_starts_fresh_but_keeps_online_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._source_checkpoint(root)
            config = DoubleDQNConfig(reward_scheme="staged",
                credit_assignment="one_step",
                total_steps=50_000,
                replay_capacity=8,
                batch_size=2,
                learning_starts=0,
                learning_rate=3e-4,
                epsilon_start=0.35,
                epsilon_end=0.05,
                epsilon_decay_steps=10_000,
                seed=99,
                device="cpu",
            )
            controller = DoubleDQNTrainerController(
                config,
                run_dir=root / "fresh-run",
                checkpoint_path=checkpoint,
                checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_ONLY,
            )

            agent, start_step, start_episode, _payload = controller._build_agent(
                ACTION_SPACE_SIZE
            )
            replay = controller._restore_replay(
                action_dim=ACTION_SPACE_SIZE,
                start_step=start_step,
                start_episode=start_episode,
            )

            self.assertEqual(start_step, 0)
            self.assertEqual(start_episode, 0)
            self.assertEqual(agent.update_count, 0)
            self.assertEqual(len(agent.optimizer.state), 0)
            self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 3e-4)
            self.assertAlmostEqual(agent.epsilon(0), 0.35)
            self.assertEqual(len(replay), 0)
            self.assertFalse(controller.summary.replay_restored)

            first_online = next(agent.online.parameters()).detach()
            first_target = next(agent.target.parameters()).detach()
            self.assertTrue(torch.allclose(first_online, torch.full_like(first_online, 0.125)))
            self.assertTrue(torch.equal(first_online, first_target))

    def test_weights_only_uses_configured_per_and_updates_priorities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._source_checkpoint(root)
            config = DoubleDQNConfig(reward_scheme="staged", credit_assignment="one_step", replay_capacity=8,
                batch_size=2, replay_scheme="per", per_alpha=0.8,
                per_beta_start=0.3, per_eps=0.002, device="cpu")
            controller = DoubleDQNTrainerController(config, run_dir=root / "transfer",
                checkpoint_path=checkpoint, checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_ONLY)
            agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
            replay = controller._restore_replay(action_dim=ACTION_SPACE_SIZE,
                start_step=step, start_episode=episode)
            self.assertEqual((replay.scheme, replay.alpha, replay.beta, replay.eps),
                             ("per", 0.8, 0.3, 0.002))
            self.assertEqual(len(replay), 0)
            observation = empty_observation()
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
            mask[0] = True
            for reward in (1.0, 10.0):
                replay.add(observation, 0, reward, observation, True, mask, ticks_advanced=100)
            batch = replay.sample(2)
            self.assertIsNotNone(batch.weights)
            loss, errors = agent.learn(batch, return_td_errors=True)
            self.assertTrue(np.isfinite(loss))
            replay.update_priorities(batch.indices, errors)
            self.assertTrue(np.any(replay._priorities[:2] != 1.0))

    def test_pre_imitater_schema_is_rejected_for_both_load_modes_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._source_checkpoint(root)
            old_schema = "pvz_observation_entities_256_zombie_grid_v2"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["observation_schema"] = old_schema
            torch.save(payload, checkpoint)
            for weights_only in (False, True):
                with self.subTest(weights_only=weights_only):
                    with self.assertRaisesRegex(ValueError, "checkpoint observation schema"):
                        DoubleDQNAgent.from_checkpoint(checkpoint, device="cpu", weights_only=weights_only)
            replay_path = root / "replay.npz"
            with np.load(replay_path, allow_pickle=False) as archive:
                replay_data = {name: archive[name] for name in archive.files}
            replay_data["observation_schema"] = np.asarray(old_schema)
            np.savez_compressed(replay_path, **replay_data)
            with self.assertRaisesRegex(ValueError, "replay observation schema"):
                ReplayBuffer.load(replay_path)

    def test_resume_keeps_saved_step_update_count_and_matching_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._source_checkpoint(root)
            config = DoubleDQNConfig(reward_scheme="staged",
                credit_assignment="one_step",
                total_steps=700_100,
                replay_capacity=8,
                batch_size=2,
                learning_starts=0,
                device="cpu",
            )
            controller = DoubleDQNTrainerController(
                config,
                run_dir=root / "resume-run",
                checkpoint_path=checkpoint,
                checkpoint_load_mode=CHECKPOINT_LOAD_RESUME,
            )

            agent, start_step, start_episode, _payload = controller._build_agent(
                ACTION_SPACE_SIZE
            )
            replay = controller._restore_replay(
                action_dim=ACTION_SPACE_SIZE,
                start_step=start_step,
                start_episode=start_episode,
            )

            self.assertEqual(start_step, 600_000)
            self.assertEqual(start_episode, 700)
            self.assertEqual(agent.update_count, 17)
            self.assertEqual(len(replay), 1)
            self.assertTrue(controller.summary.replay_restored)

    def test_legacy_run_restores_checkpoint_and_replay_without_credit_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint = self._source_checkpoint(run_dir)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["config"].pop("credit_assignment")
            for key in ("checkpoint_format_version", "checkpoint_kind", "strict_resume_supported",
                        "exact_game_resume", "replay_snapshot_step", "replay_snapshot_episode"):
                payload.pop(key, None)
            torch.save(payload, checkpoint)
            (run_dir / "config.json").write_text(json.dumps(payload["config"]), encoding="utf-8")

            config = load_run_config(run_dir)
            controller = DoubleDQNTrainerController(
                config, run_dir=run_dir, checkpoint_path=checkpoint,
            )
            agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
            replay = controller._restore_replay(
                action_dim=ACTION_SPACE_SIZE, start_step=step, start_episode=episode,
            )

            self.assertEqual((agent.credit_assignment, replay.credit_assignment), ("one_step", "one_step"))
            self.assertEqual((step, episode, agent.update_count), (600_000, 700, 17))
            self.assertEqual(len(replay), 1)
            self.assertTrue(controller.summary.replay_restored)

    def test_invalid_checkpoint_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_load_mode"):
            DoubleDQNTrainerController(
                DoubleDQNConfig(reward_scheme="staged", device="cpu"),
                checkpoint_load_mode="mystery",
            )

    def test_weights_replay_keeps_experience_and_resets_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._source_checkpoint(root)
            payload = torch.load(checkpoint, weights_only=False)
            payload["sampling_state"] = {"workers": {"0": {"episode": 99}}}
            torch.save(payload, checkpoint)
            source_bytes = (root / "replay.npz").read_bytes()
            config = DoubleDQNConfig(reward_scheme="staged", credit_assignment="one_step",
                replay_capacity=4, batch_size=2, learning_starts=0, learning_rate=3e-4,
                epsilon_start=.35, seed=99, device="cpu", replay_scheme="per", per_alpha=.8)
            controller = DoubleDQNTrainerController(config, run_dir=root / "new-run",
                checkpoint_path=checkpoint, checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_REPLAY)
            agent, step, episode, loaded = controller._build_agent(ACTION_SPACE_SIZE)
            replay = controller._restore_replay(action_dim=ACTION_SPACE_SIZE,
                start_step=step, start_episode=episode)
            self.assertEqual((step, episode, agent.update_count), (0, 0, 0))
            self.assertEqual(len(agent.optimizer.state), 0)
            self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 3e-4)
            self.assertAlmostEqual(agent.epsilon(0), .35)
            self.assertNotIn("sampling_state", loaded)
            self.assertEqual((len(replay), replay.capacity), (1, 4))
            self.assertEqual((replay.scheme, replay.alpha), ("per", .8))
            self.assertTrue(controller.summary.replay_restored)
            self.assertTrue(replay._episode_ends[0])
            self.assertFalse(replay._dones[0])
            self.assertEqual(source_bytes, (root / "replay.npz").read_bytes())
            for online, target in zip(agent.online.parameters(), agent.target.parameters()):
                self.assertTrue(torch.equal(online, target))
            first = next(agent.online.parameters()).detach()
            self.assertTrue(torch.allclose(first, torch.full_like(first, .125)))
            # A new game in the same worker must not become the old game's successor.
            replay.add(empty_observation(), 0, 1., empty_observation(), True,
                       np.ones(ACTION_SPACE_SIZE, dtype=np.bool_), ticks_advanced=100)
            self.assertEqual(replay._next_ids[0], -1)
            batch = replay.sample(2, n_step=3)
            self.assertTrue(np.isfinite(agent.learn(batch)))

    def test_weights_replay_rejects_missing_corrupt_stale_and_incompatible_pairs(self):
        for damage in ("missing", "corrupt", "stale", "reward", "schema", "credit"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint = self._source_checkpoint(root)
                sidecar = root / "replay.npz"
                if damage == "missing":
                    (root / "substitute.npz").write_bytes(sidecar.read_bytes())
                    sidecar.unlink()
                elif damage == "corrupt":
                    sidecar.write_bytes(b"broken")
                elif damage in ("stale", "schema", "credit"):
                    with np.load(sidecar, allow_pickle=False) as archive:
                        data = {name: archive[name] for name in archive.files}
                    if damage == "stale": data["env_step"] = np.asarray(1)
                    if damage == "schema": data["observation_schema"] = np.asarray("invalid")
                    if damage == "credit": data["credit_assignment"] = np.asarray("event_heads")
                    np.savez_compressed(sidecar, **data)
                config = DoubleDQNConfig(reward_scheme="staged", credit_assignment="one_step",
                                         replay_capacity=8, batch_size=2, device="cpu")
                if damage == "reward": config = replace(config, reward_scheme="sparse")
                controller = DoubleDQNTrainerController(config, run_dir=root / "new-run",
                    checkpoint_path=checkpoint, checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_REPLAY)
                _, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
                with self.assertRaisesRegex(ValueError, "weights_replay"):
                    controller._restore_replay(action_dim=ACTION_SPACE_SIZE,
                        start_step=step, start_episode=episode)
                self.assertFalse(controller.summary.replay_restored)

    def test_weights_replay_requires_source_and_independent_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DoubleDQNConfig(device="cpu", replay_capacity=8, batch_size=2)
            with self.assertRaisesRegex(ValueError, "requires a checkpoint"):
                DoubleDQNTrainerController(config, run_dir=root,
                    checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_REPLAY)
            checkpoint = self._source_checkpoint(root)
            with self.assertRaisesRegex(ValueError, "independent new run"):
                DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=checkpoint,
                    checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_REPLAY)
            existing = root / "existing"
            existing.mkdir()
            (existing / "episodes.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "already contains training history"):
                DoubleDQNTrainerController(config, run_dir=existing, checkpoint_path=checkpoint,
                    checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_REPLAY)

    def test_weights_replay_run_selection_ignores_newer_model_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._source_checkpoint(root)
            config = DoubleDQNConfig(device="cpu", replay_capacity=8, batch_size=2)
            model = DoubleDQNAgent(ACTION_SPACE_SIZE, config).save(root / "latest.pt",
                                                                  env_step=600001, episode=700)
            payload = torch.load(model, weights_only=False)
            torch.save(declare_checkpoint(payload), model)
            self.assertEqual(resolve_resume_checkpoint(root, mode="weights_replay"), full)
            controller = DoubleDQNTrainerController(config, run_dir=root / "new", checkpoint_path=model,
                checkpoint_load_mode=CHECKPOINT_LOAD_WEIGHTS_REPLAY)
            with self.assertRaisesRegex(ValueError, "model-only"):
                controller._build_agent(ACTION_SPACE_SIZE)


if __name__ == "__main__":
    unittest.main()
