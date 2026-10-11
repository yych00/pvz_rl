from __future__ import annotations

import gc
import json
import tempfile
import tkinter as tk
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import workbench.main  # noqa: F401 - install the complete Workbench wrappers
from workbench import PvzGymUnitTestGUI
from workbench.gui.checkpoint_load_mode import (
    _MODEL_LABEL, _NEW_MODEL_LABEL, _RESUME_LABEL, _STRICT_LABEL, _TRANSFER_LABEL,
    _REPLAY_TRANSFER_LABEL, _mode_key,
)
from pvz_gym.action_space import ACTION_SPACE_SIZE
from train.common import EnvConfig
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.controller import DoubleDQNTrainerController
from train.dqn.run_state import load_checkpoint_seed_types, load_run_config


class WorkbenchResumeParametersTest(unittest.TestCase):
    def setUp(self):
        # These fixtures exercise checkpoint settings, not machine-local .env.
        for target in ('workbench.gui.training_runtime.get_game_executable',
                       'pvz_gym.env_file.get_game_executable'):
            provider = patch(target, return_value='')
            provider.start()
            self.addCleanup(provider.stop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.config = DoubleDQNConfig(
            env=EnvConfig(level_id=7, playthrough=2, ticks_per_step=30,
                          max_episode_ticks=9000, simulation_mode="headless"),
            hidden_sizes=(16, 8), device="cpu", seed=42, total_steps=5000,
            replay_capacity=8, batch_size=2, learning_starts=0, n_step=5,
            learning_rate=.0003, gamma=.98, epsilon_start=.6, epsilon_end=.1,
            epsilon_decay_steps=1234, exploration_mode="q_guided_neighbor",
            exploration_ratio=(80, 20), exploration_type_weights=(3, 6, .02, .3, .4),
            train_every=4, gradient_steps=2, target_update_interval=123,
            autosave_every=555, checkpoint_every=777, reward_scheme="sparse",
            reward_scale=.015, mower_penalty=.2, max_grad_norm=2,
        )
        self.cards = (0, 1, 2)
        self.checkpoint = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config).save(
            self.directory / "latest.pt", env_step=1000, episode=20, seed_types=self.cards,
        )
        self.app = PvzGymUnitTestGUI(self.root)

    def tearDown(self):
        self.root.destroy()
        self.app = self.root = None
        gc.collect()
        self.tmp.cleanup()

    def load_checkpoint(self):
        self.app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        self.app._training_checkpoint_path = str(self.checkpoint)
        self.app._checkpoint_inspection_finished(
            str(self.checkpoint), self.config, {"seed_types": self.cards},
        )

    def start(self):
        with patch("workbench.gui.training.threading.Thread"):
            self.app._start_training()
        controller = self.app._training_controller
        self.assertIsNotNone(controller)
        agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
        self.assertEqual((step, episode), (1000, 20))
        self.assertIs(agent.config, controller.config)
        return controller, agent

    def test_checkpoint_restores_complete_config_and_actual_optimizer(self):
        self.load_checkpoint()
        controller, agent = self.start()
        self.assertEqual(asdict(controller.config), asdict(self.config))
        self.assertEqual(controller.seed_types, self.cards)
        self.assertEqual(agent.hidden_sizes, self.config.hidden_sizes)
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], self.config.learning_rate)

    def test_loaded_checkpoint_keeps_edited_sampler_runtime(self):
        self.load_checkpoint()
        values = self.app._runtime_param_vars
        values['num_envs'].set('2')
        values['process_ids'].set('110,220')
        values['inference_batch_size'].set('2')
        values['asynchronous'].set(True)
        values['asynchronous_checkpoints'].set(False)
        controller, _ = self.start()
        self.assertEqual(controller.config.runtime.num_envs, 2)
        self.assertEqual(controller.config.runtime.process_ids, (110, 220))
        self.assertEqual(controller.config.runtime.inference_batch_size, 2)
        self.assertTrue(controller.config.runtime.asynchronous)
        self.assertFalse(controller.config.runtime.asynchronous_checkpoints)

    def test_actor_learner_selection_and_cadence_survive_checkpoint_controls(self):
        self.load_checkpoint()
        values = self.app._training_param_vars
        values['training_architecture'].set('legacy')
        values['policy_sync_interval'].set('8')
        values['max_pending_updates'].set('24')
        controller, _ = self.start()
        self.assertEqual(controller.config.training_architecture, 'legacy')
        self.assertEqual(controller.config.policy_sync_interval, 8)
        self.assertEqual(controller.config.max_pending_updates, 24)

    def test_invalid_hyperparameters_allow_correction_and_restart(self):
        self.load_checkpoint()
        values = self.app._training_param_vars
        for name, invalid in (('gamma', '1.5'), ('batch_size', '99'),
                              ('learning_rate', '-0.1')):
            with self.subTest(name=name):
                original = values[name].get()
                values[name].set(invalid)
                with patch('workbench.gui.training.threading.Thread') as thread:
                    self.app._start_training()
                thread.assert_not_called()
                self.assertIsNone(self.app._training_controller)
                self.assertFalse(self.app._training_running)
                self.assertFalse(self.app._static_test_running)
                self.assertFalse(self.app.btn_training_run.is_disabled)
                self.assertIn('Config error:', self.app.lbl_training_progress.cget('text'))
                values[name].set(original)
        self.start()

    def test_run_config_remains_authoritative_after_async_checkpoint_inspection(self):
        run_config = replace(self.config, learning_rate=.0007,
                             exploration_type_weights=(1, 2, .01, .2, .3))
        (self.directory / "config.json").write_text(
            json.dumps(dict(asdict(run_config), seed_types=self.cards)), encoding="utf-8",
        )
        with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=str(self.directory)), \
             patch("workbench.gui.resume_run.threading.Thread"):
            self.app._resume_training_run()
        self.assertTrue(self.app._training_checkpoint_loading)
        self.app._start_training()
        self.assertIsNone(self.app._training_controller)
        self.app._checkpoint_inspection_finished(
            str(self.checkpoint), self.config, {"seed_types": self.cards},
        )
        controller, agent = self.start()
        self.assertEqual(asdict(controller.config), asdict(run_config))
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], .0007)

    def test_resume_run_preserves_explicit_strict_and_model_modes(self):
        (self.directory / "config.json").write_text(
            json.dumps(dict(asdict(self.config), seed_types=self.cards)), encoding="utf-8",
        )
        for label, mode in ((_STRICT_LABEL, "strict_resume"), (_MODEL_LABEL, "model_resume")):
            with self.subTest(mode=mode):
                self.app.var_training_checkpoint_load_mode.set(label)
                with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=str(self.directory)), \
                     patch("workbench.gui.resume_run.resolve_resume_checkpoint", return_value=self.checkpoint) as resolve, \
                     patch("workbench.gui.resume_run.threading.Thread"):
                    self.app._resume_training_run()
                resolve.assert_called_once_with(self.directory, mode=mode)
                self.assertEqual(self.app.var_training_checkpoint_load_mode.get(), label)
                self.assertEqual(self.app._training_resume_selected_mode, mode)
                self.app._checkpoint_inspection_finished(
                    str(self.checkpoint), self.config, {"seed_types": self.cards},
                )
                self.assertEqual(self.app.var_training_checkpoint_load_mode.get(), label)
                self.assertEqual(str(self.app.cmb_training_level.cget("state")), "disabled")

    def test_changing_run_mode_reselects_checkpoint_and_discards_old_preview(self):
        run_config = replace(self.config, learning_rate=.0007)
        (self.directory / "config.json").write_text(
            json.dumps(dict(asdict(run_config), seed_types=self.cards)), encoding="utf-8",
        )
        full_checkpoint = self.directory / "latest-full.pt"
        self.app.var_training_checkpoint_load_mode.set(_MODEL_LABEL)
        with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=str(self.directory)), \
             patch("workbench.gui.resume_run.resolve_resume_checkpoint", return_value=self.checkpoint), \
             patch("workbench.gui.resume_run.threading.Thread"):
            self.app._resume_training_run()
        self.app._checkpoint_inspection_finished(
            str(self.checkpoint), self.config,
            {"env_step": 1000, "episode": 20, "seed_types": self.cards,
             "trend_records": [{"episode": 20, "env_step": 1000, "result": "WON"}]},
        )
        with patch("workbench.gui.resume_run.resolve_resume_checkpoint", return_value=full_checkpoint) as resolve, \
             patch("workbench.gui.resume_run.threading.Thread"):
            self.app.var_training_checkpoint_load_mode.set(_STRICT_LABEL)
            self.app._checkpoint_mode_changed()
        resolve.assert_called_once_with(self.directory, mode="strict_resume")
        self.assertEqual(self.app._training_checkpoint_path, str(full_checkpoint))
        self.assertEqual(self.app._training_checkpoint_records, [])
        self.assertTrue(self.app._training_checkpoint_loading)
        self.assertTrue(self.app.btn_training_run.is_disabled)
        self.app._checkpoint_inspection_finished(
            str(self.checkpoint), self.config, {"env_step": 9999, "episode": 99},
        )
        self.assertTrue(self.app._training_checkpoint_loading)
        full_records = [{"episode": 8, "env_step": 400, "result": "LOST"}]
        self.app._checkpoint_inspection_finished(
            str(full_checkpoint), replace(self.config, learning_rate=.001),
            {"env_step": 400, "episode": 8, "seed_types": self.cards, "trend_records": full_records},
        )
        self.assertEqual(self.app._training_checkpoint_records, full_records)
        self.assertEqual(self.app._training_metric_labels["step"].cget("text"), "400")
        self.assertEqual(self.app._training_param_vars["learning_rate"].get(), "0.0007")
        self.assertEqual(self.app._training_loaded_config, run_config)
        self.assertEqual(self.app._training_checkpoint_config.learning_rate, .001)
        self.assertFalse(self.app.btn_training_run.is_disabled)

    def test_missing_full_checkpoint_blocks_run_and_switching_mode_recovers(self):
        (self.directory / "config.json").write_text(
            json.dumps(dict(asdict(self.config), seed_types=self.cards)), encoding="utf-8",
        )
        self.app.var_training_checkpoint_load_mode.set(_STRICT_LABEL)
        with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=str(self.directory)), \
             patch("workbench.gui.resume_run.resolve_resume_checkpoint", side_effect=ValueError("no matching full Replay")), \
             patch("workbench.gui.resume_run.threading.Thread") as thread:
            self.app._resume_training_run()
        thread.assert_not_called()
        self.assertEqual(self.app.var_training_checkpoint_load_mode.get(), _STRICT_LABEL)
        self.assertIsNone(self.app._training_checkpoint_path)
        self.assertEqual(self.app._training_resume_run_dir, self.directory)
        self.assertIn("no matching full Replay", self.app.lbl_training_checkpoint_mode_note.cget("text"))
        self.assertTrue(self.app.btn_training_run.is_disabled)
        with patch("workbench.gui.training.threading.Thread") as training_thread:
            self.app._start_training()
        training_thread.assert_not_called()
        self.assertIsNone(self.app._training_controller)

        with patch("workbench.gui.resume_run.resolve_resume_checkpoint", return_value=self.checkpoint) as resolve, \
             patch("workbench.gui.resume_run.threading.Thread"):
            self.app.var_training_checkpoint_load_mode.set(_MODEL_LABEL)
            self.app._checkpoint_mode_changed()
        resolve.assert_called_once_with(self.directory, mode="model_resume")
        self.app._checkpoint_inspection_finished(
            str(self.checkpoint), self.config, {"seed_types": self.cards},
        )
        self.assertIsNone(self.app._training_resume_resolution_error)
        self.assertFalse(self.app.btn_training_run.is_disabled)

    def test_failed_run_preview_requires_reload_or_new_model(self):
        (self.directory / "config.json").write_text(
            json.dumps(dict(asdict(self.config), seed_types=self.cards)), encoding="utf-8",
        )
        self.app.var_training_checkpoint_load_mode.set(_MODEL_LABEL)
        with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=str(self.directory)), \
             patch("workbench.gui.resume_run.resolve_resume_checkpoint", return_value=self.checkpoint), \
             patch("workbench.gui.resume_run.threading.Thread"):
            self.app._resume_training_run()
        self.app._checkpoint_inspection_failed(str(self.checkpoint), ValueError("unreadable model"))
        self.assertIsNone(self.app._training_checkpoint_path)
        self.assertTrue(self.app.btn_training_run.is_disabled)
        self.app.var_training_checkpoint_load_mode.set(_NEW_MODEL_LABEL)
        self.app._checkpoint_mode_changed()
        self.assertIsNone(self.app._training_resume_run_dir)
        self.assertIsNone(self.app._training_resume_resolution_error)
        self.assertFalse(self.app.btn_training_run.is_disabled)

    def test_restored_acceleration_option_and_required_boss_input(self):
        config = replace(self.config, selected_action_forward=False)
        checkpoint = DoubleDQNAgent(ACTION_SPACE_SIZE, config).save(
            self.directory / "current.pt", env_step=1000, episode=20, seed_types=self.cards)
        self.app.var_training_checkpoint_load_mode.set(_MODEL_LABEL)
        self.app._training_checkpoint_path = str(checkpoint)
        self.app._checkpoint_inspection_finished(str(checkpoint), config, {"seed_types": self.cards})
        self.assertEqual(self.app._training_param_vars["selected_action_forward"].get(), "False")
        self.assertNotIn("boss_projectiles", self.app._training_param_widgets)
        controller, agent = self.start()
        self.assertFalse(controller.config.selected_action_forward)
        self.assertTrue(controller.config.boss_projectiles)
        self.assertTrue(agent.config.boss_projectiles)

    def test_checkpoint_restore_warning_is_visible_and_logged(self):
        with patch.object(self.app, "log") as log:
            self.app._apply_dqn_training_event("checkpoint_restore_contract", {
                "checkpoint_kind": "full", "load_mode": "strict_resume",
                "restoration_warnings": ["CUDA RNG 未恢复", "第二项提示"],
            })
        text = self.app.lbl_training_progress.cget("text")
        self.assertIn("CUDA RNG 未恢复", text)
        self.assertIn("另 1 项", text)
        self.assertTrue(any("第二项提示" in str(call) for call in log.call_args_list))

    def test_safe_overrides_reach_agent_and_keep_unexposed_fields(self):
        self.load_checkpoint()
        variables = self.app._training_param_vars
        variables["learning_rate"].set("0.0009")
        variables["exploration_type_weights"].set("2,4,0.01,0.1,0.1")
        variables["gamma"].set("0.95")
        variables["n_step"].set("3")
        variables["evaluation_mode"].set("random")
        variables["evaluation_per_level"].set("6")
        variables["evaluation_random_episodes"].set("24")
        variables["adaptive_level_sampling"].set("True")
        variables["level_sampling_window"].set("50")
        variables["level_sampling_update_every"].set("10")
        variables["level_sampling_uniform_mix"].set("0.4")
        variables["level_sampling_max_weight"].set("3")
        self.app.var_training_ticks.set("60")
        controller, agent = self.start()
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], .0009)
        self.assertEqual(agent.config.exploration_type_weights, (2, 4, .01, .1, .1))
        self.assertEqual((agent.config.gamma, agent.config.n_step), (.95, 3))
        self.assertEqual(controller.config.env.ticks_per_step, 60)
        self.assertEqual(controller.config.reward_scale, .015)
        self.assertEqual(controller.config.max_grad_norm, 2)
        self.assertEqual(controller.config.evaluation_mode, "random")
        self.assertEqual(controller.config.evaluation_per_level, 6)
        self.assertEqual(controller.config.evaluation_random_episodes, 24)
        self.assertTrue(controller.config.adaptive_level_sampling)
        self.assertEqual(controller.config.level_sampling_window, 50)
        self.assertEqual(controller.config.level_sampling_update_every, 10)
        self.assertEqual(controller.config.level_sampling_uniform_mix, .4)
        self.assertEqual(controller.config.level_sampling_max_weight, 3)

    def test_resume_locks_task_fields_and_transfer_unlocks_them(self):
        self.load_checkpoint()
        for name in ("cmb_training_level", "cmb_training_playthrough",
                     "ent_training_seed", "ent_training_seed_types"):
            self.assertEqual(str(getattr(self.app, name).cget("state")), "disabled")
        self.assertEqual(str(self.app._training_param_widgets["learning_rate"].cget("state")), "normal")
        self.app.var_training_level.set("1-8")
        with patch("workbench.gui.training.threading.Thread") as thread:
            self.app._start_training()
        thread.assert_not_called()
        self.assertIsNone(self.app._training_controller)
        self.app.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
        self.app._checkpoint_mode_changed()
        self.assertEqual(str(self.app.cmb_training_level.cget("state")), "readonly")
        self.assertEqual(str(self.app.ent_training_seed.cget("state")), "normal")

    def test_replay_transfer_starts_new_run_and_displays_inherited_replay(self):
        self.load_checkpoint()
        self.app._training_resume_run_dir = self.directory
        self.app._training_resume_config = self.config
        self.app.var_training_checkpoint_load_mode.set(_REPLAY_TRANSFER_LABEL)
        self.app._checkpoint_mode_changed()
        self.assertEqual(_mode_key(_REPLAY_TRANSFER_LABEL), "weights_replay")
        self.assertIsNone(self.app._training_resume_run_dir)
        self.assertFalse(self.app._training_include_source_history)
        self.assertEqual(str(self.app.cmb_training_level.cget("state")), "readonly")
        with patch("workbench.gui.training.threading.Thread"):
            self.app._start_training()
        controller = self.app._training_controller
        self.assertIsNotNone(controller)
        self.assertEqual(controller.checkpoint_load_mode, "weights_replay")
        self.assertNotEqual(controller.run_dir.resolve(), self.directory.resolve())
        self.app._apply_dqn_training_event("checkpoint_loaded", {
            "path": str(self.checkpoint), "env_step": 0, "episode": 0,
            "replay_restored": True, "replay_size": 123,
        })
        self.assertIn("123", self.app.lbl_training_checkpoint.cget("text"))
        self.assertEqual(self.app._training_metric_labels["replay"].cget("text"), "123")
        self.assertEqual(self.app._training_metric_labels["step"].cget("text"), "0")

    def test_replay_transfer_run_picker_preserves_mode_and_creates_fresh_run(self):
        self.app.var_training_checkpoint_load_mode.set(_REPLAY_TRANSFER_LABEL)
        (self.directory / "config.json").write_text(
            json.dumps(dict(asdict(self.config), seed_types=self.cards)), encoding="utf-8")
        with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=str(self.directory)), \
             patch("workbench.gui.resume_run.resolve_resume_checkpoint", return_value=self.checkpoint) as resolve, \
             patch("workbench.gui.resume_run.threading.Thread"):
            self.assertTrue(self.app._resume_training_run())
        resolve.assert_called_once_with(self.directory, mode="weights_replay")
        self.assertEqual(self.app.var_training_checkpoint_load_mode.get(), _REPLAY_TRANSFER_LABEL)
        self.assertIsNone(self.app._training_resume_run_dir)

    def test_backend_rejects_locked_changes_even_without_gui(self):
        changes = (
            replace(self.config, hidden_sizes=(32, 16)),
            replace(self.config, seed=43),
            replace(self.config, env=replace(self.config.env, level_id=8)),
            replace(self.config, env=replace(self.config.env, playthrough=1)),
        )
        for config in changes:
            with self.subTest(config=config):
                controller = DoubleDQNTrainerController(config, checkpoint_path=self.checkpoint,
                                                       seed_types=self.cards)
                with self.assertRaisesRegex(ValueError, "接续训练不能修改"):
                    controller._build_agent(ACTION_SPACE_SIZE)
        controller = DoubleDQNTrainerController(self.config, checkpoint_path=self.checkpoint,
                                               seed_types=(0, 1, 3))
        with self.assertRaisesRegex(ValueError, "不能修改卡组"):
            controller._build_agent(ACTION_SPACE_SIZE)

    def test_hot_updates_persist_same_settings_to_run_and_checkpoint(self):
        self.load_checkpoint()
        controller, agent = self.start()
        controller._agent = agent
        controller.run_dir = self.directory / "continued-run"
        controller.run_dir.mkdir()
        controller._persist_run_config()
        controller.set_exploration_settings(mode="q_guided_mixed", ratio=(70, 30),
                                            type_weights=(2, 4, .01, .1, .1))
        saved_run = load_run_config(controller.run_dir)
        self.assertEqual(asdict(saved_run), asdict(agent.config))
        self.assertFalse((controller.run_dir / "config.json.tmp").exists())
        path = controller.save_checkpoint(controller.run_dir / "updated.pt")
        _, saved_checkpoint, payload = DoubleDQNAgent.from_checkpoint(path, device="cpu")
        self.assertEqual(asdict(saved_checkpoint), asdict(saved_run))
        self.assertEqual(tuple(payload["seed_types"]), self.cards)

    def test_invalid_paused_settings_do_not_resume_with_old_parameters(self):
        self.load_checkpoint()
        controller, agent = self.start()
        controller._agent = agent
        controller._state = "RUNNING"
        controller.pause()
        # This test covers invalid GUI settings after the runtime has confirmed
        # its pause. The pause/ACK protocol has separate controller tests.
        controller._pause_acknowledged = True
        controller._state = "PAUSED"
        self.app._training_param_vars["exploration_ratio"].set("0:0")
        self.app._continue_training()
        self.assertEqual(controller.state, "PAUSED")
        self.assertFalse(controller.pause_event.is_set())
        self.assertEqual(agent.config.exploration_ratio, (80, 20))

    def test_old_checkpoint_deck_uses_matching_sidecar_and_explicit_null_wins(self):
        path = self.directory / "config.json"
        path.write_text(json.dumps(dict(asdict(self.config), seed_types=self.cards)), encoding="utf-8")
        self.assertEqual(load_checkpoint_seed_types(self.checkpoint, self.config, {}), self.cards)
        self.assertIsNone(load_checkpoint_seed_types(self.checkpoint, self.config, {"seed_types": None}))
        other_level = replace(self.config, env=replace(self.config.env, level_id=8))
        self.assertIsNone(load_checkpoint_seed_types(self.checkpoint, other_level, {}))
        path.write_text("invalid json", encoding="utf-8")
        self.assertIsNone(load_checkpoint_seed_types(self.checkpoint, self.config, {}))

    def test_cli_checkpoint_resume_restores_saved_deck(self):
        from train.dqn import train as train_module
        with patch.object(train_module, "train") as train, \
             patch("sys.argv", ["train", "--resume", str(self.checkpoint)]):
            train_module.main()
        self.assertEqual(asdict(train.call_args.args[0]), asdict(self.config))
        self.assertEqual(train.call_args.args[3], self.cards)

    def test_resume_run_with_level_pool_restores_and_retains_mixed_mode(self):
        # 1. 构造一个包含多关关卡池的历史配置
        pool = ((1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1))
        mixed_config = replace(
            self.config,
            env=replace(self.config.env, level_id=1, playthrough=1, level_pool=pool),
        )
        mixed_checkpoint = DoubleDQNAgent(ACTION_SPACE_SIZE, mixed_config).save(
            self.directory / "mixed_latest.pt", env_step=2000, episode=30, seed_types=self.cards,
        )

        # 2. 模拟接续训练加载该检查点
        self.app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        self.app._training_checkpoint_path = str(mixed_checkpoint)
        self.app._checkpoint_inspection_finished(
            str(mixed_checkpoint), mixed_config, {"seed_types": self.cards},
        )

        # 3. 验证混合面板已被自动激活，且关卡池完整还原
        self.assertTrue(self.app.mixed_level_panel.is_mixed_mode())
        self.assertEqual(self.app.mixed_level_panel.get_level_pool(), pool)

        # 4. 启动续训，验证 controller.config.env.level_pool 绝不退回单关
        with patch("workbench.gui.training.threading.Thread"):
            self.app._start_training()
        controller = self.app._training_controller
        self.assertIsNotNone(controller)
        self.assertEqual(controller.config.env.level_pool, pool)

    def test_resume_run_prevents_silent_revert_to_single_level(self):
        # 验证即使面板被意外切到单关，接续训练也会强制恢复关卡池，绝不悄悄退回单关
        pool = ((1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1))
        mixed_config = replace(
            self.config,
            env=replace(self.config.env, level_id=1, playthrough=1, level_pool=pool),
        )
        mixed_checkpoint = DoubleDQNAgent(ACTION_SPACE_SIZE, mixed_config).save(
            self.directory / "mixed_guard.pt", env_step=2000, episode=30, seed_types=self.cards,
        )

        self.app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        self.app._training_checkpoint_path = str(mixed_checkpoint)
        self.app._checkpoint_inspection_finished(
            str(mixed_checkpoint), mixed_config, {"seed_types": self.cards},
        )

        # 人为把面板强行切成单关模式
        self.app.mixed_level_panel.set_from_level_pool(None)
        self.assertFalse(self.app.mixed_level_panel.is_mixed_mode())

        # 启动续训：必须自动强制恢复多关关卡池
        with patch("workbench.gui.training.threading.Thread"):
            self.app._start_training()
        controller = self.app._training_controller
        self.assertIsNotNone(controller)
        self.assertEqual(controller.config.env.level_pool, pool)
        self.assertTrue(self.app.mixed_level_panel.is_mixed_mode())

    def test_validate_resume_config_rejects_level_pool_tampering(self):
        from train.dqn.config import validate_resume_config
        pool = ((1, 1), (2, 1))
        saved = replace(self.config, env=replace(self.config.env, level_pool=pool))

        # 请求变为单关 (None) -> 必须报错拒绝
        req_single = replace(saved, env=replace(saved.env, level_pool=None))
        with self.assertRaises(ValueError) as ctx:
            validate_resume_config(req_single, saved)
        self.assertIn("env.level_pool", str(ctx.exception))


        # 请求变为不同关卡池 -> 必须报错拒绝
        req_diff = replace(saved, env=replace(saved.env, level_pool=((1, 1), (3, 1))))
        with self.assertRaises(ValueError) as ctx:
            validate_resume_config(req_diff, saved)
        self.assertIn("env.level_pool", str(ctx.exception))

    def test_resume_custom_pool_preserves_adaptive_shuffle(self):
        pool = ((49, 2), (41, 1), (43, 2))
        config = replace(self.config, env=replace(self.config.env,
            level_id=49, playthrough=2, level_pool=pool, seed_selection_mode="adaptive_shuffled"))
        checkpoint = DoubleDQNAgent(ACTION_SPACE_SIZE, config).save(
            self.directory / "mixed_adaptive.pt", env_step=2000, episode=30, seed_types=None)
        self.app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        self.app._training_checkpoint_path = str(checkpoint)
        self.app._checkpoint_inspection_finished(str(checkpoint), config, {"seed_types": None})
        self.assertEqual(self.app.mixed_level_panel.get_level_pool(), pool)
        self.assertEqual(self.app.mixed_level_panel.get_seed_selection_mode(), "adaptive_shuffled")
        with patch("workbench.gui.training.threading.Thread"):
            self.app._start_training()
        self.assertIsNotNone(self.app._training_controller)
        self.assertEqual(self.app._training_controller.config.env.level_pool, pool)
        self.assertEqual(self.app._training_controller.config.env.seed_selection_mode, "adaptive_shuffled")

    def test_gui_resume_rejects_modified_pool(self):
        from train.curriculum import generate_mixed_level_pool
        pool = generate_mixed_level_pool([1], [1])
        config = replace(self.config, env=replace(self.config.env, level_id=1, playthrough=1, level_pool=pool))
        checkpoint = DoubleDQNAgent(ACTION_SPACE_SIZE, config).save(
            self.directory / "mixed_modified.pt", env_step=2000, episode=30, seed_types=self.cards)
        self.app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        self.app._training_checkpoint_path = str(checkpoint)
        self.app._checkpoint_inspection_finished(str(checkpoint), config, {"seed_types": self.cards})
        self.app.mixed_level_panel.scene_vars[2].set(True)
        self.app.mixed_level_panel._handle_change()
        with patch("workbench.gui.training.threading.Thread"):
            self.app._start_training()
        self.assertIsNone(self.app._training_controller)
        self.assertIn("env.level_pool", self.app.lbl_training_progress.cget("text"))

if __name__ == "__main__":
    unittest.main()
