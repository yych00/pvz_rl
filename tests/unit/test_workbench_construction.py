import gc
import tkinter as tk
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from train.dqn.config import DoubleDQNConfig
from train.dqn.exploration import EXPLORATION_MODE_LABELS

import workbench.main  # noqa: F401 - installs the canonical Workbench patches
from workbench import PvzGymUnitTestGUI
from workbench.gui.checkpoint_load_mode import (
    _NEW_MODEL_LABEL,
    _RESUME_LABEL,
    _TRANSFER_LABEL,
)
from workbench.gui.training import _load_evaluation_records, _source_evaluation_records, _source_trend_records


class WorkbenchConstructionTest(unittest.TestCase):
    def test_configured_executable_uses_resident_and_is_not_a_modified_default(self):
        with patch('workbench.gui.training_runtime.get_game_executable', return_value='E:/pvz/PlantsVsZombies.exe'):
            app = PvzGymUnitTestGUI(self.root)
            self.assertEqual(app._runtime_param_vars['executor'].get(), 'resident')
            self.assertEqual(app._runtime_param_vars['transport'].get(), 'binary')
            self.assertIn('默认配置', app.training_runtime_advanced.button.cget('text'))

    def test_turbo_and_common_runtime_controls_round_trip(self):
        from dataclasses import replace
        from pvz_gym.runtime import RuntimeConfig, TurboConfig
        from train.runtime import TrainingRuntimeConfig
        from workbench.gui.training_runtime import runtime_settings, apply_runtime_settings
        app = PvzGymUnitTestGUI(self.root)
        app.var_training_simulation_mode.set('Turbo (最快，全程无头)')
        app.cmb_training_simulation.event_generate('<<ComboboxSelected>>')
        self.root.update()
        self.assertEqual(app._runtime_param_vars['executor'].get(), 'resident')
        self.assertEqual(app._runtime_param_vars['transport'].get(), 'binary')
        config = DoubleDQNConfig(env=replace(DoubleDQNConfig().env, simulation_mode='turbo',
            runtime=RuntimeConfig(executor='resident',transport='shared_memory',snapshot='shared'),
            turbo=TurboConfig(fast_intro=False,fast_terminal=False,mute_audio=False)),
            runtime=TrainingRuntimeConfig(num_envs=2,process_ids=(110,220),asynchronous=True,
                inference_batch_size=2,queue_capacity=16,asynchronous_checkpoints=False))
        apply_runtime_settings(app, config)
        game, training, turbo = runtime_settings(app, env=config.env, runtime=config.runtime)
        self.assertEqual(game,config.env.runtime)
        self.assertEqual(training,config.runtime)
        self.assertEqual(turbo,config.env.turbo)
        app._training_controller = type('Controller',(),{'config':config})()
        app._set_training_controls(True,paused=True)
        self.assertEqual(str(app.cmb_training_simulation.cget('state')), 'disabled')

    def setUp(self):
        provider = patch('workbench.gui.training_runtime.get_game_executable', return_value='')
        provider.start()
        self.addCleanup(provider.stop)
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.root = None
        gc.collect()

    def test_training_panel_constructs_with_turbo_four_environment_default(self):
        app = PvzGymUnitTestGUI(self.root)
        self.root.update_idletasks()

        self.assertTrue(getattr(PvzGymUnitTestGUI, "_left_tabs_installed", False))
        self.assertTrue(getattr(PvzGymUnitTestGUI, "_training_gui_installed", False))
        self.assertTrue(getattr(PvzGymUnitTestGUI, "_dqn_headless_gui_installed", False))
        self.assertTrue(getattr(PvzGymUnitTestGUI, "_resume_run_gui_installed", False))
        self.assertTrue(
            getattr(PvzGymUnitTestGUI, "_checkpoint_load_mode_gui_installed", False)
        )

        app._switch_left_tab("training")
        self.assertEqual(app._left_tab_active, "training")
        self.assertEqual(app.var_training_trainer.get(), "Double DQN")
        self.assertEqual(app._selected_training_spec().key, "double_dqn")
        self.assertTrue(app._selected_training_spec().available)
        self.assertEqual(app._training_param_vars["n_step"].get(), "3")
        for key in ("selected_action_forward",):
            self.assertEqual(app._training_param_vars[key].get(), "True")
            self.assertEqual(tuple(app._training_param_widgets[key].cget("values")), ("True", "False"))
            self.assertEqual(str(app._training_param_widgets[key].cget("state")), "readonly")
        _, parameters = app._parse_trainer_parameters()
        self.assertIs(parameters["selected_action_forward"], True)
        self.assertNotIn("boss_projectiles", parameters)
        self.assertEqual(parameters["replay_scheme"], "per")
        self.assertEqual(app.var_training_simulation_mode.get(), "Turbo (最快，全程无头)")
        self.assertFalse(app.btn_training_run.is_disabled)
        for key, expected in (("num_envs", "6"), ("inference_batch_size", "6"),
                              ("executor", "resident"), ("transport", "binary"),
                              ("snapshot", "shared")):
            self.assertEqual(app._runtime_param_vars[key].get(), expected)
        self.assertTrue(app._runtime_param_vars["asynchronous"].get())

        from train.registry import all_trainer_specs
        self.assertNotIn("ppo", {spec.key for spec in all_trainer_specs()})
        self.assertNotIn("dreamer", {spec.key for spec in all_trainer_specs()})

        app.var_training_trainer.set("Double DQN")
        app._training_trainer_changed()
        self.assertTrue(app._selected_training_spec().available)
        self.assertEqual(app.var_training_simulation_mode.get(), "Turbo (最快，全程无头)")
        self.assertFalse(app.btn_training_run.is_disabled)

        # Active runs must not allow a second thread to create a manual
        # model/replay snapshot. Autosave/milestone checkpoints remain owned by
        # the trainer loop.
        app._training_controller = object()
        app._set_training_controls(True)
        self.assertTrue(app.btn_training_save_checkpoint.is_disabled)

        app._switch_left_tab("debug")
        self.assertEqual(app._left_tab_active, "debug")
        app._switch_left_tab("live")
        self.assertEqual(app._left_tab_active, "live")

    def test_exploration_choices_ratio_and_active_controls(self):
        app = PvzGymUnitTestGUI(self.root)
        mode = app._training_param_vars["exploration_mode"]
        ratio = app._training_param_vars["exploration_ratio"]
        mode_widget = app._training_param_widgets["exploration_mode"]
        ratio_widget = app._training_param_widgets["exploration_ratio"]
        self.assertEqual(mode.get(), EXPLORATION_MODE_LABELS["q_guided_mixed"])
        self.assertEqual(ratio.get(), "50:50")
        self.assertEqual(tuple(mode_widget.cget("values")), tuple(EXPLORATION_MODE_LABELS.values()))
        self.assertIn("Q 50% / 广泛 50%", app.lbl_training_exploration_note.cget("text"))
        ratio.set("80:20")
        self.assertIn("Q 80% / 广泛 20%", app.lbl_training_exploration_note.cget("text"))
        mode.set(EXPLORATION_MODE_LABELS["epsilon_random"])
        self.assertEqual(str(ratio_widget.cget("state")), "disabled")
        mode.set(EXPLORATION_MODE_LABELS["q_guided_neighbor"])
        self.assertEqual(str(ratio_widget.cget("state")), "normal")
        self.assertEqual(str(app._training_param_widgets["exploration_type_weights"].cget("state")), "normal")
        app._set_training_controls(True)
        self.assertEqual(str(mode_widget.cget("state")), "disabled")
        self.assertEqual(str(ratio_widget.cget("state")), "disabled")
        # 暂停状态下放行探索策略参数编辑
        app._set_training_controls(True, paused=True)
        self.assertEqual(str(mode_widget.cget("state")), "readonly")
        self.assertEqual(str(ratio_widget.cget("state")), "normal")
        app._set_training_controls(False)
        self.assertEqual(str(mode_widget.cget("state")), "readonly")
        ratio.set("0:0")
        with self.assertRaises(ValueError):
            app._parse_trainer_parameters()
        mode.set(EXPLORATION_MODE_LABELS["epsilon_random"])
        self.assertEqual(app._parse_trainer_parameters()[1]["exploration_ratio"], "50:50")

    def test_exploration_reaches_training_and_restores_from_checkpoint(self):
        app = PvzGymUnitTestGUI(self.root)
        app._use_visible_training_preset()
        app._training_param_vars["exploration_mode"].set(EXPLORATION_MODE_LABELS["q_guided_neighbor"])
        app._training_param_vars["exploration_ratio"].set("80:20")
        with patch("workbench.gui.training_smoke.threading.Thread") as thread:
            app._run_training_smoke()
        self.assertEqual(thread.call_args.kwargs["args"][-1]["exploration_mode"], "q_guided_neighbor")
        self.assertEqual(thread.call_args.kwargs["args"][-1]["exploration_ratio"], (80, 20))
        app._training_running = False
        app._static_test_running = False
        with patch("workbench.gui.training.threading.Thread"):
            app._start_training()
        self.assertEqual(app._training_controller.config.exploration_mode, "q_guided_neighbor")
        self.assertEqual(app._training_controller.config.exploration_ratio, (80, 20))

        # 测试在暂停状态下修改探索策略并继续训练，控制器得到动态热同步
        app._training_param_vars["exploration_mode"].set(EXPLORATION_MODE_LABELS["q_guided_mixed"])
        app._training_param_vars["exploration_ratio"].set("70:30")
        app._training_param_vars["exploration_type_weights"].set("1, 2, 0.5, 1, 1")
        app._training_running = True
        app._training_controller._state = "RUNNING"
        app._training_controller.pause()
        self.assertEqual(app._training_controller.state, "PAUSING")
        # The worker is mocked in this GUI test; simulate its native pause ACK.
        app._training_controller._pause_acknowledged = True
        app._training_controller._state = "PAUSED"
        app._continue_training()
        self.assertEqual(app._training_controller.config.exploration_mode, "q_guided_mixed")
        self.assertEqual(app._training_controller.config.exploration_ratio, (70, 30))
        self.assertEqual(app._training_controller.config.exploration_type_weights, (1.0, 2.0, 0.5, 1.0, 1.0))

        app._training_running = False
        app._static_test_running = False
        app._training_checkpoint_path = "current.pt"
        config = DoubleDQNConfig(exploration_mode="q_guided_mixed", exploration_ratio=(60, 40))
        app._checkpoint_inspection_finished("current.pt", config, {})
        self.assertEqual(app._training_param_vars["exploration_mode"].get(), EXPLORATION_MODE_LABELS["q_guided_mixed"])
        self.assertEqual(app._training_param_vars["exploration_ratio"].get(), "60:40")

    def test_loaded_model_tests_snapshot_selected_environment_settings(self):
        app = PvzGymUnitTestGUI(self.root)
        app._use_visible_training_preset()
        app.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
        app._training_checkpoint_path = "source.pt"
        app.var_training_ticks.set("30")
        app.var_training_playthrough.set("2")
        app.var_training_simulation_mode.set("Continuous (No Pause)")

        with patch("workbench.gui.training.threading.Thread") as thread:
            app._test_training_checkpoint()
            self.assertEqual(thread.call_args.kwargs["args"][2:4], (30, 2))
            self.assertEqual(thread.call_args.kwargs["args"][-1], "continuous")

        app._training_running = False
        app._static_test_running = False
        with patch("workbench.gui.training_smoke.threading.Thread") as thread:
            app._run_training_smoke()
            self.assertEqual(thread.call_args.kwargs["args"][2:4], (30, 2))
            self.assertEqual(thread.call_args.kwargs["args"][-1], "continuous")

    def test_smoke_heatmap_wiring_checkpoint_and_environment(self):
        app = PvzGymUnitTestGUI(self.root)
        app._use_visible_training_preset()
        app.var_training_simulation_mode.set("Continuous (No Pause)")
        app.var_training_q_overlay.set("Top 5 热力图")

        # 1. 验证 Checkpoint Smoke 路径启动了 bridge，并且 _model_smoke_worker 传入了 on_event 和 visualize_q
        app._training_checkpoint_path = "model.pt"
        with patch("workbench.gui.training_smoke.threading.Thread"), \
             patch("workbench.gui.training_smoke.TrainingVisualizationBridge") as mock_bridge_cls:
            mock_bridge = mock_bridge_cls.return_value
            app._run_training_smoke()
            self.assertTrue(app._training_eval_visualize)
            mock_bridge.start.assert_called_once_with(app._training_run_token)

        with patch("workbench.gui.training_smoke.evaluate_checkpoint") as mock_eval:
            app._model_smoke_worker("model.pt", 1, 30, 1, None, 1, "continuous")
            self.assertIn("on_event", mock_eval.call_args.kwargs)
            self.assertTrue(mock_eval.call_args.kwargs["visualize_q"]())

        # 2. 验证 Environment Smoke 路径中获取并绑定了 PID
        app._training_running = False
        app._static_test_running = False
        app._training_checkpoint_path = None
        fake_snapshot = SimpleNamespace(identity=SimpleNamespace(process_id=9999))
        mock_client = Mock()
        mock_client.read_snapshot.return_value = fake_snapshot
        from train.runtime import TrainingRuntimeConfig
        app._training_smoke_runtime_settings = None
        with patch("train.sampler.native_environment") as factory, \
             patch("workbench.gui.training_smoke.get_q_heatmap_overlay") as mock_get_overlay, \
             patch("workbench.gui.training_smoke.DoubleDQNAgent"), \
             patch("workbench.gui.training_smoke.ReplayBuffer"):
            mock_env = Mock()
            mock_env.session.backend = mock_client
            instance = Mock()
            factory.return_value = (mock_env, instance)
            mock_env.reset.side_effect = RuntimeError("stop early")
            mock_overlay = mock_get_overlay.return_value
            app._environment_smoke_worker(1, 30, 1, 100, None, "continuous", 1, 0.0, {})
            mock_overlay.bind_target.assert_called_with(9999)
            instance.close.assert_called_once()

    def test_loading_weights_enables_editable_test_game_count(self):
        app = PvzGymUnitTestGUI(self.root)
        app._use_visible_training_preset()
        self.assertEqual(app.var_training_checkpoint_load_mode.get(), _NEW_MODEL_LABEL)
        with patch("workbench.gui.training.filedialog.askopenfilename", return_value="source.pt"), \
             patch("workbench.gui.training.threading.Thread"):
            app._load_training_checkpoint()
        app._checkpoint_inspection_finished("source.pt", None, {})

        self.assertEqual(app._training_checkpoint_path, "source.pt")
        self.assertEqual(app.var_training_checkpoint_load_mode.get(), _TRANSFER_LABEL)
        self.assertEqual(str(app.ent_training_eval_episodes.cget("state")), "normal")
        self.assertFalse(app.btn_training_test_checkpoint.is_disabled)

        app.ent_training_eval_episodes.delete(0, tk.END)
        app.ent_training_eval_episodes.insert(0, "3")
        with patch("workbench.gui.training.threading.Thread") as thread:
            app._test_training_checkpoint()
        self.assertEqual(thread.call_args.kwargs["args"][4], 3)

    def test_transfer_previews_history_but_only_resume_continues_old_episodes(self):
        app = PvzGymUnitTestGUI(self.root)
        record = {"episode": 1, "result": "WON", "win_rate": 1.0}
        app._training_checkpoint_records = [record]
        evaluation = {"episode": 1, "env_step": 10, "complete": True, "episodes": 20, "win_rate": .25}
        app._training_checkpoint_evaluations = [evaluation]

        app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        app._checkpoint_mode_changed()
        self.assertEqual(app._training_history["episode"], [1])
        self.assertEqual(_source_trend_records(app), [record])
        self.assertEqual(_source_evaluation_records(app), [evaluation])

        app.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
        app._checkpoint_mode_changed()
        self.assertEqual(app._training_history["episode"], [1])
        self.assertEqual(_source_trend_records(app), [])
        self.assertEqual(app._training_history["evaluation"], [evaluation])
        self.assertEqual(_source_evaluation_records(app), [])

        app.var_training_checkpoint_load_mode.set(_NEW_MODEL_LABEL)
        app._checkpoint_mode_changed()
        self.assertEqual(app._training_history["episode"], [])
        self.assertEqual(app._training_history["evaluation"], [])

        app.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
        app._checkpoint_mode_changed()
        self.assertEqual(app._training_history["episode"], [1])
        self.assertEqual(_source_trend_records(app), [])

    def test_transfer_checkpoint_inspection_displays_saved_trend(self):
        app = PvzGymUnitTestGUI(self.root)
        app.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
        app._checkpoint_mode_changed()
        app._training_checkpoint_path = "source.pt"
        records = [
            {"episode": 1, "result": "WON", "win_rate": 1.0, "training_episode_reward": 2.0},
            {"episode": 2, "result": "LOST", "win_rate": 0.5, "training_episode_reward": -1.0},
        ]

        evaluation = {"episode": 2, "env_step": 10, "complete": True, "episodes": 20, "win_rate": .25}
        app._checkpoint_inspection_finished(
            "source.pt", None, {"episode": 2, "trend_records": records, "evaluation_records": [evaluation]}
        )
        self.root.update_idletasks()

        self.assertEqual(app._training_history["episode"], [1, 2])
        self.assertEqual(app._training_history["win_rate"], [1.0, 0.5])
        self.assertEqual(app._training_history["reward"], [2.0, -1.0])
        self.assertEqual(_source_trend_records(app), [])
        self.assertEqual(app._training_history["evaluation"], [evaluation])
        self.assertTrue(app.training_chart.find_withtag("greedy_evaluation"))

    def test_evaluation_history_excludes_partial_future_and_bad_records(self):
        valid = {"episode": 10, "env_step": 100, "complete": True, "episodes": 20, "win_rate": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            records = [dict(valid, episode=20, env_step=200), valid,
                       dict(valid, complete=False), dict(valid, env_step=201),
                       dict(valid, episode=21), dict(valid, episodes=0),
                       dict(valid, win_rate="nan"), dict(valid, episode=None), []]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n{broken\n", encoding="utf-8")
            self.assertEqual(_load_evaluation_records(path, 20, 200), [valid, records[0]])
            self.assertEqual(_load_evaluation_records(path, 10, 100), [valid])
            self.assertEqual(_load_evaluation_records(path.with_name("missing.jsonl"), 20, 200), [])

    def test_resumed_checkpoint_reloads_trend_beyond_stale_preview(self):
        app = PvzGymUnitTestGUI(self.root)
        app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            records = [{"episode": episode, "result": "WON", "win_rate": 1.0,
                        "training_episode_reward": .5, "loss": .001}
                       for episode in range(1, 5)]
            evaluation = {"episode": 3, "env_step": 30, "complete": True,
                          "episodes": 20, "win_rate": .75}
            (run_dir / 'episodes.jsonl').write_text(
                '\n'.join(json.dumps(record) for record in records), encoding='utf-8')
            (run_dir / 'evaluations.jsonl').write_text(json.dumps(evaluation), encoding='utf-8')
            app._training_checkpoint_records = records[:1]
            app._restore_training_trend(records[:1])
            app._training_controller = SimpleNamespace(run_dir=run_dir)
            app._apply_dqn_training_event('checkpoint_loaded', {
                'path': str(run_dir / 'latest.pt'), 'episode': 3, 'env_step': 30,
            })
            self.assertEqual(app._training_history['episode'], [1, 2, 3])
            self.assertEqual(app._training_checkpoint_records, records[:3])
            self.assertEqual(app._training_history['evaluation'], [evaluation])

    def test_chart_labels_large_missing_log_ranges(self):
        app = PvzGymUnitTestGUI(self.root)
        records = [{"episode": episode, "result": "WON", "win_rate": 1.0}
                   for episode in (1, 2, 200, 201)]
        app._restore_training_trend(records)
        app._draw_training_chart()
        labels = [app.training_chart.itemcget(item, 'text')
                  for item in app.training_chart.find_withtag('trend_gap')]
        self.assertEqual(labels[:2], ['缺失 197 局记录'] * 2)

    def test_greedy_events_draw_sparse_points_at_training_episodes(self):
        app = PvzGymUnitTestGUI(self.root)
        records = [{"episode": episode, "result": "LOST", "win_rate": 0.0}
                   for episode in (100, 110, 200, 500)]
        app._restore_training_trend(records)
        first = {"episode": 200, "env_step": 2000, "complete": True, "episodes": 20, "win_rate": 0.0}
        app._apply_dqn_training_event("evaluation_end", first)
        self.assertEqual(app._training_history["evaluation"], [first])
        self.assertEqual(app._training_history["episode"], [100, 110, 200, 500])
        self.assertEqual(len([item for item in app.training_chart.find_withtag("greedy_evaluation")
                              if app.training_chart.type(item) == "oval"]), 1)
        app._apply_dqn_training_event("evaluation_end", dict(first, complete=False, win_rate=1.0))
        self.assertEqual(app._training_history["evaluation"], [first])
        second = dict(first, episode=500, env_step=5000, win_rate=.25)
        app._apply_dqn_training_event("evaluation_end", second)
        app._apply_dqn_training_event("evaluation_end", second)
        items = app.training_chart.find_withtag("greedy_evaluation")
        self.assertFalse(any(app.training_chart.type(item) == "line" for item in items))
        dots = [item for item in items if app.training_chart.type(item) == "oval"]
        self.assertEqual(len(dots), 2)
        left_bounds, right_bounds = [app.training_chart.coords(item) for item in dots]
        x0 = (left_bounds[0] + left_bounds[2]) / 2
        x1 = (right_bounds[0] + right_bounds[2]) / 2
        chart_right = max(app.training_chart.winfo_width(), 420) - 18
        self.assertAlmostEqual(x0, 83 + (chart_right - 83) * .25)
        self.assertAlmostEqual(x1, chart_right)
        self.assertEqual(len(app._training_history["evaluation"]), 2)
        self.assertTrue(any("Greedy 25.0%" in app.training_chart.itemcget(item, "text")
                            for item in items if app.training_chart.type(item) == "text"))
        self.assertFalse(any("games" in app.training_chart.itemcget(item, "text")
                             for item in items if app.training_chart.type(item) == "text"))
        with patch.object(app.training_chart, "winfo_width", return_value=420):
            app._draw_training_chart()
        for title, latest in zip(app.training_chart.find_withtag("trend_title"),
                                 app.training_chart.find_withtag("trend_latest")):
            self.assertLess(app.training_chart.bbox(title)[2], app.training_chart.bbox(latest)[0])
        greedy_label = next(item for item in app.training_chart.find_withtag("greedy_evaluation")
                            if app.training_chart.type(item) == "text")
        latest_label = app.training_chart.find_withtag("trend_latest")[0]
        title_label = app.training_chart.find_withtag("trend_title")[0]
        self.assertEqual(app.training_chart.coords(greedy_label)[1], 15)
        self.assertLess(app.training_chart.bbox(greedy_label)[2], app.training_chart.bbox(latest_label)[0])
        self.assertLess(app.training_chart.bbox(title_label)[2], app.training_chart.bbox(greedy_label)[0])
        self.assertTrue(app.training_chart.itemcget(latest_label, "text").startswith("Last "))



    def test_reward_selector_has_no_event_controls_and_locks_while_running(self):
        app = PvzGymUnitTestGUI(self.root)
        self.assertFalse(hasattr(app, "var_training_credit_assignment"))
        self.assertFalse(hasattr(app, "ent_training_event_credit_weight"))
        app.var_training_reward_scheme.set("dense_early")
        app._refresh_training_reward_scheme()
        app._parse_common_training_config()
        self.assertEqual(app._training_reward_options["reward_scheme"], "dense_early")
        self.assertNotIn("credit_assignment", app._training_reward_options)
        self.assertNotIn("event_credit_weight", app._training_reward_options)
        self.assertEqual(str(app.cmb_training_reward_scheme.cget("state")), "readonly")
        app._set_training_controls(True)
        self.assertEqual(str(app.cmb_training_reward_scheme.cget("state")), "disabled")
        app._set_training_controls(False)
        self.assertEqual(str(app.cmb_training_reward_scheme.cget("state")), "readonly")

    def test_checkpoint_restores_rewards_and_n_step_and_transfer_starts_fresh(self):
        app = PvzGymUnitTestGUI(self.root)
        app.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        app._training_checkpoint_path = "current.pt"
        config = DoubleDQNConfig(reward_scheme="dense_early", n_step=5)
        app._checkpoint_inspection_finished("current.pt", config, {})
        self.assertEqual(app.var_training_reward_scheme.get(), "dense_early")
        self.assertEqual(app._training_param_vars["n_step"].get(), "5")
        app._training_param_vars["n_step"].set("3")
        app._training_resume_run_dir = "existing-run"
        app.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
        app._checkpoint_mode_changed()
        self.assertIsNone(app._training_resume_run_dir)
        app.var_training_reward_scheme.set("balanced")
        app._parse_common_training_config()
        self.assertEqual(app._training_reward_options["reward_scheme"], "balanced")
        self.assertEqual(app._training_param_vars["n_step"].get(), "3")

    def test_auto_train_every_controls_linkage(self):
        app = PvzGymUnitTestGUI(self.root)
        app._use_visible_training_preset()
        self.assertIn("auto_train_every", app._training_param_vars)
        self.assertEqual(app._training_param_vars["auto_train_every"].get(), "True")
        self.assertEqual(app._training_param_vars["auto_train_high_watermark"].get(), "16")
        self.assertEqual(app._training_param_vars["auto_train_low_watermark"].get(), "4")
        self.assertEqual(str(app._training_param_widgets["auto_train_high_watermark"].cget("state")), "normal")
        self.assertEqual(str(app._training_param_widgets["auto_train_low_watermark"].cget("state")), "normal")

        # 切换关闭自适应
        app._training_param_vars["auto_train_every"].set("False")
        self.assertEqual(str(app._training_param_widgets["auto_train_high_watermark"].cget("state")), "disabled")
        self.assertEqual(str(app._training_param_widgets["auto_train_low_watermark"].cget("state")), "disabled")

        # 重新开启自适应
        app._training_param_vars["auto_train_every"].set("True")
        self.assertEqual(str(app._training_param_widgets["auto_train_high_watermark"].cget("state")), "normal")
        self.assertEqual(str(app._training_param_widgets["auto_train_low_watermark"].cget("state")), "normal")

        spec, params = app._parse_trainer_parameters()
        self.assertTrue(params["auto_train_every"])
        self.assertEqual(params["auto_train_high_watermark"], 16)
        self.assertEqual(params["auto_train_low_watermark"], 4)

    def test_evaluation_mode_single_vs_multi_level_defaults(self):
        from train.common import EnvConfig
        from train.dqn.config import DoubleDQNConfig

        # 1. 验证底层 DQNConfig 默认推断：单关为 random，多关为 per_level
        single_cfg = DoubleDQNConfig(env=EnvConfig(level_id=50))
        self.assertEqual(single_cfg.evaluation_mode, "random")

        multi_cfg = DoubleDQNConfig(env=EnvConfig(level_id=1, level_pool=((1, 1), (2, 1))))
        self.assertEqual(multi_cfg.evaluation_mode, "per_level")

        # 显式传参时尊重传入值
        explicit_single = DoubleDQNConfig(env=EnvConfig(level_id=50), evaluation_mode="per_level")
        self.assertEqual(explicit_single.evaluation_mode, "per_level")

        # 2. 验证 Workbench GUI 界面初始与联动
        app = PvzGymUnitTestGUI(self.root)
        app._switch_left_tab("training")
        self.assertIn("evaluation_mode", app._training_param_vars)

        # 单一关卡模式下默认 random
        self.assertEqual(app._training_param_vars["evaluation_mode"].get(), "random")

        # 切换到混合关卡多关模式
        app.mixed_level_panel.var_mode.set("mixed")
        app.mixed_level_panel._on_mode_toggled()
        self.assertEqual(app._training_param_vars["evaluation_mode"].get(), "per_level")

        # 切回单一关卡模式
        app.mixed_level_panel.var_mode.set("single")
        app.mixed_level_panel._on_mode_toggled()
        self.assertEqual(app._training_param_vars["evaluation_mode"].get(), "random")

    def test_evaluation_mode_off(self):
        from train.common import EnvConfig
        from train.dqn.config import DoubleDQNConfig
        from train.dqn.evaluation_plan import evaluation_plan
        from unittest.mock import MagicMock

        # 1. 配置测试
        off_cfg = DoubleDQNConfig(env=EnvConfig(level_id=50), evaluation_mode="off")
        self.assertEqual(off_cfg.evaluation_mode, "off")
        none_cfg = DoubleDQNConfig(env=EnvConfig(level_id=50), evaluation_mode="none")
        self.assertEqual(none_cfg.evaluation_mode, "off")

        # 2. evaluation_plan 测试
        mock_controller = MagicMock()
        mock_controller.config = off_cfg
        mock_controller._legacy_evaluation_protocol = False
        plan = evaluation_plan(mock_controller, random_count=20)
        self.assertEqual(plan, [])

        # 3. GUI 联动测试
        app = PvzGymUnitTestGUI(self.root)
        app._switch_left_tab("training")
        self.assertIn("evaluation_mode", app._training_param_vars)

        # 切换到 off 模式
        app._training_param_vars["evaluation_mode"].set("off")
        self.assertEqual(str(app._training_param_widgets["evaluation_per_level"].cget("state")), "disabled")
        self.assertEqual(str(app._training_param_widgets["evaluation_random_episodes"].cget("state")), "disabled")

        # 此时即便切换多关，由于是 off 模式，不应被强制覆盖为 per_level
        app.mixed_level_panel.var_mode.set("mixed")
        app.mixed_level_panel._on_mode_toggled()
        self.assertEqual(app._training_param_vars["evaluation_mode"].get(), "off")

        # 切换到 random 模式
        app._training_param_vars["evaluation_mode"].set("random")
        self.assertEqual(str(app._training_param_widgets["evaluation_per_level"].cget("state")), "disabled")
        self.assertEqual(str(app._training_param_widgets["evaluation_random_episodes"].cget("state")), "normal")

        # 切换到 per_level 模式
        app._training_param_vars["evaluation_mode"].set("per_level")
        self.assertEqual(str(app._training_param_widgets["evaluation_per_level"].cget("state")), "normal")
        self.assertEqual(str(app._training_param_widgets["evaluation_random_episodes"].cget("state")), "disabled")

    def test_non_resume_training_clears_stale_history_on_start(self):
        from workbench.gui.checkpoint_load_mode import _REPLAY_TRANSFER_LABEL, _TRANSFER_LABEL
        from workbench.gui.training import _source_trend_records, _source_evaluation_records
        app = PvzGymUnitTestGUI(self.root)
        record = {"episode": 100, "result": "WON", "win_rate": 1.0}
        app._training_checkpoint_records = [record]
        evaluation = {"episode": 100, "env_step": 10, "complete": True, "episodes": 20, "win_rate": .25}
        app._training_checkpoint_evaluations = [evaluation]

        # 在“权重 + Replay 微调 (新训练)”模式下，不应继承旧 Run 的对局历史
        app.var_training_checkpoint_load_mode.set(_REPLAY_TRANSFER_LABEL)
        self.assertEqual(_source_trend_records(app), [])
        self.assertEqual(_source_evaluation_records(app), [])

        # 在“仅继承权重微调 (新训练)”模式下，不应继承旧 Run 的对局历史
        app.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
        self.assertEqual(_source_trend_records(app), [])
        self.assertEqual(_source_evaluation_records(app), [])

        # 如果对局事件从 episode=1 重新开始，历史走势残留的旧数据必须被自动清空
        app._training_history = {
            "episode": [100, 101], "outcome": ["WON", "WON"], "win_rate": [1.0, 1.0],
            "reward": [1.0, 1.0], "recent_win_rate": [1.0, 1.0], "recent_reward": [1.0, 1.0],
            "loss": [0.1, 0.1], "recent_loss": [0.1, 0.1], "evaluation": [],
        }
        app._apply_dqn_training_event("episode_end", {"episode": 1, "result": "WON", "win_rate": 1.0})
        self.assertEqual(app._training_history["episode"], [1])


if __name__ == "__main__":
    unittest.main()
