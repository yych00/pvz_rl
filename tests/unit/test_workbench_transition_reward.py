import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from workbench.gui.reward_scheme import install_reward_scheme_gui


class WorkbenchTransitionRewardTest(unittest.TestCase):
    def test_dense_badges_and_note_show_effective_reward_semantics(self):
        hooks = ("_build_left_panel", "_training_trainer_changed", "_set_training_controls",
                 "_dqn_training_worker", "_checkpoint_inspection_finished", "_parse_common_training_config")
        gui_class = type("FakeGUI", (), {name: lambda *args, **kwargs: None for name in hooks})
        install_reward_scheme_gui(gui_class, SimpleNamespace(CLR_TEXT_MAIN="white", CLR_TEXT_MUTED="grey"))
        gui = gui_class()
        gui._selected_training_spec = lambda: SimpleNamespace(key="double_dqn")
        gui.var_training_reward_scheme = Mock()
        gui.var_training_reward_scheme.get.return_value = "transition_dense"
        gui.var_training_mower_penalty = Mock()
        gui.cmb_training_reward_scheme = Mock()
        gui.lbl_training_reward_note = Mock()
        names = ("terminal", "mower", "threat", "plants", "sun")
        gui._training_reward_badges = {name: Mock() for name in names}
        gui._training_reward_badge_titles = {name: Mock() for name in names}
        gui._refresh_training_reward_scheme()
        gui.cmb_training_reward_scheme.config.assert_called_with(state="readonly")
        expected = ("+1 / -1", "-0.50 / -0.10", "+0.40 / -0.40", "÷5000", "×0.30")
        for name, value in zip(names, expected):
            self.assertEqual(gui._training_reward_badges[name].config.call_args.kwargs["text"], value)
        self.assertIn("威胁权重 0.10，单步裁剪 ±0.01", gui.lbl_training_reward_note.config.call_args.kwargs["text"])
        self.assertIn("普通种子槽不计资产", gui.lbl_training_reward_note.config.call_args.kwargs["text"])
        self.assertIn("整局净 shaping 平滑趋近 ±0.40", gui.lbl_training_reward_note.config.call_args.kwargs["text"])
        self.assertEqual(gui._training_reward_badge_titles["threat"].config.call_args.kwargs["text"], "NET SHAPING")
        gui._parse_common_training_config()
        self.assertEqual(gui._training_reward_options["reward_scheme"], "transition_dense")
        self.assertEqual(gui._training_reward_options["shaping_limit_mode"], "smooth_total")


if __name__ == "__main__":
    unittest.main()
