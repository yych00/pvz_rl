"""Window shutdown must let the trainer release owned game processes."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from workbench.gui.main_window import request_workbench_close


class WorkbenchShutdownTest(unittest.TestCase):
    def test_running_training_stops_before_window_destruction(self):
        app = SimpleNamespace(root=Mock(), _training_running=True, _stop_training=Mock())
        with patch("workbench.gui.q_heatmap_overlay.get_q_heatmap_overlay") as overlay:
            request_workbench_close(app)
            app._stop_training.assert_called_once()
            app.root.destroy.assert_not_called()
            delay, check = app.root.after.call_args.args
            self.assertEqual(delay, 100)
            request_workbench_close(app)
            app._stop_training.assert_called_once()
            app._training_running = False
            check()
            overlay.return_value.stop.assert_called_once()
            app.root.destroy.assert_called_once()

    def test_idle_window_closes_even_when_overlay_cleanup_fails(self):
        app = SimpleNamespace(root=Mock(), _training_running=False, _stop_training=Mock())
        with patch("workbench.gui.q_heatmap_overlay.get_q_heatmap_overlay", side_effect=RuntimeError("closed")):
            request_workbench_close(app)
            app._stop_training.assert_not_called()
            app.root.after.assert_not_called()
            app.root.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
