import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import workbench.main  # noqa: F401 - install the complete Workbench wrappers
from workbench import PvzGymUnitTestGUI
from workbench.gui.checkpoint_load_mode import _RESUME_LABEL
from workbench.gui.training import _seed_history_file
from train.dqn.config import DoubleDQNConfig


class WorkbenchTrendHistoryTest(unittest.TestCase):
    def test_history_is_seeded_once_and_existing_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run' / 'episodes.jsonl'
            _seed_history_file(path, [{'episode': 1}])
            with path.open('a', encoding='utf-8') as handle:
                handle.write('{"episode": 2}\n')
            original = path.read_bytes()
            _seed_history_file(path, [{'episode': 1}])
            self.assertEqual(path.read_bytes(), original)
            empty = path.with_name('empty.jsonl')
            empty.touch()
            _seed_history_file(empty, [{'episode': 1}])
            self.assertEqual(empty.read_bytes(), b'')

    def test_resume_worker_preserves_log_beyond_cached_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = [run_dir / name for name in ('episodes.jsonl', 'evaluations.jsonl')]
            for path in paths:
                path.write_text('{"episode": 1}\n{"episode": 2}\n', encoding='utf-8')
            original = [path.read_bytes() for path in paths]
            app = PvzGymUnitTestGUI.__new__(PvzGymUnitTestGUI)
            app.root = Mock()
            app.var_training_checkpoint_load_mode = Mock(get=lambda: _RESUME_LABEL)
            app._training_resume_run_dir = run_dir
            app._training_checkpoint_records = [{'episode': 1}]
            app._training_checkpoint_evaluations = [{'episode': 1}]
            app._training_include_source_history = True
            app._training_controller = SimpleNamespace(
                run_dir=Path(directory) / 'unused', config=DoubleDQNConfig(), run=Mock(return_value='done'),
            )
            app._dqn_training_worker()
            app._training_controller.run.assert_called_once()
            self.assertEqual([path.read_bytes() for path in paths], original)
            self.assertEqual(app.root.after.call_args.args[-1], 'done')


if __name__ == '__main__':
    unittest.main()
