from dataclasses import replace
import unittest

from train.dqn.config import DoubleDQNConfig
from train.dqn.training_cadence import TrainingCadence


class TrainingCadenceTest(unittest.TestCase):
    def test_watermarks_cooldown_and_bounds(self):
        config = DoubleDQNConfig(train_every=6)
        cadence = TrainingCadence(config)
        self.assertFalse(cadence.update(config, 31, 100))
        self.assertTrue(cadence.update(config, 32, 16))
        self.assertEqual(cadence.every, 12)
        self.assertFalse(cadence.update(config, 40, 16))
        self.assertFalse(cadence.update(config, 64, 8))
        self.assertTrue(cadence.update(config, 65, 4))
        self.assertEqual(cadence.every, 6)
        for step in range(1000, 30000, 1000):
            cadence.update(config, step, 100)
        self.assertEqual(cadence.every, 384)
        for step in range(30000, 60000, 1000):
            cadence.update(config, step, 0)
        self.assertEqual(cadence.every, 1)

    def test_disabled_mode_and_changed_config(self):
        config = DoubleDQNConfig(train_every=6)
        cadence = TrainingCadence(config)
        cadence.update(config, 32, 100)
        disabled = replace(config, auto_train_every=False)
        cadence.update(disabled, 64, 100)
        self.assertEqual(cadence.every, 6)
        cadence.update(disabled, 1000, 0)
        self.assertEqual(cadence.every, 6)
        cadence.update(replace(config, train_every=10), 1010, 100)
        self.assertEqual(cadence.every, 10)

    def test_resume_preserves_cadence_and_cooldown(self):
        config = DoubleDQNConfig(train_every=6)
        cadence = TrainingCadence(config)
        cadence.update(config, 32, 100)
        restored = TrainingCadence(config, 40, cadence.snapshot())
        self.assertEqual(restored.snapshot(), cadence.snapshot())
        self.assertFalse(restored.update(config, 63, 100))
        self.assertTrue(restored.update(config, 64, 100))
        changed = TrainingCadence(replace(config, train_every=10), 40, cadence.snapshot())
        self.assertEqual(changed.every, 10)
        invalid = TrainingCadence(config, 40, ['invalid'])
        self.assertEqual(invalid.every, config.train_every)


if __name__ == '__main__':
    unittest.main()
