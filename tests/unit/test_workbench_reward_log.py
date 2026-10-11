import re
import unittest

import numpy as np

from pvz_gym.observation import empty_observation
from pvz_gym.types import Observation
from train.dqn.config import DoubleDQNConfig
from train.dqn.credit import EpisodeEventCredit
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import effective_reward_settings
from workbench.gui.training import _format_dqn_mower_log


class WorkbenchRewardLogTest(unittest.TestCase):
    def _log_transition(self, *, scheme='staged', sun=0, ticks=1000, mowers=1,
                        terminated=False, mower_penalty=0.1):
        def raw_board(sun):
            return Observation(global_state={
                'sun': sun, 'game_mode': 0, 'total_waves': 1,
                'spawn_plan_available': True, 'zombie_spawn_plan': [0] + [-1] * 49,
            })

        config = DoubleDQNConfig(reward_scheme=scheme, mower_penalty=mower_penalty)
        replay = ReplayBuffer(2, 5)
        tracker = EpisodeEventCredit(replay, config, {'raw_observation': raw_board(0)})
        row_info = {'raw_observation': raw_board(sun), 'ticks_advanced': ticks,
                    'result': 'WON' if terminated else 'PLAYING',
                    'events': {'mowers_used': mowers}}
        observation = tracker.observe(empty_observation())
        total = tracker.add(replay, observation, 0, 100 if terminated else 0,
                            empty_observation(), terminated, False, np.ones(5, dtype=np.bool_), row_info)
        data = {'mowers_used': mowers, 'training_reward': total, 'reward_scheme': scheme,
                'mower_reward': effective_reward_settings(config)['coefficients']['mower_used'] * mowers,
                'reward_breakdown': row_info.get('reward_breakdown')}
        line = _format_dqn_mower_log(data)
        fields = {name: float(value) for name, value in re.findall(r'(\w+)=([+-]\d+\.\d+)', line)}
        components = [value for name, value in fields.items() if name != 'step_reward']
        self.assertAlmostEqual(sum(components), fields['step_reward'], places=3)
        self.assertAlmostEqual(fields['step_reward'], float(replay._rewards[0]), places=3)
        return line

    def test_stage_bonus_and_mower_penalty_are_shown_separately(self):
        line = self._log_transition(sun=500)
        self.assertIn('scheme=staged', line)
        self.assertIn('mower_penalty=-0.200', line)
        self.assertIn('stage_reward=+0.100', line)
        self.assertIn('terminal_reward=+0.000', line)
        self.assertIn('step_reward=-0.100', line)

    def test_transition_dense_log_uses_actual_first_mower_penalty(self):
        line = self._log_transition(scheme='transition_dense', sun=1000, ticks=150,
                                    mowers=2, terminated=True)
        self.assertIn('mower_penalty=-0.600', line)
        self.assertIn('first_loss=yes', line)
        self.assertIn('shaping=+0.133', line)
        self.assertIn('terminal=+1.000', line)
        self.assertIn('step_reward=+0.533', line)

    def test_unsettled_stage_has_no_positive_reward_in_the_log(self):
        line = self._log_transition(sun=500, ticks=500)
        self.assertIn('stage_reward=+0.000', line)
        self.assertIn('step_reward=-0.200', line)

    def test_multiple_mowers_and_terminal_reward_are_not_conflated(self):
        line = self._log_transition(sun=500, ticks=200, mowers=2, terminated=True)
        self.assertIn('mower used x2', line)
        self.assertIn('mower_penalty=-0.400', line)
        self.assertIn('terminal_reward=+1.000', line)
        self.assertIn('step_reward=+0.700', line)

    def test_ordinary_rewards_use_the_effective_scheme_and_custom_penalty(self):
        for scheme, penalty in (('sparse', 0.35), ('balanced', 0.2), ('dense_early', 0.2)):
            with self.subTest(scheme=scheme):
                line = self._log_transition(scheme=scheme, mower_penalty=0.35, terminated=True)
                self.assertIn(f'scheme={scheme}', line)
                self.assertIn(f'mower_penalty={-penalty:+.3f}', line)
                self.assertIn('other_reward=+1.000', line)
                self.assertNotIn('stage_reward=', line)

    def test_old_event_without_breakdown_does_not_invent_a_penalty(self):
        line = _format_dqn_mower_log({'mowers_used': 1, 'training_reward': -0.1})
        self.assertIn('step_reward=-0.100', line)
        self.assertNotIn('mower_penalty=', line)


if __name__ == '__main__':
    unittest.main()
