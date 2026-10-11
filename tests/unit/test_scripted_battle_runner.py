import unittest

from agents.scripted.battle_runner import run_scripted_battle
from pvz_gym.types import ActionResult, GameStatus, Observation, ResetFacts, StepEvents, StepFacts


class BattleBackend:
    def __init__(self, terminal=None):
        self.clock = 100
        self.terminal = terminal
        self.observation = Observation(global_state={"game_clock": self.clock})
        self.actions = []
        self.headless = False

    def wait_for_battle(self, **kwargs):
        return ResetFacts(self.observation, self.clock, GameStatus.PLAYING)

    def get_action_mask(self):
        return (True,) * 757

    def step(self, action, ticks):
        self.actions.append(action)
        self.clock += ticks
        return StepFacts(self.observation, StepEvents(), ActionResult(True),
                         self.get_action_mask(), self.terminal or GameStatus.PLAYING,
                         self.clock, automation_collected=1)

    step_headless = step
    step_continuous = step

    def begin_headless_mode(self):
        self.headless = True
        return True

    def end_headless_mode(self):
        self.headless = False


class ScriptedBattleTests(unittest.TestCase):
    def test_exact_tick_budget_and_action_decode(self):
        backend = BattleBackend()
        result = run_scripted_battle(backend, lambda mask, obs: (1, 0, 0, 0, 0, "test"),
                                     max_ticks=65, ticks_per_step=30)
        self.assertEqual((result.outcome, result.ticks, result.steps), ("TICK_LIMIT", 65, 3))
        self.assertEqual(result.plants, 3)
        self.assertEqual(backend.actions[0].type, "plant")

    def test_terminal_and_unlimited(self):
        for terminal in (GameStatus.WON, GameStatus.LOST):
            backend = BattleBackend(terminal)
            result = run_scripted_battle(backend, lambda mask, obs: None,
                                         max_ticks=0, headless=True)
            self.assertEqual(result.outcome, terminal.name)
            self.assertEqual(result.auto_collected, 1)
            self.assertFalse(backend.headless)

    def test_illegal_action_cleanup(self):
        backend = BattleBackend()
        with self.assertRaisesRegex(RuntimeError, "illegal action"):
            run_scripted_battle(backend, lambda mask, obs: (900,), headless=True)
        self.assertFalse(backend.headless)
        self.assertEqual(backend.actions, [])

    def test_stalled_clock_fails(self):
        backend = BattleBackend()
        backend.step_continuous = lambda action, ticks: StepFacts(
            backend.observation, StepEvents(), ActionResult(True),
            backend.get_action_mask(), GameStatus.PLAYING, backend.clock)
        with self.assertRaisesRegex(RuntimeError, "game clock"):
            run_scripted_battle(backend, lambda mask, obs: None)


if __name__ == "__main__":
    unittest.main()
