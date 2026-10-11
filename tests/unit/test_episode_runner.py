import unittest

from eval.episode_runner import EpisodeRunner
from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.seed_types import (
    CONFIRM_ID, DESELECT_BASE, SEED_ACTION_COUNT,
    SEED_TYPE_CAPACITY, SeedOptionState, SeedSelectionResult, SeedSelectionState,
)
from pvz_gym.session import PvZSession
from pvz_gym.types import (
    Action,
    ActionResult,
    GameStatus,
    Observation,
    ResetFacts,
    RngSeedSupport,
    StepEvents,
    StepFacts,
)


class SameEpisodeBackend:
    def __init__(self, terminal=GameStatus.WON):
        self.terminal = terminal
        self.reset_calls = 0
        self.wait_calls = 0
        self.step_calls = 0
        self.selected: set[int] = set()
        self.confirmed = False
        self.closed = False

    def _observation(self, status=GameStatus.PLAYING):
        return Observation(
            global_state={
                "sun": 100, "current_wave": 1, "total_waves": 1,
                "wave_countdown": 0, "game_clock": self.step_calls * 30,
                "level_id": 4, "status": int(status),
            },
            cards=({"slot": 0, "plant_type": 0, "cost": 100,
                    "cooldown_duration": 100, "cooldown_remaining": 0, "ready": True},),
            mowers=({"row": 0, "state": "READY"},),
        )

    def reset(self, config):
        self.reset_calls += 1
        self.selected.clear()
        self.confirmed = False
        self.step_calls = 0
        if not config.get("stop_at_seed_selection"):
            raise AssertionError("EpisodeRunner must prepare at SeedChooser")
        return ResetFacts(
            None, 0, GameStatus.CHOOSING_SEEDS,
            RngSeedSupport.FULL, "seed" in config,
        )

    def wait_for_battle(self, *, timeout_ms=10000, deterministic_pause=True):
        self.wait_calls += 1
        if not self.confirmed:
            raise AssertionError("battle handoff happened before deck confirmation")
        return ResetFacts(
            self._observation(), 0, GameStatus.PLAYING,
            RngSeedSupport.FULL, False,
        )

    def step(self, action: Action, ticks: int):
        self.step_calls += 1
        status = self.terminal
        return StepFacts(
            observation=self._observation(status),
            events=StepEvents(
                won=status == GameStatus.WON,
                lost=status == GameStatus.LOST,
            ),
            action=ActionResult(True, None, "fixture"),
            action_mask=(False,) * ACTION_SPACE_SIZE,
            status=status,
            game_clock=self.step_calls * ticks,
        )

    def get_action_mask(self):
        return (True,) + (False,) * (ACTION_SPACE_SIZE - 1)

    def get_seed_type_capacity(self):
        return SEED_TYPE_CAPACITY

    def get_seed_selection(self):
        return SeedSelectionState(
            present=not self.confirmed,
            required_count=1,
            selected_count=len(self.selected),
            can_confirm=len(self.selected) == 1,
            selected_seeds=tuple(sorted(self.selected)),
            options=tuple(
                SeedOptionState(
                    plant_type=i,
                    available=True,
                    unlocked=True,
                    selected=i in self.selected,
                    disabled=False,
                )
                for i in range(2)
            ),
        )

    def get_seed_selection_mask(self):
        mask = [False] * SEED_ACTION_COUNT
        for i in range(2):
            mask[i] = len(self.selected) < 1 and i not in self.selected
            mask[DESELECT_BASE + i] = i in self.selected
        mask[CONFIRM_ID] = len(self.selected) == 1
        return mask

    def select_seed(self, plant_type):
        if len(self.selected) >= 1 or plant_type in self.selected:
            return SeedSelectionResult(False, 1, "illegal")
        self.selected.add(int(plant_type))
        return SeedSelectionResult(True, 0, "selected")

    def deselect_seed(self, plant_type):
        if plant_type not in self.selected:
            return SeedSelectionResult(False, 1, "illegal")
        self.selected.remove(int(plant_type))
        return SeedSelectionResult(True, 0, "deselected")

    def confirm_seeds(self):
        if len(self.selected) != 1:
            return SeedSelectionResult(False, 1, "incomplete")
        self.confirmed = True
        return SeedSelectionResult(True, 0, "confirmed")

    def close(self):
        self.closed = True


class EpisodeRunnerSessionTest(unittest.TestCase):
    def test_seed_selection_and_battle_are_one_native_episode(self):
        backend = SameEpisodeBackend(GameStatus.WON)
        session = PvZSession(backend)
        runner = EpisodeRunner(session=session)

        result = runner.run_episode(level_id=4, seed=123, max_battle_steps=2)

        self.assertEqual(backend.reset_calls, 1)
        self.assertEqual(backend.wait_calls, 1)
        self.assertEqual(backend.step_calls, 1)
        self.assertEqual(result.selected_deck, [0])
        self.assertTrue(result.won)
        self.assertFalse(result.lost)
        self.assertEqual(result.deck_return, 1.0)
        self.assertTrue(session.reset_facts.rng_seed_applied)

    def test_loss_is_reported_from_battle_step_result(self):
        backend = SameEpisodeBackend(GameStatus.LOST)
        session = PvZSession(backend)
        runner = EpisodeRunner(session=session)

        result = runner.run_episode(level_id=4, max_battle_steps=2)

        self.assertEqual(backend.reset_calls, 1)
        self.assertEqual(backend.wait_calls, 1)
        self.assertFalse(result.won)
        self.assertTrue(result.lost)
        self.assertEqual(result.deck_return, -1.0)

    def test_custom_views_cannot_silently_use_different_sessions(self):
        from pvz_gym.envs.battle import BattleEnv
        from pvz_gym.envs.seed_selection import SeedSelectionEnv

        session_a = PvZSession(SameEpisodeBackend())
        session_b = PvZSession(SameEpisodeBackend())
        seed_env = SeedSelectionEnv(session=session_a)
        battle_env = BattleEnv(session=session_b)
        with self.assertRaisesRegex(ValueError, "same PvZSession"):
            EpisodeRunner(seed_env=seed_env, battle_env=battle_env)


if __name__ == "__main__":
    unittest.main()
