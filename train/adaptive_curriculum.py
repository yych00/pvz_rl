"""Bounded, smoothed training-level probabilities from recent valid outcomes."""
from collections import deque
import math


class AdaptiveCurriculum:
    def __init__(self, config):
        self.pool = tuple(dict.fromkeys(config.env.level_pool or
                                        ((config.env.level_id, config.env.playthrough),)))
        self.mode = getattr(config, 'level_sampling_mode', 'squared_failure')
        self.enabled = config.adaptive_level_sampling
        self.window = config.level_sampling_window
        self.interval = config.level_sampling_update_every
        self.uniform_mix = config.level_sampling_uniform_mix
        self.max_weight = config.level_sampling_max_weight
        self.history = {pair: deque(maxlen=self.window) for pair in self.pool}
        self.counts = {pair: [0, 0] for pair in self.pool}  # episodes, decision steps
        self.valid_episodes = 0
        self.probabilities = [1 / len(self.pool)] * len(self.pool)

    def observe(self, record):
        pair = (record.get('level_id'), record.get('playthrough'))
        if (pair not in self.history or record.get('result') not in ('WON', 'LOST')
                or record.get('truncated') or record.get('runtime_error')):
            return False
        self.history[pair].append(record['result'] == 'WON')
        self.counts[pair][0] += 1
        self.counts[pair][1] += max(0, int(record.get('episode_steps', 0)))
        self.valid_episodes += 1
        if self.valid_episodes % self.interval:
            return False
        if self.enabled:
            target = self._target_probabilities()
            self.probabilities = [0.5 * old + 0.5 * new
                                  for old, new in zip(self.probabilities, target)]
        return True

    def _target_probabilities(self):
        if self.mode == "uniform":
            return [1.0 / len(self.pool)] * len(self.pool)
        weights = []
        for pair in self.pool:
            recent = self.history[pair]
            if self.mode == "bounded_difficulty":
                rate = (sum(recent) + 5) / (len(recent) + 10)
                weight = 1.0 + (self.max_weight - 1.0) * (1.0 - rate) ** 2
            else:
                rate = sum(recent) / len(recent) if recent else 0.5
                weight = 0.05 + (1.0 - rate) ** 2
            weights.append(weight)
        scale = max(weights)
        normalized = [w / scale for w in weights]
        total = sum(normalized)
        return [self.uniform_mix / len(weights) +
                (1.0 - self.uniform_mix) * w / total for w in normalized]

    def sample(self, rng):
        if not self.enabled:
            return self.pool[int(rng.integers(len(self.pool)))]
        return self.pool[int(rng.choice(len(self.pool), p=self.probabilities))]

    def rows(self):
        episodes = sum(x[0] for x in self.counts.values())
        steps = sum(x[1] for x in self.counts.values())
        return [dict(level_id=pair[0], playthrough=pair[1],
                     recent_episodes=len(self.history[pair]),
                     recent_win_rate=sum(self.history[pair]) / len(self.history[pair])
                     if self.history[pair] else None,
                     smoothed_win_rate=(sum(self.history[pair]) + 5) /
                     (len(self.history[pair]) + 10), probability=probability,
                     episode_share=self.counts[pair][0] / episodes if episodes else 0,
                     step_share=self.counts[pair][1] / steps if steps else 0)
                for pair, probability in zip(self.pool, self.probabilities)]

    def snapshot(self):
        return dict(version=4, mode=self.mode, pool=[list(pair) for pair in self.pool],
                    history=[list(self.history[pair]) for pair in self.pool],
                    counts=[list(self.counts[pair]) for pair in self.pool],
                    valid_episodes=self.valid_episodes,
                    probabilities=list(self.probabilities), enabled=self.enabled,
                    settings=[self.window, self.interval, self.uniform_mix, self.max_weight])

    def restore(self, state):
        if not isinstance(state, dict) or state.get('version') not in (1, 2, 3, 4) or state.get('pool') != [list(p) for p in self.pool]:
            return False
        history, counts = state.get('history', []), state.get('counts', [])
        probabilities = state.get('probabilities', [])
        if (not all(isinstance(x, list) for x in (history, counts, probabilities))
                or not (len(history) == len(counts) == len(probabilities) == len(self.pool))):
            return False
        if (any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-8)):
            return False
        if (any(not isinstance(recent, list) or any(type(x) is not bool for x in recent) for recent in history)
                or any(not isinstance(count, list) or len(count) != 2 or
                       any(type(x) is not int or x < 0 for x in count) for count in counts)):
            return False
        valid_episodes = state.get('valid_episodes')
        if type(valid_episodes) is not int or valid_episodes != sum(x[0] for x in counts):
            return False
        for pair, recent, count in zip(self.pool, history, counts):
            self.history[pair].clear()
            self.history[pair].extend(recent)
            self.counts[pair] = list(count)
        self.valid_episodes = valid_episodes
        if (self.enabled and state.get('version') == 4 and state.get('mode') == self.mode and state.get('enabled') and
                state.get('settings') == [self.window, self.interval, self.uniform_mix, self.max_weight]):
            self.probabilities = list(probabilities)
        elif self.enabled:
            self.probabilities = self._target_probabilities()
        else:
            self.probabilities = [1 / len(self.pool)] * len(self.pool)
        return True
