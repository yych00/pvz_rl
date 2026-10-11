"""Explicit evaluation tasks independent of training curriculum probabilities."""
import numpy as np


def evaluation_mode(controller):
    if getattr(controller, '_legacy_evaluation_protocol', False):
        return 'random'
    return controller.config.evaluation_mode


def evaluation_plan(controller, random_count):
    config = controller.config
    pool = tuple(dict.fromkeys(config.env.level_pool or
                              ((config.env.level_id, config.env.playthrough),)))
    mode = evaluation_mode(controller)
    if mode in ('off', 'none'):
        return []
    if mode == 'per_level':
        pairs = [pair for _ in range(config.evaluation_per_level) for pair in pool]
    else:
        count = (random_count if getattr(controller, '_legacy_evaluation_protocol', False)
                 or random_count != 20 else config.evaluation_random_episodes)
        rng = np.random.default_rng()
        pairs = [pool[int(rng.integers(len(pool)))] for _ in range(count)]
    # Preserve the legacy seed API for diagnostic callers; normal UI evaluation
    # uses fresh seeds and never needs a fixed-seed experiment.
    seed_start = controller.eval_seed_start
    if not getattr(controller, '_legacy_evaluation_protocol', False):
        seed_start = int(np.random.default_rng().integers(0, 2**30))
    return [(seed_start + i, pair) for i, pair in enumerate(pairs)]


def protocol_fields(controller, requested):
    return dict(evaluation_mode=evaluation_mode(controller),
                evaluation_per_level=controller.config.evaluation_per_level
                if evaluation_mode(controller) == 'per_level' else None,
                evaluation_pool=[list(pair) for pair in dict.fromkeys(
                    controller.config.env.level_pool or
                    ((controller.config.env.level_id, controller.config.env.playthrough),))],
                requested_episodes=requested)


def record_level_result(rows, pair, result):
    key = tuple(pair)
    row = rows.setdefault(key, dict(level_id=key[0], playthrough=key[1],
                                    episodes=0, wins=0, losses=0, unknown=0))
    row['episodes'] += 1
    row['wins'] += int(result == 'WON')
    row['losses'] += int(result == 'LOST')
    row['unknown'] += int(result not in ('WON', 'LOST'))
    row['win_rate'] = row['wins'] / row['episodes']


def protocol_matches(controller, record, random_count):
    mode = evaluation_mode(controller)
    if record.get('evaluation_mode', 'random') != mode:
        return False
    if mode == 'per_level':
        expected = len(tuple(dict.fromkeys(controller.config.env.level_pool or
                       ((controller.config.env.level_id, controller.config.env.playthrough),)))) * controller.config.evaluation_per_level
    else:
        expected = (random_count if getattr(controller, '_legacy_evaluation_protocol', False)
                    or random_count != 20 else controller.config.evaluation_random_episodes)
    pool = protocol_fields(controller, expected)['evaluation_pool']
    return (record.get('requested_episodes') == expected
            and record.get('max_episode_steps', -1) == controller.eval_max_episode_steps
            and record.get('max_episode_ticks') == controller.config.env.max_episode_ticks
            and (record.get('evaluation_pool') == pool or
                 (getattr(controller, '_legacy_evaluation_protocol', False) and 'evaluation_pool' not in record))
            and (not getattr(controller, '_legacy_evaluation_protocol', False) or
                 record.get('seed_start') == controller.eval_seed_start))
