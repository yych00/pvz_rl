"""Adventure deck defaults and a mask-checked, selection-only executor.

Priorities are templates, not claims of optimal play. The live chooser decides
which plants are unlocked and how many slots (including Dave's picks) exist.
"""

from __future__ import annotations

import time
import random

from .seed_types import CONFIRM_ID, DESELECT_BASE, SeedSelectionState
from .adventure_curriculum import deck_profile


PLANT_NAMES = (
    "豌豆射手", "向日葵", "樱桃炸弹", "坚果", "土豆雷", "寒冰射手",
    "大嘴花", "双发射手", "小喷菇", "阳光菇", "大喷菇", "墓碑吞噬者",
    "魅惑菇", "胆小菇", "寒冰菇", "毁灭菇", "睡莲", "窝瓜",
    "三线射手", "缠绕海草", "火爆辣椒", "地刺", "火炬树桩", "高坚果",
    "海蘑菇", "路灯花", "仙人掌", "三叶草", "裂荚射手", "杨桃",
    "南瓜头", "磁力菇", "卷心菜投手", "花盆", "玉米投手", "咖啡豆",
    "大蒜", "叶子保护伞", "金盏花", "西瓜投手",
)

# Top six work with the initial six slots; later slots extend the same deck.
# First-run availability is filtered by the original game, never force-unlocked.
FIRST_SCENE_PRIORITIES = {
    1: (1, 0, 4, 3, 7, 5, 2, 17, 20, 21),
    2: (9, 8, 10, 11, 3, 2, 14, 15, 13, 12),
    3: (16, 1, 7, 3, 17, 19, 5, 2, 23, 21),
    4: (16, 9, 8, 10, 27, 26, 3, 17, 25, 24),
    5: (33, 1, 32, 34, 17, 37, 3, 39, 2, 30),
}
SECOND_SCENE_PRIORITIES = {
    1: (1, 7, 4, 3, 5, 17, 2, 20, 21, 30),
    2: (9, 8, 10, 11, 3, 14, 15, 17, 2, 31),
    3: (16, 1, 7, 23, 17, 19, 5, 2, 21, 30),
    4: (16, 9, 8, 10, 27, 26, 23, 17, 30, 24),
    5: (33, 1, 32, 34, 17, 37, 39, 3, 2, 30),
}

# Keep the existing tutorial decks for levels 1-1..1-4.
FIRST_EARLY_PRIORITIES = {
    1: (0,), 2: (1, 0), 3: (1, 0, 2), 4: (1, 0, 2, 3),
}


def default_deck_priority(level_id: int, playthrough: int) -> tuple[int, ...]:
    if isinstance(level_id, bool) or level_id not in range(1, 51):
        raise ValueError("level_id must be 1..50")
    if isinstance(playthrough, bool) or playthrough not in (1, 2):
        raise ValueError("playthrough must be 1 or 2")
    return deck_profile(level_id, playthrough).primary


DEFAULT_DECKS = {
    playthrough: {level: default_deck_priority(level, playthrough)
                  for level in range(1, 51)}
    for playthrough in (1, 2)
}


def counter_rules(level_id: int, zombie_types):
    """Ordered threats and alternative plant packages (IDs from perception).

    Mushroom packages include Coffee Bean on day/pool/roof. Alternatives are
    evaluated as whole packages, so a sleeping Magnet-shroom is never a counter.
    """
    night = (level_id - 1) // 10 in (1, 3)
    roof = level_id > 40
    magnet = (31,) if night else (31, 35)
    fume = (10,) if night else (10, 35)
    rules = (
        ((16,), "气球", ((27,), (26,))),
        ((17,), "矿工", ((magnet, (28,), (29,)) if night else ((28,), magnet, (29,)))),
        ((18,), "跳跳", ((23,), magnet)),
        ((20, 22), "蹦极/投石车", ((37,),)),
        ((12,), "冰车", (((20,), (17,)) if roof else ((21,), (20,), (17,)))),
        ((14, 3), "海豚/撑杆", ((23,),)),
        ((6,), "铁门", (fume, (17,), (2,))),
        ((21,), "扶梯", (magnet, (2,), (20,))),
        ((23, 32), "巨人", ((17, 20), (17, 2), (2, 20), (17,), (2,), (20,))),
        ((4, 7, 15), "铁桶/橄榄球/小丑", (magnet, (17,), (2,), (20,))),
        ((11,), "潜水", ((19,),)),
    )
    roster = set(zombie_types or ())
    return tuple((label, alternatives) for ids, label, alternatives in rules
                 if roster.intersection(ids))


def deck_counter_gaps(state, level_id, deck):
    if state.zombie_types_in_level is None:
        return ("僵尸预览不可用：使用场景默认卡组，需更新原生 DLL",)
    selected = set(deck)
    return tuple(label for label, alternatives in counter_rules(level_id, state.zombie_types_in_level)
                 if not any(set(package).issubset(selected) for package in alternatives))


def _shuffle_deck(state, target, rng):
    options = {opt.plant_type: opt for opt in state.options}
    movable = [plant for plant in target if not options[plant].disabled]
    (rng if rng is not None else random.SystemRandom()).shuffle(movable)
    iterator = iter(movable)
    return tuple(plant if options[plant].disabled else next(iterator) for plant in target)


def plan_default_deck(state: SeedSelectionState, level_id: int,
                      playthrough: int, *, priority_override=None,
                      selection_mode="adaptive", rng=None) -> tuple[int, ...]:
    """Keep locked picks, then fill with available base plants in priority order."""
    if selection_mode in ("fixed", "fixed_shuffled") and priority_override is None:
        target = plan_fixed_deck(state, level_id, playthrough)
        if selection_mode == "fixed_shuffled":
            target = _shuffle_deck(state, target, rng)
        return target
    if selection_mode not in ("adaptive", "adaptive_shuffled", "custom", "fixed", "fixed_shuffled", "native"):
        raise ValueError("invalid seed selection mode")
    default_deck_priority(level_id, playthrough)  # validate arguments
    scene = (level_id - 1) // 10 + 1
    priority = ((FIRST_EARLY_PRIORITIES[level_id] if level_id in FIRST_EARLY_PRIORITIES
                 else FIRST_SCENE_PRIORITIES[scene]) if playthrough == 1
                else SECOND_SCENE_PRIORITIES[scene])
    if priority_override is not None:
        priority = tuple(priority_override) + priority
    if not state.present or state.required_count <= 0:
        raise RuntimeError("no interactive seed chooser")
    options = {opt.plant_type: opt for opt in state.options}
    chosen = [plant for plant in state.selected_seeds
              if plant in options and options[plant].disabled]

    def usable(plant):
        opt = options.get(plant)
        return (plant in chosen or (opt is not None and opt.unlocked
                and not opt.disabled and (opt.available or opt.selected)))

    def add_package(package):
        missing = [plant for plant in package if plant not in chosen]
        if len(chosen) + len(missing) > state.required_count or not all(map(usable, package)):
            return False
        chosen.extend(missing)
        return True

    if state.zombie_types_in_level is not None and priority_override is None:
        scene = (level_id - 1) // 10
        # Reserve terrain, economy and one usable attacker before tactical cards.
        if scene in (2, 3):
            add_package((16,))
        if scene == 4:
            add_package((33,))
        for economy in ((9, 1) if scene in (1, 3) else (1,)):
            if add_package((economy,)):
                break
        attackers = ((32, 34, 39) if scene == 4 else
                     (8, 10, 0, 7) if scene in (1, 3) else (0, 7, 5))
        if not set(attackers).intersection(chosen):
            for attacker in attackers:
                if add_package((attacker,)):
                    break
        for label, alternatives in counter_rules(level_id, state.zombie_types_in_level):
            if any(set(package).issubset(chosen) for package in alternatives):
                continue
            for package in alternatives:
                if add_package(package):
                    break
    # No upgrade-only or Imitater fallback: those require extra dependencies.
    fallback = tuple(i for i in range(40) if i != 38)
    for plant in (*priority, *fallback, 38):
        if len(chosen) >= state.required_count:
            break
        opt = options.get(plant)
        if (plant not in chosen and opt is not None and opt.unlocked
                and not opt.disabled and (opt.available or opt.selected)):
            if (state.zombie_types_in_level is not None and
                    plant in (*range(8, 16), 31) and (level_id - 1) // 10 not in (1, 3)):
                if not add_package((plant, 35)):
                    continue
                continue
            if (state.zombie_types_in_level is not None and plant in (26, 27)
                    and ({26, 27} & set(chosen))):
                continue
            chosen.append(plant)
    # Early first-run levels can have ten slots before Coffee Bean unlocks,
    # but fewer than ten awake plants. Fill remaining slots with unlocked
    # sleeping mushrooms only after exhausting usable daytime cards.
    if len(chosen) < state.required_count and (level_id - 1) // 10 not in (1, 3):
        for plant in (*range(8, 16), 31):
            if len(chosen) >= state.required_count:
                break
            if plant not in chosen and usable(plant):
                chosen.append(plant)
    if len(chosen) != state.required_count:
        raise RuntimeError(f"cannot fill default deck: {len(chosen)}/{state.required_count}")
    target = tuple(chosen)
    return _shuffle_deck(state, target, rng) if selection_mode == "adaptive_shuffled" else target


def plan_fixed_deck(state, level_id, playthrough):
    """Deterministic teaching deck, constrained by actual chooser availability.

    Preview zombies are diagnostic only: they never reorder the fixed profile.
    Dave's forced picks occupy slots first. Candidates replace unavailable
    primary cards; first-run candidates cannot borrow later-awarded plants.
    """
    profile = deck_profile(level_id, playthrough)
    if not state.present or state.required_count <= 0:
        raise RuntimeError("no interactive seed chooser")
    options = {opt.plant_type: opt for opt in state.options}
    chosen = [p for p in state.selected_seeds if p in options and options[p].disabled]
    for plant in (*profile.primary, *profile.candidates):
        if len(chosen) >= state.required_count:
            break
        opt = options.get(plant)
        if (plant not in chosen and opt is not None and opt.unlocked and not opt.disabled
                and (opt.available or opt.selected)):
            # Coffee precedes Magnet in daytime profiles. If unavailable or
            # displaced by Dave, don't count a sleeping Magnet as a defense.
            if plant == 31 and (level_id - 1) // 10 not in (1, 3) and 35 not in chosen:
                continue
            chosen.append(plant)
    if len(chosen) != state.required_count:
        raise RuntimeError(f"cannot fill fixed teaching deck: {len(chosen)}/{state.required_count}; "
                           "本关已解锁候选不足，不使用后续关卡植物")
    return tuple(chosen)


def select_default_deck(backend, level_id: int, playthrough: int, *,
                        confirm: bool = False, timeout_seconds: float = 10.0,
                        report=None, priority_override=None, selection_mode="adaptive", rng=None
                        ) -> SeedSelectionState:
    """Select a deck on the current chooser; enter battle only with confirm=True."""
    deadline = time.monotonic() + timeout_seconds

    def settled():
        while time.monotonic() < deadline:
            state = backend.get_seed_selection()
            if not state.present:
                raise RuntimeError("seed chooser is no longer interactive")
            if not state.seeds_in_flight:
                return state
            time.sleep(0.02)
        raise TimeoutError("default seed selection timed out")

    state = settled()
    target = plan_default_deck(state, level_id, playthrough, priority_override=priority_override,
                               selection_mode=selection_mode, rng=rng)
    if report is not None:
        report(f"本关僵尸类型：{state.zombie_types_in_level}")
        for gap in deck_counter_gaps(state, level_id, target):
            report(f"选卡应对缺口（解锁/槽位限制）：{gap}")
        if (level_id - 1) // 10 not in (1, 3) and 35 not in target:
            sleeping = [PLANT_NAMES[p] for p in target if p in (*range(8, 16), 31)]
            if sleeping:
                report("选卡补位提示：" + "、".join(sleeping) + "白天会睡觉；用于补满卡槽，不计作有效针对卡")
    locked = {opt.plant_type for opt in state.options if opt.disabled and opt.selected}
    ordered = selection_mode in ("fixed", "fixed_shuffled", "adaptive_shuffled")
    for plant in state.selected_seeds:
        if plant in locked or (plant in target and not ordered):
            continue
        settled()
        mask = backend.get_seed_selection_mask()
        if not mask[DESELECT_BASE + plant]:
            raise RuntimeError(f"Deselect({plant}) is no longer legal")
        result = backend.deselect_seed(plant)
        if not result.success:
            raise RuntimeError(f"Deselect({plant}) failed: {result.message}")
    for plant in target:
        state = settled()
        if plant in state.selected_seeds:
            continue
        if not backend.get_seed_selection_mask()[plant]:
            raise RuntimeError(f"Select({plant}) is no longer legal")
        result = backend.select_seed(plant)
        if not result.success:
            raise RuntimeError(f"Select({plant}) failed: {result.message}")
    ready = settled()
    if (set(ready.selected_seeds) != set(target) or not ready.can_confirm
            or not backend.get_seed_selection_mask()[CONFIRM_ID]):
        raise RuntimeError("selected deck does not match a confirmable default deck")
    if ordered and tuple(ready.selected_seeds) != tuple(target):
        raise RuntimeError("selected deck order does not match the requested preset order")
    if confirm:
        result = backend.confirm_seeds()
        if not result.success:
            raise RuntimeError(f"Confirm failed: {result.message}")
    return ready
