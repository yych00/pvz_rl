# Battle action space

The formal `PvZEnv` uses a fixed `Discrete(757)` battle space. Python IDs are
defined in `python/pvz_gym/action_space.py` and mirrored by the native
`ActionCodec`. Seed selection has a separate `Discrete(193)` space.

| ID | Action | Shape |
| --- | --- | --- |
| 0 | `NOOP` | 1 |
| 1–540 | `USE_CARD(slot,row,col)` | 10×6×9 |
| 541–594 | `SHOVEL_NORMAL(row,col)` | 6×9 |
| 595–648 | `SHOVEL_PUMPKIN(row,col)` | 6×9 |
| 649–702 | `COB_FIRE(target_row,target_col)` | 6×9 |
| 703–756 | `INTERACT_CELL(row,col)` | 6×9 |

`USE_CARD` addresses the current slot, not a plant type. Seed-bank and conveyor
packets occupy their original slots; live dropped seed packets use remaining
slots. Types such as Leftpeater and I, Zombie cards need native card handling
before they can be used, but do not change the action IDs.

`COB_FIRE` only selects a target. Native code picks the first cannon observed
ready and validates its instance ID again at commit. `INTERACT_CELL` currently
remains masked until the matching mode-specific native click path is installed.
Invalid rows, slots, and cells remain in the fixed shape with a false mask.

The old 3,241-action source-cannon codec is version 1 and cannot be used with
this codec or its checkpoints. Current execution and masking are defined in
[the native action validator](../native/src/action/action_validator.cpp).
