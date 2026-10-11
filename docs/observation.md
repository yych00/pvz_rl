# 观测契约

The native layer returns game facts. `pvz_gym.observation` / `projection`
convert those facts into the stable Gym observation representation.

The native zombie contract and memory offsets are implemented in
[the zombie decoder](../native/src/observation/zombie_decoder.cpp). It exposes separate armor HP, phase and
control timers, movement/height and explicit targets. The current Gym projection
keeps up to 256 individual zombies with 15 numeric features each. Body, helmet,
shield and flying HP remain separate; phase, height and target positions use
categorical fields. A separate grid summary covers all observed zombies.

Observation code may normalize/project facts but must not simulate hidden PvZ
rules or infer action legality that the native mask can provide directly.

Native `cards` places live seed coins of type 16 after the real seed-bank
packets, up to the same ten `PLANT` slots. In It's Raining Seeds there is no
seed bank, so the first coin is slot 0. The separate `dropped_cards` facts
remain available for position and coin diagnostics.

Raw reward diagnostics also expose `global.spawn_plan_available` /
`global.zombie_spawn_plan`, card `source` / `source_id`, plant `imitater_type`,
and `zombie_health` (including occupied dying/dead slots and each maximum HP
pool). `PvZEnv` supplies the complete raw observation in `info.raw_observation`.
These fields do not change the policy observation shapes or checkpoint network.

`PvZEnv.observation_space` is a `gymnasium.spaces.Dict`. The public contract is
named simply `Observation`; there is no endless-only or version-suffixed Python
type. Its fixed fields are:

| field | shape | meaning |
|---|---:|---|
| `global` | `8` | bounded sun/wave/countdown/scene-state features |
| `scene`, `zombie_roster` | `1`, `64` | 场景类别、关前可出僵尸类型列表 |
| `card_type`, `cards` | `10`, `10x6` | categorical seed type and live cost/cooldown/imitater facts |
| `plant_type`, `plant_state`, `plants` | `3x6x9`, `3x6x9`, `3x6x9x4` | layer-preserving plant category, state and HP/sleep facts |
| `grid_terrain`, `grid` | `6x9`, `6x9x9` | terrain, occupancy, grave/crater, ice and grid-item facts |
| `zombie_type/phase/height/row/target_row/target_col` | `256` each | categorical per-zombie identity-independent facts |
| `zombies`, `zombie_mask` | `256x15`, `256` | x/y, speed, altitude, separate HP pools, phase/control timers and status flags |
| `zombie_grid` | `6x9x2` | per-cell count and total body/helmet/shield/flying HP, covering visible zombies |
| `dropped_card_type`, `dropped_cards` | `16`, `16x4` | collectible card category and position |
| `mower_state` | `6` | inactive/lost/ready/rolling category per row |
| `boss_projectile_type`, `boss_projectile_row` | `8` each | 冰球/火球类别与真实行，0 为填充 |
| `boss_projectiles`, `boss_projectile_mask` | `8x2`, `8` | 连续 x/y 与有效实体掩码 |

类别字段使用 int64，连续数组使用 float32；`zombie_roster`、`zombie_mask`、
`boss_projectile_mask` 使用 int8。完整形状与 dtype 由
[OBSERVATION_SPEC](../python/pvz_gym/observation.py)定义，不能将原生 JSON 字段直接当作模型张量。

Native entity IDs remain available in the raw observation for transition/event
tracking, but are never encoded as numeric policy features. Categorical values
use zero for padding and are consumed by embeddings. Plant HP uses both a
remaining-HP ratio and bounded logarithmic magnitude. Zombie body/max/helmet/
shield/flying HP use separate bounded logarithmic features; they are not plant
HP ratios. See the ordered features in `encode_observation` for exact scales.

Card `plant_type` (and encoded `card_type`) remains the target plant for an
Imitater packet. Native JSON also exposes `is_imitater`; `cards` contains, in
order, presence, cost, cooldown progress, cooldown remaining, ready and
`is_imitater`. Missing flags default to false for older raw fixtures; live use
requires rebuilding the native DLL. The card encoder consumes all six features,
so the flag reaches the complete-action queries without changing the 757 action
IDs or native legality masks.

## Schema 与模型兼容

原生快照 `schema_version=4`；Python 当前基础输入 schema 为：

```text
pvz_observation_entities_256_zombie_grid_imitater_boss_projectiles_boss_nonspatial_v5
```

两者属于不同版本体系。当前 DQN 网络为 v9，只接受自身网络标识、对应观测结构和
`boss_projectiles=true`。`PREVIOUS_OBSERVATION_SCHEMA` 记录旧的空间化 Boss v4；
`LEGACY_OBSERVATION_SCHEMA` 是不带球字段的非空间 Boss v5，仅供底层离线契约工具。
这些常量不表示公共加载器支持旧模型。详见 [DQN 保存与加载](dqn_reference.md)。

`transition_dense` 训练在基础输入上附加 `reward_context=(3,) / float32`，schema
加 `_transition_dense_context_v1`。它由
[reward_observation.py](../train/dqn/reward_observation.py)维护；通用 Gym/native 不添加该字段。
当前 smooth_total 输入是首次丢车标记、正/负方向净余额余量，reset 为 `[0,1,1]`，
具体语义见[奖励](rewards.md)。

## Boss 本体与独立球

Boss 原始类型为 25，模型类别为 26。本体不是某一行上朝房子推进的普通僵尸：
投影保留生命、阶段、控制状态和显式目标，但将 `zombie_row` 与连续 x/y 置零。
它仍占有效实体槽，进入全局注意力，不进入普通 `zombie_grid` 和房屋接近威胁。
网络也对直接张量输入再次清除本体占位几何。

球是独立 `boss_projectiles` 实体，类型、行、连续位置进入四个固定数组；按 x、行、
类型排序后最多保留 8 个。截断数量在 info 报告，ID 只作原生追踪。
详见[原生球读取与质量契约](boss_projectile_observation_contract.md)。

### Fog visibility

Native schema 4 transports `global.fog_state_available`, `fog_offset`, and
`fog_alpha` through JSON, binary and shared-memory results. `fog_alpha` is the
actual Board `mGridCelFog[9][7]` array in column-major order; `fog_offset` is
`mFogOffset`. The native reader checks all alpha samples and the finite offset.
Reading these facts follows the existing synchronized snapshot boundary.
Source reference: [Board fog update and rendering](https://github.com/Patoke/re-plants-vs-zombies/blob/main/Lawn/Board.cpp).

On scene 3, encoding requires a valid native fog snapshot. Missing/invalid facts
raise an error instead of silently returning an unobscured observation. Other
scenes do not require fog fields. Native zero-alpha fog cells, including ordinary
fog being disabled on stormy-night boards, remain visible. This change does not
implement stormy-night darkness or lightning visibility.

The projection uses a **cell-level approximation**, not exact sprite opacity:
sample `floor((x - 40 - fog_offset) / 80)` and the entity row, treating alpha
below 128 as visible and alpha at least 128 as obscured. Samples left of the fog
map are visible; the rightmost sample repeats beyond the board. Grid cells are
sampled at their centres (`x=80+80*col`). Native alpha/offset automatically follow
Plantern/Torchwood clearance, disappearance and fade, and Blover blow/return;
the projection does not invent radii or count time since actions.

Visibility is used internally by the projection, with no additional policy
field or grid channel. Invisible zombies do not enter valid entity slots
(`zombie_mask=0` for unused slots); their features stay zero.
Own plants retain live type/state/HP/sleep facts and all three layers even
inside fog. Grid presence and base/main/pumpkin occupancy also remain public
to agree with those plant entities. Obscured hazards/items/ice, dropped seeds
and hostile entities are omitted from policy input. Terrain and
the pre-level zombie roster stay public. Zombies are filtered **before** both
the 256-entity truncation and count/HP aggregation. Hidden facts therefore
cannot consume entity slots or leak through `zombie_grid`. Native legality
masks still describe legal actions; they can reveal occupancy/plantability,
but are not constructed from zombie count/HP.

The raw snapshot is not modified. Reward/event trackers and `info.raw_observation`
retain the full state for accounting/diagnostics; those diagnostics are not
policy inputs. Lighting must be performed by the policy, not an automatic
environment action. 使用包含相应读取实现的 DLL 并重启 Workbench/workers。
离线测试覆盖遮挡、清雾/渐变阈值、雾返回、传输与当前非空间 Boss 输入；
不证明实机像素级可见性等价或胜率改善。旧的未过滤雾观测数据不应作为新策略的可见输入依据。

Zombie slots preserve the native list order; the first 256 retain individual
facts. `zombie_grid` includes the entire visible list, including entities beyond that
limit. Columns use origin 40 and `floor((x - 40) / 80)`, clamped to columns 0..8;
continuous entity x/y positions are not clamped by the projection.

当前 DQN 的 cell encoder 读取 `zombie_grid`。动作的局部分支读取目标整行及
相邻行的三列（最多 15 格）和对应空间实体；全局分支另读完整有效棋盘。
不提供额外 cell-to-zombie 相对几何向量。v9 两层全局/单层局部的完整数据流见
[网络实现](dqn_reference.md)。
