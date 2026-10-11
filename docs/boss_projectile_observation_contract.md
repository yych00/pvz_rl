# Boss 球观测与观测质量契约

本页保留已核验的 1065 布局与当前快照契约。源码核对日期：2026-10-11；地址审计记录来自 2026-10-03。

## 已核对的游戏布局

已审计样本为用户自行提供的 1065 EXE；仓库不包含该文件，目录名不是识别依据。
FileVersion 与 ProductVersion 均为 `1.2.0.1065`，SHA256：

```text
0B914385FA0341BAEAA185BEC4DFDB1D3E66973B335DD0399E257E892E3CF391
```

公开反编译的 [Zombie.h](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/master/Lawn/Zombie.h) 和 [Zombie.cpp](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/master/Lawn/Zombie.cpp) 用于确认字段含义与生命周期。该项目 [README](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/master/README.md) 明确基于 1.0.0.1051，函数地址不能直接用于本机 1065。下列 1065 地址和偏移另外使用本机 EXE 的 x86 反汇编核实；运行时只读字段，没有调用这些函数或修改 EXE。

| 事实 | 本机 1065 证据 |
| --- | --- |
| Boss 持有球 ReanimationID | `UpdateBossFireball` 在 `0x535FB0`，`0x535FB2` 读取 Zombie `+0x140` |
| 行由独立字段决定 | `0x53600D` 读取 Zombie `+0x148`，不能使用 Boss 自身 `mRow` |
| 火球/冰球选择 | `0x536111` 比较 Zombie `+0x14C`；非零为火球 |
| effect system 与 reanim holder | `0x535FB8` 读取 LawnApp `+0x820`，`0x535FBE` 读取 effect system `+0x08` |
| DataArray 容量与实例 generation | `0x535FD2` 检查 holder `+0x08`，`0x535FE3` 比较实例 `+0x9C` 和完整 ID |
| 实例步长 | `0x535FDB` 至 `0x535FE1` 计算 index × 5 × 32，即 `0xA0` |
| 连续 X | `0x535FFA` 读取并在 `0x53600A` 更新 reanim `+0x2C` |
| 连续 Y | `0x536046` 更新 reanim `+0x38`，由当前 X 与球行沿屋顶计算 |
| 创建类型 | `BossHeadSpitContact` 在 `0x535DC0`；`0x535E10` 以 `0x54` 创建火球、`0x535E71` 以 `0x53` 创建冰球 |
| 保存新 ID | `0x535EC9` 读取 reanim `+0x9C` 并在 `0x535ED1` 写回 Boss `+0x140` |
| 消失 | 离开、被消除时将 reanim 置为 dead 并清空 Boss 引用；下一次快照不发布该实体 |

当前读取实现见 [boss_projectile_snapshot.h](../native/src/observation/boss_projectile_snapshot.h)：`mDead +0x14`、overlay、完整 ID 和类型都参与校验。目标为 32 位游戏，即使 DLL 是 x64，也只按 4 字节读取远程指针。

## 原生观测与固定输入

Boss 球是 Boss 管理的独立攻击实体，不属于 `Board::mProjectiles`，也没有僵尸 HP/伤害池。原始 `boss_projectiles` 每项字段为 `id`、`owner_id`、`type`、`row`、`x`、`y`、`valid`；`type=0` 为火球，`type=1` 为冰球，坐标为真实 overlay 像素坐标。ID 只供跨 tick 追踪，不进入模型。

| 模型字段 | 形状 / dtype | 含义 |
| --- | --- | --- |
| `boss_projectile_type` | `(8,) / int64` | 0 填充，1 火球，2 冰球 |
| `boss_projectile_row` | `(8,) / int64` | 0 填充，真实行 + 1 |
| `boss_projectiles` | `(8, 2) / float32` | `[x / 900, y / 600]`，保留连续位置与负坐标 |
| `boss_projectile_mask` | `(8,) / int8` | 1 为有效实体 |

普通关无此实体时四个数组全零。正常游戏单个 Boss 最多持有一个球；对注入多个 Boss 的场景，原生保留所有有效实体，编码按 X、行、类型排序并保留最左侧八个攻击。`info.boss_projectiles_truncated` 与 `boss_projectiles_dropped` 明示截断。

`id=0`、generation 不匹配或 dead 表示不存在；引用仍有效却读不到内存、行无效、类型与引用含义冲突或坐标非有限，均拒绝整份观测。出现 Boss 的未核对版本返回不支持，不读取猜测布局；普通关无 Boss 时仍可返回空集合。

## 版本与恢复边界

原始 native observation `schema_version=4`，增加 `quality`，各传输共用同一结构。Python 模型 schema：

```text
pvz_observation_entities_256_zombie_grid_imitater_boss_projectiles_boss_nonspatial_v5
```

`LEGACY_OBSERVATION_SCHEMA` / SPEC 是不含球数组的非空间 Boss v5，用于底层离线工具；`PREVIOUS_OBSERVATION_SCHEMA` 记录带球但 Boss 尚空间化的旧 v4。当前 Boss 本体 row/x/y 清零，不参与普通局部几何和僵尸网格汇总；独立球保留真实坐标。公开训练固定启用球输入，缺少 `boss_projectiles=true` 或网络标识不是 v9 时拒绝加载。底层旧 schema 工具不是 checkpoint 兼容承诺，完整输入见[观测契约](observation.md)。

## 快照一致性与终局契约

`RuntimePhaseReader::read_nolock` 作为全局观测和运行阶段的共同输入，检查 required read，并携带 Board identity 与 clock。完整观测组装后再次核对 Board、clock、raw scene 和 phase，发生变化或任何 required reader 失败时拒绝发布。`read_cards` 与 `read_grid` 的辅助读取 bool 逐层传播，`PvzEnv::get_observation()` 在构建失败时抛出错误。

`GlobalState.status` 与 `StepResult.status` 从同一快照分类。Adventure `scene=3 + mLevelComplete=true` 为 Won；挑战/生存不将阶段性 level complete 自动当成整个任务结束。Lost/Award 仍有 Board 时要求完整观测成功。

终局已释放 Board 时，允许返回最近确认实体快照以通知结束，但明确标记：

- `observation.quality = observation_quality = terminal_fallback`。
- snapshot clock 保留该实体快照的原始 clock；`last_confirmed_game_clock` 另外表示原 Board 最后确认时钟。
- `ticks_advanced` 只使用确认时钟差，不用请求 ticks 或注入次数补全；`ticks_advanced_exact=false`、`replay_eligible=false`。
- terminal event 可结算一次，不从旧快照虚构击杀、植物损失、资源收入或支出。

读失败、进程退出和替换 Board 不触发上述回退。C/Python 成功结果缺少观测、v4 状态/质量/新鲜 clock 不一致也报错。Python info 透传质量与 duration，由训练入口排除 `replay_eligible=false`。

## 验证方式

`native/tests/boss_observation_contract_test.cpp` 以故障注入验证内存失败和实体生命周期、同 Board duration、终局回退与事件唯一结算；CMake test 名为 `boss_observation_contract`。测试 exe 传一个输出前缀可生成实际 C++ `.json`、`.pvzb` fixture。

`tests/unit/test_boss_observation_quality.py` 验证普通关空输入、独立类型和行、连续运动序列、无效实体、截断、旧 schema、bridge 失败与 JSON/PVZB/shared 解析。设置 `PVZ_GYM_NATIVE_OBSERVATION_FIXTURE` 为上述 fixture 前缀可验证真实 C++ 两种 serializer 的结果。

此文件的地址/布局证据是只读 EXE 审计与离线契约。原生构建、真实 Boss 关球进入/移动/被消除与完整训练效果，须在总体验证报告分别记录，不由这些测试推断。
