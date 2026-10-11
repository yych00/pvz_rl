# 奖励定义与记账口径

Reward is Python policy, never a native game fact.

- `pvz_gym.rewards.battle`: battle terminal reward.
- `pvz_gym.rewards.seed_selection`: seed/deck action and delayed outcome reward.

Keep game mechanics out of reward functions. Richer shaping should be added as
explicit replaceable Python policies with tests rather than hidden in C++.

Double DQN presets live in `train.dqn.reward`; ordinary TD and tick-discounted
n-step replay remain in use. New DQN training and the Workbench selector default
to `transition_dense`. Select another preset explicitly to use its own rules.

## 逐步奖励 `transition_dense`

### 目标与公式

优先通关，然后避免第一次丢车，并继续保护剩余小车。每次 transition 执行完毕后结算：

| 奖励项 | 公式 / 数值 | 结算规则 |
| --- | --- | --- |
| 资产 | `(B_after - B_before) / 5000` | 每步 |
| 伤害 | `0.30 * damage_delta / H_level` | 每步，保守快照估算 |
| 威胁 | `clip(0.10 * threat_delta, -0.01, +0.01)` | 每步，同一批僵尸的威胁变化，排除出生、小车影响和终局清场 |
| 首次丢车 | `-0.50` | 每局一次 |
| 后续丢车 | `-0.10 / 辆` | 每辆一次 |
| 通关 / 失败 | `+1.00 / -1.00` | 真实终局一次 |
| 净 shaping 上下限 | `-0.40 ～ +0.40` | 整局未折扣净累计奖励，平滑趋近边界 |

先合并本步原始变化 `r = asset + damage + threat`，累计为 `S`（每局从 0 开始）。
使用斜率在原点为 1 的 softsign：`F(S) = 0.40 * S / (0.40 + abs(S))`。
本步实际塑形奖励为 `F(S_before + r) - F(S_before)`，整局累计因此等于 `F(S_final)`。
正负变化可以抵消；有限输入下同方向变化持续给分，越接近边界反馈越小，不使用累计额度硬截断。
同号差分用等价乘积计算，避免边界附近两个几乎相等的浮点余额相减，使有效小奖励变成零。
这里的 ±0.40 限制的是**净累计**，不是分别限制正负奖励总额，也不是每步裁剪。
小车与终局独立结算。tick/step 截断不视为失败。
不另外奖励击杀、波次、收阳光、种植、植物死亡、存活时间或终局数量/伤害。

五辆小车时，最大丢车扣分为 `0.50 + 4*0.10 = 0.90`。不折扣总和下，最差通关
`1-0.90-0.40=-0.30`，优于最好失败 `-1+0.40=-0.60`（均为边界值）。
这个边界不保证 tick 折扣后的 Q 回报排序，也不适用于任意更多小车数量。

### 资产口径

`B = 当前阳光 + 场上植物原始价值 + 可消耗库存卡牌原始价值`。
普通可重复使用的种子槽不计资产；传送带、掉落卡牌用原始价格，不能用当前 `cost=0`。
掉落卡与动作槽别名按原生 ID 去重。卡牌种下后从库存转成植物资产，普通种植则由阳光
转成植物资产，均不额外加分。升级植物价值包括被消耗的底座价格。
模仿者使用目标植物价格。植物受伤不折价，死亡/铲除后移除价值。
初始资产仅作为基线，不发奖励。

### 伤害口径与限制

`H_level` 开局读取并固定为原版 Adventure 完整出怪计划的有效生命预算，沿用现有
血量表和原生 spawn-plan 校验；当前不支持任意其他模式的预算。
生命、头盔、盾牌、飞行耐久使用相同口径。按僵尸稳定 ID 保存有效生命低水位，
只对已观察对象的进一步生命下降计分，不计过量伤害，回血后重复打掉相同血量不再计分。

**目前为保守估算，不是伤害事件账本：** 新观察到的受伤对象只建立基线；消失的对象
不直接推断为击杀；终局死亡记录和 fallback 不推断为伤害。因此可能漏记出生后很快
死亡的僵尸和最终击杀。

小车首次从 READY 变为 ROLLING/LOST 的行，以及 transition 前后处于 ROLLING 的行，
本步不计伤害；它会同时舍弃这些行的正常植物伤害。如果小车事件无法对应到行，
保守抑制本步全部伤害。抑制后仍更新低水位，后续不能补领被抑制的伤害。
### 威胁口径

`p = clip((900 - x) / 900, 0, 1)`，沿用现有观察诊断的坐标约定：出生参考端 x=900，房子参考边界 x=0。
`T = sum(剩余有效生命 * p²) / H_level`。越靠近房子且剩余血量越高，威胁越大。

每步按稳定 ID 匹配前后均可观察到的敌对僵尸，在同一批对象上计算
`threat_delta = T_before_cohort - T_after_cohort`。向房子靠近得到负反馈；退远或生命减少得到正反馈。
普通非终局 transition 中观察到的死亡记录可将该对象威胁降为 0。

新生对象只建立基线，不处罚出生；无确认死亡记录的消失对象不推断为威胁下降；阵营改变不发奖励。
小车影响行前后都排除，事件无法定位到行时本步全部抑制；终局及 fallback 不发威胁变化奖励，
避免清场奖励。截断仍结算正常可观察威胁变化。每步抑制后仍更新基线，后续不能补领清除收益。

`threat_value` 记录整个当前棋盘的威胁，而 `threat_delta` 仅来自可比较的对象，二者不必满足直接相减。
这仍是保守快照估算：可能漏掉快照间已经死亡的对象；小车影响行的正常防守收益也会被舍弃。
不直接复用通用环境的 threat delta，因为它使用另一套归一化和对象统计口径。

### 小车、观察与训练

第一次同时丢 `k` 辆扣 `-0.50 - 0.10*(k-1)`，之后扣 `-0.10*k`。
有原生行状态时，以 READY 状态转换去重；同一辆从 ROLLING 消失不再次扣分。
fallback 只保留可靠的事件与终局信息，不制造资产或伤害变化。

DQN 保留 `reward_context` 三维输入以便复用旧模型权重：首次丢车标记、
`(0.40 - F(S))/0.40`、`(0.40 + F(S))/0.40`。后两项是到净余额边界的空间，范围为 `[0, 2]`。
reset 为 `[0, 1, 1]`；replay 分别保存动作前与结算后的状态。
这是 DQN 专用字段，不改变通用 Gym/native 观察。

奖励在事件发生的 transition 发放；通过现有按实际 ticks 折扣的 n-step TD 向前传播，
不复制到历史 transition。当前 smooth_total 的 reward metadata 版本为 6、metrics_version 为 4，记录 softsign 变换、实际权重和指标口径；
日志记录 `shaping_raw_total`、`shaping_balance` 和本步 `shaping_attenuation`。
缺少 `shaping_limit_mode` 的旧检查点/配置按 `episode_budget` 解读，仍使用历史独立正负额度
和当前版本 5 metadata。能否复用还需通过当前网络/观测契约，不能据此宣称旧 v6/v7/v8 网络可加载。
Workbench 的逐步奖励训练默认使用 `smooth_total`；满足当前网络/观测契约的模型可保留权重、优化器和进度，
切换后旧 Replay 因奖励语义不同而跳过并重建。严格恢复要求保持原规则。
CLI 续训默认保留检查点的规则；指定 `--shaping-limit-mode smooth_total` 可切换新规则。

**不提供跨观察结构迁移。** `transition_dense` 与其他方案之间切换须新建模型；
同方案检查点支持正常续训、replay 保存恢复和评估。

### Boss 非空间化

当前 damage/资产仍按 tracker 的生命预算与快照口径记账；Boss 本体没有普通进攻行，
不按其占位 x/y 计算向房屋靠近的 threat。球是独立攻击实体，没有可计入的僵尸 HP 池。
metrics_version 区分这类记账变化，不能只比较奖励系数是否相同。

Boss 伤害不受普通行的小车排除影响；普通僵尸仍使用小车行保守抑制。该区别由
`boss_basis=nonspatial_boss_no_threat_mower_independent_damage_v1` 记录。
评估记录通关率、零丢车通关率、通关局平均丢车数、首次丢车 tick 和奖励分量。
它们是评估指标，不参与动态修改奖励。

### 验证边界

离线回归覆盖平滑净累计边界、正负抵消、连续同向反馈、边界差分精度、历史预算、首次/后续丢车、快照伤害、fallback、截断、观察前后状态、
replay round-trip、模型学习和同方案检查点恢复。离线通过不代表真实游戏胜率或保车率提高。

## 阶段奖励 `staged`

每 1000 个实际游戏 ticks 结算一次。定义：

- `B = 当前阳光 + 尚未使用的卡牌原价总和 + 场上植物原价总和`。
- `H = 开局读取并固定的本关计划僵尸总血量`。
- `q = min(累计有效掉血 / H, 1)`。

阶段辅助奖励：

```text
stage_reward = min((B_end - B_start) / 1000 + 0.5 * (q_end - q_start), 0.10)
```

单阶段正奖励上限为 0.10，整局累计正向阶段辅助奖励上限为 1.00（防止拖延时间无限刷分）。不截断负奖励。
每使用一辆小车，立即在对应 transition 扣 `0.20`；不计小车初始资产。
车损在辅助奖励封顶后单独相加，例如 `+0.10 - 0.20 = -0.10`。
这里保证的是单阶段辅助奖励小于一辆小车，多个阶段累计收益仍可能超过车损。

真正终局额外加 `胜利 +1 / 失败 -1 + 0.5*q_final`，不足一阶段的尾段也结算。
环境 tick/step 上限截断只结算尾段，不加胜负或终局伤害奖励。
训练奖励不包含旧的击杀、波次、威胁下降、终局存活植物或终局阳光分项。

## 资源口径

卡槽、传送带、掉落种子卡均按原版 `PlantDefinition` 阳光原价计值，
不使用免费玩法下的 `cost=0`。固定卡槽无论是否冷却均计一次，因而其固定值
会在阶段差分中抵消。掉落卡的 `source_id` 与 `dropped_cards.id` 去重；
不受网络操作槽或投影实体上限影响。关卡奖励植物、奖杯不计为可用种子卡。

场上植物不因受伤减值，死亡/铲除后价值消失。升级植物计入被消耗的底座，
例如玉米加农炮 `500 + 100 + 100 = 700`。模仿者变身前按目标植物计值。
原价本来为零的植物仍为零，包括小喷菇、海蘑菇，以及原版定义中无阳光售价的
爆炸坚果/巨大坚果。不会给这些特殊种子编造售价。

## 原生整关出怪表与血量

读取版本映射的 `Board::mZombiesInWave[100][50]`：原版 Board 偏移
`0x6B4`，GOTY/汉化 GOTY 使用 `VersionedPvZ::data().spawn_list`。
`total_waves` 来自原生 `mNumWaves`。每波读取到第一个 `-1`，后面的陈旧内存
归一为 `-1`。读取失败不会用全零数组冒充普通僵尸。
Python 在开局将各计划类型的原版初始本体、头盔、盾牌、飞行耐久求和，固定 `H`。

**这里是计划总血量，不是未来所有实际生成实体的精确总血量。**
`Board+0x5598` 的 `mZombieHealthWaveStart` 仅属于当前波，且护盾权重为 0.2、
部分实体被排除，不能用于本指标。舞王召唤、小鬼、墓碑或博士动态生成实体
并不全包含在初始出怪表；雪橇队还可能因无冰路改为四只普通僵尸。
计划雪橇队按四个本体加领队头盔计 `4*270+300=1380`。
此版仅支持原版 Adventure 的非空有限出怪表；其他模式、未知类型、空表或读取失败
会明确报错，不使用已见僵尸的血量作为不断变动的分母。

掉血取未经网络截断的 `zombie_health` 原始记录，并用带 generation 的实体 ID
跨步追踪四个耐久池。新出生满血不会得分，新见实体已掉的血计入伤害；
治疗后不会重复奖励已消耗的同一段血量。正常死亡/被小车碾压的死亡阶段补齐
剩余耐久；魅惑阵营不新增伤害。此前观察到的普通敌人瞬间消失时，使用原生
`zombies_killed` 消失事件补齐；雪人和蹦极离场不据此补齐。
失败清理和原生终局观测回退不补齐伤害；回退使用最后有效资产/伤害统计。

目前是决策间隔快照记账，未安装逐 tick 伤害 hook。出生后又在同一间隔内被释放、
从未出现在任何快照中的实体可能漏计。`zombies_killed` 是现有实体消失诊断，
也不是游戏攻击函数的精确击杀通知。因此 `q` 是该口径下的伤害进度，不能作为
逐 tick 伤害日志。小车造成的可观测伤害仍计入 `q`，同时独立扣车损。

## 时钟、日志和续训

stepped/headless 模式会缩短跨阶段的动作推进，例如默认 150 ticks 在 900 后
只推进 100 ticks，准确停在 1000。continuous 模式可能自然越界，按实际观测
结算一次，不为没有中间事实的多个阶段补发奖励。

结算 transition 和每局日志记录 `reward_breakdown`，包含资产、伤害 HP、计划
总血量、伤害比例、阶段差分、车损、终局分项。评估仍保留原生胜负分 `score`，
另记录 `training_score` / `mean_training_score` 和最后的奖励明细。

新方案使用独立版本化 reward metadata。旧方案切到 `staged` 时现有恢复逻辑
跳过不匹配的 replay；旧模型权重可以作为起点，但旧 Q 数值不是新奖励的估计。
不改变动作空间或网络结构。使用 `weights_only` 时仍按现有规则重置训练状态。

原始价格、初始 HP 和结构依据以下固定版本的逆向来源；未将外部代码复制进奖励实现：
[Plant.cpp](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/325b2229a7225d73870bb69322eefcda3a45bf8d/Lawn/Plant.cpp)、
[Zombie.cpp](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/325b2229a7225d73870bb69322eefcda3a45bf8d/Lawn/Zombie.cpp)、
[Board.h](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/325b2229a7225d73870bb69322eefcda3a45bf8d/Lawn/Board.h)、
[Board.cpp](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/325b2229a7225d73870bb69322eefcda3a45bf8d/Lawn/Board.cpp)。

## 其他三种预设

以下由 `training_reward` 读取环境事件/诊断，不使用 dense/staged 的专用 tracker。
终局 scale 为 0.01，将原始 ±100 变为 ±1。

| 预设 | 额外分项 |
| --- | --- |
| `sparse` | 每次丢车 -0.10；只有该预设读取配置的 `reward_scale` / `mower_penalty` 覆盖值 |
| `balanced` | 丢车 -0.20；`0.20 * clip(zombie_threat_delta,-1,1)`；终局 `0.30 * plant_score + 0.10 * sun_score`，两种 score 限制在 [0,1] |
| `dense_early` | 丢车 -0.20；植物损失 -0.01；击杀 +0.003；推进波次 +0.03；收集阳光每单位 +0.0001 |

未列出的系数为零；balanced 的 threat 是环境诊断口径，不是 transition_dense 的
同对象/完整计划生命预算口径。截断不伪造胜负终局。调 `mower_penalty` 不会改变
其他命名预设的固定系数。

## 元数据与源码

| 方案 | reward metadata version | 说明 |
| --- | --- | --- |
| `transition_dense/smooth_total` | 6 | 净余额 softsign，metrics_version=4 |
| `transition_dense/episode_budget` | 5 | 独立正负额度，metrics_version=4 |
| `staged` | 2 | 按阶段结算，指标版本由 staged tracker 定义 |
| `sparse/balanced/dense_early` | 1 | 事件系数与小车计数口径 |

元数据描述实际奖励定义，和网络、观察、折扣版本分别校验。原始 transition
奖励语义改变时，不能仅靠重算 gamma/n 把旧 Replay 变成新奖励。
实现见 [reward.py](../train/dqn/reward.py)、[transition_reward.py](../train/dqn/transition_reward.py)、
[staged_reward.py](../train/dqn/staged_reward.py)及[reward_observation.py](../train/dqn/reward_observation.py)。
