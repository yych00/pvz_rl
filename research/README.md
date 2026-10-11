# 研究与技术报告

本项目将 PvZ retail 1.2.0.1065 的真实 Windows 进程作为强化学习环境。
这里说明当前实现的研究问题、已有训练数据的统计口径与待验证的方向。
完整参数、字段及原生接口以 [docs 文档导航](../docs/README.md) 为准。

[Double DQN 专项报告](ddqn_report.md)包含算法公式、网络结构、并发数据流、
一组 40 常规关卡历史训练统计，以及后续实验的验证方法。

## 当前实现

| 主题 | 实现及边界 | 详细文档 |
| --- | --- | --- |
| 游戏时钟与跨架构控制 | x64 Python 加载 x64 原生桥接 DLL；桥接读取 x86 游戏内存并执行生成的 x86 注入代码。Turbo 控制游戏主循环，快照在同步边界读取；没有“把 x64 DLL 注入 x86 进程”的流程 | [原生 API](../docs/native_api.md)、[Reset](../docs/reset_lifecycle.md) |
| 实例隔离 | 每路训练/评估使用独立游戏副本与专用 profile；实际存档位于 `%ProgramData%` 的实例目录。原游戏和个人存档不作为训练写入目标 | [游戏与存档](../docs/game_setup.md)、[系统结构](../docs/architecture.md) |
| 结构化观测 | 最大 6×9 网格、三层植物、最多 256 个僵尸和 8 个 Boss 球。Boss 本体去除占位空间坐标；独立球保留真实行与 x/y。雾过滤是格级可见性近似 | [观测](../docs/observation.md)、[Boss 球](../docs/boss_projectile_observation_contract.md) |
| 动作与掩码 | 战斗 757 动作，选卡另有 193 动作；战斗包含 NOOP、用卡、两种铲除和玉米炮目标。预留 INTERACT 当前被屏蔽；小车是自动运行的状态与奖励事实 | [动作契约](../docs/action_space.md) |
| Double DQN | v9 实体与动作注意力、游戏 tick 折扣、3-step 回报和 SumTree PER。当前网络使用 `Q=V+Score`，没有经典 dueling 的全动作均值中心化 | [DQN](../docs/dqn_reference.md)、[Replay](../docs/replay_memory.md) |
| 奖励 | 默认 `transition_dense`：胜负、小车与资产/伤害/威胁变化，整局净塑形默认以 `smooth_total` 限制在 ±0.4。其他奖励预设可选，不混写为默认配置 | [奖励](../docs/rewards.md) |
| 自适应课程 | 根据每个 `(level, playthrough)` 的近期有效训练对局胜率调节采样概率；独立 greedy 评估不写入该窗口。课程配置覆盖冒险 50 关 × 2 周目 | [自适应采样](../docs/adaptive_level_sampling.md)、[课程表](../docs/adventure_curriculum.md) |
| Actor–Learner | 游戏 Worker 负责环境操作；主控 Actor 推理并写 Replay；同一主控进程中的后台 Learner 训练并发布不可变权重快照。Replay 和发布通道具有同步机制 | [异步架构](../docs/actor_learner_async.md) |

Turbo 吞吐量随场景、机器、模型和决策间隔变化，这里不承诺固定加速倍数。
课程覆盖范围也不等于所有关卡都已训练收敛或完成实机验收。

## 实验如何比较

训练对局、TD loss 与固定种子 greedy 评估分别记录。固定种子有助于控制出怪等
随机因素，但真实进程与异步调度不保证逐动作复现。比较模型时应固定游戏布局、
关卡/周目、卡组、种子集合、时钟、游戏 tick 预算和 checkpoint，报告每关胜率、
截断及样本数；不能只比较总体平均数。

离线检查与显式实机验收入口见 [开发文档](../docs/development.md)。本机报告、
原始日志、模型和 Replay 不随源码发布；报告中的历史训练摘要不作为安装验收或
当前版本 greedy 性能的承诺。

## 待验证的研究方向

空间卷积与注意力的对比、同行关系特征、技能课程、专家蒸馏及增大模型容量都可以
作为实验方向。目前没有随仓库发布的同条件对照证据证明其中一种必然优于当前
网络。报告给出相应的测量方法，尚未实现的设计不属于支持功能。

长期希望智能体能掌握经济、布防、组合植物与战术时序，并进一步研究泳池无尽。
**无尽训练、分层技能和专家蒸馏尚未作为本仓库的完整训练流程实现。**
