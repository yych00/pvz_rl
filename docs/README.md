# 文档导航

首次使用从[根目录 README](../README.md)按“安装 → 放置游戏 → 检查 → 准备存档 → 训练 → 评估”操作。专题总结见[研究与技术报告](../research/README.md)；这里保留使用说明、完整算法说明和原生实现契约，默认值以当前源码为准。

## 安装与使用

| 文档 | 内容 |
| --- | --- |
| [游戏、版本与存档](game_setup.md) | 支持的 1065 布局、完整游戏目录、外部路径、训练模板与隔离存档 |
| [安装与开发](development.md) | Python/CUDA、DLL 安装、独立构建、测试与 Release 打包 |
| [选卡规则与入口](default_adventure_decks.md) | GUI、配置 API、固定/针对/乱序选卡、解锁与戴夫锁卡 |
| [逐关教学卡组](adventure_curriculum.md) | 从课程源码生成的 50 关 × 2 周目主卡、候选与教学分析 |

## 算法与训练实现

| 文档 | 内容 |
| --- | --- |
| [Double DQN](dqn_reference.md) | v9 网络、tick 折扣、n-step、PER、探索、当前默认值、保存恢复与评估 |
| [奖励](rewards.md) | 五种预设、资产/伤害/威胁口径、整局净塑形限制、奖励元数据 |
| [Actor–Learner](actor_learner_async.md) | 线程/进程职责、策略同步、Policy Lag、背压与分段计时 |
| [评估与关卡抽样](adaptive_level_sampling.md) | 单关/混合关评估、近期胜率窗口、抽样概率、动态训练间隔 |
| [Replay](replay_memory.md) | 字段压缩、episode 链、n-step 合格样本、内存与异步磁盘快照 |

## 环境与原生实现

| 文档 | 内容 |
| --- | --- |
| [系统结构](architecture.md) | 代码归属、模块依赖、私有实例与运行数据流 |
| [动作契约](action_space.md) | 战斗 757 动作、选卡 193 动作、原生合法性与 ID 布局 |
| [观测契约](observation.md) | 全部字段、native schema 4 / Python schema v5、Boss 非空间化、雾过滤 |
| [原生 API](native_api.md) | C ABI、能力检查、四种时钟、传输、单次执行结果缓冲、错误与源码位置 |
| [Reset 生命周期](reset_lifecycle.md) | 选关/选卡/终局、主循环屏障、Board 回收、图鉴与超时清理 |
| [Boss 球与快照质量](boss_projectile_observation_contract.md) | 已核验的 1065 偏移、球生命周期、快照一致性与终局回退 |

## 测试与来源

[开发文档](development.md)提供当前离线检查和显式实机验收命令。离线测试不证明游戏对局或训练效果，实机报告保存在本地 `build/`。代码许可和第三方声明见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

源码中仍存在的兼容工具或测试辅助类，不等同于公开训练支持。当前训练仅接受带 Boss 球输入的 v9 模型；旧网络、未迁入的脚本和算法占位入口不属于使用流程。
