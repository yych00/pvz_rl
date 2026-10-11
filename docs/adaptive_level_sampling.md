# 评估与近期胜率抽关

Workbench 的 Double DQN 参数区提供「评估设置」和「关卡抽样」。

- 多关训练池默认 `evaluation_mode=per_level`：每个 `(关卡, 周目)` 评估
  `evaluation_per_level=4` 局，可修改。八关共 32 局。
- 单关新配置默认 `evaluation_mode=random`；多关也可选择该模式，从训练池均匀抽取 `evaluation_random_episodes=20` 局，
  可修改。两种评估都沿用训练周目，关闭探索，使用自动生成的游戏种子。
- 每轮结果保存分关卡胜负。历史最佳仅与相同评估模式、次数、关卡池和
  截断预算比较。旧 run 未记录评估模式时沿用随机模式。
- 旧的显式控制器 `eval_seed_*` 参数继续保留诊断工具的既有协议。

普通自动评估每轮生成新种子；固定种子比较使用 `train.dqn.evaluate`。
优化控制器 full 默认每 400 个完成的训练 episode 触发，fast 默认关闭；
`evaluation_mode=off` 关闭评估。配置字段同时保存到 Run 与 checkpoint。

开启 `adaptive_level_sampling` 后，每个 `(关卡, 周目)` 独立保存最近
`level_sampling_window=40` 局正常训练胜负；评估、未知结果、截断、运行错误
不进入窗口。未开启时保持均匀抽关。

抽样模式可选 `squared_failure`（默认）、`bounded_difficulty`、`uniform`。
`bounded_difficulty` 使用 `(wins+5)/(episodes+10)` 和 `1+(M-1)*(1-p)^2`；
`uniform` 为各关等概率。三种模式共用训练窗口与新旧概率 0.5/0.5 平滑。

默认 `squared_failure`：胜率使用每关最近 40 局有效训练胜负的实际比例 `p=wins/episodes`；没有记录时取 50%。
基础权重与失败率平方：`w=0.05+(1-p)^2`。
目标抽样概率：`P=0.05/K+0.95*w/sum(w)`。
每 20 局有效训练后按 `0.5*旧概率+0.5*目标概率` 更新，下一局 reset 生效。
`level_sampling_max_weight` 仅作用于 `bounded_difficulty`。

窗口、计数和概率写入模型检查点；旧检查点可从同目录训练日志恢复。
权重微调新 run 会清空历史。界面显示每关近期胜率、抽样概率以及局数、
步数占比。概率定义在局数上，较长关卡的经验占比可能更高。

验收以独立分关卡评估为准；加权训练胜率不能直接与均匀训练总胜率比较。

Actor–Learner 的 `auto_train_every` 按 Learner 待处理更新数调整实际训练间隔：
达到高水位时加倍，降至低水位时减半，介于两者之间保持不变。
调整至少间隔 `max(32, 2*实际间隔)` 个环境步，范围为
`1` 到 `64*train_every`；只调整后续更新请求，不丢弃已有更新任务。
实际间隔显示在界面日志和训练局记录中，检查点保存间隔及冷却进度。
关闭开关或串行同步训练时使用固定的 `train_every`。

实现见 [配置](../train/dqn/config.py)、[评估任务与协议](../train/dqn/evaluation_plan.py)、
[抽样概率](../train/adaptive_curriculum.py)、[关卡池](../train/curriculum.py)和[训练节奏](../train/dqn/training_cadence.py)。
