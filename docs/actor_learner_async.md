# Double DQN Actor–Learner 异步训练

## 选择架构

新训练默认 `training_architecture="actor_learner"`。Workbench 的 Double DQN 参数区提供「采样 / 训练架构」选择；选择 `legacy` 可恢复原来的训练路径。单环境也支持新架构，多环境继续使用现有独立 Worker 进程。

CLI 示例（在 `pvz_rl` 根目录，已按 README 安装并准备训练存档）：

```powershell
.venv/Scripts/python.exe -m train.dqn.train --num-envs 2 --simulation-mode turbo --device cuda --async-sampling --training-architecture actor_learner --policy-sync-interval 32 --max-policy-lag 64 --max-pending-updates 128
```

`--training-architecture legacy` 使用现有同步 Learner。原 `--async-sampling` 仍仅控制环境的独立调度与批量推理等待，两个选项互相独立。旧 Run / 检查点缺少架构字段时按 `legacy` 读取，继续训练可显式切换。

## 执行方式

```text
游戏 Worker 进程 → 主控 Actor 接收观测 / 推理 / 写入 Replay → 派发下一动作
                                       ↓ 训练额度
                          后台 Learner 抽样 → 梯度更新
                                       ↓ 不可变权重快照
                              主控更新 Actor 推理模型
```

Actor 只持有独立的 online 推理网络；Learner 持有训练 online、target 和 optimizer。推理不获取 Learner 的模型锁。后台训练线程与主控在同一 Python 进程内；游戏采样沿用独立进程。CUDA 训练使用独立 stream，与 Actor 的模型存储分离；实际并发度取决于 GPU 负载与调度。

权重发布通过单槽 CPU 快照通道完成：新快照替换旧快照，发布后的内容不会被下一次训练修改。Actor 在动作批次之间加载快照，一个批次只使用同一版本。版本号就是已完成的梯度更新数。

## 训练比例、积压和 Policy Lag

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `train_every` | 6 | 基础请求间隔；满足 warmup 和完整 n-step 条件后发放训练额度，默认开启动态间隔 |
| `gradient_steps` | 1 | 每次发放的梯度更新数 |
| `policy_sync_interval` | 32 | 连续更新后发布快照的最大间隔；训练额度耗尽或到 Lag 边界时提前发布 |
| `runtime.max_policy_lag` | 64 | Learner 相对当前 Actor 和最老在途动作允许领先的更新数；异步架构要求至少为 1 |
| `max_pending_updates` | 128 | 未完成训练额度达到阈值后，暂停派发新动作 |

Warmup 期间不累积额度；不适合 Replay 的观测不会产生训练额度。Learner 每完成一次更新才扣除一次额度，抽样失败或暂停不丢弃额度。正常结束先关闭采样，再完成剩余额度；用户停止则保存尚未完成的额度。

积压阈值采用背压：停止派发，但继续接收已经在途的结果。因此它是派发阈值，已在途结果可以继续增加额度，额外积压最多为 `num_envs × gradient_steps`。这样不会丢掉已经采集的数据，也不会悄悄改变训练次数。Policy Lag 达到上限时，Learner 等待 Actor 同步或慢 Worker 返回。Lag 是更新数，不是环境步数，也不是 Replay 内历史样本的年龄。

背压期间主控通过短条件等待让出执行时间，避免空队列的紧轮询挤占 Learner 的优化器和权重发布工作。

## Replay 与控制流程

Replay 写入、抽样复制、n-step 链操作、PER 回写和检查点复制使用同一把 Replay 锁；GPU 训练期间不持有该锁。抽样返回独立数组。PER 回写核对槽位的 transition ID，跳过已被环形缓存覆盖的样本。

暂停确认前等待当前梯度更新结束，并暂停 Learner 与游戏 Worker；Reset 保留训练模型和 Replay，但封闭被中断的 episode 链。评估期间暂停 Learner，使用固定训练策略；保留原评估流程，其他环境结果可暂存。

检查点在更新边界冻结训练模型、优化器、Actor 权重和版本、探索 RNG 与未使用额度。滚动 autosave 为 model_only；full 保存额外冻结并配对 Replay，写入实现为 `train/checkpoint_io.py`。只有匹配 Replay 恢复成功时保留旧额度；模型恢复、缺失 Replay 或奖励不兼容而重新 warmup 时清零，避免积压阻止采样。`weights_only` / `weights_replay` 都重置进度和 Worker 状态，后者另要求匹配源 Replay。五种加载契约见 [DQN](dqn_reference.md)。

继续训练时可以收紧 Policy Lag；若保存的 Actor 超出新的上限，恢复时直接使用当前 Learner 权重初始化 Actor。

游戏现场不会精确恢复，仍标记 `exact_game_resume=False`。异步调度也不保证逐步复现相同动作顺序。

## 验证与性能观察

对应离线测试覆盖推理/训练重叠、独立权重、慢环境 Lag、额度、暂停/停止/恢复、PER 覆盖回写、错误清理、多进程采样、checkpoint 配对及 Workbench/CLI 参数。历史低层观测工具测试不代表训练加载器支持旧网络。测试与显式实机验收命令见[开发文档](development.md)。

主控耗时保存为 `timing.json`，后台训练耗时单独保存为 `learner_timing.json`；两者可能重叠，不能直接相加推算墙钟时间。`step` 事件附带 Actor / Learner 版本和未完成训练额度，`policy_sync` 事件记录同步。

### Learner 分段计时

Actor–Learner 默认启用详细计时，首次更新后写入当前运行目录的 `learner_timing.json`，随后在更新完成时每 10 秒刷新一次；正常结束、停止或失败清理时强制刷新。文件用临时文件原子替换，读取不会遇到半份 JSON。暂停期间没有更新，文件不会刷新。已有 `learn`、`replay_sample`、`policy_publish` 的字段与统计格式保留；训练更新比例、计算精度和回放数据不变。已经运行的工作台需要正常停止并重新启动进程才能加载此改动。

| 字段前缀（`learn_` 之后） | 测量内容 |
| --- | --- |
| `data_prepare` | 两份观测、动作、奖励、终止标记、mask、PER 权重转为设备张量 |
| `tensor_prepare` / `pin_memory` / `device_transfer` | NumPy 连续化和 dtype 转换、锁页内存、设备传输调用的累计墙钟时间；每次更新聚合所有张量字段 |
| `current_forward` | 当前状态的 Online Q 前向 |
| `online_next_forward` / `next_action` | 下一状态的 Online 前向、合法动作 mask 和 argmax |
| `target_forward` / `target_compute` | 下一状态的 Target 前向、tick 折扣与 TD 目标 |
| `loss_compute` | PER 加权 Huber 损失和 TD error |
| `zero_grad` / `backward` / `grad_clip` / `optimizer` | 清梯度、反向传播、梯度裁剪、Adam 更新 |
| `target_sync` | 达到配置间隔时的 Target 参数同步；计数只包含实际同步 |
| `result_copy` | loss 和 TD error 拷回 CPU；包含这里发生的既有 GPU 等待 |

所有阶段都有 `_wall` 后缀，单位毫秒，记录每次更新的 Python 墙钟耗时。CUDA 设备还提供主要阶段的 `_cuda` 字段：首次更新及每 32 次更新用当前 Learner stream 的 CUDA Event 抽样；Target 同步发生的更新也强制抽样。`tensor_prepare`、`pin_memory`、`device_transfer` 仅记录墙钟，`device_transfer_wall` 包含非阻塞复制的提交开销，不能当作实际 H2D 复制时间。

CUDA Event 在后续更新中通过 `query()` 延迟读取，退出时才等待尚未完成的事件；不会为每个阶段插入同步屏障。Event 区间包含 CPU 提交间隙和其他 stream 竞争，**不等于纯 GPU kernel 耗时**。`result_copy_wall` 可能很大，而对应的 CUDA 区间很小：前者还可能在等待此前提交的工作。所有统计保留 `count`、`total_ms`、`mean_ms`、`p50_ms`、`p95_ms`、`p99_ms` 和 `percentile_window`；分位数使用最近最多 4096 个样本，CUDA 和墙钟的样本量不同。

`data_prepare` 包含张量准备子阶段；`target_compute` 中的折扣转换也贡献张量准备统计。子阶段不要与父阶段相加，也不要把 `_wall` 和 `_cuda` 相加。判断瓶颈时先看 `learn` 总时间，再比较主要阶段的墙钟和 CUDA 区间。

另外记录 `replay_lock_wait`、`agent_lock_wait`、`priority_lock_wait` 和 `per_priority_update`，用于区分锁竞争与计算耗时。PER 样本槽被覆盖时会跳过该槽，回写计数可能少于更新计数。CPU 训练不会产生 `_cuda` 或 `pin_memory` 字段。此拆分目前接入 Actor–Learner 路径，`legacy` 路径仍保留原来的总计时。

实现见 [actor_learner.py](../train/dqn/actor_learner.py)、[训练节奏](../train/dqn/training_cadence.py)与 [learner_timing.py](../train/dqn/learner_timing.py)。真实 PvZ 吞吐、显存、长期稳定性和学习质量需按相同配置进行 A/B，架构本身不证明具体加速比例。
