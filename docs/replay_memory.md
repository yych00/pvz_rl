# Replay 内存与检查点

当前含 Boss 投射物及 transition_dense 上下文的观测，每条 transition 的
obs + next_obs 存储约 29,440 字节。20 万条仅这两组数组就占约 5.48 GiB；
还需加上掩码、流关联、优先级、模型、Python worker 和系统内存。
NPZ 压缩后文件很小，不代表训练中的内存占用小。

当前存储将浮点字段转为 float16，大于一字节的有符号类别字段转为 uint16，
mask/roster 保留 int8；抽样再恢复 float32/int64。下一动作 757 位 mask 打包为
95 字节。回报、duration、stream ID、transition ID、前后链接、episode 边界及
PER 数据另外存储，因此 5.48 GiB 不是总进程内存估计。

每个 stream 的 transition 链独立；n-step 抽样按实际 ticks 折扣并在明确边界停止。
`done` 决定是否关闭 bootstrap，`episode_end` 决定是否停止累加。Reset、恢复及
权重带 Replay 微调封闭旧流尾，避免新游戏接入上一局。完整公式见 [DQN](dqn_reference.md)。

`can_sample()` 使用增量维护的 n-step 合格计数，不再每次创建整个 Replay
的 int64 索引。PER 只屏蔽未形成完整回报的流尾；均匀采样仍保持原先
按时间排序的索引及随机数语义。

恢复时不先创建另一份空 Replay，观测逐字段解压并用连续切片重排到目标
环形缓存，避免源 Replay 和大型重排副本同时驻留。

训练中的完整快照总数组大小达到 256 MiB 后，将大数组复制到 run 目录下
`.replay-snapshot-*` 临时磁盘映射，再由后台线程压缩写成原有 NPZ。
复制发生在 Replay 锁内；后续训练可继续修改原 Replay，冻结快照保持不变。
成功或写入失败后都关闭映射并清理临时目录。需要预留约一份未压缩 Replay
的磁盘空间；强制结束宿主进程可能留下临时目录。

STOPPED / FINISHED 状态的最终保存复用已停止修改的 Replay 视图，并等待
写入完成，不额外复制整份 Replay。保存文件的格式和恢复契约保持不变。

空闲采样 worker 检测父进程退出后会关闭环境并退出，避免宿主意外退出后
长期保留 Python 进程。正在执行原生调用时，检测需等调用返回。

16 GB 内存机器使用 20 万 Replay 时，优先减少并行环境数；8 个 Python
采样进程的提交内存不可忽略。OOM 后应重启 Workbench，再选择完整检查点
及其匹配 Replay 继续训练；修复不会自动恢复失败进程的未保存进度。

源码见 [replay.py](../train/dqn/replay.py)、[sum_tree.py](../train/dqn/sum_tree.py)与
[checkpoint_io.py](../train/checkpoint_io.py)。滚动 model_only 快照不携带 Replay，
full 快照按模型中的 `replay_file` 明确配对；五种恢复/微调模式见 [DQN](dqn_reference.md)。
