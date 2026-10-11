# 原生 API 与运行时实现

原生 DLL 从真实 PvZ 进程取得游戏事实，并校验和提交动作；Python 不复制原版种植合法性规则。公共声明在 [api.h](../native/include/pvz/api.h)，数据结构在 [types.h](../native/include/pvz/types.h)，运行时枚举在 [runtime.h](../native/include/pvz/runtime.h)。Python 绑定见 [bridge/native.py](../python/pvz_gym/bridge/native.py)。

## 位数、调用约定与返回值

生产 DLL 和 Python 为 x64，目标游戏为 x86；远程指针和注入指令仍按 32 位布局处理。C 入口使用 `extern "C"` 和 `PVZ_GYM_CALL=__stdcall`，Python Windows 端使用 `ctypes.WinDLL`。Win32 构建主要用于离线执行注入代码测试，其 DLL 不给 64 位 Python 加载。

不同函数的返回值不能混用：

| 接口 | 返回约定 |
| --- | --- |
| runtime 配置/生命周期 | `Status`：0 OK、1 Unsupported、2 InvalidArgument、3 InvalidState |
| 输出字节缓冲 | 非负为已写字节数（不含末尾零）；-1 为失败；小于 -1 为负的所需容量（含末尾零） |
| attach/pause 等 bool 接口 | 布尔成功或状态，具体声明见头文件 |
| 动作查询/提交 | 动作错误码或相应结构，不能按 runtime Status 解读 |
| reset/step 缓冲 | 外层返回是字节/容量状态；payload 的 `success/error/message` 描述运行结果 |

动作错误在 `ActionError` 定义：1001 找不到游戏、1017 非战斗状态、1019 超时、1020 不支持版本、1021 卡不可用、1099 不支持操作。其余包括阳光、冷却、坐标、地形载具、升级底座、南瓜、铲除及玉米炮错误。`pvz_gym_error_message` 给出文本。

## 能力、配置与生命周期

先查询 `pvz_gym_get_runtime_capabilities_json` 再请求功能。当前接口版本为 1；DLL 宣告 stepped/headless/continuous/turbo 四种模式和以下 features：

```text
result_requests, step_profiling, batch_plant_read, attach_pid,
resident_executor, binary_transport, shared_memory_transport,
shared_snapshot, phase_advance, observation_quality, boss_projectiles
```

同一 payload 的 `observation_schema_version=4` 是原生快照版本，不是 Python 观测 v5 或 DQN 网络 v9。能力声明表示实现存在，不意味着任意游戏版本兼容；公开私有训练还需通过[1065 布局检查](game_setup.md)。

`pvz_gym_configure_runtime(process_id, executor, transport, snapshot)`：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| process_id | 0 / 正 PID | 自动附加兼容入口 / 显式绑定进程 |
| executor | 0 / 1 | legacy 逐次注入 / resident 常驻执行器 |
| transport | 0 / 1 / 2 | JSON / PVZB 二进制 / shared-memory 结果 |
| snapshot | 0 / 1 | legacy 读取 / shared 快照 |

基础 C++/Python 配置为 legacy/json/legacy；训练 CLI 的 Turbo 入口配置 resident/binary/shared。传输方式、快照方式与时钟模式是不同选项。训练 worker 使用明确的私有 PID；自动附加属于旧接口，不是私有实例创建功能。

Turbo 通过 `begin_turbo_mode(flags)` / `end_turbo_mode()` 管理，结束可重复调用；`close_runtime()` 清理原生运行时。实例池负责启动与关闭所拥有的游戏进程，DLL 的 runtime 清理不能代替 Python 实例管理。

| Turbo bit | 值 | 作用 |
| --- | --- | --- |
| SuppressRender | 1 | 抑制正常帧渲染 |
| ServiceMessages | 2 | 服务窗口消息 |
| FastIntro | 4 | 加速 intro/准备 |
| FastTerminal | 8 | 加速终局流程 |
| MuteAudio | 16 | 静音选项 |

默认 flags=31。Turbo 入口明确检查 1065 版本及执行布局；不能从 vendor 的多版本识别表推出完整训练支持。失败诊断通过 `get_runtime_error_json` 冻结在调用线程，清理后仍可读，未记录失败时为 `{}`。

## Reset、阶段与四种时钟

正式 Python 使用 `pvz_gym_env_reset_v2`。参数包括关卡、周目、可选随机种子、是否停在选卡、自动选卡、卡类型列表、确定性暂停和超时。reset 返回观测是否可用、状态、实际 clock 和种子支持/应用结果。旧 `env_reset` 仅保留 ABI 兼容。

Python 脚本选卡模式先 reset 到 SeedSelection，使用独立选卡动作域，再以 `env_wait_for_battle` 接续同一准备好的局；它不重新 reset。教程/特殊发牌关型可跳过交互选卡。细节见[选卡流程](default_adventure_decks.md)及[Reset 生命周期](reset_lifecycle.md)。

| 模式与入口 | 时间语义 |
| --- | --- |
| stepped / `env_step` | 动作后推进确定性暂停控制的游戏 ticks |
| headless / `env_step_headless` | 决策间保持暂停，短暂接管主循环并取消正常帧等待 |
| continuous / `env_step_continuous` | 不暂停，目标是距上次观测至少经过指定 ticks，延迟会增加实际时长 |
| turbo / `env_step_turbo` | resident 控制原版逻辑推进，并处理 intro/选卡/终局与渲染抑制 |

ticks 是目标量，不是可以代填的观测值。返回实际 `game_clock/ticks_advanced`，终局或教程可能不足目标，Continuous 可能超出；intro updates 也不能当作战斗 ticks。

RuntimePhase 与 raw GameUI 不同：

```text
0 Unknown, 1 Offline, 2 Loading, 3 Menu, 4 LevelIntro, 5 Dialogue,
6 SeedSelection, 7 Battle, 8 Lost, 9 Award, 10 Credit, 11 Challenge,
12 LevelComplete
```

`advance_phase(expected_phase,updates)` 是阶段推进接口；`read_runtime_snapshot` 提供带 identity 的运行时快照。原生 RNG 支持返回 0（不支持）或 2（完整），不猜测未知版本 seed 偏移。

## 单次执行与缓冲扩容

reset/step 是有副作用的调用，结果较大时**不能重新执行来扩容**。使用 [TransitionResultBuffer](../native/src/api/transition_result_buffer.h)：

1. 在执行 reset/step 的同一 OS 线程调用 `pvz_gym_begin_result_request()`，取得非零 uint64 ID。
2. 调用一次变更接口。成功返回字节；容量不足返回负的所需长度。
3. 扩大缓冲后仅调用 `pvz_gym_read_result_request(id,buf,capacity)`，不再次 reset/step。
4. 成功读取消费结果；新的 begin 使上一请求失效，其他线程不能读这个线程的 pending payload。

普通只读序列化接口可直接扩容重读。Python bridge 已实现两类路径；自定义绑定应保留这一区分。缓冲内容可含二进制零，按返回长度取 `buf.raw`，不能用 C 字符串截断。

## JSON、PVZB 与共享内存

reset/step 的输出由 [wire_result.h](../native/src/api/wire_result.h)选择格式，函数名称中的 `json` 不足以判断变更结果的实际传输。三种方式对应同一结构体事实：

- JSON：UTF-8 序列化。
- PVZB：前缀 `PVZB\x01`，little-endian 带类型树；null、bool、int64、float64、字符串、数组和 map。读取器检查长度、嵌套、重复 key 和尾部多余字节。
- Shared memory：返回 `PVZM` 加描述符，指向 Windows named mapping；头部 uint64 revision、uint32 length，后面是 PVZB 数据。发布 payload 后才发布 revision；Python 读取前后核对 revision/length，拒绝被更新的结果。映射/结果大小上限 16 MiB。

共享结果 mapping 与 remote game 的 shared snapshot 是不同机制。Python 解码见 [wire.py](../python/pvz_gym/bridge/wire.py)，原生实现见 [binary_serializer.h](../native/src/api/binary_serializer.h) 与 [shared_result.cpp](../native/src/api/shared_result.cpp)。

## 快照、动作与终局质量

StepResult 包含 observation、events、action result、下一合法 mask、状态/clock、自动收集计数，以及 `observation_quality/last_confirmed_game_clock/ticks_advanced/ticks_advanced_exact/replay_eligible`。它是同一状态转移的权威结果，不能用下一次独立读取拼回过去的一步。

required memory read 失败，或组装前后 Board、clock、scene、phase 变化，拒绝完整观测。terminal Board 已释放时可用最后确认实体快照通知结束，但明确标为 terminal_fallback、duration 不精确、不可入 Replay；进程退出或读失败不能伪装成胜负。详见 [Boss 与质量契约](boss_projectile_observation_contract.md)。

战斗编码见[动作契约](action_space.md)。mask 由原生 [action_validator.cpp](../native/src/action/action_validator.cpp)构建，提交仍重新验证实例、阳光、冷却和格子，不能把早先的 mask 当作永久授权。选卡是独立接口组，含查询、mask、选植物/模仿者、撤下及确认；不是额外塞入 757 战斗动作。

## 原生实现位置与同步

| 实现 | 职责 |
| --- | --- |
| [process_manager.cpp](../native/src/process/process_manager.cpp)、[memory_io.cpp](../native/src/process/memory_io.cpp) | 绑定进程、版本化内存访问与互斥 |
| [main_loop_barrier.cpp](../native/src/process/main_loop_barrier.cpp) | 原游戏主循环安全点与状态修改屏障 |
| [resident_executor.cpp](../native/src/runtime/resident_executor.cpp)、[resident_code.h](../native/src/runtime/resident_code.h) | x86 常驻 mailbox、命令确认、owner 与停止回收 |
| [native_runtime.cpp](../native/src/runtime/native_runtime.cpp)、[turbo_runtime.cpp](../native/src/runtime/turbo_runtime.cpp) | 配置、快照与 Turbo 生命周期 |
| [observation_builder.cpp](../native/src/observation/observation_builder.cpp)、[zombie_decoder.cpp](../native/src/observation/zombie_decoder.cpp) | 同步读取、完整性与原始实体解码 |
| [c_api.cpp](../native/src/api/c_api.cpp)、[runtime_c_api.cpp](../native/src/api/runtime_c_api.cpp) | C ABI 和结果请求边界 |
| [level_reset_code.h](../native/src/process/level_reset_code.h) | PreNewGame、图鉴与原版待释放列表回收 |

ProcessManager 互斥保护连接和原生状态；游戏线程修改使用主循环安全点，不能仅靠 Python 锁。resident owner 使用命令/ack 交接，在 owner 无法确认停止时保留隔离状态和远程内存，避免释放仍在执行的代码或启动第二个竞争 owner。Board 回收调用原版安全删除，具体 1065 地址、5 秒等待与超时内存处理见[Reset 文档](reset_lifecycle.md)。

`native/vendor/pvztoolkit` 仅保留六个必要实现及头文件，`native/vendor/miniz` 提供压缩；构建不依赖完整 Toolkit 或 FLTK。

## 验证边界

CTest 覆盖动作编码、实体事实、事件、内存失败、结果缓冲与快照质量；Win32 另执行生成的 x86 reset/resident/Board 批量推进/安全点代码。它们不要求游戏进程，不能代替实机；离线和实机检查命令见[开发文档](development.md)。
