# Reset lifecycle

`PvZClient` owns Python-side reset orchestration while native code performs the
actual game transitions. Reset must handle intro/dialogue, seed selection,
battle terminal states and Award UI without fabricating game state.

Award UI is left through the original AwardScreen continue path before entering
the next prebattle lifecycle. Deterministic pause/tick control is re-established
only after the new episode reaches the expected playable phase.

图鉴是 `DIALOG_ALMANAC=3`，属于独立模态 Dialog；`PreNewGame` 替换 Board
不会关闭它。reset 和工作台选关在跳转前通过原版 `LawnApp::KillDialog`
关闭图鉴，避免旧图鉴继续覆盖新一轮训练场景或维持模态暂停。
原生直接选关也在主循环屏障内、`PreNewGame` 前执行该清理，再由原版
`ProcessSafeDeleteList` 回收已移除的 widget。图鉴清理入口沿用已核验的
1.2.0.1065 `KillDialog` 地址，其他版本不猜测入口地址。

## 原生重开与工作台响应

远程 `PreNewGame` 必须在主循环屏障内执行，避免 Board 与窗口线程并发销毁。
屏障停在 `UpdateAppStep` 外，此时 `mUpdateAppDepth` 为 0。原版
`SafeDeleteWidget` 会将旧 Board 按该深度放入待释放列表；之后深度为 1 的正常
更新不能回收它。连续重开会因此耗尽 32 位地址空间。

`level_reset_code.h` 在重开前后调用原版 `ProcessSafeDeleteList`：先清理此前
积压的对象，再清理本次移除的 Board。没有手动释放 Board 或清零游戏指针。
目前已核验的回收入口为 **1.2.0.1065** 的 `0x00551B20`，调用约定为栈上传入
LawnApp 指针、`ret 4`；`PreNewGame` 仍使用 ESI 传入 LawnApp、`ret 8`。
vendor 中保留的其他版本重开路径不使用这个回收地址，也不属于本仓库首版完整训练支持。

重开前退出持久无头屏障，停止界面服务线程时不持有 ProcessManager 互斥锁。
重开使用局部 RAII 屏障，远程等待最多 5 秒；失败返回 `ActionTimeout`（1019），
释放屏障并恢复窗口线程。超时尚未退出的远程代码不能立刻释放，否则线程
恢复后会执行已释放内存；该小块内存在游戏进程退出时释放。旧版无超时的
`asm_code_inject` 辅助接口仍保留，这里的 5 秒限制描述 reset 远程等待，不代表所有注入调用都有相同限制。

工作台定时器及其关卡修复包装层均在原生工作线程活动时跳过 DLL 查询。
游戏退出由工作线程检测，空闲时定时器继续自动附加游戏。
桌面绑定按线程缓存：已在 Default 桌面时不开句柄；需要切换时保留一个
句柄，线程结束前恢复原绑定并关闭句柄。

## 无游戏回归验证

- `thread_desktop`：默认桌面和切换桌面各检测 10,000 次，检查句柄数稳定，
  并检查恢复绑定及释放句柄。
- Win32 `level_reset_code`：执行实际生成的 x86 指令，连续重开 2,000 次，
  交替模拟图鉴打开/未打开，检查 `KillDialog(3)` 的调用约定及其先于
  `PreNewGame` 执行，并检查旧 Board 待释放列表清空；阻塞远程线程验证
  超时和恢复后的代码有效性。
- `test_workbench_dialog_cleanup.py`：检查两种选关接口均在跳转前关闭图鉴，
  场景切换时清理图鉴，界面未切换时仍允许用户保留已打开的图鉴。
- `test_daemon_native_contention.py`：执行最终包装后的工作台定时器，检查各类
  工作线程运行时不进入原生接口，并检查空闲时仍自动检测附加。

这些测试不等同于真实游戏的长时间训练验证。替换 DLL 后需要重启工作台；
旧进程已经泄漏的桌面句柄也只能随旧进程退出回收。
