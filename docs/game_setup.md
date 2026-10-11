# 游戏版本、目录与存档

## 支持版本

首版完整训练目标为 Windows 10/11 x64 和 PvZ **retail 1.2.0.1065** 的已验证布局。EXE 是 x86；Python 和 `pvz_gym_native.dll` 使用 x64。1.0.0.1051 在原项目部分代码中可识别，但 Turbo 和若干运行时路径依赖 1065，不能视为完整训练支持。

中文/英文标题、目录名和文件属性不足以确认兼容性。工具检查 PE machine、image base、时间戳 `0x4A37D6AF`，以及 Board 更新、Board 回收和 Effect 回收入口的文件代码签名。实际启动时原生层继续检查 `PVZ_1_2_0_1065_EN` 与既有执行签名。静态签名检查只覆盖已知关键入口，不能证明任意改版都兼容；未知布局、GOTY 和检查失败的 EXE 拒绝进入私有训练流程。

## 放置完整游戏

```text
仓库根目录/
  game/pvz-1.2.0.1065/
    PlantsVsZombies.exe
    main.pak
    bass.dll                 # 若原游戏附带，保留
    properties/              # 若原游戏附带，保留
    …其余原游戏文件
```

资源展开版保留全部资源目录，包括 `images`、`sounds`、`reanim`。诊断会检查资源是否存在；最终完整性仍以游戏成功加载为准。不要混用不同版本的 EXE 与资源。

游戏可在仓库外。复制 `.env.example` 为 `.env`，例如：

```dotenv
PVZ_GAME_EXECUTABLE=C:/Games/PvZ/PlantsVsZombies.exe
PVZ_TRAINING_SAVE_TEMPLATE=local/save_template
PVZ_INSTANCE_ROOT=runtime/instances
```

相对路径以本仓库根目录为基准，与启动命令所在目录无关。环境变量优先于 `.env`，显式参数优先于环境变量。GUI 选择 EXE 后可保存配置；选择的是 EXE，其所在目录必须包含完整资源。

## 首次准备

先运行 `python -m scripts.doctor --skip-save`，再运行 `python -m scripts.prepare_training_save`，最后运行 `python -m scripts.doctor`。命令均使用已安装本仓库包的 Python。GUI 的“准备训练存档”调用同一实现。

生成 `local/save_template/users.dat`、`user1.dat` 与 `template.json`。用户索引采用 v14，专用 `PvZRL` profile 采用 retail v12、820 字节、零禅境花园记录。生成冒险已通关状态、商店升级植物及模仿者、十卡槽和泳池/屋顶清理车；不复制用户个人信息，不伪造全成就，也不提供额外阳光、免费种植或植物伤害修改。一周目固定课程选卡继续过滤本关尚未获得的植物；`native` 与针对模式的区别见[选卡规则](default_adventure_decks.md)。

文件格式参考：[retail PlayerInfo 序列化](https://github.com/ruslan831/PlantsVsZombies-decompilation/blob/master/Lawn/System/PlayerInfo.cpp)、[商店项目枚举](https://github.com/Patoke/re-plants-vs-zombies/blob/main/ConstEnums.h)。实现只使用格式信息，不嵌入游戏资源或第三方存档。

重复准备会校验并复用完全一致的模板。若目录含现有存档、模板被改动/损坏，或与原游戏、个人存档、实例池重叠，则报错并保留文件。改用新的空目录，不会自动覆盖旧存档。

## 哪些目录会产生什么

| 路径 | 用途 | Git |
|---|---|---|
| `game/pvz-1.2.0.1065/` 或外部游戏目录 | 用户提供的原游戏 | 忽略本地游戏 |
| `local/save_template/` | 自动生成的训练模板，只读来源 | 忽略 |
| `runtime/instances/` | 私有训练 EXE、资源副本及管理信息 | 忽略 |
| `runtime/evaluation/` | 手动评估使用的独立实例池 | 忽略 |
| `%ProgramData%/PopCap Games/PvZGym-<标识>/userdata` | 游戏实际写入的私有存档 | 仓库外 |
| `runs/<名称>/` | checkpoint、Replay、配置、日志 | 忽略 |
| `bin/`、`build/`、`.venv/`、`.env` | DLL、构建缓存、Python 环境、个人配置 | 忽略 |

原版 `%ProgramData%/PopCap Games/PlantsVsZombies/userdata` 不是训练写入目标。Reset 仅清理已绑定私有 profile 的冒险残局并保留可恢复隔离记录。游戏副本的产品名、互斥名和窗口标题会调整，源 EXE 保持不变。

`doctor` 默认只读：不会启动或附加到个人游戏。预检查通过后，训练才创建并启动自己的实例。游戏与资源不放进 Git、Release 或源码包。
