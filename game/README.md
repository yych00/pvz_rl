# 游戏目录

本目录只发布这份说明，游戏程序和资源由用户自行提供。

默认位置：`game/pvz-1.2.0.1065/PlantsVsZombies.exe`。请复制完整游戏文件夹（可在 [游戏资源下载 - 植僵工具箱](https://pvz.tools/download/) 查找对应版本），保留 EXE、`main.pak`、原有 DLL、`properties` 等结构；资源展开版应完整保留 `images`、`sounds`、`reanim` 等目录。不要只复制 EXE，不要将其他版本改名为 1065。

外部目录也支持：在仓库根目录 `.env` 设置 `PVZ_GAME_EXECUTABLE=C:/Games/PvZ/PlantsVsZombies.exe`。详见 [版本检查与存档准备](../docs/game_setup.md)。

支持 retail 1.2.0.1065 的已验证布局，包括通过检查的对应中文翻译版。GOTY、1.0.0.1051 和未知改版不属于首版完整训练支持范围。

安装 Python 包和 DLL 后运行：

```powershell
.venv/Scripts/python.exe -m scripts.doctor --skip-save
.venv/Scripts/python.exe -m scripts.prepare_training_save
.venv/Scripts/python.exe -m scripts.doctor
```

工具使用独立游戏副本和 `PvZRL` 训练存档。此处的原游戏与个人存档保持原样；不要把存档模板放进游戏目录。
