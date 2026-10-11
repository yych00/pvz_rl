# 安装与开发

普通用户采用 `scripts/setup.ps1` 安装依赖和对应版本的 Release DLL。默认创建 `.venv`、安装 PyTorch 2.5.1 CUDA 12.1 和 `pip install -e ".[train]"`；无 GPU 的用户可用 `-TorchIndex cpu`。Python 支持 3.10–3.12，推荐安装器默认的 64 位 3.12。

安装脚本使用已安装的 Python，不下载 Python 本身。默认执行 `py -3.12`；请在 Python 安装时包含 `py` 启动器，或用 `-PythonExe C:/Python312/python.exe` 指定解释器。已有 `.venv` 会复用。

建议使用 CUDA 版 PyTorch 跑训练。CPU 模式下模型计算与多环境采样争用 CPU，可能导致训练或 GUI 长时间无响应，表现为“卡死”，不建议用于完整训练。CPU 安装选项主要用于离线检查和有限的环境调试。

安装后确认 GPU 可用：

```powershell
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CUDA 版本应显示 `12.1`，`torch.cuda.is_available()` 应为 `True`。只有安装 CUDA wheel 而驱动/GPU 不可用时，自动设备选择仍会回退到 CPU；训练命令添加 `--device cuda` 可直接报告 GPU 不可用。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -NativeArchive C:/Downloads/pvz-rl-native-0.1.0-windows-x64.zip
```

归档同目录必须有 `.zip.sha256`，安装器同时检查整体 SHA256、DLL SHA256、包版本、架构和运行时接口。

自动下载从 `origin` 的 GitHub 地址识别仓库，使用与 Python 包版本一致的标签（如 `v0.1.0`）。公开 Release 无需认证；私有 Release 在匿名请求不可访问时，尝试 Git Credential Manager 中已有的 GitHub 登录，也可显式使用环境变量 `GH_TOKEN` / `GITHUB_TOKEN`。登录账号须有仓库访问权限，令牌只在请求时使用，不写入 `.env` 或下载文件，并在跨主机重定向时移除。

从网页下载源码 ZIP 时没有 Git remote，需要同时下载对应的 native ZIP 与 `.sha256`，使用上面的 `-NativeArchive` 命令，或从源码构建。尚未发布匹配 Release 时同样使用本地归档或 `-BuildNative`。`-ReleaseBaseUrl` 供自定义公开下载目录使用，不负责私有 GitHub 认证。

## 源码构建

安装 Visual Studio 2022/Build Tools 的“使用 C++ 的桌面开发”、Windows SDK 和 CMake。构建仅依赖本仓库内源码及 Windows 系统库：

```powershell
scripts/build_native.bat x64
```

产物复制到 `bin/pvz_gym_native.dll`；生产 DLL 使用静态 MSVC runtime。不要在 Workbench 或训练持有 DLL 时覆盖它，先退出这些进程。开发者也可手动执行 `cmake -S native -B build/native-x64 -A x64` 与 `cmake --build build/native-x64 --config Release`。

## 离线测试

```powershell
.venv/Scripts/python.exe -m pip install -e ".[train]"
.venv/Scripts/python.exe -m unittest discover -s tests/smoke -t . -v
.venv/Scripts/python.exe -m unittest discover -s tests/unit -t . -v
ctest --test-dir build/native-x64 -C Release --output-on-failure
```

`scripts/build_native.bat Win32` 构建额外的 x86 注入代码测试；其 DLL 不用于 64 位 Python。`tests/integration/manual` 的真实游戏测试需显式运行，不进入 CI。

## 实机验收

使用自己的完整兼容游戏目录，先完成 `doctor` 和训练存档准备，再执行：

```powershell
.venv/Scripts/python.exe -m tests.integration.manual.public_acceptance --game-exe "C:/Games/PvZ/PlantsVsZombies.exe" --device cuda
.venv/Scripts/python.exe -m tests.integration.manual.workbench_acceptance --game-exe "C:/Games/PvZ/PlantsVsZombies.exe"
```

第一项覆盖五种场景、单/双环境 Turbo、完整 checkpoint 与 Replay 恢复、固定种子评估；第二项在真实 Tk 事件循环中调用 GUI 的存档准备和训练入口，并核对模型 Q 帧与实际 GDI 绘制。它不模拟鼠标键盘，也不替代屏幕外观检查。两个入口都核对原游戏、个人存档和已有模板的文件哈希。报告写入被忽略的 `build/`，副本与训练结果写入 `runtime/`、`runs/`；报告含本机路径，不进入公开发布包。

## 发布包

提交审阅后的源码，再运行 `.venv/Scripts/python.exe -m scripts.package_release`。生成 `dist/pvz-rl-native-0.1.0-windows-x64.zip` 和 SHA256 sidecar，包内记录源码 commit。发布必须同时提供该 commit 的完整源码，许可证和第三方说明随 DLL 提供。

GitHub CI 在 Windows 上运行离线检查；推送与包版本一致的 `v0.1.0` 标签才触发 Release workflow。源码归档通过 `git archive` 从已跟踪文件生成，游戏、本地存档、模型和构建数据不进入发布包。
