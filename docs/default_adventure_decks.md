# 冒险选卡规则与入口

课程配置覆盖 50 关 × 2 周目，不代表已经验证全部关卡胜率。主卡、候选、新可用植物、僵尸教学与关型只在[逐关教学卡组](adventure_curriculum.md)维护；本页说明实际选卡流程，避免另一份手工表格与源码不同步。

## GUI 与配置 API

运行 `run_workbench.bat`，在训练参数中选择关卡/周目及卡组策略。单关卡组面板默认 `fixed`，混合训练提供对应策略与卡槽打乱选项。预设预览只展示所选关卡，不改变正在运行的训练局。LIVE 的“验证选卡”作用于已绑定游戏；正常 RL 训练使用私有实例。

| 配置值 | 行为 |
| --- | --- |
| `fixed` | 逐关主卡顺序，再用课程候选补位 |
| `fixed_shuffled` | 同一固定植物集合，仅打乱可移动卡槽 |
| `adaptive` | 根据真实关前僵尸预览预留地形/经济/输出，再加入应对组合 |
| `adaptive_shuffled` | 同一针对流程，之后打乱可移动卡槽 |
| `custom` | 用户优先级，结合实际可用项与 fallback |
| `native` | 原生 reset 指定卡/自动选卡路径，不经过 Python 课程规划器 |

`EnvConfig` 的基础默认是 `native`，GUI 的 `fixed` 是入口设置，不能把两者混称为同一个默认。当前训练 CLI 有 `--seed-types`，没有 `--seed-selection-mode`；需要课程模式或混合关池时使用 GUI 或配置 API。

离线查询一周目 3-4 主卡，无需启动游戏：

```powershell
.venv/Scripts/python.exe -c "from pvz_gym.default_decks import default_deck_priority, PLANT_NAMES; print([PLANT_NAMES[i] for i in default_deck_priority(24, 1)])"
```

配置 API 示例（保存为自己的脚本；多进程启动放在主程序保护内）：

```python
from train.common import EnvConfig
from train.dqn.config import DoubleDQNConfig
from train.dqn.train import train
from train.runtime import TrainingRuntimeConfig
from pvz_gym.env_file import resolve_game_paths
from pvz_gym.runtime import RuntimeConfig

if __name__ == "__main__":
    config = DoubleDQNConfig(
        env=EnvConfig(
            level_id=24,
            playthrough=1,
            simulation_mode="turbo",
            seed_selection_mode="fixed",
            runtime=RuntimeConfig(executor="resident", transport="binary", snapshot="shared"),
        ),
        runtime=TrainingRuntimeConfig(game_executable=str(resolve_game_paths().executable)),
        device="cuda",
        total_steps=10000,
    )
    train(config, run_dir="runs/fixed-deck-example")
```

该示例读取根目录 `.env` 的游戏路径，并显式设置 Turbo 后端及私有实例来源；运行前仍需按 README 完成检查和模板准备。原项目独立的 `select_default_deck.py` 没有迁入，本仓库不提供它的 `--prepare/--battle` 命令。

## 固定选卡、解锁与戴夫

`default_deck_priority(level_id, playthrough)` 查询课程主卡，`deck_profile` 还给出候选、可用集合与教学说明；`plan_fixed_deck` 才结合现场选卡状态规划最终卡组。

一周目按本关开始前已获得的植物：通关奖励下一关首次可用。课程固定模式即使看到晚期 profile 中已解锁的卡，也不加入未来奖励。二周目使用基础植物集合，不重新按一周目顺序解锁；升级卡/模仿者的购买状态与使用依赖不作通用默认假设。

先保留游戏锁定、不能撤下的戴夫卡，再依次取实际可选的主卡与候选，数量由现场 `required_count` 决定。白天主动选睡眠蘑菇时需考虑咖啡豆依赖；戴夫锁入的卡保留并给出缺口提示。实际可用卡不足槽位时明确报错，不强制解锁或静默混入未来植物。

教程、传送带、保龄球、打僵尸、砸罐子和僵王等没有普通交互选卡的关型使用游戏预置/发牌。表中存在课程记录不表示可以自由选择十张。固定选卡顺序也不是种植顺序或已证明最优策略。

## 自动针对与卡槽打乱

`counter_rules` 定义按实际预览出现的威胁及可替代的植物组合；需要咖啡豆的白天蘑菇按完整组合规划。先确保睡莲/花盆、经济和可用输出，再按威胁加入组合，最后用场景优先级及基础植物补满。无僵尸预览时保留提示并使用 fallback，不凭空推断该关阵容。

该模式与固定课程卡组不是同一算法；其场景 fallback 与教程优先级也不能代替固定模式的逐关主卡表。`deck_counter_gaps` 是选卡缺口提示，不能证明战斗策略已经正确使用针对卡。

乱序保留游戏不可移动的锁卡，仅随机排列可移动集合。Gym 使用 episode RNG，动态选卡结果记录在 reset info 与训练日志；恢复时保留来源配置与关卡池，不能只靠某次预览列表重建完整课程。

## 同步与验证

实现见 [default_decks.py](../python/pvz_gym/default_decks.py)、[adventure_curriculum.py](../python/pvz_gym/adventure_curriculum.py)、[Gym reset](../python/pvz_gym/env.py) 和 [GUI 卡组面板](../workbench/gui/deck_panel.py)。

修改课程源码后在根目录生成表格：

```powershell
.venv/Scripts/python.exe -m scripts.export_adventure_curriculum
```

生成器负责同步配置表；教学解释仍需人工核对，不是游戏通关证据。雾过滤、Boss 与动作有效性分别见[观测](observation.md)和[动作契约](action_space.md)。实机检查入口见[开发文档](development.md)。
