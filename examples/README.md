# 最小示例

安装并准备游戏/存档后，在根目录运行 `python examples/minimal_env.py`，创建独立实例并执行一次 reset/NOOP/close。

训练：`python -m train.dqn.train --level-id 4 --total-steps 10000 --simulation-mode turbo --num-envs 1 --run-dir runs/example --device cuda`。

评估：`python -m train.dqn.evaluate runs/example/latest.pt --seed-count 4 --simulation-mode turbo --device cuda`；将路径替换为实际保存的 checkpoint。更多恢复、评估和奖励参数见 `--help` 与 `docs/dqn_reference.md`。
