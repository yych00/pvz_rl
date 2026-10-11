"""Trainer metadata used by the Workbench without importing trainer backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from train.dqn.config import DEFAULT_EXPLORATION_TYPE_WEIGHTS
from train.dqn.exploration import (
    DEFAULT_EXPLORATION_MODE, DEFAULT_EXPLORATION_RATIO, EXPLORATION_MODE_LABELS,
    format_exploration_ratio,
)


@dataclass(frozen=True)
class TrainerParameter:
    key: str
    label: str
    kind: str
    default: Any
    help: str = ""
    choices: tuple[str, ...] = ()

    def parse(self, raw: str) -> Any:
        text = str(raw).strip()
        if self.kind == "int":
            return int(text)
        if self.kind == "float":
            return float(text)
        if self.kind == "bool":
            if text not in ("True", "False"):
                raise ValueError(f"{self.label} must be True or False")
            return text == "True"
        if self.kind == "choice":
            if text not in self.choices:
                raise ValueError(f"{self.label} must be one of {', '.join(self.choices)}")
            return text
        return text


@dataclass(frozen=True)
class TrainerSpec:
    key: str
    label: str
    available: bool
    description: str
    parameters: tuple[TrainerParameter, ...]
    unavailable_reason: str = ""


TRAINER_SPECS: tuple[TrainerSpec, ...] = (
    TrainerSpec(
        key="random_legal",
        label="Random Legal",
        available=True,
        description="Environment integration baseline. Samples only from the native legal-action mask.",
        parameters=(
            TrainerParameter("episodes", "Episodes", "int", 100),
            TrainerParameter("max_steps", "Max steps / episode", "int", 4000),
        ),
    ),
    TrainerSpec(
        key="double_dqn",
        label="Double DQN",
        available=True,
        description="Masked Double DQN with replay, target network and shared reward settings.",
        parameters=(
            TrainerParameter("total_steps", "Total steps", "int", 250000),
            TrainerParameter("learning_rate", "Learning rate", "float", 0.0001),
            TrainerParameter("gamma", "Gamma / 100 ticks", "float", 0.999,
                             help="Discount per game second. Resume converts the saved value; edit it to change the new training objective."),
            TrainerParameter("batch_size", "Batch size", "int", 128),
            TrainerParameter("n_step", "N-step return", "int", 3,
                             help="Decision steps per TD return: 1, 3 or 5. New replay supports changing n on resume; old replay without episode boundaries is rebuilt."),
            TrainerParameter("replay_capacity", "Replay capacity", "int", 200000),
            TrainerParameter("replay_scheme", "回放采样方案", "choice", "per",
                             choices=("uniform", "per"),
                             help="uniform: 均匀随机回放; per: 基于 TD 误差的优先经验回放。可在暂停时动态切换。"),
            TrainerParameter("per_alpha", "PER 指数 alpha", "float", 0.6,
                             help="优先级指数：0 为均匀，1 为完全按误差加权。"),
            TrainerParameter("per_beta_start", "PER 纠偏 beta 起点", "float", 0.4,
                             help="重要性采样纠偏起始权重，线性退火至 1.0。"),
            TrainerParameter("learning_starts", "Learning starts", "int", 10000),
            TrainerParameter("train_every", "Train every", "int", 6),
            TrainerParameter("auto_train_every", "自适应训练步频", "bool", True,
                             choices=("True", "False"),
                             help="开启后根据待更新任务积压动态调节 train_every，防止游戏反压卡死并榨干算力。"),
            TrainerParameter("auto_train_high_watermark", "升档避险阈值", "int", 16,
                             help="当待更新积压 >= 该阈值时，自动调大 train_every 避险减负。"),
            TrainerParameter("auto_train_low_watermark", "降档提速阈值", "int", 4,
                             help="当待更新积压 <= 该阈值时，自动调小 train_every 提速学习。"),
            TrainerParameter("training_architecture", "采样 / 训练架构", "choice", "actor_learner",
                             choices=("actor_learner", "legacy"),
                             help="actor_learner: 独立推理模型与后台训练；legacy: 原串行训练路径。"),
            TrainerParameter("policy_sync_interval", "策略同步间隔（更新数）", "int", 32,
                             help="按 Learner 更新计数发布权重；到 Policy Lag 上限时提前同步。"),
            TrainerParameter("max_pending_updates", "待训练更新上限", "int", 128,
                             help="达到上限暂停派发新动作；已在途结果仍会接收，训练额度不会丢弃。"),
            TrainerParameter("gradient_steps", "Gradient steps", "int", 1),
            TrainerParameter("evaluation_mode", "评估模式", "choice", "random",
                             choices=("random", "per_level", "off"),
                             help="单关默认 random（随机评估）；多关默认 per_level（逐关评估）；off：关闭评估。周目沿用训练。"),
            TrainerParameter("evaluation_per_level", "每关评估局数", "int", 4),
            TrainerParameter("evaluation_random_episodes", "随机评估总局数", "int", 20),
            TrainerParameter("level_sampling_mode", "关卡抽样模式", "choice", "squared_failure",
                             choices=("squared_failure", "bounded_difficulty", "uniform")),
            TrainerParameter("adaptive_level_sampling", "按近期胜率抽关", "bool", True,
                             choices=("True", "False")),
            TrainerParameter("level_sampling_window", "每关近期窗口（局）", "int", 40),
            TrainerParameter("level_sampling_update_every", "抽关更新间隔（局）", "int", 20),
            TrainerParameter("level_sampling_uniform_mix", "均匀抽样比例（0–1）", "float", 0.05),
            TrainerParameter("level_sampling_max_weight", "最大难度权重", "float", 4.0),
            TrainerParameter("target_update_interval", "Target update", "int", 2000),
            TrainerParameter("epsilon_start", "Epsilon start", "float", 0.75),
            TrainerParameter("epsilon_end", "Epsilon end", "float", 0.05),
            TrainerParameter("epsilon_decay_steps", "Epsilon decay steps", "int", 200000),
            TrainerParameter("exploration_mode", "探索方式", "choice",
                             EXPLORATION_MODE_LABELS[DEFAULT_EXPLORATION_MODE],
                             choices=tuple(EXPLORATION_MODE_LABELS.values())),
            TrainerParameter("exploration_ratio", "探索内 Q : 随机", "text",
                             format_exploration_ratio(DEFAULT_EXPLORATION_RATIO),
                             help="Q:random weights, default 50:50; epsilon controls total exploration."),
            TrainerParameter(
                "exploration_type_weights",
                "Explore NOOP,PLANT,SHOVEL,COB,INTERACT",
                "text",
                ",".join(f"{w:g}" for w in DEFAULT_EXPLORATION_TYPE_WEIGHTS),
                help="Family selection weights apply to random/Q guidance; neighbor anchors exclude shovel.",
            ),
            TrainerParameter("autosave_every", "Auto-save latest", "int", 10000),
            TrainerParameter("checkpoint_every", "Milestone every", "int", 50000),
            TrainerParameter("device", "Device", "choice", "auto", choices=("auto", "cuda", "cpu")),
            TrainerParameter("selected_action_forward", "加速 Q 学习", "bool", True,
                             choices=("True", "False")),

        ),
    ),

)


def all_trainer_specs() -> tuple[TrainerSpec, ...]:
    return TRAINER_SPECS


def get_trainer_spec(key: str) -> TrainerSpec:
    for spec in TRAINER_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def trainer_key_from_label(label: str) -> str:
    for spec in TRAINER_SPECS:
        if spec.label == label:
            return spec.key
    raise KeyError(label)


__all__ = [
    "TrainerParameter",
    "TrainerSpec",
    "TRAINER_SPECS",
    "all_trainer_specs",
    "get_trainer_spec",
    "trainer_key_from_label",
]
