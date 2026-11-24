# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Env metadata and step results, independent of how the env is served."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from lerobot.utils.feature_spec import FeatureSpec

# What an action commands: joint targets, end-effector deltas, or end-effector poses.
CONTROL_MODES = ("position", "delta", "eef_pose")


@dataclass(frozen=True)
class EnvDescriptor:
    """What an env exposes: observation and action features, control mode, fps and tasks.

    Args:
        sim_type (`str`):
            Integration that runs the env, e.g. `libero`.
        semantics (`str`):
            Name of the observation and action conventions, e.g. `libero-delta-v1`.
        features (`tuple[FeatureSpec, ...]`):
            Observation features, with LeRobot keys.
        action_feature (`FeatureSpec`):
            The action: one dimension, with ordered component names.
        control_mode (`str`):
            What an action commands: `position`, `delta` or `eef_pose`.
        fps (`float`):
            Control rate.
        tasks (`dict[str, list[int]]`):
            Task IDs of each suite.
        max_episode_steps (`int`):
            Steps before an episode is truncated.
        robot_type (`str`, *optional*, defaults to `""`):
            Body a sim robot records in its dataset.
        config (`dict[str, Any]`, *optional*):
            The `SimConfig` the env was built with. A client checks it against its own settings.
    """

    sim_type: str
    semantics: str
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec
    control_mode: str
    fps: float
    tasks: dict[str, list[int]]
    max_episode_steps: int
    robot_type: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid environment metadata."""
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control_mode {self.control_mode!r}; expected one of {CONTROL_MODES}")
        if not np.isfinite(self.fps) or self.fps <= 0 or self.max_episode_steps <= 0:
            raise ValueError("FPS and episode limit must be positive")
        if len(self.action_feature.shape) != 1 or not self.action_feature.names:
            raise ValueError("Actions require ordered component names")
        if len({feature.name for feature in self.features}) != len(self.features):
            raise ValueError("Duplicate observation features")

    def observation_space(self) -> spaces.Dict:
        """Return the observation space of one env: RGB images as uint8, other features unbounded."""
        return spaces.Dict(
            {
                f.name: spaces.Box(0, 255, f.shape, np.uint8)
                if f.kind == "rgb"
                else spaces.Box(-np.inf, np.inf, f.shape, np.dtype(f.dtype))
                for f in self.features
            }
        )

    def action_space(self) -> spaces.Box:
        """Return the action space of one env: unbounded float32."""
        return spaces.Box(-np.inf, np.inf, self.action_feature.shape, np.float32)

    def same_schema(self, other: "EnvDescriptor") -> bool:
        """Return whether `other` has the same observation features, action feature and semantics."""
        return (
            self.features == other.features
            and self.action_feature == other.action_feature
            and self.semantics == other.semantics
        )


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0102 应用装饰器 dataclass(frozen=True)，它会处理紧接着的函数或类。
@dataclass(frozen=True)
# 中文 L0103 一个 StepResult 对象保存整批环境刚重置或刚执行一步后的结果。
class StepResult:
    """Result of one reset or step of a batch of envs. Arrays have the batch dimension first.

    Args:
        obs (`dict[str, np.ndarray]`):
            Observations, with LeRobot keys. A finished env keeps its terminal observation.
        task (`tuple[str, ...]`):
            Task description of each env.
        reward (`np.ndarray`):
            Reward of each env, `float32`. Zero once its episode has finished.
        terminated (`np.ndarray`):
            Whether each env reached a terminal state, `bool`.
        truncated (`np.ndarray`):
            Whether each env's episode was cut short, e.g. by its step limit, `bool`.
        is_success (`np.ndarray`):
            Whether each env completed its task, `bool`.
        elapsed_steps (`np.ndarray`):
            Steps since the last reset, `int64`.
        metrics (`dict[str, np.ndarray]`, *optional*, defaults to `{}`):
            What the simulator reports besides success, one `float32` value per env, e.g. a progress score.
            An integration fills it from its own step info. LeRobot records the last value of each episode
            and never aggregates it, so a simulator chooses its own names and units.
    """

    # 中文 L0127 观测字典的每个 NumPy 数组都以环境批次为第 0 轴。
    obs: dict[str, np.ndarray]
    # 中文 L0128 声明“task”（任务）的类型为 tuple[str, ...]。
    task: tuple[str, ...]
    # 中文 L0129 声明“reward”（单步奖励）的类型为 np.ndarray。
    reward: np.ndarray
    # 中文 L0130 声明“terminated”（任务终止标志）的类型为 np.ndarray。
    terminated: np.ndarray
    # 中文 L0131 声明“truncated”（截断标志）的类型为 np.ndarray。
    truncated: np.ndarray
    # 中文 L0132 成功标志是布尔 NumPy 数组，原评测有一个环境，形状为 [1]。
    is_success: np.ndarray
    # 中文 L0133 声明“elapsed_steps”的类型为 np.ndarray。
    elapsed_steps: np.ndarray
    # 中文 L0134 声明“metrics”的类型为 dict[str, np.ndarray]，调用 field，用它的结果作为初始值。
    metrics: dict[str, np.ndarray] = field(default_factory=dict)

    # 中文 L0136 定义方法 frames。
    def frames(self) -> tuple[np.ndarray, ...]:
        # 中文 L0137 原文档中的说明或续行。
        """Return one frame per env from the first RGB observation, including terminal frames."""
        # 中文 L0138 调用 next，用它的结果给 “key” 赋值。
        key = next((k for k, v in self.obs.items() if v.dtype == np.uint8 and v.ndim == 4), None)
        # 中文 L0139 条件 key is None 成立时进入下方分支。
        if key is None:
            # 中文 L0140 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("This environment exposes no RGB observation")
        # 中文 L0141 调用 tuple，用它的结果作为当前调用的返回值。
        return tuple(frame.copy() for frame in self.obs[key])

    # 中文 L0143 把常用辅助数据放到符合 Gymnasium 约定的 info 字典。
    def info(self) -> dict[str, Any]:
        # 中文 L0144 原文档中的说明或续行。
        """Return the Gymnasium `info` dict for this step."""
        # 中文 L0145 创建字典，用键查找对应的值作为当前调用的返回值。
        return {
            # 中文 L0146 按 is_success 键公开成功数组，rollout 据此记录本步成功。
            "is_success": self.is_success,
            # 中文 L0147 字典项“task”（任务），冒号右边给出它的值。
            "task": self.task,
            # 中文 L0148 字典项“elapsed_steps”，冒号右边给出它的值。
            "elapsed_steps": self.elapsed_steps,
            # 中文 L0149 字典项“metrics”，冒号右边给出它的值。
            "metrics": self.metrics,
        # 中文 L0150 闭合上方尚未结束的括号。
        }
