#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0016 从模块 __future__ 导入 annotations。
from __future__ import annotations

# 中文 L0018 导入模块 logging。
import logging
# 中文 L0019 导入模块 os。
import os
# 中文 L0020 导入模块 re。
import re
# 中文 L0021 从模块 collections 导入 defaultdict。
from collections import defaultdict
# 中文 L0022 从模块 collections.abc 导入 Callable、Iterable、Mapping、Sequence。
from collections.abc import Callable, Iterable, Mapping, Sequence
# 中文 L0023 从模块 functools 导入 partial。
from functools import partial
# 中文 L0024 从模块 pathlib 导入 Path。
from pathlib import Path
# 中文 L0025 从模块 typing 导入 Any。
from typing import Any

# 中文 L0027 导入模块 gymnasium，简称 gym。
import gymnasium as gym
# 中文 L0028 导入模块 numpy，简称 np。
import numpy as np
# 中文 L0029 从模块 gymnasium 导入 spaces。
from gymnasium import spaces
# 中文 L0030 从模块 libero.libero 导入 benchmark、get_libero_path。
from libero.libero import benchmark, get_libero_path
# 中文 L0031 从安装的 hf-libero 包导入离屏渲染环境类。
from libero.libero.envs import OffScreenRenderEnv

# 中文 L0033 从模块 .utils 导入 RobotObservation、canonical_observation（本地名 _canonical_observation）、parse_camera_names。
from .utils import RobotObservation, canonical_observation as _canonical_observation, parse_camera_names

# 中文 L0035 创建列表，按给定顺序保留元素给 “__all__” 赋值。
__all__ = ["canonical_observation", "conventions", "env_fns"]

# 中文 L0037 调用 logging.getLogger，用它的结果给 “logger” 赋值。
logger = logging.getLogger(__name__)


# 中文 L0040 把形状为 […,4] 的 xyzw 四元数转换成三分量轴角表示。
def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    # 中文 L0041 原文档中的说明或续行。
    """Convert xyzw quaternions to axis-angle like robosuite's `quat2axisangle`: zeros near the identity."""
    # 中文 L0042 先把输入转换成 NumPy float32 数组。
    quat = np.asarray(quat, dtype=np.float32)
    # 中文 L0043 抽取四元数标量部 w，并把浮点误差可能超出的值限制在 arccos 的定义域。
    w = np.clip(quat[..., 3], -1, 1)
    # 中文 L0044 调用 np.sqrt，用它的结果给 “den” 赋值。
    den = np.sqrt(np.maximum(1 - w * w, 0))
    # 中文 L0045 调用 np.divide，用它的结果给 “scale” 赋值。
    scale = np.divide(2 * np.arccos(w), den, out=np.zeros_like(den), where=den > 1e-10)
    # 中文 L0046 调用 astype，用它的结果作为当前调用的返回值。
    return (quat[..., :3] * scale[..., None]).astype(np.float32)


# 中文 L0049 LIBERO 的标准观测转换，构造 8 维状态并按约定翻转相机图。
def canonical_observation(obs: Mapping) -> dict[str, np.ndarray]:
    # 中文 L0050 原文档中的说明或续行。
    """Map a LIBERO observation to LeRobot keys, rotating images by 180 degrees and building the 8-D state."""
    # 中文 L0051 调用 dict，用它的结果给 “adapted” 赋值。
    adapted = dict(obs)
    # 中文 L0052 从复制的字典取出原始的嵌套机器人状态。
    state = adapted.pop("robot_state", None)
    # 中文 L0053 条件 state is not None 成立时进入下方分支。
    if state is not None:
        # 中文 L0054 拼接末端位置 3、姿态轴角 3、左右夹爪位置 2，共 8 维。
        adapted["agent_pos"] = np.concatenate(
            # 中文 L0055 延续上面的语句，(state["eef"]["pos"], _quat_to_axisangle(state["eef"]["quat"]), state["gripper"]["qpos"]),。
            (state["eef"]["pos"], _quat_to_axisangle(state["eef"]["quat"]), state["gripper"]["qpos"]),
            # 中文 L0056 为“axis”提供这一项的值，注意它所在的参数表或外层表达式。
            axis=-1,
        # 中文 L0057 延续上面的语句，).astype(np.float32)。
        ).astype(np.float32)
    # 中文 L0058 调用通用仿真观测整理器，这里明确启用图像旋转。
    return _canonical_observation(adapted, flip_images=True)


# 中文 L0061 定义函数 _get_suite。
def _get_suite(name: str) -> benchmark.Benchmark:
    # 中文 L0062 原文档中的说明或续行。
    """Instantiate a LIBERO suite by name with clear validation."""
    # 中文 L0063 调用 benchmark.get_benchmark_dict，用它的结果给 “bench” 赋值。
    bench = benchmark.get_benchmark_dict()
    # 中文 L0064 条件 name not in bench 成立时进入下方分支。
    if name not in bench:
        # 中文 L0065 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    # 中文 L0066 调用 此行左侧的可调用对象，用它的结果给 “suite” 赋值。
    suite = bench[name]()
    # 中文 L0067 条件 not getattr(suite, "tasks", None) 成立时进入下方分支。
    if not getattr(suite, "tasks", None):
        # 中文 L0068 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Suite '{name}' has no tasks.")
    # 中文 L0069 引用已有的值 suite作为当前调用的返回值。
    return suite


# 中文 L0072 定义函数 _select_task_ids。
def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    # 中文 L0073 原文档中的说明或续行。
    """Validate/normalize task ids. If None → all tasks."""
    # 中文 L0074 条件 task_ids is None 成立时进入下方分支。
    if task_ids is None:
        # 中文 L0075 调用 list，用它的结果作为当前调用的返回值。
        return list(range(total_tasks))
    # 中文 L0076 调用 sorted，用它的结果给 “ids” 赋值。
    ids = sorted({int(t) for t in task_ids})
    # 中文 L0077 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for t in ids:
        # 中文 L0078 条件 t < 0 or t >= total_tasks 成立时进入下方分支。
        if t < 0 or t >= total_tasks:
            # 中文 L0079 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    # 中文 L0080 引用已有的值 ids作为当前调用的返回值。
    return ids


# LIBERO-plus perturbation variants encode the perturbation in the filename
# but on disk only the base `.pruned_init` exists — strip the suffix to match
# LIBERO-plus's own suite.get_task_init_states() (we reimplement it here so we
# can pass weights_only=False for PyTorch 2.6+ numpy pickles).
# 中文 L0087 调用 re.compile，用它的结果给 “_LIBERO_PERTURBATION_SUFFIX_RE” 赋值。
_LIBERO_PERTURBATION_SUFFIX_RE = re.compile(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")


# 中文 L0090 定义函数 _load_initial_states。
def _load_initial_states(path: Path) -> np.ndarray:
    # 中文 L0091 原文档中的说明或续行。
    """Load a task's initial states, from the converted `.npy` file next to `path` when there is one."""
    # 中文 L0092 调用 path.with_suffix，用它的结果给 “portable” 赋值。
    portable = path.with_suffix(path.suffix + ".npy")
    # 中文 L0093 条件 portable.exists() 成立时进入下方分支。
    if portable.exists():
        # 中文 L0094 调用 np.load，用它的结果作为当前调用的返回值。
        return np.load(portable, allow_pickle=False)
    # 中文 L0095 条件 os.environ.get("LEROBOT_SIM_REQUIRE_NUMPY") == "1" 成立时进入下方分支。
    if os.environ.get("LEROBOT_SIM_REQUIRE_NUMPY") == "1":
        # 中文 L0096 抛出 FileNotFoundError，中断当前路径并交给上层的异常处理。
        raise FileNotFoundError(f"Convert initial-state asset before serving: {portable}")
    # Only the original torch-pickle assets need torch; sim Docker images ship converted .npy files.
    # 中文 L0098 导入模块 torch。
    import torch

    # 中文 L0100 调用 np.asarray，用它的结果作为当前调用的返回值。
    return np.asarray(torch.load(path, weights_only=False))  # nosec B614


# 中文 L0103 定义函数 get_task_init_states。
def get_task_init_states(task_suite: Any, i: int, is_libero_plus: bool = False) -> np.ndarray:
    # 中文 L0104 用方括号按键或索引读取 task_suite.tasks给 “task”（任务） 赋值。
    task = task_suite.tasks[i]
    # 中文 L0105 调用 Path，用它的结果给 “filename”（文件名） 赋值。
    filename = Path(task.init_states_file)
    # 中文 L0106 调用 Path，用它的结果给 “root” 赋值。
    root = Path(get_libero_path("init_states"))

    # 中文 L0108 条件 not is_libero_plus 成立时进入下方分支。
    if not is_libero_plus:
        # 中文 L0109 计算右侧表达式 root / task.problem_folder / filename.name给 “init_states_path” 赋值。
        init_states_path = root / task.problem_folder / filename.name
        # 中文 L0110 调用 _load_initial_states，用它的结果作为当前调用的返回值。
        return _load_initial_states(init_states_path)

    # LIBERO-plus: `_add_` / `_level` variants store extra-object layouts under
    # libero_newobj/ as a flat array that must be reshaped to (1, -1).
    # 中文 L0114 条件 "_add_" in filename.name or "_level" in filename.name 成立时进入下方分支。
    if "_add_" in filename.name or "_level" in filename.name:
        # 中文 L0115 计算右侧表达式 root / "libero_newobj" / task.problem_folder / filename.name给 “init_states_path” 赋值。
        init_states_path = root / "libero_newobj" / task.problem_folder / filename.name
        # 中文 L0116 调用 _load_initial_states，用它的结果给 “init_states” 赋值。
        init_states = _load_initial_states(init_states_path)
        # 中文 L0117 调用 init_states.reshape，用它的结果作为当前调用的返回值。
        return init_states.reshape(1, -1)

    # LIBERO-plus perturbation variants encode the perturbation in the filename
    # but on disk only the base `.pruned_init` exists — strip the suffix to match.
    # 中文 L0121 计算右侧表达式 _LIBERO_PERTURBATION_SUFFIX_RE.sub("", filename.stem) + filename.suffix给 “stripped” 赋值。
    stripped = _LIBERO_PERTURBATION_SUFFIX_RE.sub("", filename.stem) + filename.suffix
    # 中文 L0122 计算右侧表达式 root / task.problem_folder / stripped给 “init_states_path” 赋值。
    init_states_path = root / task.problem_folder / stripped
    # 中文 L0123 调用 _load_initial_states，用它的结果作为当前调用的返回值。
    return _load_initial_states(init_states_path)


# 中文 L0126 定义函数 get_libero_dummy_action。
def get_libero_dummy_action():
    # 中文 L0127 原文档中的说明或续行。
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    # 中文 L0128 创建列表，按给定顺序保留元素作为当前调用的返回值。
    return [0, 0, 0, 0, 0, 0, -1]


# 中文 L0131 使用字面值 7给 “ACTION_DIM” 赋值。
ACTION_DIM = 7
# 中文 L0132 计算此行的右侧表达式给 “ACTION_LOW” 赋值。
ACTION_LOW = -1.0
# 中文 L0133 使用字面值 1.0给 “ACTION_HIGH” 赋值。
ACTION_HIGH = 1.0
# 中文 L0134 声明“TASK_SUITE_MAX_STEPS”的类型为 dict[str, int]，创建字典，用键查找对应的值作为初始值。
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    # 中文 L0135 字典项“libero_spatial”，冒号右边给出它的值。
    "libero_spatial": 280,  # longest training demo has 193 steps
    # 中文 L0136 字典项“libero_object”，冒号右边给出它的值。
    "libero_object": 280,  # longest training demo has 254 steps
    # 中文 L0137 字典项“libero_goal”，冒号右边给出它的值。
    "libero_goal": 300,  # longest training demo has 270 steps
    # 中文 L0138 字典项“libero_10”，冒号右边给出它的值。
    "libero_10": 520,  # longest training demo has 505 steps
    # 中文 L0139 字典项“libero_90”，冒号右边给出它的值。
    "libero_90": 400,  # longest training demo has 373 steps
# 中文 L0140 闭合上方尚未结束的括号。
}


# 中文 L0143 LeRobot 对单个 LIBERO 仿真实例的 Gymnasium 包装类。
class LiberoEnv(gym.Env):
    # 中文 L0144 创建字典，用键查找对应的值给 “metadata” 赋值。
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    # 中文 L0146 定义方法 __init__。作用是初始化单个 LIBERO 任务、相机、初始状态和观测配置。
    def __init__(
        # 中文 L0147 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0148 “task_suite”是当前括号中的一项，逗号将它与下一项分开。
        task_suite: Any,
        # 中文 L0149 “task_id”（套件内的任务编号）是当前括号中的一项，逗号将它与下一项分开。
        task_id: int,
        # 中文 L0150 “task_suite_name”是当前括号中的一项，逗号将它与下一项分开。
        task_suite_name: str,
        # 中文 L0151 为“episode_length”提供这一项的值，注意它所在的参数表或外层表达式。
        episode_length: int | None = None,
        # 中文 L0152 为“camera_name”提供这一项的值，注意它所在的参数表或外层表达式。
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        # 中文 L0153 为“obs_type”提供这一项的值，注意它所在的参数表或外层表达式。
        obs_type: str = "pixels",
        # 中文 L0154 为“render_mode”提供这一项的值，注意它所在的参数表或外层表达式。
        render_mode: str = "rgb_array",
        # 中文 L0155 为“observation_width”提供这一项的值，注意它所在的参数表或外层表达式。
        observation_width: int = 256,
        # 中文 L0156 为“observation_height”提供这一项的值，注意它所在的参数表或外层表达式。
        observation_height: int = 256,
        # 中文 L0157 为“visualization_width”提供这一项的值，注意它所在的参数表或外层表达式。
        visualization_width: int = 640,
        # 中文 L0158 为“visualization_height”提供这一项的值，注意它所在的参数表或外层表达式。
        visualization_height: int = 480,
        # 中文 L0159 为“init_states”提供这一项的值，注意它所在的参数表或外层表达式。
        init_states: bool = True,
        # 中文 L0160 为“episode_index”提供这一项的值，注意它所在的参数表或外层表达式。
        episode_index: int = 0,
        # 中文 L0161 为“n_envs”（环境实例数）提供这一项的值，注意它所在的参数表或外层表达式。
        n_envs: int = 1,
        # 中文 L0162 为“camera_name_mapping”提供这一项的值，注意它所在的参数表或外层表达式。
        camera_name_mapping: dict[str, str] | None = None,
        # 中文 L0163 为“num_steps_wait”提供这一项的值，注意它所在的参数表或外层表达式。
        num_steps_wait: int = 10,
        # 中文 L0164 为“control_freq”提供这一项的值，注意它所在的参数表或外层表达式。
        control_freq: int = 20,
        # 中文 L0165 为“control_mode”提供这一项的值，注意它所在的参数表或外层表达式。
        control_mode: str = "relative",
        # 中文 L0166 为“is_libero_plus”提供这一项的值，注意它所在的参数表或外层表达式。
        is_libero_plus: bool = False,
        # 中文 L0167 为“hard_reset”提供这一项的值，注意它所在的参数表或外层表达式。
        hard_reset: bool = True,
    # 中文 L0168 闭合上方尚未结束的括号。
    ):
        # 中文 L0169 调用 __init__ 执行操作。
        super().__init__()
        # 中文 L0170 条件 control_freq <= 0 成立时进入下方分支。
        if control_freq <= 0:
            # 中文 L0171 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        # 中文 L0172 条件 not hard_reset and not init_states 成立时进入下方分支。
        if not hard_reset and not init_states:
            # 中文 L0173 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("hard_reset=False requires init_states=True")
        # 中文 L0174 引用已有的值 task_id给 “self.task_id” 赋值。
        self.task_id = task_id
        # 中文 L0175 引用已有的值 is_libero_plus给 “self.is_libero_plus” 赋值。
        self.is_libero_plus = is_libero_plus
        # 中文 L0176 引用已有的值 obs_type给 “self.obs_type” 赋值。
        self.obs_type = obs_type
        # 中文 L0177 引用已有的值 render_mode给 “self.render_mode” 赋值。
        self.render_mode = render_mode
        # 中文 L0178 引用已有的值 observation_width给 “self.observation_width” 赋值。
        self.observation_width = observation_width
        # 中文 L0179 引用已有的值 observation_height给 “self.observation_height” 赋值。
        self.observation_height = observation_height
        # 中文 L0180 引用已有的值 visualization_width给 “self.visualization_width” 赋值。
        self.visualization_width = visualization_width
        # 中文 L0181 引用已有的值 visualization_height给 “self.visualization_height” 赋值。
        self.visualization_height = visualization_height
        # 中文 L0182 引用已有的值 init_states给 “self.init_states” 赋值。
        self.init_states = init_states
        # 中文 L0183 调用 parse_camera_names，用它的结果给 “self.camera_name” 赋值。
        self.camera_name = parse_camera_names(
            # 中文 L0184 延续上面的语句，camera_name。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
            camera_name
        # 中文 L0185 闭合上方尚未结束的括号。
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image1" and "image2".
        # `canonical_observation` will then prefix these with `observation.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        # 中文 L0192 条件 camera_name_mapping is None 成立时进入下方分支。
        if camera_name_mapping is None:
            # 中文 L0193 创建字典，用键查找对应的值给 “camera_name_mapping” 赋值。
            camera_name_mapping = {
                # 中文 L0194 字典项“agentview_image”，冒号右边给出它的值。
                "agentview_image": "image",
                # 中文 L0195 字典项“robot0_eye_in_hand_image”，冒号右边给出它的值。
                "robot0_eye_in_hand_image": "image2",
            # 中文 L0196 闭合上方尚未结束的括号。
            }
        # 中文 L0197 引用已有的值 camera_name_mapping给 “self.camera_name_mapping” 赋值。
        self.camera_name_mapping = camera_name_mapping
        # 中文 L0198 引用已有的值 num_steps_wait给 “self.num_steps_wait” 赋值。
        self.num_steps_wait = num_steps_wait
        # 中文 L0199 引用已有的值 control_freq给 “self.control_freq” 赋值。
        self.control_freq = control_freq
        # 中文 L0200 引用已有的值 hard_reset给 “self.hard_reset” 赋值。
        self.hard_reset = hard_reset
        # 中文 L0201 引用已有的值 episode_index给 “self.episode_index” 赋值。
        self.episode_index = episode_index
        # 中文 L0202 引用已有的值 episode_length给 “self.episode_length” 赋值。
        self.episode_length = episode_length
        # Load once and keep
        # 中文 L0204 用条件表达式在 if 前与 else 后的两个值之间选择给 “self._init_states” 赋值。
        self._init_states = (
            # 中文 L0205 延续上面的语句，get_task_init_states(task_suite, self.task_id, is_libero_plus=self.is_libero_plus)。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
            get_task_init_states(task_suite, self.task_id, is_libero_plus=self.is_libero_plus)
            # 中文 L0206 条件 self.init_states 成立时进入下方分支。
            if self.init_states
            # 中文 L0207 补充上方跨行表达式的分支值或条件。
            else None
        # 中文 L0208 闭合上方尚未结束的括号。
        )
        # 中文 L0209 引用已有的值 n_envs给 “self._reset_stride” 赋值。
        self._reset_stride = n_envs  # when performing a reset, append `_reset_stride` to `init_state_id`.

        # 中文 L0211 读取属性 self.episode_index给 “self.init_state_id” 赋值。
        self.init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        # Extract task metadata without allocating GPU resources (safe before fork).
        # 中文 L0214 调用 task_suite.get_task，用它的结果给 “task”（任务） 赋值。
        task = task_suite.get_task(task_id)
        # 中文 L0215 读取属性 task.name给 “self.task” 赋值。
        self.task = task.name
        # 中文 L0216 读取属性 task.language给 “self.task_description” 赋值。
        self.task_description = task.language
        # 中文 L0217 调用 os.path.join，用它的结果给 “self._task_bddl_file” 赋值。
        self._task_bddl_file = os.path.join(
            # 中文 L0218 延续上面的语句，get_libero_path("bddl_files"), task.problem_folder, task.bddl_file。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        # 中文 L0219 闭合上方尚未结束的括号。
        )
        # 中文 L0220 声明“self._env”的类型为 OffScreenRenderEnv | None，设为空值 None作为初始值。
        self._env: OffScreenRenderEnv | None = (
            # 中文 L0221 延续上面的语句，None # deferred — created on first reset() inside the worker subprocess。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
            None  # deferred — created on first reset() inside the worker subprocess
        # 中文 L0222 闭合上方尚未结束的括号。
        )

        # 中文 L0224 使用字面值 500给 “default_steps” 赋值。
        default_steps = 500
        # 中文 L0225 用条件表达式在 if 前与 else 后的两个值之间选择给 “self._max_episode_steps” 赋值。
        self._max_episode_steps = (
            # 中文 L0226 延续上面的语句，TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            # 中文 L0227 条件 self.episode_length is None 成立时进入下方分支。
            if self.episode_length is None
            # 中文 L0228 补充上方跨行表达式的分支值或条件。
            else self.episode_length
        # 中文 L0229 闭合上方尚未结束的括号。
        )
        # 中文 L0230 引用已有的值 control_mode给 “self.control_mode” 赋值。
        self.control_mode = control_mode
        # 中文 L0231 声明“images”的类型为 dict[str, gym.Space]，创建字典，用键查找对应的值作为初始值。
        images: dict[str, gym.Space] = {}
        # 中文 L0232 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for cam in self.camera_name:
            # 中文 L0233 调用 spaces.Box，用它的结果给 “images[self.camera_name_mapping[cam]]” 赋值。
            images[self.camera_name_mapping[cam]] = spaces.Box(
                # 中文 L0234 为“low”提供这一项的值，注意它所在的参数表或外层表达式。
                low=0,
                # 中文 L0235 为“high”提供这一项的值，注意它所在的参数表或外层表达式。
                high=255,
                # 中文 L0236 为“shape”（各轴的长度）提供这一项的值，注意它所在的参数表或外层表达式。
                shape=(self.observation_height, self.observation_width, 3),
                # 中文 L0237 为“dtype”（元素类型）提供这一项的值，注意它所在的参数表或外层表达式。
                dtype=np.uint8,
            # 中文 L0238 闭合上方尚未结束的括号。
            )

        # 中文 L0240 条件 self.obs_type == "state" 成立时进入下方分支。
        if self.obs_type == "state":
            # 中文 L0241 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
            raise NotImplementedError(
                # 中文 L0242 字符串字面量，供上方的调用、容器或错误消息使用。
                "The 'state' observation type is not supported in LiberoEnv. "
                # 中文 L0243 字符串字面量，供上方的调用、容器或错误消息使用。
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            # 中文 L0244 闭合上方尚未结束的括号。
            )

        # 中文 L0246 前面的分支均未命中时，再判断 self.obs_type == "pixels"。
        elif self.obs_type == "pixels":
            # 中文 L0247 调用 spaces.Dict，用它的结果给 “self.observation_space” 赋值。
            self.observation_space = spaces.Dict(
                # 中文 L0248 延续上面的语句，{。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
                {
                    # 中文 L0249 字典项“pixels”，冒号右边给出它的值。
                    "pixels": spaces.Dict(images),
                # 中文 L0250 闭合上方尚未结束的括号。
                }
            # 中文 L0251 闭合上方尚未结束的括号。
            )
        # 中文 L0252 前面的分支均未命中时，再判断 self.obs_type == "pixels_agent_pos"。
        elif self.obs_type == "pixels_agent_pos":
            # 中文 L0253 调用 spaces.Dict，用它的结果给 “self.observation_space” 赋值。
            self.observation_space = spaces.Dict(
                # 中文 L0254 延续上面的语句，{。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
                {
                    # 中文 L0255 字典项“pixels”，冒号右边给出它的值。
                    "pixels": spaces.Dict(images),
                    # 中文 L0256 字典项“robot_state”，冒号右边给出它的值。
                    "robot_state": spaces.Dict(
                        # 中文 L0257 延续上面的语句，{。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
                        {
                            # 中文 L0258 字典项“eef”，冒号右边给出它的值。
                            "eef": spaces.Dict(
                                # 中文 L0259 延续上面的语句，{。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
                                {
                                    # 中文 L0260 字典项“pos”，冒号右边给出它的值。
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    # 中文 L0261 字典项“quat”，冒号右边给出它的值。
                                    "quat": spaces.Box(
                                        # 中文 L0262 为“low”提供这一项的值，注意它所在的参数表或外层表达式。
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    # 中文 L0263 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                                    ),
                                    # 中文 L0264 字典项“mat”，冒号右边给出它的值。
                                    "mat": spaces.Box(
                                        # 中文 L0265 为“low”提供这一项的值，注意它所在的参数表或外层表达式。
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    # 中文 L0266 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                                    ),
                                # 中文 L0267 闭合上方尚未结束的括号。
                                }
                            # 中文 L0268 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                            ),
                            # 中文 L0269 字典项“gripper”，冒号右边给出它的值。
                            "gripper": spaces.Dict(
                                # 中文 L0270 延续上面的语句，{。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
                                {
                                    # 中文 L0271 字典项“qpos”，冒号右边给出它的值。
                                    "qpos": spaces.Box(
                                        # 中文 L0272 为“low”提供这一项的值，注意它所在的参数表或外层表达式。
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    # 中文 L0273 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                                    ),
                                    # 中文 L0274 字典项“qvel”，冒号右边给出它的值。
                                    "qvel": spaces.Box(
                                        # 中文 L0275 为“low”提供这一项的值，注意它所在的参数表或外层表达式。
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    # 中文 L0276 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                                    ),
                                # 中文 L0277 闭合上方尚未结束的括号。
                                }
                            # 中文 L0278 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                            ),
                            # 中文 L0279 字典项“joints”，冒号右边给出它的值。
                            "joints": spaces.Dict(
                                # 中文 L0280 延续上面的语句，{。本段处理初始化单个 LIBERO 任务、相机、初始状态和观测配置。
                                {
                                    # 中文 L0281 字典项“pos”，冒号右边给出它的值。
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    # 中文 L0282 字典项“vel”，冒号右边给出它的值。
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                # 中文 L0283 闭合上方尚未结束的括号。
                                }
                            # 中文 L0284 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                            ),
                        # 中文 L0285 闭合上方尚未结束的括号。
                        }
                    # 中文 L0286 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                    ),
                # 中文 L0287 闭合上方尚未结束的括号。
                }
            # 中文 L0288 闭合上方尚未结束的括号。
            )

        # 中文 L0290 调用 spaces.Box，用它的结果给 “self.action_space” 赋值。
        self.action_space = spaces.Box(
            # 中文 L0291 为“low”提供这一项的值，注意它所在的参数表或外层表达式。
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        # 中文 L0292 闭合上方尚未结束的括号。
        )

    # 中文 L0294 在真正用到时构造内部的 LIBERO 仿真环境。
    def _ensure_env(self) -> None:
        """Create the underlying OffScreenRenderEnv on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own clean EGL context rather than inheriting a stale one from the
        parent process (which causes EGL_BAD_CONTEXT crashes with AsyncVectorEnv).
        """
        # 中文 L0301 条件 self._env is not None 成立时进入下方分支。
        if self._env is not None:
            # 中文 L0302 退出当前调用并返回 None。
            return
        # 中文 L0303 调用 hf-libero 的 OffScreenRenderEnv，加载任务和相机设置。
        env = OffScreenRenderEnv(
            # 中文 L0304 把描述场景、对象和任务目标的 BDDL 文件传给底层。
            bddl_file_name=self._task_bddl_file,
            # 中文 L0305 为“camera_heights”提供这一项的值，注意它所在的参数表或外层表达式。
            camera_heights=self.observation_height,
            # 中文 L0306 为“camera_widths”提供这一项的值，注意它所在的参数表或外层表达式。
            camera_widths=self.observation_width,
            # 中文 L0307 设置控制频率，原评测为 20 Hz。
            control_freq=self.control_freq,
            # Soft resets skip LIBERO's model and renderer rebuild. They are opt-in
            # because settle steps can make their observations differ from hard resets.
            # 中文 L0310 为“hard_reset”提供这一项的值，注意它所在的参数表或外层表达式。
            hard_reset=self.hard_reset,
        # 中文 L0311 闭合上方尚未结束的括号。
        )
        # 中文 L0312 调用 env.reset，重置对象的当前状态。
        env.reset()
        # 中文 L0313 引用已有的值 env给 “self._env” 赋值。
        self._env = env

    # 中文 L0315 定义方法 render。
    def render(self):
        # 中文 L0316 调用 self._ensure_env 执行操作。
        self._ensure_env()
        # 中文 L0317 调用 self._env.env._get_observations，用它的结果给 “raw_obs” 赋值。
        raw_obs = self._env.env._get_observations()
        # 中文 L0318 用方括号按键或索引读取 容器给 “pixels” 赋值。
        pixels = self._format_raw_obs(raw_obs)["pixels"]
        # 中文 L0319 调用 next，用它的结果给 “image” 赋值。
        image = next(iter(pixels.values()))
        # 中文 L0320 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “image” 赋值。
        image = image[::-1, ::-1]  # flip both H and W for visualization
        # 中文 L0321 引用已有的值 image作为当前调用的返回值。
        return image

    # 中文 L0323 从仿真原始特征组织相机和机器人状态字典。
    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        # 中文 L0324 检查此条件，条件不成立时抛出 AssertionError。
        assert self._env is not None, "_format_raw_obs called before _ensure_env()"
        # 中文 L0325 创建字典，用键查找对应的值给 “images” 赋值。
        images = {}
        # 中文 L0326 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for camera_name in self.camera_name:
            # 中文 L0327 用方括号按键或索引读取 raw_obs给 “image” 赋值。
            image = raw_obs[camera_name]
            # 中文 L0328 用相机名称映射生成环境侧的相机观测键。
            images[self.camera_name_mapping[camera_name]] = image

        # 中文 L0330 调用 raw_obs.get，用它的结果给 “eef_pos” 赋值。
        eef_pos = raw_obs.get("robot0_eef_pos")
        # 中文 L0331 调用 raw_obs.get，用它的结果给 “eef_quat” 赋值。
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        # 中文 L0334 用条件表达式在 if 前与 else 后的两个值之间选择给 “eef_mat” 赋值。
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        # 中文 L0335 调用 raw_obs.get，用它的结果给 “gripper_qpos” 赋值。
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        # 中文 L0336 调用 raw_obs.get，用它的结果给 “gripper_qvel” 赋值。
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        # 中文 L0337 调用 raw_obs.get，用它的结果给 “joint_pos” 赋值。
        joint_pos = raw_obs.get("robot0_joint_pos")
        # 中文 L0338 调用 raw_obs.get，用它的结果给 “joint_vel” 赋值。
        joint_vel = raw_obs.get("robot0_joint_vel")
        # 中文 L0339 创建字典，用键查找对应的值给 “obs” 赋值。
        obs = {
            # 中文 L0340 字典项“pixels”，冒号右边给出它的值。
            "pixels": images,
            # 中文 L0341 字典项“robot_state”，冒号右边给出它的值。
            "robot_state": {
                # 中文 L0342 字典项“eef”，冒号右边给出它的值。
                "eef": {
                    # 中文 L0343 从底层获取末端执行器在三维空间中的位置。
                    "pos": eef_pos,  # (3,)
                    # 中文 L0344 此时姿态还是四元数，canonical_observation 会把它变成轴角。
                    "quat": eef_quat,  # (4,)
                    # 中文 L0345 字典项“mat”，冒号右边给出它的值。
                    "mat": eef_mat,  # (3, 3)
                # 中文 L0346 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                },
                # 中文 L0347 字典项“gripper”，冒号右边给出它的值。
                "gripper": {
                    # 中文 L0348 保留两个夹爪位置。它们会进入最终 8 维策略状态。
                    "qpos": gripper_qpos,  # (2,)
                    # 中文 L0349 字典项“qvel”，冒号右边给出它的值。
                    "qvel": gripper_qvel,  # (2,)
                # 中文 L0350 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                },
                # 中文 L0351 字典项“joints”，冒号右边给出它的值。
                "joints": {
                    # 中文 L0352 字典项“pos”，冒号右边给出它的值。
                    "pos": joint_pos,  # (7,)
                    # 中文 L0353 字典项“vel”，冒号右边给出它的值。
                    "vel": joint_vel,  # (7,)
                # 中文 L0354 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                },
            # 中文 L0355 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
            },
        # 中文 L0356 闭合上方尚未结束的括号。
        }
        # 中文 L0357 条件 self.obs_type == "pixels" 成立时进入下方分支。
        if self.obs_type == "pixels":
            # 中文 L0358 创建字典，用键查找对应的值作为当前调用的返回值。
            return {"pixels": images.copy()}

        # 中文 L0360 条件 self.obs_type == "pixels_agent_pos" 成立时进入下方分支。
        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            # 中文 L0362 条件 eef_pos is None or eef_quat is None or gripper_qpos is None 成立时进入下方分支。
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                # 中文 L0363 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0364 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"Missing required robot state fields in raw observation. "
                    # 中文 L0365 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    # 中文 L0366 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"gripper_qpos={gripper_qpos is not None}"
                # 中文 L0367 闭合上方尚未结束的括号。
                )
            # 中文 L0368 引用已有的值 obs作为当前调用的返回值。
            return obs

        # 中文 L0370 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError(
            # 中文 L0371 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            # 中文 L0372 字符串字面量，供上方的调用、容器或错误消息使用。
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        # 中文 L0373 闭合上方尚未结束的括号。
        )

    # 中文 L0375 执行单个环境的重置，应用种子、可选初始状态和控制模式。
    def reset(self, seed=None, **kwargs):
        # 中文 L0376 调用 self._ensure_env 执行操作。
        self._ensure_env()
        # 中文 L0377 调用 reset 执行操作。
        super().reset(seed=seed)
        # 中文 L0378 调用 self._env.seed 执行操作。
        self._env.seed(seed)
        # 中文 L0379 调用 self._env.reset，用它的结果给 “raw_obs” 赋值。
        raw_obs = self._env.reset()
        # 中文 L0380 条件 self.init_states and self._init_states is not None 成立时进入下方分支。
        if self.init_states and self._init_states is not None:
            # 中文 L0381 从任务附带的初始状态列表中选择当前索引的状态。
            raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            # 中文 L0382 用增量赋值更新 self.init_state_id，运算逻辑见左侧运算符。
            self.init_state_id += self._reset_stride  # Change init_state_id when reset

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        # 中文 L0387 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for _ in range(self.num_steps_wait):
            # 中文 L0388 给仿真若干不打算移动机械臂的预热控制步，让对象稳定。
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        # 中文 L0390 条件 self.control_mode == "absolute" 成立时进入下方分支。
        if self.control_mode == "absolute":
            # 中文 L0391 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for robot in self._env.robots:
                # 中文 L0392 使用布尔值 False给 “robot.controller.use_delta” 赋值。
                robot.controller.use_delta = False
        # 中文 L0393 原命令以 relative 模式启动仿真，进入这个控制器分支。
        elif self.control_mode == "relative":
            # 中文 L0394 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for robot in self._env.robots:
                # 中文 L0395 告诉底层机器人控制器用增量控制解释动作。
                robot.controller.use_delta = True
        # 中文 L0396 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0397 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        # 中文 L0398 调用 self._format_raw_obs，用它的结果给 “observation”（观测） 赋值。
        observation = self._format_raw_obs(raw_obs)
        # 中文 L0399 创建字典，用键查找对应的值给 “info”（额外信息字典） 赋值。
        info = {"is_success": False}
        # 中文 L0400 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return observation, info

    # 中文 L0402 单个仿真环境执行一个动作。返回下一观测和本步结果。
    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        # 中文 L0403 调用 self._ensure_env 执行操作。
        self._ensure_env()
        # 中文 L0404 检查此条件，条件不成立时抛出 AssertionError。
        assert self._env is not None
        # 中文 L0405 每个子环境接收秩为 1 的动作数组，在这里验证没有批次轴。
        if action.ndim != 1:
            # 中文 L0406 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0407 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Expected action to be 1-D (shape (action_dim,)), "
                # 中文 L0408 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"but got shape {action.shape} with ndim={action.ndim}"
            # 中文 L0409 闭合上方尚未结束的括号。
            )
        # 中文 L0410 把 NumPy 动作交给 hf-libero 的环境，仿真推进后返回新观测。
        raw_obs, reward, done, info = self._env.step(action)

        # 中文 L0412 由实际仿真检查任务目标是否达成，得到成功布尔值。
        is_success = self._env.check_success()
        # 中文 L0413 底层报告结束或满足成功条件时，这个包装层标记终止。
        terminated = done or is_success
        # 中文 L0414 调用 info.update，把新条目写入已有容器。
        info.update(
            # 中文 L0415 延续上面的语句，{。本段处理向底层递交单个动作并读取成功与新观测。
            {
                # 中文 L0416 字典项“task”（任务），冒号右边给出它的值。
                "task": self.task,
                # 中文 L0417 字典项“task_id”（套件内的任务编号），冒号右边给出它的值。
                "task_id": self.task_id,
                # 中文 L0418 字典项“done”，冒号右边给出它的值。
                "done": done,
                # 中文 L0419 字典项“is_success”（仿真的成功标志），冒号右边给出它的值。
                "is_success": is_success,
            # 中文 L0420 闭合上方尚未结束的括号。
            }
        # 中文 L0421 闭合上方尚未结束的括号。
        )
        # 中文 L0422 把仿真原始返回值整理为带相机和机器人状态的结构。
        observation = self._format_raw_obs(raw_obs)
        # Return the terminal observation unchanged. The caller owns resetting after
        # termination; vector envs created below use NEXT_STEP autoreset. Resetting here
        # would therefore reset twice and skip an initial state.
        # 中文 L0426 此包装层在这条路径中不主动设时间截断。向量层还会检查步数上限。
        truncated = False
        # 中文 L0427 依 Gymnasium 接口返回五项，供向量环境和 rollout 解包。
        return observation, reward, terminated, truncated, info

    # 中文 L0429 定义方法 close。
    def close(self):
        # 中文 L0430 条件 self._env is not None 成立时进入下方分支。
        if self._env is not None:
            # 中文 L0431 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
            try:
                # 中文 L0432 调用 self._env.close，关闭对象或释放关联资源。
                self._env.close()
            # 中文 L0433 离开 try 时执行这里的清理，即使前面的路径发生异常。
            finally:
                # LIBERO deletes its inner env on close, so this wrapper must
                # be recreated before the next reset.
                # 中文 L0436 设为空值 None给 “self._env” 赋值。
                self._env = None


# 中文 L0439 定义函数 _make_env_fns。
def _make_env_fns(
    # 中文 L0440 后续参数必须按名字传入。
    *,
    # 中文 L0441 “suite”是当前括号中的一项，逗号将它与下一项分开。
    suite,
    # 中文 L0442 “suite_name”是当前括号中的一项，逗号将它与下一项分开。
    suite_name: str,
    # 中文 L0443 “task_id”（套件内的任务编号）是当前括号中的一项，逗号将它与下一项分开。
    task_id: int,
    # 中文 L0444 “n_envs”（环境实例数）是当前括号中的一项，逗号将它与下一项分开。
    n_envs: int,
    # 中文 L0445 “camera_names”是当前括号中的一项，逗号将它与下一项分开。
    camera_names: list[str],
    # 中文 L0446 “episode_length”是当前括号中的一项，逗号将它与下一项分开。
    episode_length: int | None,
    # 中文 L0447 “init_states”是当前括号中的一项，逗号将它与下一项分开。
    init_states: bool,
    # 中文 L0448 延续上面的语句，gym_kwargs: Mapping[str, Any],。
    gym_kwargs: Mapping[str, Any],
    # 中文 L0449 “control_mode”是当前括号中的一项，逗号将它与下一项分开。
    control_mode: str,
    # 中文 L0450 为“camera_name_mapping”提供这一项的值，注意它所在的参数表或外层表达式。
    camera_name_mapping: dict[str, str] | None = None,
    # 中文 L0451 为“is_libero_plus”提供这一项的值，注意它所在的参数表或外层表达式。
    is_libero_plus: bool = False,
# 中文 L0452 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> list[Callable[[], LiberoEnv]]:
    # 中文 L0453 原文档中的说明或续行。
    """Build n_envs factory callables for a single (suite, task_id)."""

    # 中文 L0455 定义函数 _make_env。
    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        # 中文 L0456 调用 dict，用它的结果给 “local_kwargs” 赋值。
        local_kwargs = dict(kwargs)
        # 中文 L0457 调用 LiberoEnv，用它的结果作为当前调用的返回值。
        return LiberoEnv(
            # 中文 L0458 为“task_suite”提供这一项的值，注意它所在的参数表或外层表达式。
            task_suite=suite,
            # 中文 L0459 为“task_id”（套件内的任务编号）提供这一项的值，注意它所在的参数表或外层表达式。
            task_id=task_id,
            # 中文 L0460 为“task_suite_name”提供这一项的值，注意它所在的参数表或外层表达式。
            task_suite_name=suite_name,
            # 中文 L0461 为“camera_name”提供这一项的值，注意它所在的参数表或外层表达式。
            camera_name=camera_names,
            # 中文 L0462 为“init_states”提供这一项的值，注意它所在的参数表或外层表达式。
            init_states=init_states,
            # 中文 L0463 为“episode_length”提供这一项的值，注意它所在的参数表或外层表达式。
            episode_length=episode_length,
            # 中文 L0464 为“episode_index”提供这一项的值，注意它所在的参数表或外层表达式。
            episode_index=episode_index,
            # 中文 L0465 为“n_envs”（环境实例数）提供这一项的值，注意它所在的参数表或外层表达式。
            n_envs=n_envs,
            # 中文 L0466 为“control_mode”提供这一项的值，注意它所在的参数表或外层表达式。
            control_mode=control_mode,
            # 中文 L0467 为“camera_name_mapping”提供这一项的值，注意它所在的参数表或外层表达式。
            camera_name_mapping=camera_name_mapping,
            # 中文 L0468 为“is_libero_plus”提供这一项的值，注意它所在的参数表或外层表达式。
            is_libero_plus=is_libero_plus,
            # 中文 L0469 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
            **local_kwargs,
        # 中文 L0470 闭合上方尚未结束的括号。
        )

    # 中文 L0472 声明“fns”的类型为 list[Callable[[], LiberoEnv]]，创建列表，按给定顺序保留元素作为初始值。
    fns: list[Callable[[], LiberoEnv]] = []
    # 中文 L0473 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for episode_index in range(n_envs):
        # 中文 L0474 调用 fns.append，在列表末尾追加一个元素。
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    # 中文 L0475 引用已有的值 fns作为当前调用的返回值。
    return fns


# ---- Main API ----------------------------------------------------------------


# 中文 L0481 生成按套件和任务 ID 分组的环境构造函数。
def env_fns(
    # 中文 L0482 “task”（任务）是当前括号中的一项，逗号将它与下一项分开。
    task: str,
    # 中文 L0483 “n_envs”（环境实例数）是当前括号中的一项，逗号将它与下一项分开。
    n_envs: int,
    # 中文 L0484 为“gym_kwargs”提供这一项的值，注意它所在的参数表或外层表达式。
    gym_kwargs: dict[str, Any] | None = None,
    # 中文 L0485 为“camera_name”提供这一项的值，注意它所在的参数表或外层表达式。
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    # 中文 L0486 为“init_states”提供这一项的值，注意它所在的参数表或外层表达式。
    init_states: bool = True,
    # 中文 L0487 为“control_mode”提供这一项的值，注意它所在的参数表或外层表达式。
    control_mode: str = "relative",
    # 中文 L0488 为“episode_length”提供这一项的值，注意它所在的参数表或外层表达式。
    episode_length: int | None = None,
    # 中文 L0489 为“camera_name_mapping”提供这一项的值，注意它所在的参数表或外层表达式。
    camera_name_mapping: dict[str, str] | None = None,
    # 中文 L0490 为“is_libero_plus”提供这一项的值，注意它所在的参数表或外层表达式。
    is_libero_plus: bool = False,
# 中文 L0491 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict[str, dict[int, list[Callable[[], Any]]]]:
    """Return `n_envs` LIBERO env factories per task, as `{suite: {task_id: [factory, ...]}}`.

    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    # 中文 L0499 条件 not isinstance(n_envs, int) or n_envs <= 0 成立时进入下方分支。
    if not isinstance(n_envs, int) or n_envs <= 0:
        # 中文 L0500 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    # 中文 L0502 调用 dict，用它的结果给 “gym_kwargs” 赋值。
    gym_kwargs = dict(gym_kwargs or {})
    # 中文 L0503 从仿真参数里取出用户传入的任务编号过滤项。
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

    # 中文 L0505 调用 parse_camera_names，用它的结果给 “camera_names” 赋值。
    camera_names = parse_camera_names(camera_name)
    # 中文 L0506 用列表推导式从可迭代对象逐项取值并构造结果给 “suite_names” 赋值。
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    # 中文 L0507 条件 not suite_names 成立时进入下方分支。
    if not suite_names:
        # 中文 L0508 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    # 中文 L0510 调用 logger.info 执行操作。
    logger.info(
        # 中文 L0511 字符串字面量，供上方的调用、容器或错误消息使用。
        "Creating LIBERO envs | suites=%s | n_envs(per task)=%d | init_states=%s",
        # 中文 L0512 “suite_names”是当前括号中的一项，逗号将它与下一项分开。
        suite_names,
        # 中文 L0513 “n_envs”（环境实例数）是当前括号中的一项，逗号将它与下一项分开。
        n_envs,
        # 中文 L0514 “init_states”是当前括号中的一项，逗号将它与下一项分开。
        init_states,
    # 中文 L0515 闭合上方尚未结束的括号。
    )
    # 中文 L0516 条件 task_ids_filter is not None 成立时进入下方分支。
    if task_ids_filter is not None:
        # 中文 L0517 调用 logger.info 执行操作。
        logger.info("Restricting to task_ids=%s", task_ids_filter)

    # 中文 L0519 声明“out”的类型为 dict[str, dict[int, list[Callable[[], Any]]]]，调用 defaultdict，用它的结果作为初始值。
    out: dict[str, dict[int, list[Callable[[], Any]]]] = defaultdict(dict)
    # 中文 L0520 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for suite_name in suite_names:
        # 中文 L0521 调用 _get_suite，用它的结果给 “suite” 赋值。
        suite = _get_suite(suite_name)
        # 中文 L0522 调用 len，用它的结果给 “total” 赋值。
        total = len(suite.tasks)
        # 中文 L0523 验证并选出要评测的 task ID，本次保留 0。
        selected = _select_task_ids(total, task_ids_filter)
        # 中文 L0524 条件 not selected 成立时进入下方分支。
        if not selected:
            # 中文 L0525 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        # 中文 L0527 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for tid in selected:
            # 中文 L0528 为选定任务准备 n_envs 个环境构造函数。
            out[suite_name][tid] = _make_env_fns(
                # 中文 L0529 为“suite”提供这一项的值，注意它所在的参数表或外层表达式。
                suite=suite,
                # 中文 L0530 为“episode_length”提供这一项的值，注意它所在的参数表或外层表达式。
                episode_length=episode_length,
                # 中文 L0531 为“suite_name”提供这一项的值，注意它所在的参数表或外层表达式。
                suite_name=suite_name,
                # 中文 L0532 为“task_id”（套件内的任务编号）提供这一项的值，注意它所在的参数表或外层表达式。
                task_id=tid,
                # 中文 L0533 为“n_envs”（环境实例数）提供这一项的值，注意它所在的参数表或外层表达式。
                n_envs=n_envs,
                # 中文 L0534 为“camera_names”提供这一项的值，注意它所在的参数表或外层表达式。
                camera_names=camera_names,
                # 中文 L0535 为“init_states”提供这一项的值，注意它所在的参数表或外层表达式。
                init_states=init_states,
                # 中文 L0536 为“gym_kwargs”提供这一项的值，注意它所在的参数表或外层表达式。
                gym_kwargs=gym_kwargs,
                # 中文 L0537 为“control_mode”提供这一项的值，注意它所在的参数表或外层表达式。
                control_mode=control_mode,
                # 中文 L0538 为“camera_name_mapping”提供这一项的值，注意它所在的参数表或外层表达式。
                camera_name_mapping=camera_name_mapping,
                # 中文 L0539 为“is_libero_plus”提供这一项的值，注意它所在的参数表或外层表达式。
                is_libero_plus=is_libero_plus,
            # 中文 L0540 闭合上方尚未结束的括号。
            )

    # 中文 L0542 用字典推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
    return {suite: dict(task_map) for suite, task_map in out.items()}


# 中文 L0545 定义函数 conventions。
def conventions(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return LIBERO's default fps, control mode, state and action names, and robot type.

    LIBERO's own `control_mode` option, in `kwargs`, is `relative` (eef deltas) or `absolute` (eef poses).
    """
    # 中文 L0550 打包成元组，元素顺序由逗号分隔的位置决定给 “pose” 赋值。
    pose = ("eef.x", "eef.y", "eef.z", "eef.axis_angle.x", "eef.axis_angle.y", "eef.axis_angle.z")
    # 中文 L0551 计算此行的右侧表达式给 “absolute” 赋值。
    absolute = kwargs.get("control_mode", "relative") == "absolute"
    # 中文 L0552 用条件表达式在 if 前与 else 后的两个值之间选择给 “commanded” 赋值。
    commanded = pose if absolute else tuple(name.replace("eef.", "eef.delta.") for name in pose)
    # 中文 L0553 创建字典，用键查找对应的值作为当前调用的返回值。
    return {
        # 中文 L0554 字典项“fps”，冒号右边给出它的值。
        "fps": (kwargs.get("gym_kwargs") or {}).get("control_freq", 20),
        # 中文 L0555 公布仿真控制模式约定，relative 在通用描述符里对应 delta。
        "control_mode": "eef_pose" if absolute else "delta",
        # 中文 L0556 定义 8 个状态分量的名字，末端位姿和两个夹爪位置。
        "state_names": (*pose, "gripper.left.pos", "gripper.right.pos"),
        # 中文 L0557 定义 7 个动作分量的名字，末端位姿控制的前六项加夹爪命令。
        "action_names": (*commanded, "gripper.command"),
        # 中文 L0558 字典项“robot_type”，冒号右边给出它的值。
        "robot_type": "panda",
    # 中文 L0559 闭合上方尚未结束的括号。
    }
