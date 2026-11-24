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

"""Config for the simulator an env runs."""

import importlib
import math
from dataclasses import dataclass, field
from typing import Any

from lerobot.sim.integrations import simulator_required
from lerobot.sim.types import CONTROL_MODES


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0026 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0027 定义类 SimConfig。
class SimConfig:
    """The simulator an env runs: integration, task, options and control settings.

    Args:
        type (`str`):
            Integration, a module of `lerobot.sim.integrations`, e.g. `libero`.
        task (`str`):
            Task to run, as the integration names it, e.g. `libero_spatial`. Some integrations take a
            comma-separated list.
        fps (`float`, *optional*, defaults to `30`):
            Control rate.
        control_mode (`str`, *optional*, defaults to `"position"`):
            What an action commands: `position`, `delta` or `eef_pose`.
        semantics (`str`, *optional*):
            Name of the observation and action conventions. Defaults to `<type>-<control_mode>-v1`.
        kwargs (`dict[str, Any]`, *optional*):
            Options passed to the integration's `env_fns`.
        state_names (`tuple[str, ...]`, *optional*):
            Component names of `observation.state`. Left empty, they are `state_<i>`.
        action_names (`tuple[str, ...]`, *optional*):
            Component names of the action. Left empty, they are `command_<i>`.
        robot_type (`str`, *optional*, defaults to `""`):
            Body a sim robot records in its dataset. Left empty, it records `sim_robot`.
    """

    # 中文 L0052 声明“type”的类型为 str。
    type: str
    # 中文 L0053 声明“task”（任务）的类型为 str。
    task: str
    # 中文 L0054 声明“fps”的类型为 float，使用字面值 30作为初始值。
    fps: float = 30
    # 中文 L0055 声明“control_mode”的类型为 str，使用字面值 'position'作为初始值。
    control_mode: str = "position"
    # 中文 L0056 声明“semantics”的类型为 str，使用字面值 ''作为初始值。
    semantics: str = ""
    # 中文 L0057 声明“kwargs”（额外的关键字参数）的类型为 dict[str, Any]，调用 field，用它的结果作为初始值。
    kwargs: dict[str, Any] = field(default_factory=dict)
    # 中文 L0058 声明“state_names”的类型为 tuple[str, ...]，打包成元组，元素顺序由逗号分隔的位置决定作为初始值。
    state_names: tuple[str, ...] = ()
    # 中文 L0059 声明“action_names”的类型为 tuple[str, ...]，打包成元组，元素顺序由逗号分隔的位置决定作为初始值。
    action_names: tuple[str, ...] = ()
    # 中文 L0060 声明“robot_type”的类型为 str，使用字面值 ''作为初始值。
    robot_type: str = ""

    # 中文 L0062 定义方法 __post_init__。
    def __post_init__(self) -> None:
        # 中文 L0063 原文档中的说明或续行。
        """Check `control_mode` and `fps`, and fill in the default `semantics`."""
        # 中文 L0064 调用 tuple，用它的结果给 “self.state_names” 赋值。
        self.state_names = tuple(self.state_names)
        # 中文 L0065 调用 tuple，用它的结果给 “self.action_names” 赋值。
        self.action_names = tuple(self.action_names)
        # 中文 L0066 条件 self.control_mode not in CONTROL_MODES 成立时进入下方分支。
        if self.control_mode not in CONTROL_MODES:
            # 中文 L0067 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Unknown control_mode {self.control_mode!r}; expected one of {CONTROL_MODES}")
        # 中文 L0068 条件 not math.isfinite(self.fps) or self.fps <= 0 成立时进入下方分支。
        if not math.isfinite(self.fps) or self.fps <= 0:
            # 中文 L0069 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"A simulator needs a positive fps, got {self.fps}")
        # 中文 L0070 条件 not self.semantics 成立时进入下方分支。
        if not self.semantics:
            # 中文 L0071 计算此行的右侧表达式给 “self.semantics” 赋值。
            self.semantics = f"{self.type}-{self.control_mode.replace('_', '-')}-v1"

    # 中文 L0073 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0074 用环境的类型名解析模块和默认控制约定。
    def resolve(cls, settings: dict[str, Any]) -> "SimConfig":
        # 中文 L0075 原文档中的说明或续行。
        """Build a config from `settings`, taking missing values from the integration's `conventions`."""
        # 中文 L0076 进入上下文管理器，缩进块结束时由它处理退出动作。
        with simulator_required(settings["type"]):
            # 中文 L0077 按 type 动态导入仿真接入。此例是 lerobot.sim.integrations.libero。
            integration = importlib.import_module(f"lerobot.sim.integrations.{settings['type']}")
            # 中文 L0078 调用 getattr，用它的结果给 “conventions” 赋值。
            conventions = getattr(integration, "conventions", None)
            # 中文 L0079 用条件表达式在 if 前与 else 后的两个值之间选择给 “declared” 赋值。
            declared = conventions(settings.get("kwargs") or {}) if conventions else {}
        # 中文 L0080 调用 cls，用它的结果作为当前调用的返回值。
        return cls(**{**declared, **{key: value for key, value in settings.items() if value is not None}})


@dataclass
class SimSettings:
    """`SimConfig`'s fields as a config file gives them; `SimConfig.resolve` fills in the unset (`None`) ones.

    They come from the integration's conventions, so a file only names the simulator, its task and its options.
    """

    type: str = ""
    task: str = ""
    fps: float | None = None
    control_mode: str | None = None
    semantics: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    robot_type: str | None = None
