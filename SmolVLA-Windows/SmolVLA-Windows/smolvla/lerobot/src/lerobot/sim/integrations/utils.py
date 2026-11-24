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

# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0015 原文档中的说明或续行。
"""Helpers shared by simulator integrations."""

# 中文 L0017 从模块 collections.abc 导入 Mapping、Sequence。
from collections.abc import Mapping, Sequence
# 中文 L0018 从模块 typing 导入 Any。
from typing import Any

# 中文 L0020 导入模块 numpy，简称 np。
import numpy as np

# 中文 L0022 从模块 lerobot.utils.constants 导入 OBS_ENV_STATE、OBS_IMAGE、OBS_IMAGES、OBS_STATE、OBS_STR。
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR

# Same as lerobot.lerobot_types.RobotObservation, redefined here so integrations don't import torch.
# 中文 L0025 用方括号按键或索引读取 dict给 “RobotObservation” 赋值。
RobotObservation = dict[str, Any]


# 中文 L0028 按统一协议整理仿真图像和机器人状态。
def canonical_observation(
    # 中文 L0029 “obs”是当前括号中的一项，逗号将它与下一项分开。
    obs: Mapping,
    # 中文 L0030 后续参数必须按名字传入。
    *,
    # 中文 L0031 为“flip_images”提供这一项的值，注意它所在的参数表或外层表达式。
    flip_images: bool = False,
# 中文 L0032 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict[str, np.ndarray]:
    """Map a simulator observation to LeRobot keys; images must be HWC `uint8` with three channels.

    With `flip_images`, images are rotated by 180 degrees, as LIBERO needs.
    """
    # 中文 L0037 创建字典，用键查找对应的值给 “result” 赋值。
    result = {}
    # 中文 L0038 调用 obs.get，用它的结果给 “pixels” 赋值。
    pixels = obs.get("pixels")
    # 中文 L0039 条件 pixels is not None 成立时进入下方分支。
    if pixels is not None:
        # 中文 L0040 用条件表达式在 if 前与 else 后的两个值之间选择给 “images” 赋值。
        images = pixels if isinstance(pixels, dict) else {None: pixels}
        # 中文 L0041 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for name, image in images.items():
            # 中文 L0042 用标准前缀拼出相机键，此时名字是 image 和 image2 之类的环境侧名字。
            key = f"{OBS_IMAGES}.{name}" if name is not None else OBS_IMAGE
            # 中文 L0043 调用 np.asarray，用它的结果给 “image” 赋值。
            image = np.asarray(image)
            # 中文 L0044 要求单张相机图是 NumPy HWC 布局的三通道 uint8 数组。
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                # 中文 L0045 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0046 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"Feature {key!r}: images must be HWC uint8 with three channels, "
                    # 中文 L0047 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"got {image.shape}, {image.dtype}"
                # 中文 L0048 闭合上方尚未结束的括号。
                )
            # 中文 L0049 条件 flip_images 成立时进入下方分支。
            if flip_images:
                # 中文 L0050 在高和宽两轴都倒序，等效在图像平面旋转 180 度。
                image = image[::-1, ::-1]
            # 中文 L0051 保证返回数组占用连续内存，以便后续批处理和转换。
            result[key] = np.ascontiguousarray(image)
    # 中文 L0052 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for raw, canonical in (("agent_pos", OBS_STATE), ("environment_state", OBS_ENV_STATE)):
        # 中文 L0053 条件 raw in obs 成立时进入下方分支。
        if raw in obs:
            # 中文 L0054 为环境状态或机器人状态写入 LeRobot 标准键，并转成 NumPy float32。
            result[canonical] = np.asarray(obs[raw], dtype=np.float32)
    # 中文 L0055 计算此行的右侧表达式给 “handled” 赋值。
    handled = {"pixels", "agent_pos", "environment_state"}
    # 中文 L0056 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key, value in obs.items():
        # 中文 L0057 条件 key not in handled and isinstance(value, np.ndarray) 成立时进入下方分支。
        if key not in handled and isinstance(value, np.ndarray):
            # 中文 L0058 用条件表达式在 if 前与 else 后的两个值之间选择给 “target” 赋值。
            target = key if key.startswith(f"{OBS_STR}.") else f"{OBS_STR}.{key}"
            # 中文 L0059 调用 np.asarray，用它的结果给 “result[target]” 赋值。
            result[target] = np.asarray(value, dtype=np.float32)
    # 中文 L0060 引用已有的值 result作为当前调用的返回值。
    return result


# 中文 L0063 定义函数 parse_camera_names。
def parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize ``camera_name`` into a non-empty list of strings.

    Accepts a comma-separated string (``"cam_a,cam_b"``) or a sequence of
    strings (tuples/lists). Whitespace is stripped; empty entries are
    dropped. Raises ``TypeError`` for unsupported input types and
    ``ValueError`` when the normalized list is empty.
    """
    # 中文 L0071 条件 isinstance(camera_name, str) 成立时进入下方分支。
    if isinstance(camera_name, str):
        # 中文 L0072 用列表推导式从可迭代对象逐项取值并构造结果给 “cams” 赋值。
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    # 中文 L0073 前面的分支均未命中时，再判断 isinstance(camera_name, (list | tuple))。
    elif isinstance(camera_name, (list | tuple)):
        # 中文 L0074 用列表推导式从可迭代对象逐项取值并构造结果给 “cams” 赋值。
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    # 中文 L0075 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
    else:
        # 中文 L0076 抛出 TypeError，中断当前路径并交给上层的异常处理。
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    # 中文 L0077 条件 not cams 成立时进入下方分支。
    if not cams:
        # 中文 L0078 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("camera_name resolved to an empty list.")
    # 中文 L0079 引用已有的值 cams作为当前调用的返回值。
    return cams
