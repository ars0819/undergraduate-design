#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# 中文 L0016 导入模块 logging。
import logging
# 中文 L0017 导入模块 warnings。
import warnings
# 中文 L0018 从模块 collections.abc 导入 Mapping、Sequence。
from collections.abc import Mapping, Sequence
# 中文 L0019 从模块 functools 导入 singledispatch。
from functools import singledispatch
# 中文 L0020 从模块 typing 导入 Any。
from typing import Any

# 中文 L0022 导入模块 einops。
import einops
# 中文 L0023 导入模块 gymnasium，简称 gym。
import gymnasium as gym
# 中文 L0024 导入模块 numpy，简称 np。
import numpy as np
# 中文 L0025 导入模块 torch。
import torch
# 中文 L0026 从模块 torch 导入 Tensor。
from torch import Tensor

# 中文 L0028 从模块 lerobot.configs 导入 FeatureType、PipelineFeatureType、PolicyFeature、PreTrainedConfig。
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature, PreTrainedConfig
# 中文 L0029 从模块 lerobot.processor 导入 PolicyAction、PolicyProcessorPipeline、rename_keys。
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, rename_keys
# 中文 L0030 从模块 lerobot.sim 导入 EnvDescriptor。
from lerobot.sim import EnvDescriptor
# 中文 L0031 从模块 lerobot.utils.constants 导入 ACTION、OBS_ENV_STATE、OBS_IMAGE、OBS_IMAGES、OBS_STATE、OBS_STR。
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR
# 中文 L0032 从模块 lerobot.utils.utils 导入 get_channel_first_image_shape。
from lerobot.utils.utils import get_channel_first_image_shape

# 中文 L0034 从模块 .configs 导入 EnvConfig。
from .configs import EnvConfig
# 中文 L0035 从模块 .factory 导入 descriptor_features。
from .factory import descriptor_features

# 中文 L0037 调用 logging.getLogger，用它的结果给 “logger” 赋值。
logger = logging.getLogger(__name__)


# 中文 L0040 定义函数 _convert_nested_dict。
def _convert_nested_dict(d):
    # 中文 L0041 创建字典，用键查找对应的值给 “result” 赋值。
    result = {}
    # 中文 L0042 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for k, v in d.items():
        # 中文 L0043 条件 isinstance(v, dict) 成立时进入下方分支。
        if isinstance(v, dict):
            # 中文 L0044 调用 _convert_nested_dict，用它的结果给 “result[k]” 赋值。
            result[k] = _convert_nested_dict(v)
        # 中文 L0045 前面的分支均未命中时，再判断 isinstance(v, np.ndarray)。
        elif isinstance(v, np.ndarray):
            # 中文 L0046 调用 torch.from_numpy，用它的结果给 “result[k]” 赋值。
            result[k] = torch.from_numpy(v)
        # 中文 L0047 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0048 引用已有的值 v给 “result[k]” 赋值。
            result[k] = v
    # 中文 L0049 引用已有的值 result作为当前调用的返回值。
    return result


# 中文 L0052 转换尚未采用 LeRobot 标准键名的通用观测。原 LIBERO 主路径使用下面的标准转换函数。
def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(jadechoghari, imstevenpmwork): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    # 中文 L0061 创建字典，用键查找对应的值给 “return_observations” 赋值。
    return_observations = {}
    # 中文 L0062 条件 "pixels" in observations 成立时进入下方分支。
    if "pixels" in observations:
        # 中文 L0063 条件 isinstance(observations["pixels"], dict) 成立时进入下方分支。
        if isinstance(observations["pixels"], dict):
            # 中文 L0064 用字典推导式从可迭代对象逐项取值并构造结果给 “imgs” 赋值。
            imgs = {f"{OBS_IMAGES}.{key}": img for key, img in observations["pixels"].items()}
        # 中文 L0065 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0066 创建字典，用键查找对应的值给 “imgs” 赋值。
            imgs = {OBS_IMAGE: observations["pixels"]}

        # 中文 L0068 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            # 中文 L0070 调用 torch.from_numpy，用它的结果给 “img_tensor” 赋值。
            img_tensor = torch.from_numpy(img)

            # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
            # This is the case for human-in-the-loop RL where there is only one environment.
            # 中文 L0074 条件 img_tensor.ndim == 3 成立时进入下方分支。
            if img_tensor.ndim == 3:
                # 中文 L0075 调用 img_tensor.unsqueeze，用它的结果给 “img_tensor” 赋值。
                img_tensor = img_tensor.unsqueeze(0)
            # 中文 L0076 读取属性 img_tensor.shape给 “(_, h, w, c)” 赋值。
            _, h, w, c = img_tensor.shape
            # 中文 L0077 条件 not (c < h and c < w) 成立时进入下方分支。
            if not (c < h and c < w):
                # 中文 L0078 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(f"Expected channel-last images, got shape {tuple(img_tensor.shape)}")
            # 中文 L0079 条件 img_tensor.dtype != torch.uint8 成立时进入下方分支。
            if img_tensor.dtype != torch.uint8:
                # 中文 L0080 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(f"Expected uint8 images, got {img_tensor.dtype}")

            # convert to channel first of type float32 in range [0,1]
            # 中文 L0083 调用 contiguous，用它的结果给 “img_tensor” 赋值。
            img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()
            # 中文 L0084 调用 img_tensor.type，用它的结果给 “img_tensor” 赋值。
            img_tensor = img_tensor.type(torch.float32)
            # 中文 L0085 用增量赋值更新 img_tensor，运算逻辑见左侧运算符。
            img_tensor /= 255

            # 中文 L0087 引用已有的值 img_tensor给 “return_observations[imgkey]” 赋值。
            return_observations[imgkey] = img_tensor

    # 中文 L0089 条件 "environment_state" in observations 成立时进入下方分支。
    if "environment_state" in observations:
        # 中文 L0090 调用 float，用它的结果给 “env_state” 赋值。
        env_state = torch.from_numpy(observations["environment_state"]).float()
        # 中文 L0091 条件 env_state.dim() == 1 成立时进入下方分支。
        if env_state.dim() == 1:
            # 中文 L0092 调用 env_state.unsqueeze，用它的结果给 “env_state” 赋值。
            env_state = env_state.unsqueeze(0)

        # 中文 L0094 引用已有的值 env_state给 “return_observations[OBS_ENV_STATE]” 赋值。
        return_observations[OBS_ENV_STATE] = env_state

    # 中文 L0096 条件 "agent_pos" in observations 成立时进入下方分支。
    if "agent_pos" in observations:
        # 中文 L0097 调用 float，用它的结果给 “agent_pos” 赋值。
        agent_pos = torch.from_numpy(observations["agent_pos"]).float()
        # 中文 L0098 条件 agent_pos.dim() == 1 成立时进入下方分支。
        if agent_pos.dim() == 1:
            # 中文 L0099 调用 agent_pos.unsqueeze，用它的结果给 “agent_pos” 赋值。
            agent_pos = agent_pos.unsqueeze(0)
        # 中文 L0100 引用已有的值 agent_pos给 “return_observations[OBS_STATE]” 赋值。
        return_observations[OBS_STATE] = agent_pos

    # 中文 L0102 条件 "robot_state" in observations 成立时进入下方分支。
    if "robot_state" in observations:
        # 中文 L0103 调用 _convert_nested_dict，用它的结果给 “return_observations[f'{OBS_STR}.robot_state']” 赋值。
        return_observations[f"{OBS_STR}.robot_state"] = _convert_nested_dict(observations["robot_state"])

    # Handle IsaacLab Arena format: observations have 'policy' and 'camera_obs' keys
    # 中文 L0106 条件 "policy" in observations 成立时进入下方分支。
    if "policy" in observations:
        # 中文 L0107 用方括号按键或索引读取 observations给 “return_observations[f'{OBS_STR}.policy']” 赋值。
        return_observations[f"{OBS_STR}.policy"] = observations["policy"]

    # 中文 L0109 条件 "camera_obs" in observations 成立时进入下方分支。
    if "camera_obs" in observations:
        # 中文 L0110 用方括号按键或索引读取 observations给 “return_observations[f'{OBS_STR}.camera_obs']” 赋值。
        return_observations[f"{OBS_STR}.camera_obs"] = observations["camera_obs"]

    # Pass through any remaining ndarray/tensor keys not already handled above,
    # so env plugins can expose extra observation keys via get_env_processors().
    # 中文 L0114 计算此行的右侧表达式给 “_handled” 赋值。
    _handled = {"pixels", "environment_state", "agent_pos", "robot_state", "policy", "camera_obs"}
    # 中文 L0115 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key, value in observations.items():
        # 中文 L0116 条件 key in _handled 成立时进入下方分支。
        if key in _handled:
            # 中文 L0117 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0118 计算此行的右侧表达式给 “target” 赋值。
        target = f"{OBS_STR}.{key}"
        # 中文 L0119 条件 target in return_observations 成立时进入下方分支。
        if target in return_observations:
            # 中文 L0120 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0121 条件 isinstance(value, np.ndarray) 成立时进入下方分支。
        if isinstance(value, np.ndarray):
            # 中文 L0122 调用 float，用它的结果给 “val” 赋值。
            val = torch.from_numpy(value).float()
            # 中文 L0123 条件 val.dim() == 1 成立时进入下方分支。
            if val.dim() == 1:
                # 中文 L0124 调用 val.unsqueeze，用它的结果给 “val” 赋值。
                val = val.unsqueeze(0)
            # 中文 L0125 引用已有的值 val给 “return_observations[target]” 赋值。
            return_observations[target] = val
        # 中文 L0126 前面的分支均未命中时，再判断 isinstance(value, Tensor)。
        elif isinstance(value, Tensor):
            # 中文 L0127 调用 value.float，用它的结果给 “val” 赋值。
            val = value.float()
            # 中文 L0128 条件 val.dim() == 1 成立时进入下方分支。
            if val.dim() == 1:
                # 中文 L0129 调用 val.unsqueeze，用它的结果给 “val” 赋值。
                val = val.unsqueeze(0)
            # 中文 L0130 引用已有的值 val给 “return_observations[target]” 赋值。
            return_observations[target] = val

    # 中文 L0132 引用已有的值 return_observations作为当前调用的返回值。
    return return_observations


# 中文 L0135 仿真已整理过键名时，把回合提供的批量 NumPy 观测转换为 Tensor。
def prepare_canonical_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # 中文 L0136 原文档概述本段功能，将标准仿真 NumPy 观测转换为 torch Tensor。
    """Convert batched sim observations to tensors; uint8 BHWC images become float BCHW in [0, 1]."""
    # 中文 L0137 创建字典，用键查找对应的值给 “result” 赋值。
    result = {}
    # 中文 L0138 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for name, value in observations.items():
        # 中文 L0139 用 from_numpy 把 NumPy 数组封装为 CPU Tensor。数据类型沿用 NumPy。
        tensor = torch.from_numpy(value)
        # 中文 L0140 四轴 uint8 对应批量的彩色相机输入，源轴顺序为 [B,H,W,C]。
        if value.dtype == np.uint8 and value.ndim == 4:
            # 中文 L0141 把通道移到第二轴，再转为 float32 和 [0,1] 范围，设备仍为 CPU。
            tensor = tensor.permute(0, 3, 1, 2).contiguous().float() / 255
        # 中文 L0142 前面的分支均未命中时，再判断 tensor.is_floating_point()。
        elif tensor.is_floating_point():
            # 中文 L0143 保持非图像浮点特征原有的轴，将元素类型统一为 float32。
            tensor = tensor.float()
        # 中文 L0144 引用已有的值 tensor给 “result[name]” 赋值。
        result[name] = tensor
    # 中文 L0145 引用已有的值 result作为当前调用的返回值。
    return result


# 中文 L0148 定义函数 env_to_policy_features。
def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    # TODO(jadechoghari, imstevenpmwork): remove this hardcoding of keys and just use the nested keys as is
    # (need to also refactor preprocess_observation and externalize normalization from policies)
    # 中文 L0151 创建字典，用键查找对应的值给 “policy_features” 赋值。
    policy_features = {}
    # 中文 L0152 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key, ft in env_cfg.features.items():
        # 中文 L0153 条件 ft.type is FeatureType.VISUAL 成立时进入下方分支。
        if ft.type is FeatureType.VISUAL:
            # 中文 L0154 条件 len(ft.shape) != 3 成立时进入下方分支。
            if len(ft.shape) != 3:
                # 中文 L0155 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={ft.shape})")

            # 中文 L0157 调用 get_channel_first_image_shape，用它的结果给 “shape”（各轴的长度） 赋值。
            shape = get_channel_first_image_shape(ft.shape)
            # 中文 L0158 调用 PolicyFeature，用它的结果给 “feature”（输入或输出特征描述） 赋值。
            feature = PolicyFeature(type=ft.type, shape=shape)
        # 中文 L0159 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0160 引用已有的值 ft给 “feature”（输入或输出特征描述） 赋值。
            feature = ft

        # Only envs that declare their own features map them; the others keep their keys.
        # 中文 L0163 调用 env_cfg.features_map.get，用它的结果给 “policy_key” 赋值。
        policy_key = env_cfg.features_map.get(key, key)
        # 中文 L0164 引用已有的值 feature给 “policy_features[policy_key]” 赋值。
        policy_features[policy_key] = feature

    # 中文 L0166 引用已有的值 policy_features作为当前调用的返回值。
    return policy_features


# 中文 L0169 定义函数 _observation_features。
def _observation_features(descriptor: EnvDescriptor) -> dict[str, PolicyFeature]:
    # 中文 L0170 原文档中的说明或续行。
    """Return the sim's observation features as the policy side receives them, images channel first."""
    # 中文 L0171 用字典推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
    return {
        # 中文 L0172 延续上面的语句，name: PolicyFeature(feature.type, (feature.shape[2], *feature.shape[:2]))。
        name: PolicyFeature(feature.type, (feature.shape[2], *feature.shape[:2]))
        # 中文 L0173 条件 feature.type is FeatureType.VISUAL 成立时进入下方分支。
        if feature.type is FeatureType.VISUAL
        # 中文 L0174 补充上方跨行表达式的分支值或条件。
        else feature
        # 中文 L0175 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for name, feature in descriptor_features(descriptor).items()
        # 中文 L0176 条件 name != ACTION 成立时进入下方分支。
        if name != ACTION
    # 中文 L0177 闭合上方尚未结束的括号。
    }


# TODO: Consider refactoring the validation ways in the future.
# 中文 L0181 在加载模型前核对仿真所供特征和 checkpoint 需要的输入输出。
def validate_checkpoint(
    # 中文 L0182 “descriptor”是当前括号中的一项，逗号将它与下一项分开。
    descriptor: EnvDescriptor,
    # 中文 L0183 “config”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    config: PreTrainedConfig,
    # 中文 L0184 延续上面的语句，rename_map: dict[str, str],。本段处理对照环境与权重配置的观测、动作和控制约定。
    rename_map: dict[str, str],
    # 中文 L0185 后续参数必须按名字传入。
    *,
    # 中文 L0186 为“env_preprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    # 中文 L0187 为“env_postprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
    env_postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
# 中文 L0188 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> None:
    """Raise if the checkpoint's features don't match the sim's `descriptor`.

    The checkpoint is admitted on what the environment processors produce, read from their
    `transform_features`: the observation after `env_preprocessor` and then `rename_map`, and the action
    after `env_postprocessor`. Image sizes and the policy's placeholder cameras (`empty_cameras`) are not checked.
    """
    # 中文 L0195 调用 _observation_features，用它的结果给 “observed” 赋值。
    observed = _observation_features(descriptor)
    # 中文 L0196 条件 env_preprocessor is not None 成立时进入下方分支。
    if env_preprocessor is not None:
        # 中文 L0197 创建字典，用键查找对应的值给 “features” 赋值。
        features = {PipelineFeatureType.OBSERVATION: observed, PipelineFeatureType.ACTION: {}}
        # 中文 L0198 用方括号按键或索引读取 容器给 “observed” 赋值。
        observed = env_preprocessor.transform_features(features)[PipelineFeatureType.OBSERVATION]
    # 中文 L0199 调用 rename_keys，用它的结果给 “available” 赋值。
    available = rename_keys(observed, rename_map)
    # 中文 L0200 用集合推导式从可迭代对象逐项取值并构造结果给 “placeholders” 赋值。
    placeholders = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(getattr(config, "empty_cameras", 0))}
    # 中文 L0201 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for name, expected in (config.input_features or {}).items():
        # 中文 L0202 条件 name in placeholders 成立时进入下方分支。
        if name in placeholders:
            # 中文 L0203 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0204 条件 name not in available 成立时进入下方分支。
        if name not in available:
            # 中文 L0205 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Checkpoint feature missing from the simulator (check --rename_map): {name}")
        # 中文 L0206 用方括号按键或索引读取 available给 “actual” 赋值。
        actual = available[name]
        # 中文 L0207 条件 actual.type is FeatureType.VISUAL 成立时进入下方分支。
        if actual.type is FeatureType.VISUAL:
            # 中文 L0208 组合多个布尔判断，按 Python 的短路规则求值给 “compatible” 赋值。
            compatible = len(expected.shape) == 3 and expected.shape[0] == actual.shape[0]
        # 中文 L0209 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0210 计算此行的右侧表达式给 “compatible” 赋值。
            compatible = tuple(expected.shape) == tuple(actual.shape)
        # 中文 L0211 条件 not compatible 成立时进入下方分支。
        if not compatible:
            # 中文 L0212 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0213 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Checkpoint feature shape mismatch for {name}: expected {expected.shape}, got {actual.shape}"
            # 中文 L0214 闭合上方尚未结束的括号。
            )
    # 中文 L0215 调用 get，用它的结果给 “action”（单步动作） 赋值。
    action = (config.output_features or {}).get(ACTION)
    # 中文 L0216 条件 action is None 成立时进入下方分支。
    if action is None:
        # 中文 L0217 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("Checkpoint must declare the canonical action feature")
    # 中文 L0218 引用已有的值 action给 “commanded” 赋值。
    commanded = action
    # 中文 L0219 条件 env_postprocessor is not None 成立时进入下方分支。
    if env_postprocessor is not None:
        # 中文 L0220 创建字典，用键查找对应的值给 “features” 赋值。
        features = {PipelineFeatureType.ACTION: {ACTION: action}, PipelineFeatureType.OBSERVATION: {}}
        # 中文 L0221 用方括号按键或索引读取 容器给 “commanded” 赋值。
        commanded = env_postprocessor.transform_features(features)[PipelineFeatureType.ACTION][ACTION]
    # 中文 L0222 条件 tuple(commanded.shape) != descriptor.action_feature.shape 成立时进入下方分支。
    if tuple(commanded.shape) != descriptor.action_feature.shape:
        # 中文 L0223 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0224 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Checkpoint action dimension differs from the simulator: the checkpoint predicts "
            # 中文 L0225 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"{tuple(action.shape)}, its environment processors command {tuple(commanded.shape)} and the "
            # 中文 L0226 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"simulator expects {descriptor.action_feature.shape}"
        # 中文 L0227 闭合上方尚未结束的括号。
        )


# 中文 L0230 定义函数 _sub_env_has_attr。
def _sub_env_has_attr(env: gym.vector.VectorEnv, attr: str) -> bool:
    # 中文 L0231 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0232 调用 env.get_attr 执行操作。
        env.get_attr(attr)
        # 中文 L0233 使用布尔值 True作为当前调用的返回值。
        return True
    # 中文 L0234 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except Exception:
        # 中文 L0235 使用布尔值 False作为当前调用的返回值。
        return False


# 中文 L0238 定义函数 check_env_attributes_and_types。
def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    # 中文 L0239 进入上下文管理器，缩进块结束时由它处理退出动作。
    with warnings.catch_warnings():
        # 中文 L0240 调用 warnings.simplefilter 执行操作。
        warnings.simplefilter("once", UserWarning)

        # 中文 L0242 条件 not (_sub_env_has_attr(env, "task_description") and _sub_env_has_attr(env, "task")) 成立时进入下方分支。
        if not (_sub_env_has_attr(env, "task_description") and _sub_env_has_attr(env, "task")):
            # 中文 L0243 调用 warnings.warn 执行操作。
            warnings.warn(
                # 中文 L0244 字符串字面量，供上方的调用、容器或错误消息使用。
                "The environment does not have 'task_description' and 'task'. Some policies require these features.",
                # 中文 L0245 “UserWarning”是当前括号中的一项，逗号将它与下一项分开。
                UserWarning,
                # 中文 L0246 为“stacklevel”提供这一项的值，注意它所在的参数表或外层表达式。
                stacklevel=2,
            # 中文 L0247 闭合上方尚未结束的括号。
            )


# 中文 L0250 定义函数 _close_single_env。
def _close_single_env(env: Any) -> None:
    # 中文 L0251 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0252 调用 env.close，关闭对象或释放关联资源。
        env.close()
    # 中文 L0253 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except Exception as exc:
        # 中文 L0254 调用 logger.warning 执行操作。
        logger.warning("Failed to close env %s: %s", env, exc)


# 中文 L0257 根据第一个参数的运行时类型分派到不同的关闭实现。
@singledispatch
# 中文 L0258 通用关闭入口。具体容器类型和单个环境可以注册专用实现。
def close_envs(obj: Any) -> None:
    # 中文 L0259 原文档中的说明或续行。
    """Default: raise if the type is not recognized."""
    # 中文 L0260 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
    raise NotImplementedError(f"close_envs not implemented for type {type(obj).__name__}")


# 中文 L0263 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@close_envs.register
# 中文 L0264 定义函数 _。
def _(env: Mapping) -> None:
    # 中文 L0265 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for v in env.values():
        # 中文 L0266 条件 isinstance(v, Mapping) 成立时进入下方分支。
        if isinstance(v, Mapping):
            # 中文 L0267 调用 close_envs 执行操作。
            close_envs(v)
        # 中文 L0268 前面的分支均未命中时，再判断 hasattr(v, "close")。
        elif hasattr(v, "close"):
            # 中文 L0269 调用 _close_single_env 执行操作。
            _close_single_env(v)


# 中文 L0272 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@close_envs.register
# 中文 L0273 定义函数 _。
def _(envs: Sequence) -> None:
    # 中文 L0274 条件 isinstance(envs, (str | bytes)) 成立时进入下方分支。
    if isinstance(envs, (str | bytes)):
        # 中文 L0275 退出当前调用并返回 None。
        return
    # 中文 L0276 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for v in envs:
        # 中文 L0277 条件 isinstance(v, Mapping) or isinstance(v, Sequence) and not isinstance(v, (str | bytes)) 成立时进入下方分支。
        if isinstance(v, Mapping) or isinstance(v, Sequence) and not isinstance(v, (str | bytes)):
            # 中文 L0278 调用 close_envs 执行操作。
            close_envs(v)
        # 中文 L0279 前面的分支均未命中时，再判断 hasattr(v, "close")。
        elif hasattr(v, "close"):
            # 中文 L0280 调用 _close_single_env 执行操作。
            _close_single_env(v)


# 中文 L0283 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@close_envs.register
# 中文 L0284 定义函数 _。
def _(env: gym.Env) -> None:
    # 中文 L0285 调用 _close_single_env 执行操作。
    _close_single_env(env)
