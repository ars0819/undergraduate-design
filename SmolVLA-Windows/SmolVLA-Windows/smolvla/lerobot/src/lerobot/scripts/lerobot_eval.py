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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Requires: pip install 'lerobot[evaluation]' plus the policy extra (e.g. lerobot[pi])
          and the environment extra (e.g. lerobot[pusht]) if evaluating in simulation.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0052 导入模块 concurrent.futures，简称 cf。
import concurrent.futures as cf
# 中文 L0053 导入模块 json。
import json
# 中文 L0054 导入模块 logging。
import logging
# 中文 L0055 导入模块 threading。
import threading
# 中文 L0056 导入模块 time。
import time
# 中文 L0057 从模块 collections 导入 defaultdict。
from collections import defaultdict
# 中文 L0058 从模块 collections.abc 导入 Callable。
from collections.abc import Callable
# 中文 L0059 从模块 contextlib 导入 nullcontext。
from contextlib import nullcontext
# 中文 L0060 从模块 copy 导入 deepcopy。
from copy import deepcopy
# 中文 L0061 从模块 dataclasses 导入 asdict。
from dataclasses import asdict
# 中文 L0062 从模块 functools 导入 partial。
from functools import partial
# 中文 L0063 从模块 pathlib 导入 Path。
from pathlib import Path
# 中文 L0064 从模块 pprint 导入 pformat。
from pprint import pformat
# 中文 L0065 从模块 typing 导入 TYPE_CHECKING、Any、TypedDict。
from typing import TYPE_CHECKING, Any, TypedDict

# 中文 L0067 导入模块 einops。
import einops
# 中文 L0068 导入模块 gymnasium，简称 gym。
import gymnasium as gym
# 中文 L0069 导入模块 numpy，简称 np。
import numpy as np
# 中文 L0070 导入模块 torch。
import torch
# 中文 L0071 从模块 termcolor 导入 colored。
from termcolor import colored
# 中文 L0072 从模块 torch 导入 Tensor、nn。
from torch import Tensor, nn
# 中文 L0073 从模块 tqdm 导入 trange。
from tqdm import trange

# 中文 L0075 从配置包导入配置类型和命令行包装模块 parser。
from lerobot.configs import FeatureType, PolicyFeature, parser
# 中文 L0076 从 configs/eval.py 导入实际的评测配置类。eval_main 的参数类型会引用它。
from lerobot.configs.eval import EvalPipelineConfig
# 中文 L0077 从模块 lerobot.datasets.lerobot_dataset 导入 LeRobotDataset。
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# 中文 L0078 从环境包公开的接口导入下列环境工具。找定义时可先点进 envs/__init__.py。
from lerobot.envs import (
    # 中文 L0079 “check_env_attributes_and_types”是当前括号中的一项，逗号将它与下一项分开。
    check_env_attributes_and_types,
    # 中文 L0080 导入关闭环境的接口。可以从环境包的导出位置继续定位实现。
    close_envs,
    # 中文 L0081 导入创建任务环境集合的工厂函数 make_env。
    make_env,
    # 中文 L0082 导入选择环境专属前后处理器的工厂函数。
    make_env_pre_post_processors,
    # 中文 L0083 导入通用观测转换。此处是导入语句，具体评测走哪个分支要看第 292 行。
    preprocess_observation,
# 中文 L0084 闭合上方尚未结束的括号。
)
# 中文 L0085 从模块 lerobot.envs.factory 导入 descriptor_features。
from lerobot.envs.factory import descriptor_features
# 中文 L0086 同时导入标准仿真观测转换和 checkpoint 兼容性检查。
from lerobot.envs.utils import prepare_canonical_observation, validate_checkpoint
# 中文 L0087 从模块 lerobot.lerobot_types 导入 PolicyAction。
from lerobot.lerobot_types import PolicyAction
# 中文 L0088 导入策略基类及策略、处理流水线的工厂。
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
# 中文 L0089 从模块 lerobot.processor 导入下方列出的多个名字。
from lerobot.processor import (
    # 中文 L0090 “PolicyProcessorPipeline”是当前括号中的一项，逗号将它与下一项分开。
    PolicyProcessorPipeline,
    # 中文 L0091 “bind_relative_anchor”是当前括号中的一项，逗号将它与下一项分开。
    bind_relative_anchor,
    # 中文 L0092 “policy_output_to_transition”是当前括号中的一项，逗号将它与下一项分开。
    policy_output_to_transition,
    # 中文 L0093 “transition_to_policy_action”是当前括号中的一项，逗号将它与下一项分开。
    transition_to_policy_action,
# 中文 L0094 闭合上方尚未结束的括号。
)
# 中文 L0095 从模块 lerobot.utils.constants 导入 ACTION、DONE、OBS_IMAGE、OBS_IMAGES、OBS_STR、REWARD。
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, OBS_IMAGES, OBS_STR, REWARD
# 中文 L0096 从模块 lerobot.utils.device_utils 导入 get_safe_torch_device。
from lerobot.utils.device_utils import get_safe_torch_device
# 中文 L0097 从模块 lerobot.utils.eval_stats 导入 success_summary。
from lerobot.utils.eval_stats import success_summary
# 中文 L0098 从模块 lerobot.utils.import_utils 导入 _peft_available、register_third_party_plugins、require_package。
from lerobot.utils.import_utils import _peft_available, register_third_party_plugins, require_package
# 中文 L0099 从模块 lerobot.utils.io_utils 导入 write_video。
from lerobot.utils.io_utils import write_video
# 中文 L0100 从模块 lerobot.utils.random_utils 导入 set_seed。
from lerobot.utils.random_utils import set_seed
# 中文 L0101 从模块 lerobot.utils.utils 导入 init_logging、inside_slurm。
from lerobot.utils.utils import (
    # 中文 L0102 “init_logging”是当前括号中的一项，逗号将它与下一项分开。
    init_logging,
    # 中文 L0103 “inside_slurm”是当前括号中的一项，逗号将它与下一项分开。
    inside_slurm,
# 中文 L0104 闭合上方尚未结束的括号。
)

# 中文 L0106 条件 TYPE_CHECKING or _peft_available 成立时进入下方分支。
if TYPE_CHECKING or _peft_available:
    # 中文 L0107 从模块 peft 导入 PeftModel。
    from peft import PeftModel
# 中文 L0108 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
else:
    # 中文 L0109 设为空值 None给 “PeftModel” 赋值。
    PeftModel = None


# 中文 L0112 调用 logging.getLogger，用它的结果给 “logger” 赋值。
logger = logging.getLogger(__name__)


# 中文 L0115 定义函数 _env_features_to_dataset_features。
def _env_features_to_dataset_features(env_features: dict[str, PolicyFeature]) -> dict[str, dict[str, Any]]:
    # 中文 L0116 原文档中的说明或续行。
    """Convert EnvConfig.features to the dict format expected by LeRobotDataset.create()."""
    # 中文 L0117 声明“features”的类型为 dict[str, dict[str, Any]]，创建字典，用键查找对应的值作为初始值。
    features: dict[str, dict[str, Any]] = {}
    # 中文 L0118 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key, ft in env_features.items():
        # 中文 L0119 调用 tuple，用它的结果给 “shape”（各轴的长度） 赋值。
        shape = tuple(ft.shape)
        # 中文 L0120 条件 ft.type is FeatureType.VISUAL 成立时进入下方分支。
        if ft.type is FeatureType.VISUAL:
            # 中文 L0121 创建字典，用键查找对应的值给 “features[key]” 赋值。
            features[key] = {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"]}
        # 中文 L0122 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0123 创建字典，用键查找对应的值给 “features[key]” 赋值。
            features[key] = {"dtype": "float32", "shape": shape, "names": None}
    # 中文 L0124 创建字典，用键查找对应的值给 “features['next.reward']” 赋值。
    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    # 中文 L0125 创建字典，用键查找对应的值给 “features['next.success']” 赋值。
    features["next.success"] = {"dtype": "bool", "shape": (1,), "names": None}
    # 中文 L0126 创建字典，用键查找对应的值给 “features['next.done']” 赋值。
    features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
    # 中文 L0127 引用已有的值 features作为当前调用的返回值。
    return features


# 中文 L0130 定义函数 _build_raw_frame。
def _build_raw_frame(
    # 中文 L0131 “raw_obs”是当前括号中的一项，逗号将它与下一项分开。
    raw_obs: dict,
    # 中文 L0132 “env_idx”是当前括号中的一项，逗号将它与下一项分开。
    env_idx: int,
    # 中文 L0133 “action”（单步动作）是当前括号中的一项，逗号将它与下一项分开。
    action: np.ndarray,
    # 中文 L0134 “reward”（单步奖励）是当前括号中的一项，逗号将它与下一项分开。
    reward: float,
    # 中文 L0135 “success”是当前括号中的一项，逗号将它与下一项分开。
    success: bool,
    # 中文 L0136 “done”是当前括号中的一项，逗号将它与下一项分开。
    done: bool,
    # 中文 L0137 “task”（任务）是当前括号中的一项，逗号将它与下一项分开。
    task: str,
    # 中文 L0138 “env_features”是当前括号中的一项，逗号将它与下一项分开。
    env_features: dict,
# 中文 L0139 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict:
    """Build a dataset frame from raw env observations for one env index.

    Keys in the frame match the keys in env_features so they align with the
    dataset schema created by _env_features_to_dataset_features().
    """
    # 中文 L0145 声明“frame”的类型为 dict[str, Any]，创建字典，用键查找对应的值作为初始值。
    frame: dict[str, Any] = {}
    # 中文 L0146 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key in env_features:
        # 中文 L0147 条件 key == ACTION 成立时进入下方分支。
        if key == ACTION:
            # 中文 L0148 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0149 条件 key.startswith("next.") 成立时进入下方分支。
        if key.startswith("next."):
            # 中文 L0150 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0151 条件 "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict) 成立时进入下方分支。
        if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
            # 中文 L0152 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for cam_name, img in raw_obs["pixels"].items():
                # 中文 L0153 计算此行的右侧表达式给 “candidate” 赋值。
                candidate = f"{OBS_IMAGES}.{cam_name}"
                # 中文 L0154 条件 candidate == key 成立时进入下方分支。
                if candidate == key:
                    # 中文 L0155 用方括号按键或索引读取 img给 “frame[key]” 赋值。
                    frame[key] = img[env_idx]
            # 中文 L0156 条件 key in frame 成立时进入下方分支。
            if key in frame:
                # 中文 L0157 跳过这一轮剩余操作，进入下一轮循环。
                continue
        # 中文 L0158 条件 "pixels" in raw_obs and not isinstance(raw_obs["pixels"], dict) and key in ("pixels", OBS_IMAGE) 成立时进入下方分支。
        if "pixels" in raw_obs and not isinstance(raw_obs["pixels"], dict) and key in ("pixels", OBS_IMAGE):
            # 中文 L0159 用方括号按键或索引读取 容器给 “frame[key]” 赋值。
            frame[key] = raw_obs["pixels"][env_idx]
            # 中文 L0160 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0161 条件 key in raw_obs and isinstance(raw_obs[key], np.ndarray) 成立时进入下方分支。
        if key in raw_obs and isinstance(raw_obs[key], np.ndarray):
            # 中文 L0162 用方括号按键或索引读取 容器给 “val” 赋值。
            val = raw_obs[key][env_idx]
            # 中文 L0163 条件 val.dtype == np.float64 成立时进入下方分支。
            if val.dtype == np.float64:
                # 中文 L0164 调用 val.astype，用它的结果给 “val” 赋值。
                val = val.astype(np.float32)
            # 中文 L0165 引用已有的值 val给 “frame[key]” 赋值。
            frame[key] = val
    # 中文 L0166 引用已有的值 action给 “frame[ACTION]” 赋值。
    frame[ACTION] = action
    # 中文 L0167 调用 np.atleast_1d，用它的结果给 “frame['next.reward']” 赋值。
    frame["next.reward"] = np.atleast_1d(np.float32(reward))
    # 中文 L0168 调用 np.atleast_1d，用它的结果给 “frame['next.success']” 赋值。
    frame["next.success"] = np.atleast_1d(np.bool_(success))
    # 中文 L0169 调用 np.atleast_1d，用它的结果给 “frame['next.done']” 赋值。
    frame["next.done"] = np.atleast_1d(np.bool_(done))
    # 中文 L0170 引用已有的值 task给 “frame['task']” 赋值。
    frame["task"] = task
    # 中文 L0171 引用已有的值 frame作为当前调用的返回值。
    return frame


# 中文 L0174 定义 rollout，一次处理一批回合，在循环里把每一步观测转成动作。
def rollout(
    # 中文 L0175 “env”是当前括号中的一项，逗号将它与下一项分开。
    env: gym.vector.VectorEnv,
    # 中文 L0176 “policy”是当前括号中的一项，逗号将它与下一项分开。
    policy: PreTrainedPolicy,
    # 中文 L0177 延续上面的语句，env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0178 延续上面的语句，env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0179 延续上面的语句，preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0180 延续上面的语句，postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    # 中文 L0181 为“seeds”提供这一项的值，注意它所在的参数表或外层表达式。
    seeds: list[int] | None = None,
    # 中文 L0182 为“return_observations”提供这一项的值，注意它所在的参数表或外层表达式。
    return_observations: bool = False,
    # 中文 L0183 为“render_callback”提供这一项的值，注意它所在的参数表或外层表达式。
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    # 中文 L0184 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_dir: Path | None = None,
    # 中文 L0185 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
    env_features: dict | None = None,
    # 中文 L0186 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_repo_id: str | None = None,
    # 中文 L0187 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_private: bool = False,
    # 中文 L0188 为“predicted_latents_callback”提供这一项的值，注意它所在的参数表或外层表达式。
    predicted_latents_callback: Callable[[PreTrainedPolicy], None] | None = None,
    # 中文 L0189 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
    max_episode_steps: int | None = None,
# 中文 L0190 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
        predicted_latents_callback: Optional callback invoked after every ``select_action`` with the policy
            itself. World-model policies (e.g. LingBot-VA) stash predicted video latents on
            ``policy.last_predicted_latents``; this lets the caller concatenate chunks and decode once.
    Returns:
        The dictionary described above.
    """
    # 中文 L0225 检查此条件，条件不成立时抛出 AssertionError。
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy, its processors and the environments. The processors matter: a step that
    # latches per-episode state -- e.g. ``RelativeActionsProcessorStep``'s cached anchor -- would
    # otherwise carry the previous batch of episodes' value into this one.
    # ``SyncInferenceEngine`` already does this on the robot.
    # 中文 L0231 调用实际策略的 reset 方法。SmolVLA 在这里清空动作队列。
    policy.reset()
    # 中文 L0232 重置策略前处理流水线保存的跨步状态。
    preprocessor.reset()
    # 中文 L0233 重置动作后处理流水线保存的跨步状态。
    postprocessor.reset()
    # 中文 L0234 重置向量环境，得到批量初始观测和额外信息。
    observation, info = env.reset(seed=seeds)
    # 中文 L0235 条件 render_callback is not None 成立时进入下方分支。
    if render_callback is not None:
        # 中文 L0236 调用 render_callback 执行操作。
        render_callback(env)

    # 中文 L0238 声明“recording_datasets”的类型为 list[LeRobotDataset] | None，设为空值 None作为初始值。
    recording_datasets: list[LeRobotDataset] | None = None
    # 中文 L0239 设为空值 None给 “raw_observation” 赋值。
    raw_observation = None
    # 中文 L0240 使用字面值 ''给 “task_desc” 赋值。
    task_desc = ""
    # 中文 L0241 条件 recording_dir is not None and env_features is not None 成立时进入下方分支。
    if recording_dir is not None and env_features is not None:
        # 中文 L0242 调用 _env_features_to_dataset_features，用它的结果给 “features” 赋值。
        features = _env_features_to_dataset_features(env_features)
        # 中文 L0243 调用 env.unwrapped.metadata.get，用它的结果给 “fps” 赋值。
        fps = env.unwrapped.metadata.get("render_fps", 30)
        # 中文 L0244 创建列表，按给定顺序保留元素给 “recording_datasets” 赋值。
        recording_datasets = []
        # 中文 L0245 计算此行的右侧表达式给 “multi_env” 赋值。
        multi_env = env.num_envs > 1
        # 中文 L0246 组合多个布尔判断，按 Python 的短路规则求值给 “base_repo_id” 赋值。
        base_repo_id = recording_repo_id or "eval_recording"
        # 中文 L0247 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for i in range(env.num_envs):
            # 中文 L0248 用条件表达式在 if 前与 else 后的两个值之间选择给 “root” 赋值。
            root = str(recording_dir / f"env_{i}") if multi_env else str(recording_dir)
            # 中文 L0249 用条件表达式在 if 前与 else 后的两个值之间选择给 “repo_id”（Hub 仓库标识） 赋值。
            repo_id = f"{base_repo_id}_env_{i}" if multi_env else base_repo_id
            # 中文 L0250 调用 recording_datasets.append，在列表末尾追加一个元素。
            recording_datasets.append(
                # 中文 L0251 延续上面的语句，LeRobotDataset.create(。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                LeRobotDataset.create(
                    # 中文 L0252 为“repo_id”（Hub 仓库标识）提供这一项的值，注意它所在的参数表或外层表达式。
                    repo_id=repo_id,
                    # 中文 L0253 为“fps”提供这一项的值，注意它所在的参数表或外层表达式。
                    fps=fps,
                    # 中文 L0254 为“features”提供这一项的值，注意它所在的参数表或外层表达式。
                    features=features,
                    # 中文 L0255 为“root”提供这一项的值，注意它所在的参数表或外层表达式。
                    root=root,
                    # 中文 L0256 为“use_videos”提供这一项的值，注意它所在的参数表或外层表达式。
                    use_videos=True,
                # 中文 L0257 闭合上方尚未结束的括号。
                )
            # 中文 L0258 闭合上方尚未结束的括号。
            )
        # 中文 L0259 调用 deepcopy，用它的结果给 “raw_observation” 赋值。
        raw_observation = deepcopy(observation)
        # 中文 L0260 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0261 用方括号按键或索引读取 容器给 “task_desc” 赋值。
            task_desc = list(env.call("task_description"))[0]
        # 中文 L0262 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except (AttributeError, NotImplementedError):
            # 中文 L0263 使用字面值 ''给 “task_desc” 赋值。
            task_desc = ""

    # 中文 L0265 创建列表，按给定顺序保留元素给 “all_observations” 赋值。
    all_observations = []
    # 中文 L0266 创建列表，按给定顺序保留元素给 “all_actions” 赋值。
    all_actions = []
    # 中文 L0267 创建列表，按给定顺序保留元素给 “all_rewards” 赋值。
    all_rewards = []
    # 中文 L0268 创建列表，按给定顺序保留元素给 “all_successes” 赋值。
    all_successes = []
    # 中文 L0269 创建列表，按给定顺序保留元素给 “all_dones” 赋值。
    all_dones = []
    # What the simulator reports besides success, one row per step; empty for benchmarks that report nothing.
    # 中文 L0271 声明“all_metrics”的类型为 list[dict[str, np.ndarray]]，创建列表，按给定顺序保留元素作为初始值。
    all_metrics: list[dict[str, np.ndarray]] = []

    # A relative-action policy predicts a chunk of offsets anchored to the state at prediction
    # time, but this loop reruns the pre/post pipeline every step, which would re-anchor queued
    # actions to the current (moved) state. Bind the step to the policy's queue depth so it holds
    # the anchor until the chunk drains. Done here rather than in `eval_main` so that every caller
    # of this public loop is covered; rebinding the same callable each rollout is a no-op.
    # The anchor is [B, state_dim] and tracks exactly what the single shared action queue tracks,
    # so it inherits the queue's batching assumptions rather than adding any: every sub-env
    # refills on the same step, and a sub-env that finished early stays frozen at its final state
    # and its transitions are discarded.
    # 中文 L0282 为使用相对动作处理步骤的策略连接队列深度与锚点。此分支由处理流水线是否包含对应步骤决定。
    if bind_relative_anchor(policy, preprocessor) is not None:
        # 中文 L0283 调用 logging.info 执行操作。
        logging.info("Relative actions enabled: chunk anchor held until the action queue drains")

    # 中文 L0285 使用字面值 0给 “step” 赋值。
    step = 0
    # Keep track of which environments are done.
    # 中文 L0287 创建形状为 [B] 的 NumPy 布尔数组，开始时每个环境都未结束。
    done = np.array([False] * env.num_envs)
    # 中文 L0288 向底层环境询问每回合的步数上限。
    max_steps = env.call("_max_episode_steps")[0]
    # 中文 L0289 条件 max_episode_steps is not None 成立时进入下方分支。
    if max_episode_steps is not None:
        # 中文 L0290 调用 min，用它的结果给 “max_steps” 赋值。
        max_steps = min(max_steps, max_episode_steps)
    # Sim envs already return observations with LeRobot keys; other envs return their own format.
    # 中文 L0292 用条件表达式在 if 前与 else 后的两个值之间选择给 “prepare_observation” 赋值。
    prepare_observation = (
        # 中文 L0293 对已使用 LeRobot 观测键的仿真，选择 NumPy 到 Tensor 的转换。
        prepare_canonical_observation
        # 中文 L0294 LocalVectorEnv 的 metadata 将此标记设为 True，本次 LIBERO 因而使用上一行的函数。
        if env.metadata.get("canonical_observations", False)
        # 中文 L0295 补充上方跨行表达式的分支值或条件。
        else preprocess_observation
    # 中文 L0296 闭合上方尚未结束的括号。
    )
    # 中文 L0297 调用 trange，用它的结果给 “progbar” 赋值。
    progbar = trange(
        # 中文 L0298 “max_steps”是当前括号中的一项，逗号将它与下一项分开。
        max_steps,
        # 中文 L0299 为“desc”提供这一项的值，注意它所在的参数表或外层表达式。
        desc=f"Running rollout with at most {max_steps} steps",
        # 中文 L0300 为“disable”提供这一项的值，注意它所在的参数表或外层表达式。
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        # 中文 L0301 为“leave”提供这一项的值，注意它所在的参数表或外层表达式。
        leave=False,
    # 中文 L0302 闭合上方尚未结束的括号。
    )
    # 中文 L0303 调用 check_env_attributes_and_types 执行操作。
    check_env_attributes_and_types(env)
    # 中文 L0304 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0305 只有还存在未结束的环境且没达到步数上限时才继续控制循环。
        while not np.all(done) and step < max_steps:
            # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
            # 中文 L0307 将本轮观测变成 Tensor。双相机分别是 [1,3,360,360] 的 CPU float32，状态是 [1,8] 的 CPU float32。
            observation = prepare_observation(observation)
            # 中文 L0308 条件 return_observations 成立时进入下方分支。
            if return_observations:
                # 中文 L0309 调用 all_observations.append，在列表末尾追加一个元素。
                all_observations.append(deepcopy(observation))

            # Infer "task" from sub-environments (prefer natural language description).
            # env.call() works with both SyncVectorEnv and AsyncVectorEnv.
            # 中文 L0313 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
            try:
                # 中文 L0314 从底层环境取自然语言任务描述，按批次存成长度为 B 的 Python 列表。
                observation["task"] = list(env.call("task_description"))
            # 中文 L0315 捕获这里列出的异常类型，然后执行下方的处理逻辑。
            except (AttributeError, NotImplementedError):
                # 中文 L0316 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
                try:
                    # 中文 L0317 调用 list，用它的结果给 “observation['task']” 赋值。
                    observation["task"] = list(env.call("task"))
                # 中文 L0318 捕获这里列出的异常类型，然后执行下方的处理逻辑。
                except (AttributeError, NotImplementedError):
                    # 中文 L0319 计算右侧表达式 [""] * env.num_envs给 “observation['task']” 赋值。
                    observation["task"] = [""] * env.num_envs

            # Apply environment-specific preprocessing (e.g., IsaaclabArenaProcessorStep for Isaac Lab Arena)
            # 中文 L0322 运行环境级前处理，本次 SmolVLA + LIBERO 对应空流水线。
            observation = env_preprocessor(observation)

            # 中文 L0324 运行 checkpoint 的策略前处理。相机改名，任务文字分词，数据移到 CUDA，状态按保存的统计量归一化。
            observation = preprocessor(observation)
            # 中文 L0325 进入 PyTorch 推理模式，关闭梯度追踪并省去额外的张量追踪开销。
            with torch.inference_mode():
                # 中文 L0326 调用 transition_to_policy_action，用它的结果给 “action”（单步动作） 赋值。
                action = transition_to_policy_action(
                    # 中文 L0327 调用策略对象的 select_action，获得本控制步的动作。SmolVLA 返回形状为 [B,A] 的张量。
                    policy_output_to_transition(policy.select_action(observation))
                # 中文 L0328 闭合上方尚未结束的括号。
                )
            # 中文 L0329 条件 predicted_latents_callback is not None 成立时进入下方分支。
            if predicted_latents_callback is not None:
                # 中文 L0330 调用 predicted_latents_callback 执行操作。
                predicted_latents_callback(policy)
            # 中文 L0331 将动作反归一化并移回 CPU。此时动作仍为 Tensor。
            action = postprocessor(action)

            # 中文 L0333 创建字典，用键查找对应的值给 “action_transition” 赋值。
            action_transition = {ACTION: action}
            # 中文 L0334 运行环境级动作后处理。本次配置走空流水线。
            action_transition = env_postprocessor(action_transition)
            # 中文 L0335 用方括号按键或索引读取 action_transition给 “action”（单步动作） 赋值。
            action = action_transition[ACTION]

            # Convert to CPU / numpy.
            # 中文 L0338 把动作转换成 CPU 上的 NumPy 数组，保留批次轴。原评测形状为 [1,7]。
            action_numpy: np.ndarray = action.to("cpu").numpy()
            # 中文 L0339 断言动作有且只有两个轴，先区分环境实例，再区分动作分量。
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

            # Apply the next action.
            # 中文 L0342 把动作送入 LocalVectorEnv，返回下一批观测、奖励、终止、截断和辅助信息。
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            # 中文 L0343 条件 render_callback is not None 成立时进入下方分支。
            if render_callback is not None:
                # 中文 L0344 调用 render_callback 执行操作。
                render_callback(env)

            # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
            # available if none of the envs finished.
            # 中文 L0348 条件 "final_info" in info 成立时进入下方分支。
            if "final_info" in info:
                # 中文 L0349 用方括号按键或索引读取 info给 “final_info” 赋值。
                final_info = info["final_info"]
                # 中文 L0350 条件 isinstance(final_info, dict) 成立时进入下方分支。
                if isinstance(final_info, dict):
                    # 中文 L0351 调用 final_info.get，用它的结果给 “is_success”（仿真的成功标志） 赋值。
                    is_success = final_info.get("is_success", [False] * env.num_envs)
                    # 中文 L0352 用条件表达式在 if 前与 else 后的两个值之间选择给 “successes” 赋值。
                    successes = (
                        # 中文 L0353 延续上面的语句，is_success.tolist()。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                        is_success.tolist()
                        # 中文 L0354 条件 hasattr(is_success, "tolist") 成立时进入下方分支。
                        if hasattr(is_success, "tolist")
                        # 中文 L0355 补充上方跨行表达式的分支值或条件。
                        else [bool(is_success)] * env.num_envs
                    # 中文 L0356 闭合上方尚未结束的括号。
                    )
                # 中文 L0357 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
                else:
                    # Gymnasium < 1.0 returns final_info as a per-env sequence/object array,
                    # with entries set to a dict only for envs that just finished.
                    # 中文 L0360 创建列表，按给定顺序保留元素给 “successes” 赋值。
                    successes = []
                    # 中文 L0361 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                    for item in final_info:
                        # 中文 L0362 条件 isinstance(item, dict) and "is_success" in item 成立时进入下方分支。
                        if isinstance(item, dict) and "is_success" in item:
                            # 中文 L0363 调用 successes.append，在列表末尾追加一个元素。
                            successes.append(bool(item["is_success"]))
                        # 中文 L0364 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
                        else:
                            # 中文 L0365 调用 successes.append，在列表末尾追加一个元素。
                            successes.append(False)
            # 中文 L0366 在结果没有 final_info 而直接提供 is_success 时采用这一分支。LocalVectorEnv 通过 StepResult.info 提供这个字段。
            elif "is_success" in info:
                # 中文 L0367 取出一批环境的成功标志。
                is_success = info["is_success"]
                # 中文 L0368 用条件表达式在 if 前与 else 后的两个值之间选择给 “successes” 赋值。
                successes = (
                    # 中文 L0369 延续上面的语句，is_success.tolist()。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                    is_success.tolist()
                    # 中文 L0370 条件 hasattr(is_success, "tolist") 成立时进入下方分支。
                    if hasattr(is_success, "tolist")
                    # 中文 L0371 补充上方跨行表达式的分支值或条件。
                    else [bool(is_success)] * env.num_envs
                # 中文 L0372 闭合上方尚未结束的括号。
                )
            # 中文 L0373 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0374 计算右侧表达式 [False] * env.num_envs给 “successes” 赋值。
                successes = [False] * env.num_envs

            # 中文 L0376 按环境对终止和截断做逐元素逻辑或，得到本步回合结束标记。
            episode_ends = terminated | truncated
            # The rollout can end before the simulator declares the episode over.
            # Use the same boundary in the recorded datasets and returned transitions.
            # 中文 L0379 最后一个许可控制步达到上限时，由评测器结束剩余回合。
            if step + 1 == max_steps:
                # 中文 L0380 调用 np.ones_like，用它的结果给 “episode_ends” 赋值。
                episode_ends = np.ones_like(episode_ends, dtype=bool)

            # 中文 L0382 条件 recording_datasets is not None and raw_observation is not None 成立时进入下方分支。
            if recording_datasets is not None and raw_observation is not None:
                # 中文 L0383 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for env_idx in range(env.num_envs):
                    # 中文 L0384 条件 done[env_idx] 成立时进入下方分支。
                    if done[env_idx]:
                        # 中文 L0385 跳过这一轮剩余操作，进入下一轮循环。
                        continue
                    # 中文 L0386 调用 _build_raw_frame，用它的结果给 “frame” 赋值。
                    frame = _build_raw_frame(
                        # 中文 L0387 “raw_observation”是当前括号中的一项，逗号将它与下一项分开。
                        raw_observation,
                        # 中文 L0388 “env_idx”是当前括号中的一项，逗号将它与下一项分开。
                        env_idx,
                        # 中文 L0389 延续上面的语句，action_numpy[env_idx],。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                        action_numpy[env_idx],
                        # 中文 L0390 延续上面的语句，reward[env_idx],。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                        reward[env_idx],
                        # 中文 L0391 延续上面的语句，successes[env_idx],。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                        successes[env_idx],
                        # 中文 L0392 延续上面的语句，bool(episode_ends[env_idx]),。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                        bool(episode_ends[env_idx]),
                        # 中文 L0393 “task_desc”是当前括号中的一项，逗号将它与下一项分开。
                        task_desc,
                        # 中文 L0394 延续上面的语句，recording_datasets[env_idx].features,。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                        recording_datasets[env_idx].features,
                    # 中文 L0395 闭合上方尚未结束的括号。
                    )
                    # 中文 L0396 调用 add_frame 执行操作。
                    recording_datasets[env_idx].add_frame(frame)
                    # 中文 L0397 条件 episode_ends[env_idx] 成立时进入下方分支。
                    if episode_ends[env_idx]:
                        # 中文 L0398 调用 save_episode 执行操作。
                        recording_datasets[env_idx].save_episode()
                # 中文 L0399 调用 deepcopy，用它的结果给 “raw_observation” 赋值。
                raw_observation = deepcopy(observation)

            # 中文 L0401 计算右侧表达式 episode_ends | done给 “done” 赋值。
            done = episode_ends | done

            # 中文 L0403 调用 all_actions.append，在列表末尾追加一个元素。
            all_actions.append(torch.from_numpy(action_numpy))
            # 中文 L0404 调用 all_rewards.append，在列表末尾追加一个元素。
            all_rewards.append(torch.from_numpy(reward))
            # 中文 L0405 调用 all_dones.append，在列表末尾追加一个元素。
            all_dones.append(torch.from_numpy(done))
            # 中文 L0406 调用 all_successes.append，在列表末尾追加一个元素。
            all_successes.append(torch.tensor(successes))
            # 中文 L0407 调用 all_metrics.append，在列表末尾追加一个元素。
            all_metrics.append(
                # 中文 L0408 延续上面的语句，{。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                {
                    # 中文 L0409 为“key”提供这一项的值，注意它所在的参数表或外层表达式。
                    key: np.asarray(value, dtype=np.float32)
                    # Gymnasium's vector envs add a boolean presence mask per key of a sub-env's info.
                    # 中文 L0411 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                    for key, value in (info.get("metrics") or {}).items()
                    # 中文 L0412 条件 not key.startswith("_") 成立时进入下方分支。
                    if not key.startswith("_")
                # 中文 L0413 闭合上方尚未结束的括号。
                }
            # 中文 L0414 闭合上方尚未结束的括号。
            )

            # 中文 L0416 用增量赋值更新 step，运算逻辑见左侧运算符。
            step += 1
            # 中文 L0417 调用 mean，用它的结果给 “running_success_rate” 赋值。
            running_success_rate = (
                # 中文 L0418 延续上面的语句，einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()。本段处理重置环境并逐控制步执行动作、保存观察和成功标志。
                einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
            # 中文 L0419 闭合上方尚未结束的括号。
            )
            # 中文 L0420 调用 progbar.set_postfix 执行操作。
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            # 中文 L0421 调用 progbar.update，把新条目写入已有容器。
            progbar.update()
    # 中文 L0422 离开 try 时执行这里的清理，即使前面的路径发生异常。
    finally:
        # 中文 L0423 条件 recording_datasets is not None 成立时进入下方分支。
        if recording_datasets is not None:
            # 中文 L0424 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for ds in recording_datasets:
                # 中文 L0425 调用 ds.finalize 执行操作。
                ds.finalize()
                # 中文 L0426 条件 recording_repo_id is not None 成立时进入下方分支。
                if recording_repo_id is not None:
                    # 中文 L0427 条件 ds.num_episodes > 0 成立时进入下方分支。
                    if ds.num_episodes > 0:
                        # 中文 L0428 调用 ds.push_to_hub 执行操作。
                        ds.push_to_hub(private=recording_private)
                    # 中文 L0429 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
                    else:
                        # 中文 L0430 调用 logging.warning 执行操作。
                        logging.warning("No episodes recorded for %s — skipping push to hub.", ds.repo_id)

    # Track the final observation.
    # 中文 L0433 条件 return_observations 成立时进入下方分支。
    if return_observations:
        # 中文 L0434 调用 prepare_observation，用它的结果给 “observation”（观测） 赋值。
        observation = prepare_observation(observation)
        # 中文 L0435 调用 all_observations.append，在列表末尾追加一个元素。
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    # 中文 L0438 创建字典，用键查找对应的值给 “ret” 赋值。
    ret = {
        # 中文 L0439 为“ACTION”提供这一项的值，注意它所在的参数表或外层表达式。
        ACTION: torch.stack(all_actions, dim=1),
        # 中文 L0440 字典项“reward”（单步奖励），冒号右边给出它的值。
        "reward": torch.stack(all_rewards, dim=1),
        # 中文 L0441 字典项“success”，冒号右边给出它的值。
        "success": torch.stack(all_successes, dim=1),
        # 中文 L0442 字典项“done”，冒号右边给出它的值。
        "done": torch.stack(all_dones, dim=1),
    # 中文 L0443 闭合上方尚未结束的括号。
    }
    # 中文 L0444 调用 sorted，用它的结果给 “metric_names” 赋值。
    metric_names = sorted({key for metrics in all_metrics for key in metrics})
    # 中文 L0445 条件 metric_names 成立时进入下方分支。
    if metric_names:
        # 中文 L0446 调用 np.zeros，用它的结果给 “missing” 赋值。
        missing = np.zeros(env.num_envs, dtype=np.float32)
        # 中文 L0447 用字典推导式从可迭代对象逐项取值并构造结果给 “ret['metrics']” 赋值。
        ret["metrics"] = {
            # 中文 L0448 为“name”提供这一项的值，注意它所在的参数表或外层表达式。
            name: torch.from_numpy(np.stack([metrics.get(name, missing) for metrics in all_metrics], axis=1))
            # 中文 L0449 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for name in metric_names
        # 中文 L0450 闭合上方尚未结束的括号。
        }
    # 中文 L0451 条件 return_observations 成立时进入下方分支。
    if return_observations:
        # 中文 L0452 创建字典，用键查找对应的值给 “stacked_observations” 赋值。
        stacked_observations = {}
        # 中文 L0453 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key in all_observations[0]:
            # 中文 L0454 调用 torch.stack，用它的结果给 “stacked_observations[key]” 赋值。
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        # 中文 L0455 引用已有的值 stacked_observations给 “ret[OBS_STR]” 赋值。
        ret[OBS_STR] = stacked_observations

    # 中文 L0457 条件 hasattr(policy, "use_original_modules") 成立时进入下方分支。
    if hasattr(policy, "use_original_modules"):
        # 中文 L0458 调用 policy.use_original_modules 执行操作。
        policy.use_original_modules()

    # 中文 L0460 引用已有的值 ret作为当前调用的返回值。
    return ret


# 中文 L0463 定义给一组相同任务环境评测多个回合的函数。
def eval_policy(
    # 中文 L0464 “env”是当前括号中的一项，逗号将它与下一项分开。
    env: gym.vector.VectorEnv,
    # 中文 L0465 “policy”是当前括号中的一项，逗号将它与下一项分开。
    policy: PreTrainedPolicy,
    # 中文 L0466 延续上面的语句，env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理按批次跑指定回合数，汇总奖励与成功。
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0467 延续上面的语句，env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理按批次跑指定回合数，汇总奖励与成功。
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0468 延续上面的语句，preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理按批次跑指定回合数，汇总奖励与成功。
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0469 延续上面的语句，postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],。本段处理按批次跑指定回合数，汇总奖励与成功。
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    # 中文 L0470 “n_episodes”（评测回合数）是当前括号中的一项，逗号将它与下一项分开。
    n_episodes: int,
    # 中文 L0471 为“max_episodes_rendered”提供这一项的值，注意它所在的参数表或外层表达式。
    max_episodes_rendered: int = 0,
    # 中文 L0472 为“videos_dir”提供这一项的值，注意它所在的参数表或外层表达式。
    videos_dir: Path | None = None,
    # 中文 L0473 为“return_episode_data”提供这一项的值，注意它所在的参数表或外层表达式。
    return_episode_data: bool = False,
    # 中文 L0474 为“start_seed”提供这一项的值，注意它所在的参数表或外层表达式。
    start_seed: int | None = None,
    # 中文 L0475 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_dir: Path | None = None,
    # 中文 L0476 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
    env_features: dict | None = None,
    # 中文 L0477 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_repo_id: str | None = None,
    # 中文 L0478 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_private: bool = False,
    # 中文 L0479 为“save_predicted_video”提供这一项的值，注意它所在的参数表或外层表达式。
    save_predicted_video: bool = False,
    # 中文 L0480 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
    max_episode_steps: int | None = None,
# 中文 L0481 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    # 中文 L0496 条件 max_episodes_rendered > 0 and not videos_dir 成立时进入下方分支。
    if max_episodes_rendered > 0 and not videos_dir:
        # 中文 L0497 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    # World-model policies (e.g. LingBot-VA) opt into predicted-video saving via their config.
    # 中文 L0500 组合多个布尔判断，按 Python 的短路规则求值给 “save_predicted_video” 赋值。
    save_predicted_video = save_predicted_video or bool(
        # 中文 L0501 延续上面的语句，getattr(getattr(policy, "config", None), "save_predicted_video", False)。本段处理按批次跑指定回合数，汇总奖励与成功。
        getattr(getattr(policy, "config", None), "save_predicted_video", False)
    # 中文 L0502 闭合上方尚未结束的括号。
    )

    # 中文 L0504 条件 not isinstance(policy, PreTrainedPolicy) 成立时进入下方分支。
    if not isinstance(policy, PreTrainedPolicy):
        # 中文 L0505 调用 ValueError，用它的结果给 “exc” 赋值。
        exc = ValueError(
            # 中文 L0506 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        # 中文 L0507 闭合上方尚未结束的括号。
        )
        # 中文 L0508 条件 not _peft_available 成立时进入下方分支。
        if not _peft_available:
            # 中文 L0509 抛出 exc，中断当前路径并交给上层的异常处理。
            raise exc
        # 中文 L0510 调用 require_package 执行操作。
        require_package("peft", extra="peft")
        # 中文 L0511 条件 not isinstance(policy, PeftModel) 成立时进入下方分支。
        if not isinstance(policy, PeftModel):
            # 中文 L0512 抛出 exc，中断当前路径并交给上层的异常处理。
            raise exc

    # 中文 L0514 调用 time.time，用它的结果给 “start” 赋值。
    start = time.time()
    # Preserve the mode for direct callers. eval_policy_all scopes the mode
    # around all tasks so parallel evaluations cannot race with each other.
    # 中文 L0517 读取属性 policy.training给 “was_training” 赋值。
    was_training = policy.training
    # 中文 L0518 设置策略为评测态，例如让 dropout 退出训练时的随机行为。
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    # 中文 L0522 计算右侧表达式 n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)给 “n_batches” 赋值。
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    # 中文 L0525 创建列表，按给定顺序保留元素给 “sum_rewards” 赋值。
    sum_rewards = []
    # 中文 L0526 创建列表，按给定顺序保留元素给 “max_rewards” 赋值。
    max_rewards = []
    # 中文 L0527 创建列表，按给定顺序保留元素给 “all_successes” 赋值。
    all_successes = []
    # 中文 L0528 声明“all_seeds”的类型为 list[int | None]，创建列表，按给定顺序保留元素作为初始值。
    all_seeds: list[int | None] = []
    # 中文 L0529 声明“episode_metrics”的类型为 list[dict[str, float]]，创建列表，按给定顺序保留元素作为初始值。
    episode_metrics: list[dict[str, float]] = []
    # 中文 L0530 创建列表，按给定顺序保留元素给 “threads” 赋值。
    threads = []  # for video saving threads
    # 中文 L0531 使用字面值 0给 “n_episodes_rendered” 赋值。
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    # 中文 L0534 定义函数 render_frame。
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        # 中文 L0536 条件 n_episodes_rendered >= max_episodes_rendered 成立时进入下方分支。
        if n_episodes_rendered >= max_episodes_rendered:
            # 中文 L0537 退出当前调用并返回 None。
            return
        # 中文 L0538 调用 min，用它的结果给 “n_to_render_now” 赋值。
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        # 中文 L0539 条件 isinstance(env, gym.vector.SyncVectorEnv) 成立时进入下方分支。
        if isinstance(env, gym.vector.SyncVectorEnv):
            # 中文 L0540 调用 ep_frames.append，在列表末尾追加一个元素。
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        # 中文 L0541 前面的分支均未命中时，再判断 hasattr(env, "call")。
        elif hasattr(env, "call"):
            # Here we must render all frames and discard any we don't need.
            # 中文 L0543 调用 ep_frames.append，在列表末尾追加一个元素。
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    # 中文 L0545 条件 max_episodes_rendered > 0 成立时进入下方分支。
    if max_episodes_rendered > 0:
        # 中文 L0546 声明“video_paths”的类型为 list[str]，创建列表，按给定顺序保留元素作为初始值。
        video_paths: list[str] = []

    # 中文 L0548 条件 save_predicted_video 成立时进入下方分支。
    if save_predicted_video:
        # 中文 L0549 条件 not videos_dir 成立时进入下方分支。
        if not videos_dir:
            # 中文 L0550 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("If save_predicted_video is True, videos_dir must be provided.")
        # 中文 L0551 声明“predicted_video_paths”的类型为 list[str]，创建列表，按给定顺序保留元素作为初始值。
        predicted_video_paths: list[str] = []
        # 中文 L0552 使用字面值 0给 “n_predicted_rendered” 赋值。
        n_predicted_rendered = 0

    # Collect predicted-video latents across a rollout (world-model policies only). The latents are
    # concatenated and decoded once after the rollout, matching upstream LingBot-VA's visualization path.
    # 中文 L0556 定义函数 collect_predicted_latents。
    def collect_predicted_latents(policy: PreTrainedPolicy):
        # 中文 L0557 条件 _peft_available and isinstance(policy, PeftModel) 成立时进入下方分支。
        if _peft_available and isinstance(policy, PeftModel):
            # 中文 L0558 调用 policy.get_base_model，用它的结果给 “policy” 赋值。
            policy = policy.get_base_model()
        # 中文 L0559 调用 getattr，用它的结果给 “latents” 赋值。
        latents = getattr(policy, "last_predicted_latents", None)
        # 中文 L0560 条件 latents is not None 成立时进入下方分支。
        if latents is not None:
            # 中文 L0561 调用 pred_latents.append，在列表末尾追加一个元素。
            pred_latents.append(
                # 中文 L0562 延续上面的语句，latents.detach().to("cpu") if hasattr(latents, "detach") else torch.as_tensor(latents).cpu()。
                latents.detach().to("cpu") if hasattr(latents, "detach") else torch.as_tensor(latents).cpu()
            # 中文 L0563 闭合上方尚未结束的括号。
            )
            # 中文 L0564 设为空值 None给 “policy.last_predicted_latents” 赋值。
            policy.last_predicted_latents = None

    # 中文 L0566 条件 return_episode_data 成立时进入下方分支。
    if return_episode_data:
        # 中文 L0567 声明“episode_data”的类型为 dict | None，设为空值 None作为初始值。
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    # 中文 L0570 调用 trange，用它的结果给 “progbar” 赋值。
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    # 中文 L0571 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        # 中文 L0574 条件 max_episodes_rendered > 0 成立时进入下方分支。
        if max_episodes_rendered > 0:
            # 中文 L0575 声明“ep_frames”的类型为 list[np.ndarray]，创建列表，按给定顺序保留元素作为初始值。
            ep_frames: list[np.ndarray] = []

        # 中文 L0577 条件 save_predicted_video 成立时进入下方分支。
        if save_predicted_video:
            # 中文 L0578 声明“pred_latents”的类型为 list[torch.Tensor]，创建列表，按给定顺序保留元素作为初始值。
            pred_latents: list[torch.Tensor] = []

        # 中文 L0580 条件 start_seed is None 成立时进入下方分支。
        if start_seed is None:
            # 中文 L0581 设为空值 None给 “seeds” 赋值。
            seeds = None
        # 中文 L0582 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0583 按起始种子和批次下标分配回合种子。本次单环境单回合得到 1000。
            seeds = range(
                # 中文 L0584 延续上面的语句，start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)。本段处理按批次跑指定回合数，汇总奖励与成功。
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            # 中文 L0585 闭合上方尚未结束的括号。
            )
        # 中文 L0586 将环境、策略、处理器与种子交给闭环执行函数 rollout。
        rollout_data = rollout(
            # 中文 L0587 为“env”提供这一项的值，注意它所在的参数表或外层表达式。
            env=env,
            # 中文 L0588 为“policy”提供这一项的值，注意它所在的参数表或外层表达式。
            policy=policy,
            # 中文 L0589 为“env_preprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
            env_preprocessor=env_preprocessor,
            # 中文 L0590 为“env_postprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
            env_postprocessor=env_postprocessor,
            # 中文 L0591 为“preprocessor”（策略前处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
            preprocessor=preprocessor,
            # 中文 L0592 为“postprocessor”（动作后处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
            postprocessor=postprocessor,
            # 中文 L0593 为“seeds”提供这一项的值，注意它所在的参数表或外层表达式。
            seeds=list(seeds) if seeds else None,
            # 中文 L0594 为“return_observations”提供这一项的值，注意它所在的参数表或外层表达式。
            return_observations=return_episode_data,
            # 中文 L0595 为“render_callback”提供这一项的值，注意它所在的参数表或外层表达式。
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            # 中文 L0596 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
            recording_dir=recording_dir,
            # 中文 L0597 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
            env_features=env_features,
            # 中文 L0598 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
            recording_repo_id=recording_repo_id,
            # 中文 L0599 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
            recording_private=recording_private,
            # 中文 L0600 为“predicted_latents_callback”提供这一项的值，注意它所在的参数表或外层表达式。
            predicted_latents_callback=collect_predicted_latents if save_predicted_video else None,
            # 中文 L0601 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            max_episode_steps=max_episode_steps,
        # 中文 L0602 闭合上方尚未结束的括号。
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        # 中文 L0606 用方括号按键或索引读取 shape给 “n_steps” 赋值。
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        # 中文 L0608 对每条轨迹查找 done 第一次出现的位置，用于指标掩码。
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        # 中文 L0612 创建 [B,步骤数] 的掩码，与后续的奖励、成功矩阵逐元素相乘。
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        # 中文 L0614 沿步骤轴汇总每个回合的奖励。
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        # 中文 L0615 调用 sum_rewards.extend，将可迭代对象展开后追加到末尾。
        sum_rewards.extend(batch_sum_rewards.tolist())
        # 中文 L0616 调用 einops.reduce，用它的结果给 “batch_max_rewards” 赋值。
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        # 中文 L0617 调用 max_rewards.extend，将可迭代对象展开后追加到末尾。
        max_rewards.extend(batch_max_rewards.tolist())
        # 中文 L0618 沿步骤轴检查一个回合是否出现过成功标志。
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        # 中文 L0619 调用 all_successes.extend，将可迭代对象展开后追加到末尾。
        all_successes.extend(batch_successes.tolist())
        # 中文 L0620 条件 seeds 成立时进入下方分支。
        if seeds:
            # 中文 L0621 调用 all_seeds.extend，将可迭代对象展开后追加到末尾。
            all_seeds.extend(seeds)
        # 中文 L0622 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0623 调用 all_seeds.extend，将可迭代对象展开后追加到末尾。
            all_seeds.extend([None] * env.num_envs)
        # 中文 L0624 用字典推导式从可迭代对象逐项取值并构造结果给 “batch_metrics” 赋值。
        batch_metrics = {
            # 中文 L0625 延续上面的语句，name: values.gather(1, done_indices[:, None]).squeeze(1).tolist()。本段处理按批次跑指定回合数，汇总奖励与成功。
            name: values.gather(1, done_indices[:, None]).squeeze(1).tolist()
            # 中文 L0626 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for name, values in rollout_data.get("metrics", {}).items()
        # 中文 L0627 闭合上方尚未结束的括号。
        }
        # 中文 L0628 调用 episode_metrics.extend，将可迭代对象展开后追加到末尾。
        episode_metrics.extend(
            # 中文 L0629 延续上面的语句，{name: values[index] for name, values in batch_metrics.items()} for index in range(env.num_envs)。本段处理按批次跑指定回合数，汇总奖励与成功。
            {name: values[index] for name, values in batch_metrics.items()} for index in range(env.num_envs)
        # 中文 L0630 闭合上方尚未结束的括号。
        )

        # FIXME: episode_data is either None or it doesn't exist
        # 中文 L0633 条件 return_episode_data 成立时进入下方分支。
        if return_episode_data:
            # 中文 L0634 调用 _compile_episode_data，用它的结果给 “this_episode_data” 赋值。
            this_episode_data = _compile_episode_data(
                # 中文 L0635 “rollout_data”是当前括号中的一项，逗号将它与下一项分开。
                rollout_data,
                # 中文 L0636 “done_indices”是当前括号中的一项，逗号将它与下一项分开。
                done_indices,
                # 中文 L0637 为“start_episode_index”提供这一项的值，注意它所在的参数表或外层表达式。
                start_episode_index=batch_ix * env.num_envs,
                # 中文 L0638 为“start_data_index”提供这一项的值，注意它所在的参数表或外层表达式。
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                # 中文 L0639 为“fps”提供这一项的值，注意它所在的参数表或外层表达式。
                fps=env.unwrapped.metadata["render_fps"],
            # 中文 L0640 闭合上方尚未结束的括号。
            )
            # 中文 L0641 条件 episode_data is None 成立时进入下方分支。
            if episode_data is None:
                # 中文 L0642 引用已有的值 this_episode_data给 “episode_data” 赋值。
                episode_data = this_episode_data
            # 中文 L0643 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                # 中文 L0645 检查此条件，条件不成立时抛出 AssertionError。
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                # 中文 L0646 检查此条件，条件不成立时抛出 AssertionError。
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                # 中文 L0648 用字典推导式从可迭代对象逐项取值并构造结果给 “episode_data” 赋值。
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        # 中文 L0651 条件 max_episodes_rendered > 0 and len(ep_frames) > 0 成立时进入下方分支。
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            # 中文 L0652 调用 np.stack，用它的结果给 “batch_stacked_frames” 赋值。
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            # 中文 L0653 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for stacked_frames, done_index in zip(
                # 中文 L0654 延续上面的语句，batch_stacked_frames, done_indices.flatten().tolist(), strict=False。本段处理按批次跑指定回合数，汇总奖励与成功。
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            # 中文 L0655 闭合上方尚未结束的括号。
            ):
                # 中文 L0656 条件 n_episodes_rendered >= max_episodes_rendered 成立时进入下方分支。
                if n_episodes_rendered >= max_episodes_rendered:
                    # 中文 L0657 结束所在的最近一层循环。
                    break
                # 中文 L0658 条件 videos_dir is None: # already validated above 成立时进入下方分支。
                if videos_dir is None:  # already validated above
                    # 中文 L0659 抛出 ValueError，中断当前路径并交给上层的异常处理。
                    raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

                # 中文 L0661 调用 videos_dir.mkdir 执行操作。
                videos_dir.mkdir(parents=True, exist_ok=True)
                # 中文 L0662 计算右侧表达式 videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"给 “video_path” 赋值。
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                # 中文 L0663 调用 video_paths.append，在列表末尾追加一个元素。
                video_paths.append(str(video_path))
                # 中文 L0664 调用 threading.Thread，用它的结果给 “thread” 赋值。
                thread = threading.Thread(
                    # 中文 L0665 为“target”提供这一项的值，注意它所在的参数表或外层表达式。
                    target=write_video,
                    # 中文 L0666 为“args”（位置参数序列）提供这一项的值，注意它所在的参数表或外层表达式。
                    args=(
                        # 中文 L0667 延续上面的语句，str(video_path),。本段处理按批次跑指定回合数，汇总奖励与成功。
                        str(video_path),
                        # 中文 L0668 延续上面的语句，stacked_frames[: done_index + 1], # + 1 to capture the last observation。本段处理按批次跑指定回合数，汇总奖励与成功。
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        # 中文 L0669 延续上面的语句，env.unwrapped.metadata["render_fps"],。本段处理按批次跑指定回合数，汇总奖励与成功。
                        env.unwrapped.metadata["render_fps"],
                    # 中文 L0670 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                    ),
                # 中文 L0671 闭合上方尚未结束的括号。
                )
                # 中文 L0672 调用 thread.start，启动线程或进程。
                thread.start()
                # 中文 L0673 调用 threads.append，在列表末尾追加一个元素。
                threads.append(thread)
                # 中文 L0674 用增量赋值更新 n_episodes_rendered，运算逻辑见左侧运算符。
                n_episodes_rendered += 1

        # Maybe save the policy's predicted (imagined) video for this batch's rollout.
        # 中文 L0677 条件 save_predicted_video and len(pred_latents) > 0 成立时进入下方分支。
        if save_predicted_video and len(pred_latents) > 0:
            # 中文 L0678 调用 torch.cat，用它的结果给 “predicted_latent” 赋值。
            predicted_latent = torch.cat(pred_latents, dim=2)
            # 中文 L0679 组合多个布尔判断，按 Python 的短路规则求值给 “decoder” 赋值。
            decoder = getattr(policy, "decode_predicted_latents", None) or getattr(
                # 中文 L0680 延续上面的语句，policy, "_decode_predicted_video", None。本段处理按批次跑指定回合数，汇总奖励与成功。
                policy, "_decode_predicted_video", None
            # 中文 L0681 闭合上方尚未结束的括号。
            )
            # 中文 L0682 条件 decoder is None 成立时进入下方分支。
            if decoder is None:
                # 中文 L0683 抛出 AttributeError，中断当前路径并交给上层的异常处理。
                raise AttributeError(
                    # 中文 L0684 字符串字面量，供上方的调用、容器或错误消息使用。
                    "Policy config requested predicted-video saving, but the policy does not expose "
                    # 中文 L0685 字符串字面量，供上方的调用、容器或错误消息使用。
                    "`decode_predicted_latents` or `_decode_predicted_video`."
                # 中文 L0686 闭合上方尚未结束的括号。
                )
            # 中文 L0687 调用 decoder，用它的结果给 “predicted_video” 赋值。
            predicted_video = decoder(predicted_latent)
            # 中文 L0688 条件 hasattr(predicted_video, "detach") 成立时进入下方分支。
            if hasattr(predicted_video, "detach"):
                # 中文 L0689 调用 numpy，用它的结果给 “predicted_video” 赋值。
                predicted_video = predicted_video.detach().to("cpu").numpy()
            # 中文 L0690 条件 videos_dir is None: # already validated above 成立时进入下方分支。
            if videos_dir is None:  # already validated above
                # 中文 L0691 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError("If save_predicted_video is True, videos_dir must be provided.")
            # 中文 L0692 调用 videos_dir.mkdir 执行操作。
            videos_dir.mkdir(parents=True, exist_ok=True)
            # 中文 L0693 计算右侧表达式 videos_dir / f"pred_episode_{n_predicted_rendered}.mp4"给 “predicted_video_path” 赋值。
            predicted_video_path = videos_dir / f"pred_episode_{n_predicted_rendered}.mp4"
            # 中文 L0694 调用 predicted_video_paths.append，在列表末尾追加一个元素。
            predicted_video_paths.append(str(predicted_video_path))
            # 中文 L0695 调用 threading.Thread，用它的结果给 “thread” 赋值。
            thread = threading.Thread(
                # 中文 L0696 为“target”提供这一项的值，注意它所在的参数表或外层表达式。
                target=write_video,
                # 中文 L0697 为“args”（位置参数序列）提供这一项的值，注意它所在的参数表或外层表达式。
                args=(
                    # 中文 L0698 延续上面的语句，str(predicted_video_path),。本段处理按批次跑指定回合数，汇总奖励与成功。
                    str(predicted_video_path),
                    # 中文 L0699 “predicted_video”是当前括号中的一项，逗号将它与下一项分开。
                    predicted_video,
                    # 中文 L0700 延续上面的语句，env.unwrapped.metadata["render_fps"],。本段处理按批次跑指定回合数，汇总奖励与成功。
                    env.unwrapped.metadata["render_fps"],
                # 中文 L0701 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                ),
            # 中文 L0702 闭合上方尚未结束的括号。
            )
            # 中文 L0703 调用 thread.start，启动线程或进程。
            thread.start()
            # 中文 L0704 调用 threads.append，在列表末尾追加一个元素。
            threads.append(thread)
            # 中文 L0705 用增量赋值更新 n_predicted_rendered，运算逻辑见左侧运算符。
            n_predicted_rendered += 1

        # 中文 L0707 调用 progbar.set_postfix 执行操作。
        progbar.set_postfix(
            # 中文 L0708 延续上面的语句，{"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}。本段处理按批次跑指定回合数，汇总奖励与成功。
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        # 中文 L0709 闭合上方尚未结束的括号。
        )

    # Wait till all video rendering threads are done.
    # 中文 L0712 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for thread in threads:
        # 中文 L0713 调用 thread.join，等待对应线程或进程结束。
        thread.join()

    # Compile eval info.
    # 中文 L0716 对实际请求的回合计算成功数量和置信区间等统计。
    success_stats = success_summary(all_successes[:n_episodes])
    # 中文 L0717 声明“info”（额外信息字典）的类型为 dict[str, Any]，创建字典，用键查找对应的值作为初始值。
    info: dict[str, Any] = {
        # 中文 L0718 字典项“per_episode”，冒号右边给出它的值。
        "per_episode": [
            # 中文 L0719 延续上面的语句，{。本段处理按批次跑指定回合数，汇总奖励与成功。
            {
                # 中文 L0720 字典项“episode_ix”，冒号右边给出它的值。
                "episode_ix": i,
                # 中文 L0721 字典项“sum_reward”，冒号右边给出它的值。
                "sum_reward": sum_reward,
                # 中文 L0722 字典项“max_reward”，冒号右边给出它的值。
                "max_reward": max_reward,
                # 中文 L0723 字典项“success”，冒号右边给出它的值。
                "success": success,
                # 中文 L0724 字典项“seed”（随机种子），冒号右边给出它的值。
                "seed": seed,
            # 中文 L0725 闭合上方尚未结束的括号。
            }
            # 中文 L0726 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                # 中文 L0727 延续上面的语句，zip(。本段处理按批次跑指定回合数，汇总奖励与成功。
                zip(
                    # 中文 L0728 延续上面的语句，sum_rewards[:n_episodes],。本段处理按批次跑指定回合数，汇总奖励与成功。
                    sum_rewards[:n_episodes],
                    # 中文 L0729 延续上面的语句，max_rewards[:n_episodes],。本段处理按批次跑指定回合数，汇总奖励与成功。
                    max_rewards[:n_episodes],
                    # 中文 L0730 延续上面的语句，all_successes[:n_episodes],。本段处理按批次跑指定回合数，汇总奖励与成功。
                    all_successes[:n_episodes],
                    # 中文 L0731 延续上面的语句，all_seeds[:n_episodes],。本段处理按批次跑指定回合数，汇总奖励与成功。
                    all_seeds[:n_episodes],
                    # 中文 L0732 为“strict”提供这一项的值，注意它所在的参数表或外层表达式。
                    strict=True,
                # 中文 L0733 闭合上方尚未结束的括号。
                )
            # 中文 L0734 闭合上方尚未结束的括号。
            )
        # 中文 L0735 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
        ],
        # 中文 L0736 字典项“aggregated”，冒号右边给出它的值。
        "aggregated": {
            # 中文 L0737 字典项“avg_sum_reward”，冒号右边给出它的值。
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            # 中文 L0738 字典项“avg_max_reward”，冒号右边给出它的值。
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            # 中文 L0739 字典项“pc_success”，冒号右边给出它的值。
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            # 中文 L0740 字典项“n_episodes”（评测回合数），冒号右边给出它的值。
            "n_episodes": success_stats["n_episodes"],
            # 中文 L0741 字典项“n_success”，冒号右边给出它的值。
            "n_success": success_stats["n_success"],
            # 中文 L0742 字典项“pc_success_ci95”，冒号右边给出它的值。
            "pc_success_ci95": success_stats["pc_success_ci95"],
            # 中文 L0743 字典项“eval_s”，冒号右边给出它的值。
            "eval_s": time.time() - start,
            # 中文 L0744 字典项“eval_ep_s”，冒号右边给出它的值。
            "eval_ep_s": (time.time() - start) / n_episodes,
        # 中文 L0745 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
        },
    # 中文 L0746 闭合上方尚未结束的括号。
    }

    # 中文 L0748 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for episode, metrics in zip(info["per_episode"], episode_metrics[:n_episodes], strict=True):
        # 中文 L0749 条件 metrics 成立时进入下方分支。
        if metrics:
            # 中文 L0750 引用已有的值 metrics给 “episode['metrics']” 赋值。
            episode["metrics"] = metrics

    # 中文 L0752 条件 return_episode_data 成立时进入下方分支。
    if return_episode_data:
        # 中文 L0753 引用已有的值 episode_data给 “info['episodes']” 赋值。
        info["episodes"] = episode_data

    # 中文 L0755 条件 max_episodes_rendered > 0 成立时进入下方分支。
    if max_episodes_rendered > 0:
        # 中文 L0756 引用已有的值 video_paths给 “info['video_paths']” 赋值。
        info["video_paths"] = video_paths

    # 中文 L0758 条件 save_predicted_video 成立时进入下方分支。
    if save_predicted_video:
        # 中文 L0759 引用已有的值 predicted_video_paths给 “info['predicted_video_paths']” 赋值。
        info["predicted_video_paths"] = predicted_video_paths

    # 中文 L0761 调用 policy.train 执行操作。
    policy.train(was_training)

    # 中文 L0763 引用已有的值 info作为当前调用的返回值。
    return info


# 中文 L0766 定义函数 _compile_episode_data。作用是把可保存的各回合步骤汇成回合数据。
def _compile_episode_data(
    # 中文 L0767 延续上面的语句，rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float。本段处理把可保存的各回合步骤汇成回合数据。
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
# 中文 L0768 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    # 中文 L0775 创建列表，按给定顺序保留元素给 “ep_dicts” 赋值。
    ep_dicts = []
    # 中文 L0776 使用字面值 0给 “total_frames” 赋值。
    total_frames = 0
    # 中文 L0777 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        # 中文 L0779 计算右侧表达式 done_indices[ep_ix].item() + 2给 “num_frames” 赋值。
        num_frames = done_indices[ep_ix].item() + 2
        # 中文 L0780 用增量赋值更新 total_frames，运算逻辑见左侧运算符。
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        # 中文 L0783 创建字典，用键查找对应的值给 “ep_dict” 赋值。
        ep_dict = {
            # 中文 L0784 延续上面的语句，ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],。本段处理把可保存的各回合步骤汇成回合数据。
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            # 中文 L0785 字典项“episode_index”，冒号右边给出它的值。
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            # 中文 L0786 字典项“frame_index”，冒号右边给出它的值。
            "frame_index": torch.arange(0, num_frames - 1, 1),
            # 中文 L0787 字典项“timestamp”，冒号右边给出它的值。
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            # 中文 L0788 延续上面的语句，DONE: rollout_data["done"][ep_ix, : num_frames - 1],。本段处理把可保存的各回合步骤汇成回合数据。
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            # 中文 L0789 字典项“next.success”，冒号右边给出它的值。
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            # 中文 L0790 延续上面的语句，REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),。本段处理把可保存的各回合步骤汇成回合数据。
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        # 中文 L0791 闭合上方尚未结束的括号。
        }

        # For the last observation frame, all other keys will just be copy padded.
        # 中文 L0794 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for k in ep_dict:
            # 中文 L0795 调用 torch.cat，用它的结果给 “ep_dict[k]” 赋值。
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        # 中文 L0797 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key in rollout_data[OBS_STR]:
            # 中文 L0798 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “ep_dict[key]” 赋值。
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        # 中文 L0800 调用 ep_dicts.append，在列表末尾追加一个元素。
        ep_dicts.append(ep_dict)

    # 中文 L0802 创建字典，用键查找对应的值给 “data_dict” 赋值。
    data_dict = {}
    # 中文 L0803 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key in ep_dicts[0]:
        # 中文 L0804 调用 torch.cat，用它的结果给 “data_dict[key]” 赋值。
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    # 中文 L0806 调用 torch.arange，用它的结果给 “data_dict['index']” 赋值。
    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    # 中文 L0808 引用已有的值 data_dict作为当前调用的返回值。
    return data_dict


# 中文 L0811 装饰器为 eval_main 外包一层参数解析。终端启动时由这层构造 cfg，再调用下面的函数体。
@parser.wrap()
# 中文 L0812 评测编排入口。cfg 是 EvalPipelineConfig 实例，后续代码用 cfg.env、cfg.policy 等属性。
def eval_main(cfg: EvalPipelineConfig) -> None:
    # 中文 L0813 把 dataclass 实例转成字典并打印。这是复现实际生效配置的日志入口。
    logging.info(pformat(asdict(cfg)))

    # 中文 L0815 条件 cfg.policy is None 成立时进入下方分支。
    if cfg.policy is None:
        # 中文 L0816 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0817 字符串字面量，供上方的调用、容器或错误消息使用。
            "Evaluation requires a policy: pass --policy.path=<pretrained_dir> or --policy.type=<name>."
        # 中文 L0818 闭合上方尚未结束的括号。
        )
    # 中文 L0819 条件 cfg.policy.device is None 成立时进入下方分支。
    if cfg.policy.device is None:
        # PreTrainedConfig.__post_init__ always resolves the device, so reaching this is a programming error.
        # 中文 L0821 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("Policy config has no device set.")
    # 中文 L0822 条件 cfg.output_dir is None 成立时进入下方分支。
    if cfg.output_dir is None:
        # EvalPipelineConfig.__post_init__ always assigns a default output_dir.
        # 中文 L0824 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("EvalPipelineConfig.output_dir is not set.")

    # Check device is available
    # 中文 L0827 把配置中的设备名解析为可用的 PyTorch 设备。
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # 中文 L0829 使用布尔值 True给 “torch.backends.cudnn.benchmark” 赋值。
    torch.backends.cudnn.benchmark = True
    # 中文 L0830 条件 cfg.tf32 is not None 成立时进入下方分支。
    if cfg.tf32 is not None:
        # 中文 L0831 读取属性 cfg.tf32给 “torch.backends.cuda.matmul.allow_tf32” 赋值。
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        # 中文 L0832 读取属性 cfg.tf32给 “torch.backends.cudnn.allow_tf32” 赋值。
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    # 中文 L0833 设置此次评测用的随机种子。
    set_seed(cfg.seed)

    # 中文 L0835 调用 logging.info 执行操作。
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # 中文 L0837 调用 logging.info 执行操作。
    logging.info(f"Making environment (batch_size={cfg.eval.batch_size}, async={cfg.eval.use_async_envs}).")
    # 中文 L0838 按 cfg.env 创建环境字典，外层键是套件名，内层键是任务号。
    envs = make_env(
        # 中文 L0839 延续上面的语句，cfg.env,。本段处理解析完成配置后组装环境、策略和流水线，再运行评测并写结果。
        cfg.env,
        # 中文 L0840 测试命令 batch_size=1，使每个任务只启动一个环境实例。
        n_envs=cfg.eval.batch_size,
        # 中文 L0841 为“use_async_envs”提供这一项的值，注意它所在的参数表或外层表达式。
        use_async_envs=cfg.eval.use_async_envs,
    # 中文 L0842 闭合上方尚未结束的括号。
    )

    # Environment-specific preprocessor and postprocessor (e.g. for LIBERO environments), built before
    # admission: a checkpoint is admitted on the observation and the action they produce.
    # 中文 L0846 选择环境级的前处理和后处理。本次 LIBERO + SmolVLA 走环境配置提供的默认空步骤。
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    # Check the checkpoint against the sim before loading the policy. Hub-only and native-batch envs have
    # no descriptor.
    # 中文 L0850 调用 next，用它的结果给 “first_env” 赋值。
    first_env = next(env for tasks in envs.values() for env in tasks.values())
    # 中文 L0851 调用 getattr，用它的结果给 “descriptor” 赋值。
    descriptor = getattr(first_env, "descriptor", None)
    # 中文 L0852 条件 descriptor is not None 成立时进入下方分支。
    if descriptor is not None:
        # The policy and the recording datasets take the env's features; envs without a descriptor declare them.
        # 中文 L0854 从仿真描述符生成特征定义，后面的兼容性检查和策略工厂会用到。
        cfg.env.features = descriptor_features(descriptor)
        # 中文 L0855 检查 checkpoint 输入输出与仿真是否兼容，包含图像键的重命名映射。
        validate_checkpoint(
            # 中文 L0856 “descriptor”是当前括号中的一项，逗号将它与下一项分开。
            descriptor,
            # 中文 L0857 延续上面的语句，cfg.policy,。本段处理解析完成配置后组装环境、策略和流水线，再运行评测并写结果。
            cfg.policy,
            # 中文 L0858 延续上面的语句，cfg.rename_map,。本段处理解析完成配置后组装环境、策略和流水线，再运行评测并写结果。
            cfg.rename_map,
            # 中文 L0859 为“env_preprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
            env_preprocessor=env_preprocessor,
            # 中文 L0860 为“env_postprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
            env_postprocessor=env_postprocessor,
        # 中文 L0861 闭合上方尚未结束的括号。
        )

    # 中文 L0863 调用 logging.info 执行操作。
    logging.info("Making policy.")
    # 中文 L0864 在环境配置和特征已确定后创建策略，预训练路径触发权重加载。
    policy = make_policy(
        # 中文 L0865 为“cfg”（配置对象）提供这一项的值，注意它所在的参数表或外层表达式。
        cfg=cfg.policy,
        # 中文 L0866 为“env_cfg”提供这一项的值，注意它所在的参数表或外层表达式。
        env_cfg=cfg.env,
        # 中文 L0867 为“rename_map”提供这一项的值，注意它所在的参数表或外层表达式。
        rename_map=cfg.rename_map,
    # 中文 L0868 闭合上方尚未结束的括号。
    )

    # 中文 L0870 把策略设置为评测模式。eval 控制模块行为，关闭梯度由后面的上下文控制。
    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    # 中文 L0873 创建前处理覆盖项。此字典的键是已保存流水线中处理步骤的注册名。
    preprocessor_overrides = {
        # 中文 L0874 用实际策略设备覆盖已保存的 device_processor 配置。
        "device_processor": {"device": str(policy.config.device)},
        # 中文 L0875 用命令行 rename_map 覆盖相机键重命名处理器的映射。
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    # 中文 L0876 闭合上方尚未结束的括号。
    }

    # 中文 L0878 从指定 checkpoint 版本恢复策略的前处理和后处理。
    preprocessor, postprocessor = make_pre_post_processors(
        # 中文 L0879 为“policy_cfg”提供这一项的值，注意它所在的参数表或外层表达式。
        policy_cfg=cfg.policy,
        # 中文 L0880 为“pretrained_path”提供这一项的值，注意它所在的参数表或外层表达式。
        pretrained_path=cfg.policy.pretrained_path,
        # 中文 L0881 为“pretrained_revision”提供这一项的值，注意它所在的参数表或外层表达式。
        pretrained_revision=cfg.policy.pretrained_revision,
        # 中文 L0882 为“preprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
        preprocessor_overrides=preprocessor_overrides,
    # 中文 L0883 闭合上方尚未结束的括号。
    )

    # 中文 L0885 用条件表达式在 if 前与 else 后的两个值之间选择给 “recording_dir” 赋值。
    recording_dir = Path(cfg.output_dir) / "recordings" if cfg.eval.recording else None
    # 中文 L0886 用条件表达式在 if 前与 else 后的两个值之间选择给 “max_episodes_rendered” 赋值。
    max_episodes_rendered = 0 if cfg.eval.recording else 10
    # 中文 L0887 用条件表达式在 if 前与 else 后的两个值之间选择给 “videos_dir” 赋值。
    videos_dir = None if cfg.eval.recording else Path(cfg.output_dir) / "videos"

    # 中文 L0889 整个评测放在无梯度上下文中。use_amp 决定是否再进入自动混合精度，这次为 False。
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        # 中文 L0890 运行各任务并汇总回合结果。
        info = eval_policy_all(
            # 中文 L0891 为“envs”提供这一项的值，注意它所在的参数表或外层表达式。
            envs=envs,
            # 中文 L0892 为“policy”提供这一项的值，注意它所在的参数表或外层表达式。
            policy=policy,
            # 中文 L0893 为“env_preprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
            env_preprocessor=env_preprocessor,
            # 中文 L0894 为“env_postprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
            env_postprocessor=env_postprocessor,
            # 中文 L0895 为“preprocessor”（策略前处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
            preprocessor=preprocessor,
            # 中文 L0896 为“postprocessor”（动作后处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
            postprocessor=postprocessor,
            # 中文 L0897 为“n_episodes”（评测回合数）提供这一项的值，注意它所在的参数表或外层表达式。
            n_episodes=cfg.eval.n_episodes,
            # 中文 L0898 为“max_episodes_rendered”提供这一项的值，注意它所在的参数表或外层表达式。
            max_episodes_rendered=max_episodes_rendered,
            # 中文 L0899 为“videos_dir”提供这一项的值，注意它所在的参数表或外层表达式。
            videos_dir=videos_dir,
            # 中文 L0900 为“return_episode_data”提供这一项的值，注意它所在的参数表或外层表达式。
            return_episode_data=False,
            # 中文 L0901 为“start_seed”提供这一项的值，注意它所在的参数表或外层表达式。
            start_seed=cfg.seed,
            # 中文 L0902 给调度器传入并行任务数上限。本次设为 1，走顺序执行分支。
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            # 中文 L0903 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
            recording_dir=recording_dir,
            # 中文 L0904 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
            env_features=cfg.env.features if cfg.eval.recording else None,
            # 中文 L0905 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
            recording_repo_id=cfg.eval.recording_repo_id,
            # 中文 L0906 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
            recording_private=cfg.eval.recording_private,
            # 中文 L0907 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            max_episode_steps=cfg.eval.max_episode_steps,
        # 中文 L0908 闭合上方尚未结束的括号。
        )
        # 中文 L0909 调用 logger.info 执行操作。
        logger.info("Overall Aggregated Metrics:")
        # 中文 L0910 调用 logger.info 执行操作。
        logger.info(info["overall"])
        # 中文 L0911 用方括号按键或索引读取 容器给 “(ci_low, ci_high)” 赋值。
        ci_low, ci_high = info["overall"]["pc_success_ci95"]
        # 中文 L0912 调用 logger.info 执行操作。
        logger.info(
            # 中文 L0913 字符串字面量，供上方的调用、容器或错误消息使用。
            "Success rate %.1f%% (%d/%d episodes, 95%% Wilson interval %.1f%% to %.1f%%)",
            # 中文 L0914 延续上面的语句，info["overall"]["pc_success"],。本段处理解析完成配置后组装环境、策略和流水线，再运行评测并写结果。
            info["overall"]["pc_success"],
            # 中文 L0915 延续上面的语句，info["overall"]["n_success"],。本段处理解析完成配置后组装环境、策略和流水线，再运行评测并写结果。
            info["overall"]["n_success"],
            # 中文 L0916 延续上面的语句，info["overall"]["n_episodes"],。本段处理解析完成配置后组装环境、策略和流水线，再运行评测并写结果。
            info["overall"]["n_episodes"],
            # 中文 L0917 “ci_low”是当前括号中的一项，逗号将它与下一项分开。
            ci_low,
            # 中文 L0918 “ci_high”是当前括号中的一项，逗号将它与下一项分开。
            ci_high,
        # 中文 L0919 闭合上方尚未结束的括号。
        )

        # Print per-suite stats
        # 中文 L0922 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for task_group, task_group_info in info.items():
            # 中文 L0923 调用 logger.info 执行操作。
            logger.info(f"\nAggregated Metrics for {task_group}:")
            # 中文 L0924 调用 logger.info 执行操作。
            logger.info(task_group_info)
    # 中文 L0925 评测结束后负责清理环境。
    # Close all vec envs
    # 中文 L0926 调用 close_envs 执行操作。
    close_envs(envs)

    # Save each task's language instruction next to the metrics; the benchmark CI report reads it.
    # 中文 L0929 用字典推导式从可迭代对象逐项取值并构造结果给 “descriptions” 赋值。
    descriptions = {
        # 中文 L0930 f 字符串片段，花括号中的表达式会被求值并插入文本。
        f"{group}_{task_id}": env.latest.task[0]
        # 中文 L0931 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for group, tasks in envs.items()
        # 中文 L0932 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for task_id, env in tasks.items()
        # 中文 L0933 条件 getattr(env, "latest", None) is not None 成立时进入下方分支。
        if getattr(env, "latest", None) is not None
    # 中文 L0934 闭合上方尚未结束的括号。
    }
    # 中文 L0935 条件 descriptions 成立时进入下方分支。
    if descriptions:
        # 中文 L0936 把收集到的自然语言任务描述写成 JSON。
        (Path(cfg.output_dir) / "task_descriptions.json").write_text(json.dumps(descriptions, indent=2))

    # Save info
    # 中文 L0939 打开评测输出目录中的指标文件。
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        # 中文 L0940 把汇总字典序列化为可读的 JSON。
        json.dump(info, f, indent=2)

    # 中文 L0942 调用 logging.info 执行操作。
    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
# 中文 L0946 定义类 TaskMetrics，括号列出基类 TypedDict。
class TaskMetrics(TypedDict):
    # 中文 L0947 声明“sum_rewards”的类型为 list[float]。
    sum_rewards: list[float]
    # 中文 L0948 声明“max_rewards”的类型为 list[float]。
    max_rewards: list[float]
    # 中文 L0949 声明“successes”的类型为 list[bool]。
    successes: list[bool]
    # 中文 L0950 声明“video_paths”的类型为 list[str]。
    video_paths: list[str]
    # 中文 L0951 声明“predicted_video_paths”的类型为 list[str]。
    predicted_video_paths: list[str]
    # One entry per episode: seed, success, rewards, and info["metrics"] if the benchmark reports any.
    # 中文 L0953 声明“per_episode”的类型为 list[dict[str, Any]]。
    per_episode: list[dict[str, Any]]


# 中文 L0956 打包成元组，元素顺序由逗号分隔的位置决定给 “ACC_KEYS” 赋值。
ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths", "predicted_video_paths")


# 中文 L0959 定义函数 eval_one。作用是适配一个套件中一个任务的评测。
def eval_one(
    # 中文 L0960 “env”是当前括号中的一项，逗号将它与下一项分开。
    env: gym.vector.VectorEnv,
    # 中文 L0961 后续参数必须按名字传入。
    *,
    # 中文 L0962 “policy”是当前括号中的一项，逗号将它与下一项分开。
    policy: PreTrainedPolicy,
    # 中文 L0963 延续上面的语句，env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理适配一个套件中一个任务的评测。
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0964 延续上面的语句，env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理适配一个套件中一个任务的评测。
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0965 延续上面的语句，preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理适配一个套件中一个任务的评测。
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0966 延续上面的语句，postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],。本段处理适配一个套件中一个任务的评测。
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    # 中文 L0967 “n_episodes”（评测回合数）是当前括号中的一项，逗号将它与下一项分开。
    n_episodes: int,
    # 中文 L0968 “max_episodes_rendered”是当前括号中的一项，逗号将它与下一项分开。
    max_episodes_rendered: int,
    # 中文 L0969 “videos_dir”是当前括号中的一项，逗号将它与下一项分开。
    videos_dir: Path | None,
    # 中文 L0970 “return_episode_data”是当前括号中的一项，逗号将它与下一项分开。
    return_episode_data: bool,
    # 中文 L0971 “start_seed”是当前括号中的一项，逗号将它与下一项分开。
    start_seed: int | None,
    # 中文 L0972 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_dir: Path | None = None,
    # 中文 L0973 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
    env_features: dict | None = None,
    # 中文 L0974 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_repo_id: str | None = None,
    # 中文 L0975 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_private: bool = False,
    # 中文 L0976 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
    max_episode_steps: int | None = None,
# 中文 L0977 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> TaskMetrics:
    # 中文 L0978 原文档概述本段功能，适配一个套件中一个任务的评测。
    """Evaluates one task_id of one suite using the provided vec env."""

    # 中文 L0980 引用已有的值 videos_dir给 “task_videos_dir” 赋值。
    task_videos_dir = videos_dir

    # 中文 L0982 把当前单个任务交给 eval_policy 评测一个或多个回合。
    task_result = eval_policy(
        # 中文 L0983 为“env”提供这一项的值，注意它所在的参数表或外层表达式。
        env=env,
        # 中文 L0984 为“policy”提供这一项的值，注意它所在的参数表或外层表达式。
        policy=policy,
        # 中文 L0985 为“env_preprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
        env_preprocessor=env_preprocessor,
        # 中文 L0986 为“env_postprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
        env_postprocessor=env_postprocessor,
        # 中文 L0987 为“preprocessor”（策略前处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
        preprocessor=preprocessor,
        # 中文 L0988 为“postprocessor”（动作后处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
        postprocessor=postprocessor,
        # 中文 L0989 为“n_episodes”（评测回合数）提供这一项的值，注意它所在的参数表或外层表达式。
        n_episodes=n_episodes,
        # 中文 L0990 为“max_episodes_rendered”提供这一项的值，注意它所在的参数表或外层表达式。
        max_episodes_rendered=max_episodes_rendered,
        # 中文 L0991 为“videos_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        videos_dir=task_videos_dir,
        # 中文 L0992 为“return_episode_data”提供这一项的值，注意它所在的参数表或外层表达式。
        return_episode_data=return_episode_data,
        # 中文 L0993 为“start_seed”提供这一项的值，注意它所在的参数表或外层表达式。
        start_seed=start_seed,
        # 中文 L0994 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_dir=recording_dir,
        # 中文 L0995 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
        env_features=env_features,
        # 中文 L0996 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_repo_id=recording_repo_id,
        # 中文 L0997 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_private=recording_private,
        # 中文 L0998 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
        max_episode_steps=max_episode_steps,
    # 中文 L0999 闭合上方尚未结束的括号。
    )

    # 中文 L1001 用方括号按键或索引读取 task_result给 “per_episode” 赋值。
    per_episode = task_result["per_episode"]
    # 中文 L1002 调用 TaskMetrics，用它的结果给 “metrics” 赋值。
    metrics = TaskMetrics(
        # 中文 L1003 为“sum_rewards”提供这一项的值，注意它所在的参数表或外层表达式。
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        # 中文 L1004 为“max_rewards”提供这一项的值，注意它所在的参数表或外层表达式。
        max_rewards=[ep["max_reward"] for ep in per_episode],
        # 中文 L1005 为“successes”提供这一项的值，注意它所在的参数表或外层表达式。
        successes=[ep["success"] for ep in per_episode],
        # 中文 L1006 为“video_paths”提供这一项的值，注意它所在的参数表或外层表达式。
        video_paths=task_result.get("video_paths", []),
        # 中文 L1007 为“predicted_video_paths”提供这一项的值，注意它所在的参数表或外层表达式。
        predicted_video_paths=task_result.get("predicted_video_paths", []),
        # 中文 L1008 为“per_episode”提供这一项的值，注意它所在的参数表或外层表达式。
        per_episode=per_episode,
    # 中文 L1009 闭合上方尚未结束的括号。
    )
    # 中文 L1010 引用已有的值 metrics作为当前调用的返回值。
    return metrics


# 中文 L1013 定义函数 run_one。作用是把一个任务交给 eval_one，记录任务与结果的关系。
def run_one(
    # 中文 L1014 “task_group”（任务套件名）是当前括号中的一项，逗号将它与下一项分开。
    task_group: str,
    # 中文 L1015 “task_id”（套件内的任务编号）是当前括号中的一项，逗号将它与下一项分开。
    task_id: int,
    # 中文 L1016 “env”是当前括号中的一项，逗号将它与下一项分开。
    env,
    # 中文 L1017 后续参数必须按名字传入。
    *,
    # 中文 L1018 “policy”是当前括号中的一项，逗号将它与下一项分开。
    policy,
    # 中文 L1019 “env_preprocessor”是当前括号中的一项，逗号将它与下一项分开。
    env_preprocessor,
    # 中文 L1020 “env_postprocessor”是当前括号中的一项，逗号将它与下一项分开。
    env_postprocessor,
    # 中文 L1021 “preprocessor”（策略前处理流水线）是当前括号中的一项，逗号将它与下一项分开。
    preprocessor,
    # 中文 L1022 “postprocessor”（动作后处理流水线）是当前括号中的一项，逗号将它与下一项分开。
    postprocessor,
    # 中文 L1023 “n_episodes”（评测回合数）是当前括号中的一项，逗号将它与下一项分开。
    n_episodes: int,
    # 中文 L1024 “max_episodes_rendered”是当前括号中的一项，逗号将它与下一项分开。
    max_episodes_rendered: int,
    # 中文 L1025 “videos_dir”是当前括号中的一项，逗号将它与下一项分开。
    videos_dir: Path | None,
    # 中文 L1026 “return_episode_data”是当前括号中的一项，逗号将它与下一项分开。
    return_episode_data: bool,
    # 中文 L1027 “start_seed”是当前括号中的一项，逗号将它与下一项分开。
    start_seed: int | None,
    # 中文 L1028 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_dir: Path | None = None,
    # 中文 L1029 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
    env_features: dict | None = None,
    # 中文 L1030 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_repo_id: str | None = None,
    # 中文 L1031 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_private: bool = False,
    # 中文 L1032 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
    max_episode_steps: int | None = None,
# 中文 L1033 闭合上方尚未结束的括号。
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    # 中文 L1039 设为空值 None给 “task_videos_dir” 赋值。
    task_videos_dir = None
    # 中文 L1040 条件 videos_dir is not None 成立时进入下方分支。
    if videos_dir is not None:
        # 中文 L1041 计算右侧表达式 videos_dir / f"{task_group}_{task_id}"给 “task_videos_dir” 赋值。
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        # 中文 L1042 调用 task_videos_dir.mkdir 执行操作。
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # 中文 L1044 设为空值 None给 “task_recording_dir” 赋值。
    task_recording_dir = None
    # 中文 L1045 设为空值 None给 “task_repo_id” 赋值。
    task_repo_id = None
    # 中文 L1046 条件 recording_dir is not None and env_features is not None 成立时进入下方分支。
    if recording_dir is not None and env_features is not None:
        # 中文 L1047 计算右侧表达式 recording_dir / f"{task_group}_{task_id}"给 “task_recording_dir” 赋值。
        task_recording_dir = recording_dir / f"{task_group}_{task_id}"
        # 中文 L1048 条件 recording_repo_id is not None 成立时进入下方分支。
        if recording_repo_id is not None:
            # 中文 L1049 计算此行的右侧表达式给 “task_repo_id” 赋值。
            task_repo_id = f"{recording_repo_id}_{task_group}_{task_id}"

    # 中文 L1051 把当前套件中的任务交给 eval_one。
    metrics = eval_one(
        # 中文 L1052 “env”是当前括号中的一项，逗号将它与下一项分开。
        env,
        # 中文 L1053 为“policy”提供这一项的值，注意它所在的参数表或外层表达式。
        policy=policy,
        # 中文 L1054 为“env_preprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
        env_preprocessor=env_preprocessor,
        # 中文 L1055 为“env_postprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
        env_postprocessor=env_postprocessor,
        # 中文 L1056 为“preprocessor”（策略前处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
        preprocessor=preprocessor,
        # 中文 L1057 为“postprocessor”（动作后处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
        postprocessor=postprocessor,
        # 中文 L1058 为“n_episodes”（评测回合数）提供这一项的值，注意它所在的参数表或外层表达式。
        n_episodes=n_episodes,
        # 中文 L1059 为“max_episodes_rendered”提供这一项的值，注意它所在的参数表或外层表达式。
        max_episodes_rendered=max_episodes_rendered,
        # 中文 L1060 为“videos_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        videos_dir=task_videos_dir,
        # 中文 L1061 为“return_episode_data”提供这一项的值，注意它所在的参数表或外层表达式。
        return_episode_data=return_episode_data,
        # 中文 L1062 为“start_seed”提供这一项的值，注意它所在的参数表或外层表达式。
        start_seed=start_seed,
        # 中文 L1063 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_dir=task_recording_dir,
        # 中文 L1064 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
        env_features=env_features,
        # 中文 L1065 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_repo_id=task_repo_id,
        # 中文 L1066 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_private=recording_private,
        # 中文 L1067 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
        max_episode_steps=max_episode_steps,
    # 中文 L1068 闭合上方尚未结束的括号。
    )

    # 中文 L1070 条件 max_episodes_rendered > 0 成立时进入下方分支。
    if max_episodes_rendered > 0:
        # 中文 L1071 调用 metrics.setdefault 执行操作。
        metrics.setdefault("video_paths", [])
    # 中文 L1072 调用 metrics.setdefault 执行操作。
    metrics.setdefault("predicted_video_paths", [])
    # 中文 L1073 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
    return task_group, task_id, metrics


# 中文 L1076 定义函数 _task_info。
def _task_info(task_group: str, task_id: int, metrics: dict) -> dict:
    # 中文 L1077 原文档中的说明或续行。
    """One `per_task` entry: the raw per-episode metrics plus the task's success count and interval."""
    # 中文 L1078 创建字典，用键查找对应的值作为当前调用的返回值。
    return {
        # 中文 L1079 字典项“task_group”（任务套件名），冒号右边给出它的值。
        "task_group": task_group,
        # 中文 L1080 字典项“task_id”（套件内的任务编号），冒号右边给出它的值。
        "task_id": task_id,
        # 中文 L1081 字典项“metrics”，冒号右边给出它的值。
        "metrics": metrics,
        # 中文 L1082 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **success_summary(metrics.get("successes") or []),
    # 中文 L1083 闭合上方尚未结束的括号。
    }


# 中文 L1086 定义函数 eval_policy_all。作用是按任务调度评测，汇总套件和总体数据。
def eval_policy_all(
    # 中文 L1087 延续上面的语句，envs: dict[str, dict[int, gym.vector.VectorEnv]],。本段处理按任务调度评测，汇总套件和总体数据。
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    # 中文 L1088 “policy”是当前括号中的一项，逗号将它与下一项分开。
    policy,
    # 中文 L1089 延续上面的语句，env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理按任务调度评测，汇总套件和总体数据。
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L1090 延续上面的语句，env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理按任务调度评测，汇总套件和总体数据。
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L1091 延续上面的语句，preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理按任务调度评测，汇总套件和总体数据。
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L1092 延续上面的语句，postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],。本段处理按任务调度评测，汇总套件和总体数据。
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    # 中文 L1093 “n_episodes”（评测回合数）是当前括号中的一项，逗号将它与下一项分开。
    n_episodes: int,
    # 中文 L1094 后续参数必须按名字传入。
    *,
    # 中文 L1095 为“max_episodes_rendered”提供这一项的值，注意它所在的参数表或外层表达式。
    max_episodes_rendered: int = 0,
    # 中文 L1096 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_dir: Path | None = None,
    # 中文 L1097 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
    env_features: dict | None = None,
    # 中文 L1098 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_repo_id: str | None = None,
    # 中文 L1099 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
    recording_private: bool = False,
    # 中文 L1100 为“videos_dir”提供这一项的值，注意它所在的参数表或外层表达式。
    videos_dir: Path | None = None,
    # 中文 L1101 为“return_episode_data”提供这一项的值，注意它所在的参数表或外层表达式。
    return_episode_data: bool = False,
    # 中文 L1102 为“start_seed”提供这一项的值，注意它所在的参数表或外层表达式。
    start_seed: int | None = None,
    # 中文 L1103 为“max_parallel_tasks”提供这一项的值，注意它所在的参数表或外层表达式。
    max_parallel_tasks: int = 1,
    # 中文 L1104 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
    max_episode_steps: int | None = None,
# 中文 L1105 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    # 中文 L1113 调用 time.time，用它的结果给 “start_t” 赋值。
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    # 中文 L1116 用列表推导式从可迭代对象逐项取值并构造结果给 “tasks” 赋值。
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    # 中文 L1119 声明“group_acc”的类型为 dict[str, dict[str, list]]，调用 defaultdict，用它的结果作为初始值。
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    # 中文 L1120 声明“overall”的类型为 dict[str, list]，用字典推导式从可迭代对象逐项取值并构造结果作为初始值。
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    # 中文 L1121 声明“per_task_infos”的类型为 list[dict]，创建列表，按给定顺序保留元素作为初始值。
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    # 中文 L1124 定义函数 _accumulate_to。
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        # 中文 L1128 定义函数 _append。
        def _append(key, value):
            # 中文 L1129 条件 value is None 成立时进入下方分支。
            if value is None:
                # 中文 L1130 退出当前调用并返回 None。
                return
            # 中文 L1131 条件 isinstance(value, list) 成立时进入下方分支。
            if isinstance(value, list):
                # 中文 L1132 调用 extend 执行操作。
                group_acc[group][key].extend(value)
                # 中文 L1133 调用 extend 执行操作。
                overall[key].extend(value)
            # 中文 L1134 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L1135 调用 append 执行操作。
                group_acc[group][key].append(value)
                # 中文 L1136 调用 append 执行操作。
                overall[key].append(value)

        # 中文 L1138 调用 _append 执行操作。
        _append("sum_rewards", metrics.get("sum_rewards"))
        # 中文 L1139 调用 _append 执行操作。
        _append("max_rewards", metrics.get("max_rewards"))
        # 中文 L1140 调用 _append 执行操作。
        _append("successes", metrics.get("successes"))
        # 中文 L1141 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key in ("video_paths", "predicted_video_paths"):
            # 中文 L1142 调用 metrics.get，用它的结果给 “paths” 赋值。
            paths = metrics.get(key, [])
            # 中文 L1143 条件 paths 成立时进入下方分支。
            if paths:
                # 中文 L1144 调用 extend 执行操作。
                group_acc[group][key].extend(paths)
                # 中文 L1145 调用 extend 执行操作。
                overall[key].extend(paths)

    # Choose runner (sequential vs threaded)
    # 中文 L1148 partial 先固定 run_one 的一组共享参数，调度时再传入任务名、编号和环境。
    task_runner = partial(
        # 中文 L1149 “run_one”是当前括号中的一项，逗号将它与下一项分开。
        run_one,
        # 中文 L1150 为“policy”提供这一项的值，注意它所在的参数表或外层表达式。
        policy=policy,
        # 中文 L1151 为“env_preprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
        env_preprocessor=env_preprocessor,
        # 中文 L1152 为“env_postprocessor”提供这一项的值，注意它所在的参数表或外层表达式。
        env_postprocessor=env_postprocessor,
        # 中文 L1153 为“preprocessor”（策略前处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
        preprocessor=preprocessor,
        # 中文 L1154 为“postprocessor”（动作后处理流水线）提供这一项的值，注意它所在的参数表或外层表达式。
        postprocessor=postprocessor,
        # 中文 L1155 为“n_episodes”（评测回合数）提供这一项的值，注意它所在的参数表或外层表达式。
        n_episodes=n_episodes,
        # 中文 L1156 为“max_episodes_rendered”提供这一项的值，注意它所在的参数表或外层表达式。
        max_episodes_rendered=max_episodes_rendered,
        # 中文 L1157 为“videos_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        videos_dir=videos_dir,
        # 中文 L1158 为“return_episode_data”提供这一项的值，注意它所在的参数表或外层表达式。
        return_episode_data=return_episode_data,
        # 中文 L1159 为“start_seed”提供这一项的值，注意它所在的参数表或外层表达式。
        start_seed=start_seed,
        # 中文 L1160 为“recording_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_dir=recording_dir,
        # 中文 L1161 为“env_features”提供这一项的值，注意它所在的参数表或外层表达式。
        env_features=env_features,
        # 中文 L1162 为“recording_repo_id”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_repo_id=recording_repo_id,
        # 中文 L1163 为“recording_private”提供这一项的值，注意它所在的参数表或外层表达式。
        recording_private=recording_private,
        # 中文 L1164 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
        max_episode_steps=max_episode_steps,
    # 中文 L1165 闭合上方尚未结束的括号。
    )

    # Set the shared policy's mode before launching any workers. Restoring it
    # inside individual tasks would let one task enable training mode while
    # another task is still evaluating.
    # 中文 L1170 读取属性 policy.training给 “was_training” 赋值。
    was_training = policy.training
    # 中文 L1171 调用 policy.eval，切换 PyTorch 模块到评测模式。
    policy.eval()
    # 中文 L1172 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L1173 并行任务上限为 1 时走顺序分支，本次运行进入此处。
        if max_parallel_tasks <= 1:
            # Each task builds its simulator on its first reset, in this thread, after the previous
            # task's env is closed: consecutive tasks never hold the GPU at once, and simulators
            # that must run on the main thread (e.g. Isaac Sim) are built there.
            # 中文 L1177 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for task_group, task_id, env in tasks:
                # 中文 L1178 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
                try:
                    # 中文 L1179 调用预先绑定好公共参数的 run_one。
                    tg, tid, metrics = task_runner(task_group, task_id, env)
                    # 中文 L1180 调用 _accumulate_to 执行操作。
                    _accumulate_to(tg, metrics)
                    # 中文 L1181 调用 per_task_infos.append，在列表末尾追加一个元素。
                    per_task_infos.append(_task_info(tg, tid, metrics))
                # 中文 L1182 离开 try 时执行这里的清理，即使前面的路径发生异常。
                finally:
                    # 中文 L1183 当前任务执行完或抛出异常后关闭该任务的环境。
                    env.close()
        # 中文 L1184 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L1185 进入上下文管理器，缩进块结束时由它处理退出动作。
            with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
                # 中文 L1186 创建字典，用键查找对应的值给 “fut2meta” 赋值。
                fut2meta = {}
                # 中文 L1187 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for task_group, task_id, env in tasks:
                    # 中文 L1188 调用 executor.submit，用它的结果给 “fut” 赋值。
                    fut = executor.submit(task_runner, task_group, task_id, env)
                    # 中文 L1189 打包成元组，元素顺序由逗号分隔的位置决定给 “fut2meta[fut]” 赋值。
                    fut2meta[fut] = (task_group, task_id, env)
                # 中文 L1190 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for fut in cf.as_completed(fut2meta):
                    # 中文 L1191 用方括号按键或索引读取 fut2meta给 “(tg, tid, env)” 赋值。
                    tg, tid, env = fut2meta[fut]
                    # 中文 L1192 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
                    try:
                        # 中文 L1193 调用 fut.result，用它的结果给 “(tg, tid, metrics)” 赋值。
                        tg, tid, metrics = fut.result()
                        # 中文 L1194 调用 _accumulate_to 执行操作。
                        _accumulate_to(tg, metrics)
                        # 中文 L1195 调用 per_task_infos.append，在列表末尾追加一个元素。
                        per_task_infos.append(_task_info(tg, tid, metrics))
                    # 中文 L1196 离开 try 时执行这里的清理，即使前面的路径发生异常。
                    finally:
                        # 中文 L1197 调用 env.close，关闭对象或释放关联资源。
                        env.close()
    # 中文 L1198 离开 try 时执行这里的清理，即使前面的路径发生异常。
    finally:
        # 中文 L1199 调用 policy.train 执行操作。
        policy.train(was_training)

    # compute aggregated metrics helper (robust to lists/scalars)
    # 中文 L1202 定义函数 _agg_from_list。
    def _agg_from_list(xs):
        # 中文 L1203 条件 not xs 成立时进入下方分支。
        if not xs:
            # 中文 L1204 调用 float，用它的结果作为当前调用的返回值。
            return float("nan")
        # 中文 L1205 调用 np.array，用它的结果给 “arr” 赋值。
        arr = np.array(xs, dtype=float)
        # 中文 L1206 调用 float，用它的结果作为当前调用的返回值。
        return float(np.nanmean(arr))

    # compute per-group aggregates
    # 中文 L1209 创建字典，用键查找对应的值给 “groups_aggregated” 赋值。
    groups_aggregated = {}
    # 中文 L1210 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for group, acc in group_acc.items():
        # 中文 L1211 调用 success_summary，用它的结果给 “group_success” 赋值。
        group_success = success_summary(acc["successes"])
        # 中文 L1212 创建字典，用键查找对应的值给 “groups_aggregated[group]” 赋值。
        groups_aggregated[group] = {
            # 中文 L1213 字典项“avg_sum_reward”，冒号右边给出它的值。
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            # 中文 L1214 字典项“avg_max_reward”，冒号右边给出它的值。
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            # 中文 L1215 字典项“pc_success”，冒号右边给出它的值。
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            # 中文 L1216 字典项“n_episodes”（评测回合数），冒号右边给出它的值。
            "n_episodes": len(acc["sum_rewards"]),
            # 中文 L1217 字典项“n_success”，冒号右边给出它的值。
            "n_success": group_success["n_success"],
            # 中文 L1218 字典项“pc_success_ci95”，冒号右边给出它的值。
            "pc_success_ci95": group_success["pc_success_ci95"],
            # 中文 L1219 字典项“video_paths”，冒号右边给出它的值。
            "video_paths": list(acc["video_paths"]),
            # 中文 L1220 字典项“predicted_video_paths”，冒号右边给出它的值。
            "predicted_video_paths": list(acc["predicted_video_paths"]),
        # 中文 L1221 闭合上方尚未结束的括号。
        }

    # overall aggregates
    # 中文 L1224 对所有任务合并后的成功布尔值做汇总。
    overall_success = success_summary(overall["successes"])
    # 中文 L1225 创建字典，用键查找对应的值给 “overall_agg” 赋值。
    overall_agg = {
        # 中文 L1226 字典项“avg_sum_reward”，冒号右边给出它的值。
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        # 中文 L1227 字典项“avg_max_reward”，冒号右边给出它的值。
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        # 中文 L1228 字典项“pc_success”，冒号右边给出它的值。
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        # 中文 L1229 字典项“n_episodes”（评测回合数），冒号右边给出它的值。
        "n_episodes": len(overall["sum_rewards"]),
        # 中文 L1230 字典项“n_success”，冒号右边给出它的值。
        "n_success": overall_success["n_success"],
        # 中文 L1231 字典项“pc_success_ci95”，冒号右边给出它的值。
        "pc_success_ci95": overall_success["pc_success_ci95"],
        # 中文 L1232 字典项“eval_s”，冒号右边给出它的值。
        "eval_s": time.time() - start_t,
        # 中文 L1233 字典项“eval_ep_s”，冒号右边给出它的值。
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        # 中文 L1234 字典项“video_paths”，冒号右边给出它的值。
        "video_paths": list(overall["video_paths"]),
        # 中文 L1235 字典项“predicted_video_paths”，冒号右边给出它的值。
        "predicted_video_paths": list(overall["predicted_video_paths"]),
    # 中文 L1236 闭合上方尚未结束的括号。
    }

    # 中文 L1238 创建字典，用键查找对应的值作为当前调用的返回值。
    return {
        # 中文 L1239 字典项“per_task”，冒号右边给出它的值。
        "per_task": per_task_infos,
        # 中文 L1240 字典项“per_group”，冒号右边给出它的值。
        "per_group": groups_aggregated,
        # 中文 L1241 字典项“overall”，冒号右边给出它的值。
        "overall": overall_agg,
    # 中文 L1242 闭合上方尚未结束的括号。
    }


# 中文 L1245 定义终端命令 lerobot-eval 导入的 main 函数。
def main():
    # 中文 L1246 初始化日志格式与输出。
    init_logging()
    # 中文 L1247 注册评测可以发现的第三方插件。
    register_third_party_plugins()
    # 中文 L1248 执行 eval_main 的包装器，由它解析参数后进入评测编排。
    eval_main()


# 中文 L1251 当文件直接作为主程序运行时，Python 将其模块名设为 __main__。
if __name__ == "__main__":
    # 中文 L1252 直接运行模块时也走相同 main 入口。安装的命令由独立启动脚本导入并调用 main。
    main()
