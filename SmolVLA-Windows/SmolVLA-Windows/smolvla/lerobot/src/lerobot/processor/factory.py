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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.lerobot_types import PolicyAction, RobotAction, RobotObservation
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import (
    batch_to_transition,
    observation_to_transition,
    policy_action_to_transition,
    robot_action_observation_to_transition,
    transition_to_batch,
    transition_to_observation,
    transition_to_policy_action,
    transition_to_robot_action,
)
from .device_processor import DeviceProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from .pipeline import (
    IdentityProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    RobotProcessorPipeline,
)
from .relative_action_processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from .rename_processor import RenameObservationsProcessorStep


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors():
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0088 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0089 定义类 DefaultPolicyProcessorSteps。
class DefaultPolicyProcessorSteps:
    """The canonical processor steps shared by most policies' pre/post pipelines.

    Policies compose these in their own order (step ORDER is a Hub-serialized contract
    and intentionally stays explicit per policy) and interleave their custom steps.
    """

    # 中文 L0096 声明“rename_observations”的类型为 RenameObservationsProcessorStep。
    rename_observations: RenameObservationsProcessorStep
    # 中文 L0097 声明“add_batch_dim”的类型为 AddBatchDimensionProcessorStep。
    add_batch_dim: AddBatchDimensionProcessorStep
    # 中文 L0098 声明“to_device”的类型为 DeviceProcessorStep。
    to_device: DeviceProcessorStep
    # 中文 L0099 声明“normalize”的类型为 NormalizerProcessorStep。
    normalize: NormalizerProcessorStep
    # 中文 L0100 声明“unnormalize”的类型为 UnnormalizerProcessorStep。
    unnormalize: UnnormalizerProcessorStep
    # 中文 L0101 声明“to_cpu”的类型为 DeviceProcessorStep。
    to_cpu: DeviceProcessorStep


# 中文 L0104 定义函数 make_default_policy_processor_steps。
def make_default_policy_processor_steps(
    # 中文 L0105 “config”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    config: PreTrainedConfig,
    # 中文 L0106 为“dataset_stats”提供这一项的值，注意它所在的参数表或外层表达式。
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    # 中文 L0107 后续参数必须按名字传入。
    *,
    # 中文 L0108 为“normalizer_device”提供这一项的值，注意它所在的参数表或外层表达式。
    normalizer_device: torch.device | str | None = None,
# 中文 L0109 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> DefaultPolicyProcessorSteps:
    """Construct the canonical policy processor steps from a policy config.

    Args:
        config: A `PreTrainedConfig` providing `device`, `input_features`,
            `output_features` and `normalization_mapping`.
        dataset_stats: Dataset statistics used for (un)normalization.
        normalizer_device: Device passed to `NormalizerProcessorStep` (some policies pin
            their normalization stats to the policy device; most leave it unset).
    """
    # 中文 L0119 条件 config.device is None or config.input_features is None or config.output_features is None 成立时进入下方分支。
    if config.device is None or config.input_features is None or config.output_features is None:
        # 中文 L0120 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0121 字符串字面量，供上方的调用、容器或错误消息使用。
            "PreTrainedConfig.device, input_features and output_features must be resolved before "
            # 中文 L0122 字符串字面量，供上方的调用、容器或错误消息使用。
            "building the default policy processor steps."
        # 中文 L0123 闭合上方尚未结束的括号。
        )
    # 中文 L0124 调用 DefaultPolicyProcessorSteps，用它的结果作为当前调用的返回值。
    return DefaultPolicyProcessorSteps(
        # 中文 L0125 为“rename_observations”提供这一项的值，注意它所在的参数表或外层表达式。
        rename_observations=RenameObservationsProcessorStep(rename_map={}),
        # 中文 L0126 为“add_batch_dim”提供这一项的值，注意它所在的参数表或外层表达式。
        add_batch_dim=AddBatchDimensionProcessorStep(),
        # 中文 L0127 为“to_device”提供这一项的值，注意它所在的参数表或外层表达式。
        to_device=DeviceProcessorStep(device=config.device),
        # 中文 L0128 为“normalize”提供这一项的值，注意它所在的参数表或外层表达式。
        normalize=NormalizerProcessorStep(
            # 中文 L0129 为“features”提供这一项的值，注意它所在的参数表或外层表达式。
            features={**config.input_features, **config.output_features},
            # 中文 L0130 为“norm_map”提供这一项的值，注意它所在的参数表或外层表达式。
            norm_map=config.normalization_mapping,
            # 中文 L0131 为“stats”（归一化统计量）提供这一项的值，注意它所在的参数表或外层表达式。
            stats=dataset_stats,
            # 中文 L0132 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
            device=normalizer_device,
        # 中文 L0133 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
        ),
        # 中文 L0134 为“unnormalize”提供这一项的值，注意它所在的参数表或外层表达式。
        unnormalize=UnnormalizerProcessorStep(
            # 中文 L0135 为“features”提供这一项的值，注意它所在的参数表或外层表达式。
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        # 中文 L0136 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
        ),
        # 中文 L0137 为“to_cpu”提供这一项的值，注意它所在的参数表或外层表达式。
        to_cpu=DeviceProcessorStep(device="cpu"),
    # 中文 L0138 闭合上方尚未结束的括号。
    )


# 中文 L0141 定义函数 make_policy_processor_pipelines。
def make_policy_processor_pipelines(
    # 中文 L0142 “input_steps”是当前括号中的一项，逗号将它与下一项分开。
    input_steps: list[ProcessorStep],
    # 中文 L0143 “output_steps”是当前括号中的一项，逗号将它与下一项分开。
    output_steps: list[ProcessorStep],
# 中文 L0144 延续上面的语句，) -> tuple[。
) -> tuple[
    # 中文 L0145 延续上面的语句，PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0146 延续上面的语句，PolicyProcessorPipeline[PolicyAction, PolicyAction],。
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
# 中文 L0147 闭合上方尚未结束的括号。
]:
    """Wrap pre/post step lists into the canonical policy pipeline pair.

    Uses the standard pipeline names (which determine the serialized JSON filenames on
    the Hub) and the standard policy-action converters on the postprocessor.
    """
    # 中文 L0153 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
    return (
        # 中文 L0154 延续上面的语句，PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](。
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            # 中文 L0155 为“steps”提供这一项的值，注意它所在的参数表或外层表达式。
            steps=input_steps,
            # 中文 L0156 为“name”提供这一项的值，注意它所在的参数表或外层表达式。
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        # 中文 L0157 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
        ),
        # 中文 L0158 延续上面的语句，PolicyProcessorPipeline[PolicyAction, PolicyAction](。
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            # 中文 L0159 为“steps”提供这一项的值，注意它所在的参数表或外层表达式。
            steps=output_steps,
            # 中文 L0160 为“name”提供这一项的值，注意它所在的参数表或外层表达式。
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            # 中文 L0161 为“to_transition”提供这一项的值，注意它所在的参数表或外层表达式。
            to_transition=policy_action_to_transition,
            # 中文 L0162 为“to_output”提供这一项的值，注意它所在的参数表或外层表达式。
            to_output=transition_to_policy_action,
        # 中文 L0163 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
        ),
    # 中文 L0164 闭合上方尚未结束的括号。
    )


def make_default_pre_post_processors(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    normalizer_device: torch.device | str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """The pure-scaffold policy pipeline pair: Rename -> Batch -> Device -> Normalize,
    and Unnormalize -> Device(cpu). Policies with custom steps or a different step order
    compose `make_default_policy_processor_steps` themselves instead.
    """
    s = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=normalizer_device)
    return make_policy_processor_pipelines(
        input_steps=[s.rename_observations, s.add_batch_dim, s.to_device, s.normalize],
        output_steps=[s.unnormalize, s.to_cpu],
    )


# 中文 L0187 定义函数 _reconnect_relative_absolute_steps。
def _reconnect_relative_absolute_steps(
    # 中文 L0188 延续上面的语句，preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline。
    preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
# 中文 L0189 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> None:
    """Wire AbsoluteActionsProcessorStep.relative_step to the RelativeActionsProcessorStep after deserialization.

    After a policy is loaded from disk, the preprocessor and postprocessor are reconstructed
    independently from their configs. AbsoluteActionsProcessorStep needs a live reference to
    the RelativeActionsProcessorStep so it can read the cached state at inference time.
    That reference is not serializable, so we re-establish it here after loading.
    """
    # 中文 L0197 检查前处理器中是否存在相对动作处理步骤，本次 checkpoint 没有这一步。
    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    # 中文 L0198 条件 relative_step is None 成立时进入下方分支。
    if relative_step is None:
        # 中文 L0199 退出当前调用并返回 None。
        return
    # 中文 L0200 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for step in postprocessor.steps:
        # 中文 L0201 条件 isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None 成立时进入下方分支。
        if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
            # 中文 L0202 引用已有的值 relative_step给 “step.relative_step” 赋值。
            step.relative_step = relative_step


# 中文 L0205 从预训练模型目录或 Hub 缓存恢复策略的前后处理流水线。
def load_pretrained_policy_processors(
    # 中文 L0206 “pretrained_path”是当前括号中的一项，逗号将它与下一项分开。
    pretrained_path: str | Path,
    # 中文 L0207 后续参数必须按名字传入。
    *,
    # 中文 L0208 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
    revision: str | None = None,
    # 中文 L0209 为“preprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
    preprocessor_overrides: dict[str, Any] | None = None,
    # 中文 L0210 为“postprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
    postprocessor_overrides: dict[str, Any] | None = None,
    # 中文 L0211 为“preprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    # 中文 L0212 为“postprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
# 中文 L0213 延续上面的语句，) -> tuple[。
) -> tuple[
    # 中文 L0214 延续上面的语句，PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0215 延续上面的语句，PolicyProcessorPipeline[PolicyAction, PolicyAction],。
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
# 中文 L0216 闭合上方尚未结束的括号。
]:
    # 中文 L0217 原文档中的说明或续行。
    """Load a serialized policy pipeline pair and re-establish links that do not survive saving."""
    # 中文 L0218 读预训练前处理配置并创建有序的处理器对象。
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        # 中文 L0219 为“pretrained_model_name_or_path”提供这一项的值，注意它所在的参数表或外层表达式。
        pretrained_model_name_or_path=pretrained_path,
        # 中文 L0220 为“config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
        config_filename=preprocessor_config_filename,
        # 中文 L0221 用评测参数覆盖相机改名表和目标设备等保存字段。
        overrides=preprocessor_overrides,
        # 中文 L0222 把外部批量观测字典转换成处理器统一使用的 transition 结构。
        to_transition=batch_to_transition,
        # 中文 L0223 把前处理后的 transition 转回策略期望的批量字典。
        to_output=transition_to_batch,
        # 中文 L0224 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
        revision=revision,
    # 中文 L0225 闭合上方尚未结束的括号。
    )
    # 中文 L0226 读取对应 checkpoint 的后处理配置。
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        # 中文 L0227 为“pretrained_model_name_or_path”提供这一项的值，注意它所在的参数表或外层表达式。
        pretrained_model_name_or_path=pretrained_path,
        # 中文 L0228 为“config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
        config_filename=postprocessor_config_filename,
        # 中文 L0229 为“overrides”提供这一项的值，注意它所在的参数表或外层表达式。
        overrides=postprocessor_overrides,
        # 中文 L0230 把外部动作封装进 transition，以便复用同一套处理接口。
        to_transition=policy_action_to_transition,
        # 中文 L0231 执行完后从 transition 提取处理过的动作。
        to_output=transition_to_policy_action,
        # 中文 L0232 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
        revision=revision,
    # 中文 L0233 闭合上方尚未结束的括号。
    )
    # 中文 L0234 重新连接有需要的相对动作步骤。本次保存的步骤没有这对关联。
    _reconnect_relative_absolute_steps(preprocessor, postprocessor)
    # 中文 L0235 返回可直接调用的前处理和后处理对象。
    return preprocessor, postprocessor
