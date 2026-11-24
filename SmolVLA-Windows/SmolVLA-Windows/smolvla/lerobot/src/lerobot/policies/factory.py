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
# 中文 L0017 从模块 __future__ 导入 annotations。
from __future__ import annotations

# 中文 L0019 导入模块 importlib。
import importlib
# 中文 L0020 导入模块 inspect。
import inspect
# 中文 L0021 导入模块 logging。
import logging
# 中文 L0022 导入模块 os。
import os
# 中文 L0023 从模块 pathlib 导入 Path。
from pathlib import Path
# 中文 L0024 从模块 types 导入 ModuleType。
from types import ModuleType
# 中文 L0025 从模块 typing 导入 TYPE_CHECKING、Any、TypedDict、Unpack。
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

# 中文 L0027 导入模块 torch。
import torch
# 中文 L0028 从模块 huggingface_hub 导入 hf_hub_download。
from huggingface_hub import hf_hub_download
# 中文 L0029 从模块 huggingface_hub.errors 导入 EntryNotFoundError。
from huggingface_hub.errors import EntryNotFoundError

# 中文 L0031 条件 TYPE_CHECKING 成立时进入下方分支。
if TYPE_CHECKING:
    # 中文 L0032 从模块 lerobot.datasets 导入 LeRobotDatasetMetadata。
    from lerobot.datasets import LeRobotDatasetMetadata

# 中文 L0034 从模块 lerobot.configs 导入 FeatureType、PreTrainedConfig。
from lerobot.configs import FeatureType, PreTrainedConfig
# 中文 L0035 从模块 lerobot.envs 导入 EnvConfig、env_to_policy_features。
from lerobot.envs import EnvConfig, env_to_policy_features
# 中文 L0036 从模块 lerobot.lerobot_types 导入 PolicyAction。
from lerobot.lerobot_types import PolicyAction
# 中文 L0037 从模块 lerobot.processor 导入 PolicyProcessorPipeline、load_pretrained_policy_processors、rename_keys。
from lerobot.processor import (
    # 中文 L0038 “PolicyProcessorPipeline”是当前括号中的一项，逗号将它与下一项分开。
    PolicyProcessorPipeline,
    # 中文 L0039 “load_pretrained_policy_processors”是当前括号中的一项，逗号将它与下一项分开。
    load_pretrained_policy_processors,
    # 中文 L0040 “rename_keys”是当前括号中的一项，逗号将它与下一项分开。
    rename_keys,
# 中文 L0041 闭合上方尚未结束的括号。
)
# 中文 L0042 从模块 lerobot.utils.constants 导入 ACTION、POLICY_POSTPROCESSOR_DEFAULT_NAME、POLICY_PREPROCESSOR_DEFAULT_NAME。
from lerobot.utils.constants import (
    # 中文 L0043 “ACTION”是当前括号中的一项，逗号将它与下一项分开。
    ACTION,
    # 中文 L0044 “POLICY_POSTPROCESSOR_DEFAULT_NAME”是当前括号中的一项，逗号将它与下一项分开。
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    # 中文 L0045 “POLICY_PREPROCESSOR_DEFAULT_NAME”是当前括号中的一项，逗号将它与下一项分开。
    POLICY_PREPROCESSOR_DEFAULT_NAME,
# 中文 L0046 闭合上方尚未结束的括号。
)
# 中文 L0047 从模块 lerobot.utils.feature_utils 导入 dataset_to_policy_features。
from lerobot.utils.feature_utils import dataset_to_policy_features
# 中文 L0048 从模块 lerobot.utils.import_utils 导入 _peft_available、require_package。
from lerobot.utils.import_utils import _peft_available, require_package

# 中文 L0050 从模块 .pretrained 导入 PreTrainedPolicy。
from .pretrained import PreTrainedPolicy
# 中文 L0051 从模块 .utils 导入 validate_visual_features_consistency。
from .utils import validate_visual_features_consistency

# 中文 L0053 条件 TYPE_CHECKING or _peft_available 成立时进入下方分支。
if TYPE_CHECKING or _peft_available:
    # 中文 L0054 从模块 peft 导入 PeftConfig、PeftModel、get_peft_model。
    from peft import PeftConfig, PeftModel, get_peft_model
# 中文 L0055 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
else:
    # 中文 L0056 设为空值 None给 “PeftConfig” 赋值。
    PeftConfig = None
    # 中文 L0057 设为空值 None给 “PeftModel” 赋值。
    PeftModel = None
    # 中文 L0058 设为空值 None给 “get_peft_model” 赋值。
    get_peft_model = None

# 中文 L0060 使用字面值 'adapter_config.json'给 “PEFT_ADAPTER_CONFIG_NAME” 赋值。
PEFT_ADAPTER_CONFIG_NAME = "adapter_config.json"


# 中文 L0063 根据已注册的策略名定位并返回一个类，实例在后面的 make_policy 创建。
def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    Resolution is convention-based: the draccus-registered config class of ``name`` is
    looked up, its ``configuration_*`` module path is rewritten to ``modeling_*``, and
    the ``<X>Policy`` class is imported from there. The modeling module is only imported
    at call time, keeping heavy optional dependencies lazy. This works for both built-in
    policies and third-party lerobot plugins (anything registered via
    ``@PreTrainedConfig.register_subclass``).

    Args:
        name: The registered name of the policy (e.g. "act", "diffusion", "pi0").
    Returns:
        The policy class corresponding to the given name.

    Raises:
        ValueError: If the policy name is not registered.
        ImportError: If the policy's optional dependencies are not installed.
    """
    # 中文 L0083 条件 name not in PreTrainedConfig.get_known_choices() 成立时进入下方分支。
    if name not in PreTrainedConfig.get_known_choices():
        # 中文 L0084 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0085 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        # 中文 L0086 闭合上方尚未结束的括号。
        )

    # 中文 L0088 先通过名称查到配置类。smolvla 在配置模块里注册为 SmolVLAConfig。
    config_cls = PreTrainedConfig.get_choice_class(name)
    # 中文 L0089 读取属性 config_cls.__name__给 “config_cls_name” 赋值。
    config_cls_name = config_cls.__name__

    # 中文 L0091 从类名后缀移除 Config，计算模型类的共同词干 SmolVLA。
    model_name = config_cls_name.removesuffix("Config")  # e.g., DiffusionConfig -> Diffusion
    # 中文 L0092 条件 model_name == config_cls_name 成立时进入下方分支。
    if model_name == config_cls_name:
        # 中文 L0093 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0094 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            # 中文 L0095 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Make sure it ends with 'Config'!"
        # 中文 L0096 闭合上方尚未结束的括号。
        )
    # 中文 L0097 在配置类的词干后加 Policy，得到应导入的类名 SmolVLAPolicy。
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy

    # 中文 L0099 把配置类的模块路径转换成同包的 modeling 模块路径并导入。
    module = _import_sibling_policy_module(config_cls, "modeling")
    # 中文 L0100 条件 module is None 成立时进入下方分支。
    if module is None:
        # 中文 L0101 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Policy class for '{name}' is not implemented.")
    # 中文 L0102 按刚得到的字符串从模块取策略类。
    policy_cls = getattr(module, cls_name, None)
    # 中文 L0103 条件 policy_cls is None 成立时进入下方分支。
    if policy_cls is None:
        # 中文 L0104 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0105 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Policy class '{cls_name}' not found in '{module.__name__}'. "
            # 中文 L0106 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Policies must expose '<Name>Policy' in the sibling 'modeling_*' module by naming convention."
        # 中文 L0107 闭合上方尚未结束的括号。
        )
    # 中文 L0108 引用已有的值 policy_cls作为当前调用的返回值。
    return policy_cls


# 中文 L0111 定义函数 make_policy_config。
def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The registered type of the policy (any name registered via
                     ``@PreTrainedConfig.register_subclass``, e.g. "act", "diffusion", "pi0").
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    # 中文 L0129 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0130 调用 PreTrainedConfig.get_choice_class，用它的结果给 “config_cls” 赋值。
        config_cls = PreTrainedConfig.get_choice_class(policy_type)
    # 中文 L0131 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except Exception as e:
        # 中文 L0132 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Policy type '{policy_type}' is not available.") from e
    # 中文 L0133 调用 config_cls，用它的结果作为当前调用的返回值。
    return config_cls(**kwargs)


# 中文 L0136 定义类 ProcessorConfigKwargs，括号列出基类 TypedDict。
class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    # 中文 L0151 声明“preprocessor_config_filename”的类型为 str | None。
    preprocessor_config_filename: str | None
    # 中文 L0152 声明“postprocessor_config_filename”的类型为 str | None。
    postprocessor_config_filename: str | None
    # 中文 L0153 声明“preprocessor_overrides”的类型为 dict[str, Any] | None。
    preprocessor_overrides: dict[str, Any] | None
    # 中文 L0154 声明“postprocessor_overrides”的类型为 dict[str, Any] | None。
    postprocessor_overrides: dict[str, Any] | None
    # 中文 L0155 声明“dataset_stats”的类型为 dict[str, dict[str, torch.Tensor]] | None。
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None
    # 中文 L0156 声明“dataset_meta”的类型为 Any | None。
    dataset_meta: Any | None


# 中文 L0159 选择从训练产物恢复处理步骤或按照当前配置从头组装。
def make_pre_post_processors(
    # 中文 L0160 “policy_cfg”是当前括号中的一项，逗号将它与下一项分开。
    policy_cfg: PreTrainedConfig,
    # 中文 L0161 为“pretrained_path”提供这一项的值，注意它所在的参数表或外层表达式。
    pretrained_path: str | Path | None = None,
    # 中文 L0162 为“pretrained_revision”提供这一项的值，注意它所在的参数表或外层表达式。
    pretrained_revision: str | None = None,
    # 中文 L0163 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
    **kwargs: Unpack[ProcessorConfigKwargs],
# 中文 L0164 延续上面的语句，) -> tuple[。本段处理恢复或新建一对策略处理流水线。
) -> tuple[
    # 中文 L0165 延续上面的语句，PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。本段处理恢复或新建一对策略处理流水线。
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0166 延续上面的语句，PolicyProcessorPipeline[PolicyAction, PolicyAction],。本段处理恢复或新建一对策略处理流水线。
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
# 中文 L0167 闭合上方尚未结束的括号。
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        ValueError: If no processor factory exists for the given policy configuration type.
    """
    # 中文 L0189 有预训练来源时，优先使用该 checkpoint 保存的流水线配置和统计量。
    if pretrained_path:
        # 中文 L0190 组合多个布尔判断，按 Python 的短路规则求值给 “preprocessor_config_filename” 赋值。
        preprocessor_config_filename = (
            # 中文 L0191 延续上面的语句，kwargs.get("preprocessor_config_filename") or f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"。本段处理恢复或新建一对策略处理流水线。
            kwargs.get("preprocessor_config_filename") or f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        # 中文 L0192 闭合上方尚未结束的括号。
        )
        # 中文 L0193 组合多个布尔判断，按 Python 的短路规则求值给 “postprocessor_config_filename” 赋值。
        postprocessor_config_filename = (
            # 中文 L0194 延续上面的语句，kwargs.get("postprocessor_config_filename") or f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"。本段处理恢复或新建一对策略处理流水线。
            kwargs.get("postprocessor_config_filename") or f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
        # 中文 L0195 闭合上方尚未结束的括号。
        )
        # 中文 L0196 尝试调用策略特有的预训练处理器工厂。SmolVLA 没有这种专用函数。
        custom_processors = _make_pretrained_processors_from_policy_config(
            # 中文 L0197 为“config”（配置对象）提供这一项的值，注意它所在的参数表或外层表达式。
            config=policy_cfg,
            # 中文 L0198 为“pretrained_path”提供这一项的值，注意它所在的参数表或外层表达式。
            pretrained_path=pretrained_path,
            # 中文 L0199 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
            revision=pretrained_revision,
            # 中文 L0200 为“dataset_stats”提供这一项的值，注意它所在的参数表或外层表达式。
            dataset_stats=kwargs.get("dataset_stats"),
            # 中文 L0201 为“dataset_meta”提供这一项的值，注意它所在的参数表或外层表达式。
            dataset_meta=kwargs.get("dataset_meta"),
            # 中文 L0202 为“preprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
            preprocessor_overrides=kwargs.get("preprocessor_overrides"),
            # 中文 L0203 为“postprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
            postprocessor_overrides=kwargs.get("postprocessor_overrides"),
            # 中文 L0204 为“preprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
            preprocessor_config_filename=preprocessor_config_filename,
            # 中文 L0205 为“postprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
            postprocessor_config_filename=postprocessor_config_filename,
        # 中文 L0206 闭合上方尚未结束的括号。
        )
        # 中文 L0207 条件 custom_processors is not None 成立时进入下方分支。
        if custom_processors is not None:
            # 中文 L0208 引用已有的值 custom_processors作为当前调用的返回值。
            return custom_processors

        # 中文 L0210 本次评测从序列化的 JSON 恢复通用前后处理流水线。
        return load_pretrained_policy_processors(
            # 中文 L0211 “pretrained_path”是当前括号中的一项，逗号将它与下一项分开。
            pretrained_path,
            # 中文 L0212 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
            revision=pretrained_revision,
            # 中文 L0213 为“preprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
            preprocessor_overrides=kwargs.get("preprocessor_overrides"),
            # 中文 L0214 为“postprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
            postprocessor_overrides=kwargs.get("postprocessor_overrides"),
            # 中文 L0215 为“preprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
            preprocessor_config_filename=preprocessor_config_filename,
            # 中文 L0216 为“postprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
            postprocessor_config_filename=postprocessor_config_filename,
        # 中文 L0217 闭合上方尚未结束的括号。
        )

    # Create new processors from the policy config, resolving the per-policy factory
    # function by naming convention (lazy import keeps optional dependencies optional).
    # 中文 L0221 调用 _make_processors_from_policy_config，用它的结果作为当前调用的返回值。
    return _make_processors_from_policy_config(
        # 中文 L0222 为“config”（配置对象）提供这一项的值，注意它所在的参数表或外层表达式。
        config=policy_cfg,
        # 中文 L0223 为“dataset_stats”提供这一项的值，注意它所在的参数表或外层表达式。
        dataset_stats=kwargs.get("dataset_stats"),
        # 中文 L0224 为“dataset_meta”提供这一项的值，注意它所在的参数表或外层表达式。
        dataset_meta=kwargs.get("dataset_meta"),
    # 中文 L0225 闭合上方尚未结束的括号。
    )


# 中文 L0228 定义函数 has_peft_adapter_config。
def has_peft_adapter_config(
    # 中文 L0229 “pretrained_path”是当前括号中的一项，逗号将它与下一项分开。
    pretrained_path: str | Path,
    # 中文 L0230 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
    revision: str | None = None,
# 中文 L0231 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> bool:
    """Return whether ``pretrained_path`` points to an existing PEFT adapter.

    A PEFT adapter checkpoint always ships an ``adapter_config.json``.
    A plain base-model checkpoint does not. This distinction lets us tell apart
    two very different ``use_peft=True`` scenarios that both set ``pretrained_path``:

    * loading/resuming a previously trained adapter (config lives at ``pretrained_path``)
    * starting a *fresh* PEFT fine-tune on top of a base model

    Works for both local directories and Hub repo ids. Hub repos go through the local cache, so
    a cached adapter is recognised offline. Only a missing ``adapter_config.json`` means "not an
    adapter": any other Hub error (unknown repo, auth, network) is raised as-is.
    """
    # 中文 L0245 调用 str，用它的结果给 “pretrained_path” 赋值。
    pretrained_path = str(pretrained_path)
    # 中文 L0246 条件 os.path.isdir(pretrained_path) 成立时进入下方分支。
    if os.path.isdir(pretrained_path):
        # 中文 L0247 调用 os.path.isfile，用它的结果作为当前调用的返回值。
        return os.path.isfile(os.path.join(pretrained_path, PEFT_ADAPTER_CONFIG_NAME))

    # 中文 L0249 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0250 调用 hf_hub_download 执行操作。
        hf_hub_download(pretrained_path, PEFT_ADAPTER_CONFIG_NAME, revision=revision)
    # 中文 L0251 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except EntryNotFoundError:  # also raised offline when the file is not cached
        # 中文 L0252 使用布尔值 False作为当前调用的返回值。
        return False
    # 中文 L0253 使用布尔值 True作为当前调用的返回值。
    return True


# 中文 L0256 整合已经解析的策略配置和环境或数据集特征，并创建策略。
def make_policy(
    # 中文 L0257 “cfg”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    cfg: PreTrainedConfig,
    # 中文 L0258 为“ds_meta”提供这一项的值，注意它所在的参数表或外层表达式。
    ds_meta: LeRobotDatasetMetadata | None = None,
    # 中文 L0259 为“env_cfg”提供这一项的值，注意它所在的参数表或外层表达式。
    env_cfg: EnvConfig | None = None,
    # 中文 L0260 为“rename_map”提供这一项的值，注意它所在的参数表或外层表达式。
    rename_map: dict[str, str] | None = None,
    # 中文 L0261 为“defer_weight_load”提供这一项的值，注意它所在的参数表或外层表达式。
    defer_weight_load: bool = False,
    # 中文 L0262 为“pretrained_path”提供这一项的值，注意它所在的参数表或外层表达式。
    pretrained_path: str | Path | None = None,
# 中文 L0263 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg (PreTrainedConfig): The configuration for the policy to be created. If
            `cfg.pretrained_path` is set, the policy will be loaded with weights from that path.
        ds_meta (LeRobotDatasetMetadata | None): Dataset metadata used to infer feature shapes and
            types. Also provides statistics for normalization layers.
        env_cfg (EnvConfig | None): Environment configuration used to infer feature shapes and
            types. One of `ds_meta` or `env_cfg` must be provided.
        rename_map (dict[str, str] | None): Optional mapping of dataset or environment feature
            keys to match expected policy feature names (e.g., `"left"` → `"camera1"`).
        defer_weight_load (bool): Build the exact policy `from_pretrained` would build — same
            config resolution, same stats-derived buffers, same device placement and eval mode —
            but skip the safetensors weight load. Used when resuming from a DCP checkpoint, whose
            sharded weights stream in after `accelerator.prepare()` (the distributed checkpoint
            engine overwrites the random init). For a PEFT policy only the adapter weights are
            skipped, the base weights load as usual.
        pretrained_path (str | Path | None): Load the weights (or PEFT adapter) from here instead
            of `cfg.pretrained_path`, which keeps naming the model the policy was fine-tuned from
            once the policy is built. Used when resuming from a checkpoint.

    Returns:
        PreTrainedPolicy: An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend combination
            (e.g., VQBeT with 'mps').
    """
    # 中文 L0299 条件 bool(ds_meta) == bool(env_cfg) 成立时进入下方分支。
    if bool(ds_meta) == bool(env_cfg):
        # 中文 L0300 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    # 中文 L0309 条件 cfg.type == "vqbet" and cfg.device == "mps" 成立时进入下方分支。
    if cfg.type == "vqbet" and cfg.device == "mps":
        # 中文 L0310 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError(
            # 中文 L0311 字符串字面量，供上方的调用、容器或错误消息使用。
            "Current implementation of VQBeT does not support `mps` backend. "
            # 中文 L0312 字符串字面量，供上方的调用、容器或错误消息使用。
            "Please use `cpu` or `cuda` backend."
        # 中文 L0313 闭合上方尚未结束的括号。
        )

    # 中文 L0315 取到具体的策略类。结合原运行的 type 字段，这里拿到 SmolVLAPolicy。
    policy_cls = get_policy_class(cfg.type)
    # Policies read `pretrained_path` while building to skip initializing weights the load will
    # overwrite, so it names the weight source until the policy is built, then the parent again.
    # 中文 L0318 读取属性 cfg.pretrained_path给 “parent_path” 赋值。
    parent_path = cfg.pretrained_path
    # 中文 L0319 条件 pretrained_path is not None 成立时进入下方分支。
    if pretrained_path is not None:
        # 中文 L0320 调用 Path，用它的结果给 “cfg.pretrained_path” 赋值。
        cfg.pretrained_path = Path(pretrained_path)

    # 中文 L0322 声明“kwargs”（额外的关键字参数）的类型为 dict[str, Any]，创建字典，用键查找对应的值作为初始值。
    kwargs: dict[str, Any] = {}
    # 中文 L0323 条件 ds_meta is not None 成立时进入下方分支。
    if ds_meta is not None:
        # 中文 L0324 调用 dataset_to_policy_features，用它的结果给 “features” 赋值。
        features = dataset_to_policy_features(ds_meta.features)
    # 中文 L0325 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
    else:
        # 中文 L0326 条件 not cfg.pretrained_path 成立时进入下方分支。
        if not cfg.pretrained_path:
            # 中文 L0327 调用 logging.warning 执行操作。
            logging.warning(
                # 中文 L0328 字符串字面量，供上方的调用、容器或错误消息使用。
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                # 中文 L0329 字符串字面量，供上方的调用、容器或错误消息使用。
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                # 中文 L0330 字符串字面量，供上方的调用、容器或错误消息使用。
                "by default without stats from a dataset."
            # 中文 L0331 闭合上方尚未结束的括号。
            )
        # 中文 L0332 条件 env_cfg is None 成立时进入下方分支。
        if env_cfg is None:
            # 中文 L0333 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        # 中文 L0334 调用 env_to_policy_features，用它的结果给 “features” 赋值。
        features = env_to_policy_features(env_cfg)

    # 中文 L0336 条件 rename_map 成立时进入下方分支。
    if rename_map:
        # 中文 L0337 调用 rename_keys，用它的结果给 “features” 赋值。
        features = rename_keys(features, rename_map)

    # 中文 L0339 条件 ds_meta is not None or not cfg.output_features 成立时进入下方分支。
    if ds_meta is not None or not cfg.output_features:
        # 中文 L0340 用字典推导式从可迭代对象逐项取值并构造结果给 “cfg.output_features” 赋值。
        cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    # 中文 L0341 条件 not cfg.input_features 成立时进入下方分支。
    if not cfg.input_features:
        # 中文 L0342 用字典推导式从可迭代对象逐项取值并构造结果给 “cfg.input_features” 赋值。
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    # Store action feature names for relative_exclude_joints support
    # 中文 L0345 条件 ds_meta is not None and hasattr(cfg, "action_feature_names") 成立时进入下方分支。
    if ds_meta is not None and hasattr(cfg, "action_feature_names"):
        # 中文 L0346 调用 next，用它的结果给 “raw_action_feature” 赋值。
        raw_action_feature = next(
            # 中文 L0347 延续上面的语句，(。本段处理统一构造策略并判断权重加载路径。
            (
                # 中文 L0348 延续上面的语句，feature。本段处理统一构造策略并判断权重加载路径。
                feature
                # 中文 L0349 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for raw_key, feature in ds_meta.features.items()
                # 中文 L0350 条件 (rename_map or {}).get(raw_key, raw_key) == ACTION 成立时进入下方分支。
                if (rename_map or {}).get(raw_key, raw_key) == ACTION
            # 中文 L0351 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
            ),
            # 中文 L0352 “None”是当前括号中的一项，逗号将它与下一项分开。
            None,
        # 中文 L0353 闭合上方尚未结束的括号。
        )
        # 中文 L0354 用条件表达式在 if 前与 else 后的两个值之间选择给 “action_names” 赋值。
        action_names = raw_action_feature.get("names") if raw_action_feature is not None else None
        # 中文 L0355 条件 action_names is not None 成立时进入下方分支。
        if action_names is not None:
            # Grouped metadata stores dimension names in the values, not the group keys.
            # 中文 L0357 条件 isinstance(action_names, dict) and all( 成立时进入下方分支。
            if isinstance(action_names, dict) and all(
                # 中文 L0358 延续上面的语句，isinstance(group, (list, tuple)) for group in action_names.values()。本段处理统一构造策略并判断权重加载路径。
                isinstance(group, (list, tuple)) for group in action_names.values()
            # 中文 L0359 闭合上方尚未结束的括号。
            ):
                # 中文 L0360 用列表推导式从可迭代对象逐项取值并构造结果给 “action_names” 赋值。
                action_names = [name for group in action_names.values() for name in group]
            # 中文 L0361 调用 list，用它的结果给 “cfg.action_feature_names” 赋值。
            cfg.action_feature_names = list(action_names)
    # 中文 L0362 条件 ds_meta is not None 成立时进入下方分支。
    if ds_meta is not None:
        # 中文 L0363 调用 getattr，用它的结果给 “set_dataset_feature_metadata” 赋值。
        set_dataset_feature_metadata = getattr(cfg, "set_dataset_feature_metadata", None)
        # 中文 L0364 条件 callable(set_dataset_feature_metadata) 成立时进入下方分支。
        if callable(set_dataset_feature_metadata):
            # 中文 L0365 读取属性 ds_meta.features给 “ds_feature” 赋值。
            ds_feature = ds_meta.features
            # 中文 L0366 条件 rename_map: # use the policy-side (renamed) keys 成立时进入下方分支。
            if rename_map:  # use the policy-side (renamed) keys
                # 中文 L0367 调用 rename_keys，用它的结果给 “ds_feature” 赋值。
                ds_feature = rename_keys(ds_meta.features, rename_map)
            # 中文 L0368 调用 set_dataset_feature_metadata 执行操作。
            set_dataset_feature_metadata(ds_feature)
        # 中文 L0369 引用已有的值 ds_meta给 “cfg._runtime_dataset_meta” 赋值。
        cfg._runtime_dataset_meta = ds_meta

    # 中文 L0371 引用已有的值 cfg给 “kwargs['config']” 赋值。
    kwargs["config"] = cfg

    # Pass dataset_stats to the policy if available (needed for some policies like SARM)
    # 中文 L0374 条件 ds_meta is not None and hasattr(ds_meta, "stats") 成立时进入下方分支。
    if ds_meta is not None and hasattr(ds_meta, "stats"):
        # 中文 L0375 读取属性 ds_meta.stats给 “kwargs['dataset_stats']” 赋值。
        kwargs["dataset_stats"] = ds_meta.stats

    # 中文 L0377 条件 ds_meta is not None 成立时进入下方分支。
    if ds_meta is not None:
        # 中文 L0378 引用已有的值 ds_meta给 “kwargs['dataset_meta']” 赋值。
        kwargs["dataset_meta"] = ds_meta

    # 中文 L0380 条件 not cfg.pretrained_path and cfg.use_peft 成立时进入下方分支。
    if not cfg.pretrained_path and cfg.use_peft:
        # 中文 L0381 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0382 字符串字面量，供上方的调用、容器或错误消息使用。
            "Instantiating a policy with `use_peft=True` without a checkpoint is not supported since that requires "
            # 中文 L0383 字符串字面量，供上方的调用、容器或错误消息使用。
            "the PEFT config parameters to be set. For training with PEFT, see `lerobot_train.py` on how to do that."
        # 中文 L0384 闭合上方尚未结束的括号。
        )

    # When `use_peft=True` and a checkpoint is given, the checkpoint can be one of two things:
    # 1. A base model checkpoint (e.g., a pretrained policy) on which we want to start a fresh PEFT fine-tune.
    # 2. A PEFT adapter checkpoint (e.g., a previously trained PEFT adapter)
    # We distinguish between these two cases
    # 中文 L0390 组合多个布尔判断，按 Python 的短路规则求值给 “load_existing_adapter” 赋值。
    load_existing_adapter = (
        # 中文 L0391 延续上面的语句，cfg.pretrained_path。本段处理统一构造策略并判断权重加载路径。
        cfg.pretrained_path
        # 中文 L0392 续写上一行的布尔表达式，按 and / or 的短路规则判断。
        and cfg.use_peft
        # 中文 L0393 续写上一行的布尔表达式，按 and / or 的短路规则判断。
        and has_peft_adapter_config(cfg.pretrained_path, cfg.pretrained_revision)
    # 中文 L0394 闭合上方尚未结束的括号。
    )

    # 中文 L0396 预训练但没有已有 PEFT 适配器时进入此分支。
    if cfg.pretrained_path and not load_existing_adapter:
        # 中文 L0397 条件 defer_weight_load 成立时进入下方分支。
        if defer_weight_load:
            # Same construction path as from_pretrained (config already resolved from the
            # checkpoint by the caller; dataset_stats/dataset_meta kwargs identical), minus the
            # weight load — parity by construction.
            # 中文 L0401 调用 policy_cls，用它的结果给 “policy” 赋值。
            policy = policy_cls(**kwargs)
            # 中文 L0402 调用 policy.eval，切换 PyTorch 模块到评测模式。
            policy.eval()
        # 中文 L0403 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # Load a pretrained policy and override the config if needed (for example, if there
            # are inference-time hyperparameters that we want to vary). This also covers starting
            # a fresh PEFT fine-tune on top of a base model: the base weights are loaded here and
            # `wrap_with_peft` builds the adapter afterwards.
            # 中文 L0408 传入预训练模型来源，原命令的值为 lerobot/smolvla_libero。
            kwargs["pretrained_name_or_path"] = cfg.pretrained_path
            # 中文 L0409 把固定的版本号传给权重加载逻辑。
            kwargs["revision"] = cfg.pretrained_revision
            # 中文 L0410 调具体策略类的 from_pretrained。实现来自 PreTrainedPolicy。
            policy = policy_cls.from_pretrained(**kwargs)
    # 中文 L0411 前面的分支均未命中时，再判断 load_existing_adapter。
    elif load_existing_adapter:
        # Load a pretrained PEFT model on top of the policy. The pretrained path points to the folder/repo
        # of the adapter and the adapter's config contains the path to the base policy. So we need the
        # adapter config first, then load the correct policy and then apply PEFT.
        # 中文 L0415 调用 require_package 执行操作。
        require_package("peft", extra="peft")

        # 中文 L0417 调用 logging.info 执行操作。
        logging.info("Loading policy's PEFT adapter.")

        # 中文 L0419 调用 str，用它的结果给 “peft_pretrained_path” 赋值。
        peft_pretrained_path = str(cfg.pretrained_path)
        # 中文 L0420 调用 PeftConfig.from_pretrained，用它的结果给 “peft_config” 赋值。
        peft_config = PeftConfig.from_pretrained(
            # 中文 L0421 “peft_pretrained_path”是当前括号中的一项，逗号将它与下一项分开。
            peft_pretrained_path,
            # 中文 L0422 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
            revision=cfg.pretrained_revision,
        # 中文 L0423 闭合上方尚未结束的括号。
        )

        # 中文 L0425 读取属性 peft_config.base_model_name_or_path给 “kwargs['pretrained_name_or_path']” 赋值。
        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        # 中文 L0426 条件 not kwargs["pretrained_name_or_path"] 成立时进入下方分支。
        if not kwargs["pretrained_name_or_path"]:
            # This means that there's a bug or we trained a policy from scratch using PEFT.
            # It is more likely that this is a bug so we'll raise an error.
            # 中文 L0429 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0430 字符串字面量，供上方的调用、容器或错误消息使用。
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                # 中文 L0431 字符串字面量，供上方的调用、容器或错误消息使用。
                "the adapter was trained."
            # 中文 L0432 闭合上方尚未结束的括号。
            )

        # 中文 L0434 读取属性 peft_config.revision给 “kwargs['revision']” 赋值。
        kwargs["revision"] = peft_config.revision
        # 中文 L0435 调用 policy_cls.from_pretrained，用它的结果给 “policy” 赋值。
        policy = policy_cls.from_pretrained(**kwargs)
        # 中文 L0436 条件 defer_weight_load 成立时进入下方分支。
        if defer_weight_load:
            # 中文 L0437 使用布尔值 False给 “peft_config.inference_mode” 赋值。
            peft_config.inference_mode = False  # trainable, as `is_trainable=True` below
            # 中文 L0438 读取属性 peft_config.base_model_name_or_path给 “policy.name_or_path” 赋值。
            policy.name_or_path = peft_config.base_model_name_or_path
            # 中文 L0439 调用 get_peft_model，用它的结果给 “policy” 赋值。
            policy = get_peft_model(policy, peft_config)
        # 中文 L0440 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0441 调用 PeftModel.from_pretrained，用它的结果给 “policy” 赋值。
            policy = PeftModel.from_pretrained(
                # 中文 L0442 “policy”是当前括号中的一项，逗号将它与下一项分开。
                policy,
                # 中文 L0443 “peft_pretrained_path”是当前括号中的一项，逗号将它与下一项分开。
                peft_pretrained_path,
                # 中文 L0444 为“config”（配置对象）提供这一项的值，注意它所在的参数表或外层表达式。
                config=peft_config,
                # 中文 L0445 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
                revision=cfg.pretrained_revision,
                # 中文 L0446 为“is_trainable”提供这一项的值，注意它所在的参数表或外层表达式。
                is_trainable=True,
            # 中文 L0447 闭合上方尚未结束的括号。
            )

    # 中文 L0449 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
    else:
        # Make a fresh policy.
        # 中文 L0451 调用 policy_cls，用它的结果给 “policy” 赋值。
        policy = policy_cls(**kwargs)

    # 中文 L0453 条件 pretrained_path is not None 成立时进入下方分支。
    if pretrained_path is not None:
        # 中文 L0454 引用已有的值 parent_path给 “cfg.pretrained_path” 赋值。
        cfg.pretrained_path = parent_path
        # 中文 L0455 用条件表达式在 if 前与 else 后的两个值之间选择给 “base_policy” 赋值。
        base_policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        # 中文 L0456 引用已有的值 parent_path给 “base_policy.config.pretrained_path” 赋值。
        base_policy.config.pretrained_path = parent_path

    # 中文 L0458 调用 policy.to 执行操作。
    policy.to(cfg.device)
    # 中文 L0459 检查此条件，条件不成立时抛出 AssertionError。
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    # 中文 L0463 条件 not rename_map 成立时进入下方分支。
    if not rename_map:
        # 中文 L0464 调用 validate_visual_features_consistency 执行操作。
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    # 中文 L0467 引用已有的值 policy作为当前调用的返回值。
    return policy


# 中文 L0470 定义函数 _import_sibling_policy_module。
def _import_sibling_policy_module(config_cls: type[PreTrainedConfig], prefix: str) -> ModuleType | None:
    # 中文 L0471 原文档中的说明或续行。
    """Import a config class' sibling ``{prefix}_*`` module, or None when that module does not exist."""
    # 中文 L0472 用配置类的 __module__ 拼出同目录的实现模块，动态导入。
    module_path = config_cls.__module__.replace("configuration_", f"{prefix}_")
    # 中文 L0473 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0474 调用 importlib.import_module，用它的结果作为当前调用的返回值。
        return importlib.import_module(module_path)
    # 中文 L0475 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except ModuleNotFoundError as e:
        # 中文 L0476 条件 e.name == module_path 成立时进入下方分支。
        if e.name == module_path:
            # The sibling module itself does not exist for this policy type. A missing optional
            # dependency inside an existing module propagates unchanged instead, so its
            # actionable install hint stays visible.
            # 中文 L0480 设为空值 None作为当前调用的返回值。
            return None
        # 中文 L0481 抛出 异常，中断当前路径并交给上层的异常处理。
        raise


# 中文 L0484 定义函数 _make_pretrained_processors_from_policy_config。作用是查找策略特有的预训练处理器创建入口。
def _make_pretrained_processors_from_policy_config(
    # 中文 L0485 “config”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    config: PreTrainedConfig,
    # 中文 L0486 “pretrained_path”是当前括号中的一项，逗号将它与下一项分开。
    pretrained_path: str | Path,
    # 中文 L0487 后续参数必须按名字传入。
    *,
    # 中文 L0488 “revision”（固定的 checkpoint 版本）是当前括号中的一项，逗号将它与下一项分开。
    revision: str | None,
    # 中文 L0489 延续上面的语句，dataset_stats: dict[str, dict[str, torch.Tensor]] | None,。本段处理查找策略特有的预训练处理器创建入口。
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    # 中文 L0490 “dataset_meta”是当前括号中的一项，逗号将它与下一项分开。
    dataset_meta: Any | None,
    # 中文 L0491 延续上面的语句，preprocessor_overrides: dict[str, Any] | None,。本段处理查找策略特有的预训练处理器创建入口。
    preprocessor_overrides: dict[str, Any] | None,
    # 中文 L0492 延续上面的语句，postprocessor_overrides: dict[str, Any] | None,。本段处理查找策略特有的预训练处理器创建入口。
    postprocessor_overrides: dict[str, Any] | None,
    # 中文 L0493 “preprocessor_config_filename”是当前括号中的一项，逗号将它与下一项分开。
    preprocessor_config_filename: str,
    # 中文 L0494 “postprocessor_config_filename”是当前括号中的一项，逗号将它与下一项分开。
    postprocessor_config_filename: str,
# 中文 L0495 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> tuple[Any, Any] | None:
    # 中文 L0496 原文档概述本段功能，查找策略特有的预训练处理器创建入口。
    """Let a policy rebuild pretrained processors when its current runtime requires it."""
    # 中文 L0497 拼出策略特定的恢复函数名。普通的 make_smolvla_pre_post_processors 少了 from_pretrained 这个后缀。
    function_name = f"make_{config.type}_pre_post_processors_from_pretrained"
    # 中文 L0498 调用 _import_sibling_policy_module，用它的结果给 “module” 赋值。
    module = _import_sibling_policy_module(config.__class__, "processor")
    # 中文 L0499 条件 module is None 成立时进入下方分支。
    if module is None:
        # 中文 L0500 设为空值 None作为当前调用的返回值。
        return None
    # 中文 L0501 若找不到指定的恢复函数，下面将回退到 JSON 恢复路径。
    function = getattr(module, function_name, None)
    # 中文 L0502 条件 function is None 成立时进入下方分支。
    if function is None:
        # 中文 L0503 设为空值 None作为当前调用的返回值。
        return None
    # 中文 L0504 调用 function，用它的结果作为当前调用的返回值。
    return function(
        # 中文 L0505 为“config”（配置对象）提供这一项的值，注意它所在的参数表或外层表达式。
        config=config,
        # 中文 L0506 为“pretrained_path”提供这一项的值，注意它所在的参数表或外层表达式。
        pretrained_path=pretrained_path,
        # 中文 L0507 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
        revision=revision,
        # 中文 L0508 为“dataset_stats”提供这一项的值，注意它所在的参数表或外层表达式。
        dataset_stats=dataset_stats,
        # 中文 L0509 为“dataset_meta”提供这一项的值，注意它所在的参数表或外层表达式。
        dataset_meta=dataset_meta,
        # 中文 L0510 为“preprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
        preprocessor_overrides=preprocessor_overrides,
        # 中文 L0511 为“postprocessor_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
        postprocessor_overrides=postprocessor_overrides,
        # 中文 L0512 为“preprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
        preprocessor_config_filename=preprocessor_config_filename,
        # 中文 L0513 为“postprocessor_config_filename”提供这一项的值，注意它所在的参数表或外层表达式。
        postprocessor_config_filename=postprocessor_config_filename,
    # 中文 L0514 闭合上方尚未结束的括号。
    )


# 中文 L0517 定义函数 _make_processors_from_policy_config。
def _make_processors_from_policy_config(
    # 中文 L0518 “config”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    config: PreTrainedConfig,
    # 中文 L0519 为“dataset_stats”提供这一项的值，注意它所在的参数表或外层表达式。
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    # 中文 L0520 为“dataset_meta”提供这一项的值，注意它所在的参数表或外层表达式。
    dataset_meta: Any | None = None,
# 中文 L0521 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a policy configuration using dynamic imports.

    Resolves ``make_{type}_pre_post_processors`` from the policy's ``processor_*`` module
    by naming convention. Works for built-in policies and 3rd party lerobot plugins.

    Args:
        config: The policy configuration object.
        dataset_stats: Dataset statistics for normalization.
        dataset_meta: Dataset metadata, forwarded only to factories that declare a
            ``dataset_meta`` parameter (e.g. groot, molmoact2).
    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.
    """

    # 中文 L0536 读取属性 config.type给 “policy_type” 赋值。
    policy_type = config.type
    # 中文 L0537 计算此行的右侧表达式给 “function_name” 赋值。
    function_name = f"make_{policy_type}_pre_post_processors"
    # 中文 L0538 调用 logging.debug 执行操作。
    logging.debug(f"Instantiating pre/post processors using function '{function_name}'")
    # 中文 L0539 调用 _import_sibling_policy_module，用它的结果给 “module” 赋值。
    module = _import_sibling_policy_module(config.__class__, "processor")
    # 中文 L0540 条件 module is None 成立时进入下方分支。
    if module is None:
        # 中文 L0541 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Processor for policy type '{policy_type}' is not implemented.")
    # 中文 L0542 调用 getattr，用它的结果给 “function” 赋值。
    function = getattr(module, function_name, None)
    # 中文 L0543 条件 function is None 成立时进入下方分支。
    if function is None:
        # 中文 L0544 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Processor for policy type '{policy_type}' is not implemented.")
    # 中文 L0545 声明“call_kwargs”的类型为 dict[str, Any]，创建字典，用键查找对应的值作为初始值。
    call_kwargs: dict[str, Any] = {"dataset_stats": dataset_stats}
    # 中文 L0546 条件 "dataset_meta" in inspect.signature(function).parameters 成立时进入下方分支。
    if "dataset_meta" in inspect.signature(function).parameters:
        # 中文 L0547 引用已有的值 dataset_meta给 “call_kwargs['dataset_meta']” 赋值。
        call_kwargs["dataset_meta"] = dataset_meta
    # 中文 L0548 调用 function，用它的结果作为当前调用的返回值。
    return function(config, **call_kwargs)
