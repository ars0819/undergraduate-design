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

"""
This script defines a processor for tokenizing natural language instructions from an environment transition.

It uses a tokenizer from the Hugging Face `transformers` library to convert task descriptions (text) into
token IDs and attention masks, which are then added to the observation dictionary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, RobotObservation, TransitionKey
from lerobot.utils.constants import (
    ACTION_CODE_TOKEN_MASK,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available

from .pipeline import ActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
else:
    AutoProcessor = None
    AutoTokenizer = None


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0056 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0057 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="tokenizer_processor")
# 中文 L0058 从任务字符串生成 token ID 和有效位，并作为模型的观测字段输出。
class TokenizerProcessorStep(ObservationProcessorStep):
    """
    Processor step to tokenize a natural language task description.

    This step extracts a task string from the `complementary_data` of an `EnvTransition`,
    tokenizes it using a Hugging Face `transformers` tokenizer, and adds the resulting
    token IDs and attention mask to the `observation` dictionary.

    Requires the `transformers` library to be installed.

    Attributes:
        tokenizer_name: The name of a pretrained tokenizer from the Hugging Face Hub (e.g., "bert-base-uncased").
        tokenizer: A pre-initialized tokenizer object. If provided, `tokenizer_name` is ignored.
        max_length: The maximum length to pad or truncate sequences to.
        task_key: The key in `complementary_data` where the task string is stored.
        padding_side: The side to pad on ('left' or 'right').
        padding: The padding strategy ('max_length', 'longest', etc.).
        truncation: Whether to truncate sequences longer than `max_length`.
        input_tokenizer: The internal tokenizer instance, loaded during initialization.
    """

    # 中文 L0079 声明“tokenizer_name”的类型为 str | None，设为空值 None作为初始值。
    tokenizer_name: str | None = None
    # 中文 L0080 声明“tokenizer”的类型为 Any | None，设为空值 None作为初始值。
    tokenizer: Any | None = None  # Use `Any` for compatibility without a hard dependency
    # 中文 L0081 声明“max_length”的类型为 int，使用字面值 512作为初始值。
    max_length: int = 512
    # 中文 L0082 声明“task_key”的类型为 str，使用字面值 'task'作为初始值。
    task_key: str = "task"
    # 中文 L0083 声明“padding_side”的类型为 str，使用字面值 'right'作为初始值。
    padding_side: str = "right"
    # 中文 L0084 声明“padding”的类型为 str，使用字面值 'max_length'作为初始值。
    padding: str = "max_length"
    # 中文 L0085 声明“truncation”的类型为 bool，使用布尔值 True作为初始值。
    truncation: bool = True

    # Internal tokenizer instance (not part of the config)
    # 中文 L0088 声明“input_tokenizer”的类型为 Any，调用 field，用它的结果作为初始值。
    input_tokenizer: Any = field(default=None, init=False, repr=False)

    # 中文 L0090 初始化数据类时选择已提供的 tokenizer 或按名字加载。
    def __post_init__(self):
        """
        Initializes the tokenizer after the dataclass is created.

        It checks for the availability of the `transformers` library and loads the tokenizer
        either from a provided object or by name from the Hugging Face Hub.

        Raises:
            ImportError: If the `transformers` library is not installed.
            ValueError: If neither `tokenizer` nor `tokenizer_name` is provided.
        """
        # 中文 L0101 条件 not _transformers_available 成立时进入下方分支。
        if not _transformers_available:
            # 中文 L0102 抛出 ImportError，中断当前路径并交给上层的异常处理。
            raise ImportError(
                # 中文 L0103 字符串字面量，供上方的调用、容器或错误消息使用。
                "The 'transformers' library is not installed. "
                # 中文 L0104 字符串字面量，供上方的调用、容器或错误消息使用。
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessorStep."
            # 中文 L0105 闭合上方尚未结束的括号。
            )

        # 中文 L0107 条件 self.tokenizer is not None 成立时进入下方分支。
        if self.tokenizer is not None:
            # Use provided tokenizer object directly
            # 中文 L0109 读取属性 self.tokenizer给 “self.input_tokenizer” 赋值。
            self.input_tokenizer = self.tokenizer
        # 中文 L0110 前面的分支均未命中时，再判断 self.tokenizer_name is not None。
        elif self.tokenizer_name is not None:
            # 中文 L0111 条件 AutoTokenizer is None 成立时进入下方分支。
            if AutoTokenizer is None:
                # 中文 L0112 抛出 ImportError，中断当前路径并交给上层的异常处理。
                raise ImportError("AutoTokenizer is not available")
            # 中文 L0113 调用 AutoTokenizer.from_pretrained，用它的结果给 “self.input_tokenizer” 赋值。
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # 中文 L0114 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0115 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0116 字符串字面量，供上方的调用、容器或错误消息使用。
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                # 中文 L0117 字符串字面量，供上方的调用、容器或错误消息使用。
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            # 中文 L0118 闭合上方尚未结束的括号。
            )

    # 中文 L0120 定义方法 get_task。
    def get_task(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the task description(s) from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of task strings, or None if the task key is not found or the value is None.
        """
        # 中文 L0130 调用 transition.get，用它的结果给 “complementary_data” 赋值。
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        # 中文 L0131 条件 complementary_data is None 成立时进入下方分支。
        if complementary_data is None:
            # 中文 L0132 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("Complementary data is None so no task can be extracted from it")

        # 中文 L0134 用方括号按键或索引读取 complementary_data给 “task”（任务） 赋值。
        task = complementary_data[self.task_key]
        # 中文 L0135 条件 task is None 成立时进入下方分支。
        if task is None:
            # 中文 L0136 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("Task extracted from Complementary data is None")

        # Standardize to a list of strings for the tokenizer
        # 中文 L0139 条件 isinstance(task, str) 成立时进入下方分支。
        if isinstance(task, str):
            # 中文 L0140 创建列表，按给定顺序保留元素作为当前调用的返回值。
            return [task]
        # 中文 L0141 前面的分支均未命中时，再判断 isinstance(task, list | tuple) and all(isinstance(t, str) for t in task)。
        elif isinstance(task, list | tuple) and all(isinstance(t, str) for t in task):
            # 中文 L0142 调用 list，用它的结果作为当前调用的返回值。
            return list(task)

        # 中文 L0144 设为空值 None作为当前调用的返回值。
        return None

    # 中文 L0146 定义方法 get_subtask。
    def get_subtask(self, transition: EnvTransition) -> list[str] | None:
        """
        Extracts the subtask from the transition's complementary data.

        Args:
            transition: The environment transition.

        Returns:
            A list of subtask strings, or None if the subtask key is not found or the value is None.
        """
        # 中文 L0156 调用 transition.get，用它的结果给 “complementary_data” 赋值。
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        # 中文 L0157 条件 complementary_data is None 成立时进入下方分支。
        if complementary_data is None:
            # 中文 L0158 设为空值 None作为当前调用的返回值。
            return None

        # 中文 L0160 调用 complementary_data.get，用它的结果给 “subtask” 赋值。
        subtask = complementary_data.get("subtask")
        # 中文 L0161 条件 subtask is None 成立时进入下方分支。
        if subtask is None:
            # 中文 L0162 设为空值 None作为当前调用的返回值。
            return None

        # Standardize to a list of strings for the tokenizer
        # 中文 L0165 条件 isinstance(subtask, str) 成立时进入下方分支。
        if isinstance(subtask, str):
            # 中文 L0166 创建列表，按给定顺序保留元素作为当前调用的返回值。
            return [subtask]
        # 中文 L0167 前面的分支均未命中时，再判断 isinstance(subtask, list) and all(isinstance(t, str) for t in subtask)。
        elif isinstance(subtask, list) and all(isinstance(t, str) for t in subtask):
            # 中文 L0168 引用已有的值 subtask作为当前调用的返回值。
            return subtask

        # 中文 L0170 设为空值 None作为当前调用的返回值。
        return None

    # 中文 L0172 从当前 transition 找任务文字并将编码结果添到观测。
    def observation(self, observation: RobotObservation) -> RobotObservation:
        """
        Tokenizes the task description and adds it to the observation dictionary.

        This method retrieves the task, tokenizes it, moves the resulting tensors to the
        same device as other data in the transition, and updates the observation.

        Args:
            observation: The original observation dictionary.

        Returns:
            The updated observation dictionary including token IDs and an attention mask.
        """
        # 中文 L0185 调用 self.get_task，用它的结果给 “task”（任务） 赋值。
        task = self.get_task(self.transition)
        # 中文 L0186 条件 task is None 成立时进入下方分支。
        if task is None:
            # 中文 L0187 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("Task cannot be None")

        # Tokenize the task (this will create CPU tensors)
        # 中文 L0190 调用 self._tokenize_text，用它的结果给 “tokenized_prompt” 赋值。
        tokenized_prompt = self._tokenize_text(task)

        # Detect the device from existing tensors in the transition to ensure consistency
        # 中文 L0193 调用 self._detect_device，用它的结果给 “target_device” 赋值。
        target_device = self._detect_device(self.transition)

        # Move new tokenized tensors to the detected device
        # 中文 L0196 条件 target_device is not None 成立时进入下方分支。
        if target_device is not None:
            # 中文 L0197 用字典推导式从可迭代对象逐项取值并构造结果给 “tokenized_prompt” 赋值。
            tokenized_prompt = {
                # 中文 L0198 延续上面的语句，k: v.to(target_device) if isinstance(v, torch.Tensor) else v。
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                # 中文 L0199 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for k, v in tokenized_prompt.items()
            # 中文 L0200 闭合上方尚未结束的括号。
            }

        # Create a new observation dict to avoid modifying the original in place
        # 中文 L0203 调用 dict，用它的结果给 “new_observation” 赋值。
        new_observation = dict(observation)

        # Add tokenized data to the observation
        # 中文 L0206 保存 token ID。每个整数是词表中的索引，形状为 [B,L]。
        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        # 中文 L0207 把 tokenizer 的有效位转成布尔型 Tensor，保持与文字 token 一一对应。
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        # Tokenize subtask if available
        # 中文 L0210 调用 self.get_subtask，用它的结果给 “subtask” 赋值。
        subtask = self.get_subtask(self.transition)
        # 中文 L0211 条件 subtask is not None 成立时进入下方分支。
        if subtask is not None:
            # 中文 L0212 调用 self._tokenize_text，用它的结果给 “tokenized_subtask” 赋值。
            tokenized_subtask = self._tokenize_text(subtask)

            # Move new tokenized tensors to the detected device
            # 中文 L0215 条件 target_device is not None 成立时进入下方分支。
            if target_device is not None:
                # 中文 L0216 用字典推导式从可迭代对象逐项取值并构造结果给 “tokenized_subtask” 赋值。
                tokenized_subtask = {
                    # 中文 L0217 延续上面的语句，k: v.to(target_device) if isinstance(v, torch.Tensor) else v。
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    # 中文 L0218 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                    for k, v in tokenized_subtask.items()
                # 中文 L0219 闭合上方尚未结束的括号。
                }

            # Add tokenized subtask to the observation
            # 中文 L0222 用方括号按键或索引读取 tokenized_subtask给 “new_observation[OBS_LANGUAGE_SUBTASK_TOKENS]” 赋值。
            new_observation[OBS_LANGUAGE_SUBTASK_TOKENS] = tokenized_subtask["input_ids"]
            # 中文 L0223 调用 to，用它的结果给 “new_observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]” 赋值。
            new_observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = tokenized_subtask["attention_mask"].to(
                # 中文 L0224 为“dtype”（元素类型）提供这一项的值，注意它所在的参数表或外层表达式。
                dtype=torch.bool
            # 中文 L0225 闭合上方尚未结束的括号。
            )

        # 中文 L0227 引用已有的值 new_observation作为当前调用的返回值。
        return new_observation

    # 中文 L0229 定义方法 _detect_device。
    def _detect_device(self, transition: EnvTransition) -> torch.device | None:
        """
        Detects the torch.device from existing tensors in the transition.

        It checks tensors in the observation dictionary first, then the action tensor.

        Args:
            transition: The environment transition.

        Returns:
            The detected `torch.device`, or None if no tensors are found.
        """
        # Check observation tensors first (most likely place to find tensors)
        # 中文 L0242 调用 transition.get，用它的结果给 “observation”（观测） 赋值。
        observation = transition.get(TransitionKey.OBSERVATION)
        # 中文 L0243 条件 observation 成立时进入下方分支。
        if observation:
            # 中文 L0244 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for value in observation.values():
                # 中文 L0245 条件 isinstance(value, torch.Tensor) 成立时进入下方分支。
                if isinstance(value, torch.Tensor):
                    # 中文 L0246 读取属性 value.device作为当前调用的返回值。
                    return value.device

        # Fallback to checking the action tensor
        # 中文 L0249 调用 transition.get，用它的结果给 “action”（单步动作） 赋值。
        action = transition.get(TransitionKey.ACTION)
        # 中文 L0250 条件 isinstance(action, torch.Tensor) 成立时进入下方分支。
        if isinstance(action, torch.Tensor):
            # 中文 L0251 读取属性 action.device作为当前调用的返回值。
            return action.device

        # 中文 L0253 设为空值 None作为当前调用的返回值。
        return None  # No tensors found, default will be CPU

    # 中文 L0255 将 Python 字符串或字符串列表交给 tokenizer，返回 PyTorch Tensor。
    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        """
        A wrapper around the tokenizer call.

        Args:
            text: A string or list of strings to tokenize.

        Returns:
            A dictionary containing tokenized 'input_ids' and 'attention_mask' as PyTorch tensors.
        """
        # 中文 L0265 调用实际 tokenizer 对任务文字编码，同时应用补齐和截断设置。
        return self.input_tokenizer(
            # 中文 L0266 “text”是当前括号中的一项，逗号将它与下一项分开。
            text,
            # 中文 L0267 为“max_length”提供这一项的值，注意它所在的参数表或外层表达式。
            max_length=self.max_length,
            # 中文 L0268 为“truncation”提供这一项的值，注意它所在的参数表或外层表达式。
            truncation=self.truncation,
            # 中文 L0269 为“padding”提供这一项的值，注意它所在的参数表或外层表达式。
            padding=self.padding,
            # 中文 L0270 为“padding_side”提供这一项的值，注意它所在的参数表或外层表达式。
            padding_side=self.padding_side,
            # 中文 L0271 为“return_tensors”提供这一项的值，注意它所在的参数表或外层表达式。
            return_tensors="pt",
        # 中文 L0272 闭合上方尚未结束的括号。
        )

    # 中文 L0274 定义方法 get_config。
    def get_config(self) -> dict[str, Any]:
        """
        Returns the serializable configuration of the processor.

        Note: The tokenizer object itself is not serialized. If the processor was initialized
        with a tokenizer name, that name will be included in the config.

        Returns:
            A dictionary with the processor's configuration parameters.
        """
        # 中文 L0284 创建字典，用键查找对应的值给 “config”（配置对象） 赋值。
        config = {
            # 中文 L0285 字典项“max_length”，冒号右边给出它的值。
            "max_length": self.max_length,
            # 中文 L0286 字典项“task_key”，冒号右边给出它的值。
            "task_key": self.task_key,
            # 中文 L0287 字典项“padding_side”，冒号右边给出它的值。
            "padding_side": self.padding_side,
            # 中文 L0288 字典项“padding”，冒号右边给出它的值。
            "padding": self.padding,
            # 中文 L0289 字典项“truncation”，冒号右边给出它的值。
            "truncation": self.truncation,
        # 中文 L0290 闭合上方尚未结束的括号。
        }

        # Only save tokenizer_name if it was used to create the tokenizer
        # 中文 L0293 条件 self.tokenizer_name is not None and self.tokenizer is None 成立时进入下方分支。
        if self.tokenizer_name is not None and self.tokenizer is None:
            # 中文 L0294 读取属性 self.tokenizer_name给 “config['tokenizer_name']” 赋值。
            config["tokenizer_name"] = self.tokenizer_name

        # 中文 L0296 引用已有的值 config作为当前调用的返回值。
        return config

    # 中文 L0298 定义方法 save_artifacts。
    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        # 中文 L0299 原文档中的说明或续行。
        """Save the tokenizer so object-provided instances reload without overrides."""
        # 中文 L0300 调用 Path，用它的结果给 “artifact_path” 赋值。
        artifact_path = Path("tokenizer")
        # 中文 L0301 调用 getattr，用它的结果给 “save_pretrained” 赋值。
        save_pretrained = getattr(self.input_tokenizer, "save_pretrained", None)
        # 中文 L0302 条件 save_pretrained is None 成立时进入下方分支。
        if save_pretrained is None:
            # 中文 L0303 抛出 TypeError，中断当前路径并交给上层的异常处理。
            raise TypeError("Tokenizer must implement save_pretrained() to save a portable pipeline.")
        # 中文 L0304 调用 save_pretrained 执行操作。
        save_pretrained(save_directory / artifact_path)
        # 中文 L0305 创建字典，用键查找对应的值作为当前调用的返回值。
        return {"tokenizer_name": artifact_path.as_posix()}

    # 中文 L0307 定义方法 transform_features。
    def transform_features(
        # 中文 L0308 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0309 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds feature definitions for the language tokens and attention mask.

        This updates the policy features dictionary to include the new data added to the
        observation, ensuring downstream components are aware of their shape and type.

        Args:
            features: The dictionary of existing policy features.

        Returns:
            The updated dictionary of policy features.
        """
        # Add a feature for the token IDs if it doesn't already exist
        # 中文 L0323 条件 OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION] 成立时进入下方分支。
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            # 中文 L0324 调用 PolicyFeature，用它的结果给 “features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS]” 赋值。
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                # 中文 L0325 为“type”提供这一项的值，注意它所在的参数表或外层表达式。
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            # 中文 L0326 闭合上方尚未结束的括号。
            )

        # Add a feature for the attention mask if it doesn't already exist
        # 中文 L0329 条件 OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION] 成立时进入下方分支。
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            # 中文 L0330 调用 PolicyFeature，用它的结果给 “features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION…” 赋值。
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                # 中文 L0331 为“type”提供这一项的值，注意它所在的参数表或外层表达式。
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            # 中文 L0332 闭合上方尚未结束的括号。
            )

        # 中文 L0334 引用已有的值 features作为当前调用的返回值。
        return features


@dataclass
@ProcessorStepRegistry.register(name="action_tokenizer_processor")
class ActionTokenizerProcessorStep(ActionProcessorStep):
    """
    Processor step to tokenize action data using a fast action tokenizer.

    This step takes action tensors from an `EnvTransition`, tokenizes them using
    a Hugging Face `transformers` AutoProcessor (such as the Physical Intelligence "fast" tokenizer),
    and returns the tokenized action.

    Requires the `transformers` library to be installed.

    Attributes:
        tokenizer_name: The name of a pretrained processor from the Hugging Face Hub (e.g., "lerobot/fast-action-tokenizer").
        tokenizer: A pre-initialized processor/tokenizer object. If provided, `tokenizer_name` is ignored.
        trust_remote_code: Whether to trust remote code when loading the tokenizer (required for some tokenizers).
        action_tokenizer: The internal tokenizer/processor instance, loaded during initialization.
        paligemma_tokenizer_name: The name of a pretrained PaliGemma tokenizer from the Hugging Face Hub (e.g., "google/paligemma-3b-pt-224").
    """

    action_tokenizer_name: str | None = None
    action_tokenizer_input_object: Any | None = None
    trust_remote_code: bool = True
    max_action_tokens: int = 256
    fast_skip_tokens: int = 128
    paligemma_tokenizer_name: str = "google/paligemma-3b-pt-224"
    allow_truncation: bool = True
    # Internal tokenizer instance (not part of the config)
    action_tokenizer: Any = field(default=None, init=False, repr=False)
    _paligemma_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the action tokenizer after the dataclass is created.

        It checks for the availability of the `transformers` library and loads the tokenizer
        either from a provided object or by name from the Hugging Face Hub.

        Raises:
            ImportError: If the `transformers` library is not installed.
            ValueError: If neither `tokenizer` nor `tokenizer_name` is provided.
        """
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use ActionTokenizerProcessorStep."
            )

        if self.action_tokenizer_input_object is not None:
            self.action_tokenizer = self.action_tokenizer_input_object

        elif self.action_tokenizer_name is not None:
            if AutoProcessor is None:
                raise ImportError("AutoProcessor is not available")
            self.action_tokenizer = AutoProcessor.from_pretrained(
                self.action_tokenizer_name, trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(
                "Either 'action_tokenizer' or 'action_tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

        self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
            self.paligemma_tokenizer_name,
            trust_remote_code=self.trust_remote_code,
            add_eos_token=True,
            add_bos_token=False,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Applies action tokenization to the transition.

        This overrides the base class to handle both tokens and mask.

        Args:
            transition: The input transition with action data.

        Returns:
            The processed transition with tokenized actions and mask in complementary data.
        """
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            # During inference, no action is available, skip tokenization
            return new_transition

        # Tokenize and get masks for the full formatted sequence and the discrete action codes.
        tokens, mask, code_mask = self._tokenize_action(action)

        # Store mask in complementary data
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if complementary_data is None:
            complementary_data = {}
        complementary_data[ACTION_TOKEN_MASK] = mask
        complementary_data[ACTION_CODE_TOKEN_MASK] = code_mask
        complementary_data[ACTION_TOKENS] = tokens
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Converts action tokens to PaliGemma tokens.
        """
        return self._paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes the action tensor and creates a mask.

        Args:
            action: The input action tensor to tokenize. Shape: (B, H, action_dim) or (H, action_dim,)

        Returns:
            A tuple of (tokens, mask) where:
            - tokens: Tensor of token IDs with shape (B, max_action_tokens)
            - mask: Boolean mask with shape (B, max_action_tokens), True for real tokens, False for padding
        """
        if action is None:
            raise ValueError("Action cannot be None")

        # Get the device and dtype of the input action
        device = action.device if isinstance(action, torch.Tensor) else None

        # Handle single sample (add batch dimension)
        single_sample = action.dim() == 1
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]

        # Tokenize the action batch
        # The fast tokenizer expects action data and returns token IDs
        tokens_list = []
        masks_list = []
        code_masks_list = []

        for i in range(batch_size):
            # Tokenize single action (move to CPU first as tokenizer uses scipy which requires numpy)
            action_cpu = action[i : i + 1].cpu()
            tokens = self.action_tokenizer(action_cpu)

            # Convert to numpy array if it's a list
            if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=action.device)
            else:
                # Move tokens back to the same device as input action
                tokens = tokens.to(device=action.device)

            # Flatten to 1D if needed
            if tokens.dim() > 1:
                tokens = tokens.flatten()

            action_code_tokens = self._act_tokens_to_paligemma_tokens(tokens)
            bos_id = self._paligemma_tokenizer.bos_token_id
            prompt_tokens = torch.tensor(
                self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                device=action.device,
            )
            end_tokens = torch.tensor(self._paligemma_tokenizer.encode("|"), device=action.device)

            code_start = 1 + len(prompt_tokens)
            code_end = code_start + len(action_code_tokens)
            tokens = torch.cat(
                [
                    torch.tensor([bos_id], device=action.device),
                    prompt_tokens,
                    action_code_tokens,
                    end_tokens,
                ]
            )
            code_mask = torch.zeros(len(tokens), dtype=torch.bool, device=action.device)
            code_mask[code_start:code_end] = True

            # Truncate or pad to max_action_tokens
            if len(tokens) > self.max_action_tokens:
                if not self.allow_truncation:
                    raise ValueError(
                        f"FAST action sequence has {len(tokens)} tokens, exceeding "
                        f"max_action_tokens={self.max_action_tokens}."
                    )
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self.max_action_tokens}), truncating. "
                    "Consider increasing the `max_action_tokens` in your model config if this happens frequently."
                )
                tokens = tokens[: self.max_action_tokens]
                code_mask = code_mask[: self.max_action_tokens]
                mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=action.device)
            else:
                pad_len = self.max_action_tokens - len(tokens)
                mask = torch.cat(
                    [
                        torch.ones(len(tokens), dtype=torch.bool, device=action.device),
                        torch.zeros(pad_len, dtype=torch.bool, device=action.device),
                    ]
                )
                code_mask = torch.nn.functional.pad(code_mask, (0, pad_len), value=False)
                # Pad tokens with zeros
                tokens = torch.nn.functional.pad(tokens, (0, pad_len), value=0)

            tokens_list.append(tokens)
            masks_list.append(mask)
            code_masks_list.append(code_mask)

        # Stack into batched tensors
        tokens_batch = torch.stack(tokens_list, dim=0)  # (B, max_action_tokens)
        masks_batch = torch.stack(masks_list, dim=0)  # (B, max_action_tokens)
        code_masks_batch = torch.stack(code_masks_list, dim=0)  # (B, max_action_tokens)

        # Remove batch dimension if input was single sample
        if single_sample:
            tokens_batch = tokens_batch.squeeze(0)
            masks_batch = masks_batch.squeeze(0)
            code_masks_batch = code_masks_batch.squeeze(0)

        # Move to the same device as the input
        if device is not None:
            tokens_batch = tokens_batch.to(device)
            masks_batch = masks_batch.to(device)
            code_masks_batch = code_masks_batch.to(device)

        return tokens_batch, masks_batch, code_masks_batch

    def action(self, action: torch.Tensor) -> torch.Tensor:
        """
        This method is not used since we override __call__.
        Required by ActionProcessorStep ABC.
        """
        tokens, _, _ = self._tokenize_action(action)
        return tokens

    def get_config(self) -> dict[str, Any]:
        """
        Returns the serializable configuration of the processor.

        Note: The tokenizer object itself is not serialized. If the processor was initialized
        with a tokenizer name, that name will be included in the config.

        Returns:
            A dictionary with the processor's configuration parameters.
        """
        config = {
            "trust_remote_code": self.trust_remote_code,
            "max_action_tokens": self.max_action_tokens,
            "fast_skip_tokens": self.fast_skip_tokens,
            "paligemma_tokenizer_name": self.paligemma_tokenizer_name,
            "allow_truncation": self.allow_truncation,
        }

        # Only save tokenizer_name if it was used to create the tokenizer
        if self.action_tokenizer_name is not None and self.action_tokenizer_input_object is None:
            config["action_tokenizer_name"] = self.action_tokenizer_name

        return config

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        artifact_path = Path("action_tokenizer")
        save_pretrained = getattr(self.action_tokenizer, "save_pretrained", None)
        if save_pretrained is None:
            raise TypeError("Action tokenizer must implement save_pretrained() to save a portable pipeline.")
        save_pretrained(save_directory / artifact_path)
        return {"action_tokenizer_name": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates feature definitions to reflect tokenized actions.

        This updates the policy features dictionary to indicate that the action
        has been tokenized into a sequence of token IDs with shape (max_action_tokens,).

        Args:
            features: The dictionary of existing policy features.

        Returns:
            The updated dictionary of policy features.
        """
        return features
