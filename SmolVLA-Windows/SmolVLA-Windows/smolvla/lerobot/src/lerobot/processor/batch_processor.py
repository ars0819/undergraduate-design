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
This script defines processor steps for adding a batch dimension to various components of an environment transition.

These steps are designed to process actions, observations, and complementary data, making them suitable for batch processing by adding a leading dimension. This is a common requirement before feeding data into a neural network model.
"""

# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0023 从模块 dataclasses 导入 dataclass、field。
from dataclasses import dataclass, field

# 中文 L0025 从模块 torch 导入 Tensor。
from torch import Tensor

# 中文 L0027 从模块 lerobot.configs 导入 PipelineFeatureType、PolicyFeature。
from lerobot.configs import PipelineFeatureType, PolicyFeature
# 中文 L0028 从模块 lerobot.lerobot_types 导入 EnvTransition、PolicyAction、TransitionKey。
from lerobot.lerobot_types import EnvTransition, PolicyAction, TransitionKey
# 中文 L0029 从模块 lerobot.utils.constants 导入 MESSAGES_RENDERED、OBS_ENV_STATE、OBS_IMAGE、OBS_IMAGES、OBS_STATE。
from lerobot.utils.constants import MESSAGES_RENDERED, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

# 中文 L0031 从模块 .pipeline 导入下方列出的多个名字。
from .pipeline import (
    # 中文 L0032 “ComplementaryDataProcessorStep”是当前括号中的一项，逗号将它与下一项分开。
    ComplementaryDataProcessorStep,
    # 中文 L0033 “ObservationProcessorStep”是当前括号中的一项，逗号将它与下一项分开。
    ObservationProcessorStep,
    # 中文 L0034 “PolicyActionProcessorStep”是当前括号中的一项，逗号将它与下一项分开。
    PolicyActionProcessorStep,
    # 中文 L0035 “ProcessorStep”是当前括号中的一项，逗号将它与下一项分开。
    ProcessorStep,
    # 中文 L0036 “ProcessorStepRegistry”是当前括号中的一项，逗号将它与下一项分开。
    ProcessorStepRegistry,
# 中文 L0037 闭合上方尚未结束的括号。
)


# 中文 L0040 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0041 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="to_batch_processor_action")
# 中文 L0042 定义类 AddBatchDimensionActionStep，括号列出基类 PolicyActionProcessorStep。
class AddBatchDimensionActionStep(PolicyActionProcessorStep):
    """
    Processor step to add a batch dimension to a 1D tensor action.

    This is useful for creating a batch of size 1 from a single action sample.
    """

    # 中文 L0049 定义方法 action。
    def action(self, action: PolicyAction) -> PolicyAction:
        """
        Adds a batch dimension to the action if it's a 1D tensor.

        Args:
            action: The action tensor.

        Returns:
            The action tensor with an added batch dimension.
        """
        # 中文 L0059 条件 action.dim() != 1 成立时进入下方分支。
        if action.dim() != 1:
            # 中文 L0060 引用已有的值 action作为当前调用的返回值。
            return action
        # 中文 L0061 调用 action.unsqueeze，用它的结果作为当前调用的返回值。
        return action.unsqueeze(0)

    # 中文 L0063 定义方法 transform_features。
    def transform_features(
        # 中文 L0064 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0065 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        # 中文 L0077 引用已有的值 features作为当前调用的返回值。
        return features


# 中文 L0080 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0081 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="to_batch_processor_observation")
# 中文 L0082 针对没有批次轴的观测增加前导批次轴。
class AddBatchDimensionObservationStep(ObservationProcessorStep):
    """
    Processor step to add a batch dimension to observations.

    It handles different types of observations:
    - State vectors (1D tensors).
    - Single images (3D tensors).
    - Dictionaries of multiple images (3D tensors).
    """

    # 中文 L0092 定义方法 observation。
    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Adds a batch dimension to tensor-based observations in the observation dictionary.

        Args:
            observation: The observation dictionary.

        Returns:
            The observation dictionary with batch dimensions added to tensors.
        """
        # Process state observations - add batch dim if 1D
        # 中文 L0103 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for state_key in [OBS_STATE, OBS_ENV_STATE]:
            # 中文 L0104 条件 state_key in observation 成立时进入下方分支。
            if state_key in observation:
                # 中文 L0105 用方括号按键或索引读取 observation给 “state_value” 赋值。
                state_value = observation[state_key]
                # 中文 L0106 条件 isinstance(state_value, Tensor) and state_value.dim() == 1 成立时进入下方分支。
                if isinstance(state_value, Tensor) and state_value.dim() == 1:
                    # 中文 L0107 对一维状态向量加批次轴，从 [S] 变成 [1,S]。
                    observation[state_key] = state_value.unsqueeze(0)

        # Process single image observation - add batch dim if 3D
        # 中文 L0110 条件 OBS_IMAGE in observation 成立时进入下方分支。
        if OBS_IMAGE in observation:
            # 中文 L0111 用方括号按键或索引读取 observation给 “image_value” 赋值。
            image_value = observation[OBS_IMAGE]
            # 中文 L0112 条件 isinstance(image_value, Tensor) and image_value.dim() == 3 成立时进入下方分支。
            if isinstance(image_value, Tensor) and image_value.dim() == 3:
                # 中文 L0113 对一张 CHW 图像加批次轴，从 [C,H,W] 变成 [1,C,H,W]。
                observation[OBS_IMAGE] = image_value.unsqueeze(0)

        # Process multiple image observations - add batch dim if 3D
        # 中文 L0116 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, value in observation.items():
            # 中文 L0117 条件 key.startswith(f"{OBS_IMAGES}.") and isinstance(value, Tensor) and value.dim() == 3 成立时进入下方分支。
            if key.startswith(f"{OBS_IMAGES}.") and isinstance(value, Tensor) and value.dim() == 3:
                # 中文 L0118 调用 value.unsqueeze，用它的结果给 “observation[key]” 赋值。
                observation[key] = value.unsqueeze(0)
        # 中文 L0119 引用已有的值 observation作为当前调用的返回值。
        return observation

    # 中文 L0121 定义方法 transform_features。
    def transform_features(
        # 中文 L0122 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0123 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        # 中文 L0135 引用已有的值 features作为当前调用的返回值。
        return features


# 中文 L0138 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0139 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="to_batch_processor_complementary_data")
# 中文 L0140 定义类 AddBatchDimensionComplementaryDataStep，括号列出基类 ComplementaryDataProcessorStep。
class AddBatchDimensionComplementaryDataStep(ComplementaryDataProcessorStep):
    """
    Processor step to add a batch dimension to complementary data fields.

    Handles specific keys like 'task', 'index', and 'task_index' to make them batched.
    - 'task' (str) is wrapped in a list.
    - 'index' and 'task_index' (0D tensors) get a batch dimension.
    """

    # 中文 L0149 定义方法 complementary_data。
    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Adds a batch dimension to specific fields in the complementary data dictionary.

        Args:
            complementary_data: The complementary data dictionary.

        Returns:
            The complementary data dictionary with batch dimensions added.
        """
        # Process task field - wrap string in list to add batch dimension
        # 中文 L0160 条件 "task" in complementary_data 成立时进入下方分支。
        if "task" in complementary_data:
            # 中文 L0161 用方括号按键或索引读取 complementary_data给 “task_value” 赋值。
            task_value = complementary_data["task"]
            # 中文 L0162 条件 isinstance(task_value, str) 成立时进入下方分支。
            if isinstance(task_value, str):
                # 中文 L0163 创建列表，按给定顺序保留元素给 “complementary_data['task']” 赋值。
                complementary_data["task"] = [task_value]

        # Process index field - add batch dim if 0D
        # 中文 L0166 条件 "index" in complementary_data 成立时进入下方分支。
        if "index" in complementary_data:
            # 中文 L0167 用方括号按键或索引读取 complementary_data给 “index_value” 赋值。
            index_value = complementary_data["index"]
            # 中文 L0168 条件 isinstance(index_value, Tensor) and index_value.dim() == 0 成立时进入下方分支。
            if isinstance(index_value, Tensor) and index_value.dim() == 0:
                # 中文 L0169 调用 index_value.unsqueeze，用它的结果给 “complementary_data['index']” 赋值。
                complementary_data["index"] = index_value.unsqueeze(0)

        # Process task_index field - add batch dim if 0D
        # 中文 L0172 条件 "task_index" in complementary_data 成立时进入下方分支。
        if "task_index" in complementary_data:
            # 中文 L0173 用方括号按键或索引读取 complementary_data给 “task_index_value” 赋值。
            task_index_value = complementary_data["task_index"]
            # 中文 L0174 条件 isinstance(task_index_value, Tensor) and task_index_value.dim() == 0 成立时进入下方分支。
            if isinstance(task_index_value, Tensor) and task_index_value.dim() == 0:
                # 中文 L0175 调用 task_index_value.unsqueeze，用它的结果给 “complementary_data['task_index']” 赋值。
                complementary_data["task_index"] = task_index_value.unsqueeze(0)

        # 中文 L0177 条件 MESSAGES_RENDERED in complementary_data 成立时进入下方分支。
        if MESSAGES_RENDERED in complementary_data:
            # 中文 L0178 用方括号按键或索引读取 complementary_data给 “messages” 赋值。
            messages = complementary_data[MESSAGES_RENDERED]
            # 中文 L0179 条件 isinstance(messages, list) and (not messages or isinstance(messages[0], dict)) 成立时进入下方分支。
            if isinstance(messages, list) and (not messages or isinstance(messages[0], dict)):
                # 中文 L0180 创建列表，按给定顺序保留元素给 “complementary_data[MESSAGES_RENDERED]” 赋值。
                complementary_data[MESSAGES_RENDERED] = [messages]

        # 中文 L0182 条件 "message_streams" in complementary_data 成立时进入下方分支。
        if "message_streams" in complementary_data:
            # 中文 L0183 用方括号按键或索引读取 complementary_data给 “streams” 赋值。
            streams = complementary_data["message_streams"]
            # 中文 L0184 条件 isinstance(streams, list) and (not streams or isinstance(streams[0], str)) 成立时进入下方分支。
            if isinstance(streams, list) and (not streams or isinstance(streams[0], str)):
                # 中文 L0185 创建列表，按给定顺序保留元素给 “complementary_data['message_streams']” 赋值。
                complementary_data["message_streams"] = [streams]

        # 中文 L0187 条件 "target_message_indices" in complementary_data 成立时进入下方分支。
        if "target_message_indices" in complementary_data:
            # 中文 L0188 用方括号按键或索引读取 complementary_data给 “indices” 赋值。
            indices = complementary_data["target_message_indices"]
            # 中文 L0189 条件 isinstance(indices, list) and (not indices or isinstance(indices[0], int)) 成立时进入下方分支。
            if isinstance(indices, list) and (not indices or isinstance(indices[0], int)):
                # 中文 L0190 创建列表，按给定顺序保留元素给 “complementary_data['target_message_indices']” 赋值。
                complementary_data["target_message_indices"] = [indices]
        # 中文 L0191 引用已有的值 complementary_data作为当前调用的返回值。
        return complementary_data

    # 中文 L0193 定义方法 transform_features。
    def transform_features(
        # 中文 L0194 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0195 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        # 中文 L0207 引用已有的值 features作为当前调用的返回值。
        return features


# 中文 L0210 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0211 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="to_batch_processor")
# 中文 L0212 把动作、观测和附加数据的批次处理整合成一个可调用步骤。
class AddBatchDimensionProcessorStep(ProcessorStep):
    """
    A composite processor step that adds a batch dimension to the entire environment transition.

    This step combines individual processors for actions, observations, and complementary data
    to create a batched transition (batch size 1) from a single-instance transition.

    Attributes:
        to_batch_action_processor: Processor for the action component.
        to_batch_observation_processor: Processor for the observation component.
        to_batch_complementary_data_processor: Processor for the complementary data component.
    """

    # 中文 L0225 声明“to_batch_action_processor”的类型为 AddBatchDimensionActionStep，调用 field，用它的结果作为初始值。
    to_batch_action_processor: AddBatchDimensionActionStep = field(
        # 中文 L0226 为“default_factory”提供这一项的值，注意它所在的参数表或外层表达式。
        default_factory=AddBatchDimensionActionStep
    # 中文 L0227 闭合上方尚未结束的括号。
    )
    # 中文 L0228 声明“to_batch_observation_processor”的类型为 AddBatchDimensionObservationStep，调用 field，用它的结果作为初始值。
    to_batch_observation_processor: AddBatchDimensionObservationStep = field(
        # 中文 L0229 为“default_factory”提供这一项的值，注意它所在的参数表或外层表达式。
        default_factory=AddBatchDimensionObservationStep
    # 中文 L0230 闭合上方尚未结束的括号。
    )
    # 中文 L0231 声明“to_batch_complementary_data_processor”的类型为 AddBatchDimensionComplementaryDataStep，调用 field，用它的结果作为初始值。
    to_batch_complementary_data_processor: AddBatchDimensionComplementaryDataStep = field(
        # 中文 L0232 为“default_factory”提供这一项的值，注意它所在的参数表或外层表达式。
        default_factory=AddBatchDimensionComplementaryDataStep
    # 中文 L0233 闭合上方尚未结束的括号。
    )

    # 中文 L0235 定义方法 __call__。
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Applies the batching process to all relevant parts of an environment transition.

        Args:
            transition: The environment transition to process.

        Returns:
            The environment transition with a batch dimension added.
        """
        # 中文 L0245 条件 transition[TransitionKey.ACTION] is not None 成立时进入下方分支。
        if transition[TransitionKey.ACTION] is not None:
            # 中文 L0246 调用 self.to_batch_action_processor，用它的结果给 “transition” 赋值。
            transition = self.to_batch_action_processor(transition)
        # 中文 L0247 条件 transition[TransitionKey.OBSERVATION] is not None 成立时进入下方分支。
        if transition[TransitionKey.OBSERVATION] is not None:
            # 中文 L0248 调用 self.to_batch_observation_processor，用它的结果给 “transition” 赋值。
            transition = self.to_batch_observation_processor(transition)
        # 中文 L0249 条件 transition[TransitionKey.COMPLEMENTARY_DATA] is not None 成立时进入下方分支。
        if transition[TransitionKey.COMPLEMENTARY_DATA] is not None:
            # 中文 L0250 调用 self.to_batch_complementary_data_processor，用它的结果给 “transition” 赋值。
            transition = self.to_batch_complementary_data_processor(transition)
        # 中文 L0251 引用已有的值 transition作为当前调用的返回值。
        return transition

    # 中文 L0253 定义方法 transform_features。
    def transform_features(
        # 中文 L0254 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0255 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        # NOTE: We ignore the batch dimension when transforming features
        # 中文 L0268 引用已有的值 features作为当前调用的返回值。
        return features
