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
This script defines a processor step for moving environment transition data to a specific torch device and casting
its floating-point precision.
"""

# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0022 从模块 dataclasses 导入 dataclass。
from dataclasses import dataclass
# 中文 L0023 从模块 typing 导入 Any。
from typing import Any

# 中文 L0025 导入模块 torch。
import torch

# 中文 L0027 从模块 lerobot.configs 导入 PipelineFeatureType、PolicyFeature。
from lerobot.configs import PipelineFeatureType, PolicyFeature
# 中文 L0028 从模块 lerobot.lerobot_types 导入 EnvTransition、TransitionKey。
from lerobot.lerobot_types import EnvTransition, TransitionKey
# 中文 L0029 从模块 lerobot.utils.device_utils 导入 get_safe_torch_device。
from lerobot.utils.device_utils import get_safe_torch_device

# 中文 L0031 从模块 .pipeline 导入 ProcessorStep、ProcessorStepRegistry。
from .pipeline import ProcessorStep, ProcessorStepRegistry


# 中文 L0034 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register("device_processor")
# 中文 L0035 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0036 根据阶段决定 Tensor 目的设备或浮点 dtype。策略前处理使用 CUDA，动作后处理用 CPU。
class DeviceProcessorStep(ProcessorStep):
    """
    Processor step to move all tensors within an `EnvTransition` to a specified device and optionally cast their
    floating-point data type.

    This is crucial for preparing data for model training or inference on hardware like GPUs.

    Attributes:
        device: The target device for tensors (e.g., "cpu", "cuda", "cuda:0").
        float_dtype: The target floating-point dtype as a string (e.g., "float32", "float16", "bfloat16").
                     If None, the dtype is not changed.
    """

    # 中文 L0049 声明“device”（运算设备）的类型为 str，使用字面值 'cpu'作为初始值。
    device: str = "cpu"
    # 中文 L0050 声明“float_dtype”的类型为 str | None，设为空值 None作为初始值。
    float_dtype: str | None = None

    # 中文 L0052 创建字典，用键查找对应的值给 “DTYPE_MAPPING” 赋值。
    DTYPE_MAPPING = {
        # 中文 L0053 字典项“float16”，冒号右边给出它的值。
        "float16": torch.float16,
        # 中文 L0054 字典项“float32”，冒号右边给出它的值。
        "float32": torch.float32,
        # 中文 L0055 字典项“float64”，冒号右边给出它的值。
        "float64": torch.float64,
        # 中文 L0056 字典项“bfloat16”，冒号右边给出它的值。
        "bfloat16": torch.bfloat16,
        # 中文 L0057 字典项“half”，冒号右边给出它的值。
        "half": torch.float16,
        # 中文 L0058 字典项“float”，冒号右边给出它的值。
        "float": torch.float32,
        # 中文 L0059 字典项“double”，冒号右边给出它的值。
        "double": torch.float64,
    # 中文 L0060 闭合上方尚未结束的括号。
    }

    # 中文 L0062 定义方法 __post_init__。
    def __post_init__(self) -> None:
        """
        Initializes the processor by converting string configurations to torch objects.

        This method sets up the `torch.device`, determines if transfers can be non-blocking, and validates the
        `float_dtype` string, converting it to a `torch.dtype` object.
        """
        # 中文 L0069 把配置里的设备名称转换为可用的 torch.device。
        self.tensor_device: torch.device = get_safe_torch_device(self.device)
        # Preserve the resolved device index when serializing the processor configuration.
        # 中文 L0071 调用 str，用它的结果给 “self.device” 赋值。
        self.device = str(self.tensor_device)
        # 中文 L0072 计算此行的右侧表达式给 “self.non_blocking” 赋值。
        self.non_blocking = "cuda" in str(self.device)

        # Validate and convert float_dtype string to torch dtype
        # 中文 L0075 条件 self.float_dtype is not None 成立时进入下方分支。
        if self.float_dtype is not None:
            # 中文 L0076 条件 self.float_dtype not in self.DTYPE_MAPPING 成立时进入下方分支。
            if self.float_dtype not in self.DTYPE_MAPPING:
                # 中文 L0077 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0078 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"Invalid float_dtype '{self.float_dtype}'. Available options: {list(self.DTYPE_MAPPING.keys())}"
                # 中文 L0079 闭合上方尚未结束的括号。
                )
            # 中文 L0080 用方括号按键或索引读取 self.DTYPE_MAPPING给 “self._target_float_dtype” 赋值。
            self._target_float_dtype = self.DTYPE_MAPPING[self.float_dtype]
        # 中文 L0081 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0082 设为空值 None给 “self._target_float_dtype” 赋值。
            self._target_float_dtype = None

    # 中文 L0084 根据输入和设置决定单个 Tensor 的类型及设备转换。
    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Moves a single tensor to the target device and casts its dtype.

        Handles multi-GPU scenarios by not moving a tensor if it's already on a different CUDA device than
        the target, which is useful when using frameworks like Accelerate.

        Args:
            tensor: The input torch.Tensor.

        Returns:
            The processed tensor on the correct device and with the correct dtype.
        """
        # Determine target device
        # 中文 L0098 条件 tensor.is_cuda and self.tensor_device.type == "cuda" 成立时进入下方分支。
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            # Both tensor and target are on GPU - preserve tensor's GPU placement.
            # This handles multi-GPU scenarios where Accelerate has already placed
            # tensors on the correct GPU for each process.
            # 中文 L0102 读取属性 tensor.device给 “target_device” 赋值。
            target_device = tensor.device
        # 中文 L0103 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # Either tensor is on CPU, or we're configured for CPU.
            # In both cases, use the configured device.
            # 中文 L0106 读取属性 self.tensor_device给 “target_device” 赋值。
            target_device = self.tensor_device

        # MPS workaround: Convert float64 to float32 since MPS doesn't support float64
        # 中文 L0109 条件 target_device.type == "mps" and tensor.dtype == torch.float64 成立时进入下方分支。
        if target_device.type == "mps" and tensor.dtype == torch.float64:
            # 中文 L0110 调用 tensor.to，用它的结果给 “tensor” 赋值。
            tensor = tensor.to(dtype=torch.float32)

        # Only move if necessary
        # 中文 L0113 条件 tensor.device != target_device 成立时进入下方分支。
        if tensor.device != target_device:
            # 中文 L0114 调用 Tensor.to 把 Tensor 移到目标设备。形状不变。
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)

        # Convert float dtype if specified and tensor is floating point
        # 中文 L0117 条件 self._target_float_dtype is not None and tensor.is_floating_point() 成立时进入下方分支。
        if self._target_float_dtype is not None and tensor.is_floating_point():
            # 中文 L0118 配置指定目标浮点类型时转换 dtype。
            tensor = tensor.to(dtype=self._target_float_dtype)

        # 中文 L0120 引用已有的值 tensor作为当前调用的返回值。
        return tensor

    # 中文 L0122 按 transition 内各区域逐个处理需要设备转换的张量。
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Applies device and dtype conversion to all tensors in an environment transition.

        It iterates through the transition, finds all `torch.Tensor` objects (including those nested in
        dictionaries like `observation`), and processes them.

        Args:
            transition: The input `EnvTransition` object.

        Returns:
            A new `EnvTransition` object with all tensors moved to the target device and dtype.
        """
        # 中文 L0135 调用 transition.copy，用它的结果给 “new_transition” 赋值。
        new_transition = transition.copy()
        # 中文 L0136 调用 new_transition.get，用它的结果给 “action”（单步动作） 赋值。
        action = new_transition.get(TransitionKey.ACTION)

        # 中文 L0138 条件 action is not None and not isinstance(action, torch.Tensor) 成立时进入下方分支。
        if action is not None and not isinstance(action, torch.Tensor):
            # 中文 L0139 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"If action is not None should be a PolicyAction type got {type(action)}")

        # 中文 L0141 打包成元组，元素顺序由逗号分隔的位置决定给 “simple_tensor_keys” 赋值。
        simple_tensor_keys = (
            # 中文 L0142 延续上面的语句，TransitionKey.ACTION,。
            TransitionKey.ACTION,
            # 中文 L0143 延续上面的语句，TransitionKey.REWARD,。
            TransitionKey.REWARD,
            # 中文 L0144 延续上面的语句，TransitionKey.DONE,。
            TransitionKey.DONE,
            # 中文 L0145 延续上面的语句，TransitionKey.TRUNCATED,。
            TransitionKey.TRUNCATED,
        # 中文 L0146 闭合上方尚未结束的括号。
        )

        # 中文 L0148 打包成元组，元素顺序由逗号分隔的位置决定给 “dict_tensor_keys” 赋值。
        dict_tensor_keys = (
            # 中文 L0149 延续上面的语句，TransitionKey.OBSERVATION,。
            TransitionKey.OBSERVATION,
            # 中文 L0150 延续上面的语句，TransitionKey.COMPLEMENTARY_DATA,。
            TransitionKey.COMPLEMENTARY_DATA,
        # 中文 L0151 闭合上方尚未结束的括号。
        )

        # Process simple, top-level tensors
        # 中文 L0154 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for tensor_key in simple_tensor_keys:
            # 中文 L0155 调用 transition.get，用它的结果给 “value” 赋值。
            value = transition.get(tensor_key)
            # 中文 L0156 条件 isinstance(value, torch.Tensor) 成立时进入下方分支。
            if isinstance(value, torch.Tensor):
                # 中文 L0157 调用 self._process_tensor，用它的结果给 “new_transition[tensor_key]” 赋值。
                new_transition[tensor_key] = self._process_tensor(value)

        # Process tensors nested within dictionaries
        # 中文 L0160 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for dict_key in dict_tensor_keys:
            # 中文 L0161 调用 transition.get，用它的结果给 “data_dict” 赋值。
            data_dict = transition.get(dict_key)
            # 中文 L0162 条件 data_dict is not None 成立时进入下方分支。
            if data_dict is not None:
                # 中文 L0163 用字典推导式从可迭代对象逐项取值并构造结果给 “new_data_dict” 赋值。
                new_data_dict = {
                    # 中文 L0164 延续上面的语句，k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v。
                    k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
                    # 中文 L0165 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                    for k, v in data_dict.items()
                # 中文 L0166 闭合上方尚未结束的括号。
                }
                # 中文 L0167 引用已有的值 new_data_dict给 “new_transition[dict_key]” 赋值。
                new_transition[dict_key] = new_data_dict

        # 中文 L0169 引用已有的值 new_transition作为当前调用的返回值。
        return new_transition

    # 中文 L0171 定义方法 get_config。
    def get_config(self) -> dict[str, Any]:
        """
        Returns the serializable configuration of the processor.

        Returns:
            A dictionary containing the device and float_dtype settings.
        """
        # 中文 L0178 创建字典，用键查找对应的值作为当前调用的返回值。
        return {"device": self.device, "float_dtype": self.float_dtype}

    # 中文 L0180 定义方法 transform_features。
    def transform_features(
        # 中文 L0181 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0182 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Device and dtype transformations do not alter the fundamental definition of the features (e.g., shape).

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        # 中文 L0194 引用已有的值 features作为当前调用的返回值。
        return features
