#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
# 中文 L0017 从模块 typing 导入 Any。
from typing import Any

# 中文 L0019 导入模块 torch。
import torch

# 中文 L0021 从模块 lerobot.processor 导入下方列出的多个名字。
from lerobot.processor import (
    # 中文 L0022 “NewLineTaskProcessorStep”是当前括号中的一项，逗号将它与下一项分开。
    NewLineTaskProcessorStep,
    # 中文 L0023 “PolicyAction”是当前括号中的一项，逗号将它与下一项分开。
    PolicyAction,
    # 中文 L0024 “PolicyProcessorPipeline”是当前括号中的一项，逗号将它与下一项分开。
    PolicyProcessorPipeline,
    # 中文 L0025 “TokenizerProcessorStep”是当前括号中的一项，逗号将它与下一项分开。
    TokenizerProcessorStep,
    # 中文 L0026 “make_default_policy_processor_steps”是当前括号中的一项，逗号将它与下一项分开。
    make_default_policy_processor_steps,
    # 中文 L0027 “make_policy_processor_pipelines”是当前括号中的一项，逗号将它与下一项分开。
    make_policy_processor_pipelines,
# 中文 L0028 闭合上方尚未结束的括号。
)

# 中文 L0030 相对导入前的一个点指当前包，用它从同目录配置模块取 SmolVLAConfig。
from .configuration_smolvla import SmolVLAConfig


# 中文 L0033 策略从头构建时创建前处理与后处理。这也是理解 checkpoint 中同类步骤的对照代码。
def make_smolvla_pre_post_processors(
    # 中文 L0034 “config”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    config: SmolVLAConfig,
    # 中文 L0035 为“dataset_stats”提供这一项的值，注意它所在的参数表或外层表达式。
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
# 中文 L0036 延续上面的语句，) -> tuple[。
) -> tuple[
    # 中文 L0037 延续上面的语句，PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],。
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # 中文 L0038 延续上面的语句，PolicyProcessorPipeline[PolicyAction, PolicyAction],。
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
# 中文 L0039 闭合上方尚未结束的括号。
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # 中文 L0063 先按通用策略配置和统计量建立可复用的处理步骤对象。
    steps = make_default_policy_processor_steps(config, dataset_stats)

    # 中文 L0065 前处理代码按列表顺序运行，每一步的输出会传给下一步。
    input_steps = [
        # 中文 L0066 把当前观测键转换成策略期望的键名。
        steps.rename_observations,  # To mimic the same processor as pretrained one
        # 中文 L0067 为缺少批次轴的输入补上批次轴。
        steps.add_batch_dim,
        # 中文 L0068 为自然语言任务补齐末尾换行，以匹配训练时的文本格式。
        NewLineTaskProcessorStep(),
        # 中文 L0069 构造负责把任务文字转成模型词表编号的分词步骤。
        TokenizerProcessorStep(
            # 中文 L0070 为“tokenizer_name”提供这一项的值，注意它所在的参数表或外层表达式。
            tokenizer_name=config.vlm_model_name,
            # 中文 L0071 配置分词的补齐方式。
            padding=config.pad_language_to,
            # 中文 L0072 用右侧填充对齐短文本序列。
            padding_side="right",
            # 中文 L0073 把最大文本 token 数传给分词器。
            max_length=config.tokenizer_max_length,
        # 中文 L0074 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
        ),
        # 中文 L0075 将流水线中的张量移到策略设备。
        steps.to_device,
        # 中文 L0076 用保存的统计量归一化配置指定的特征。
        steps.normalize,
    # 中文 L0077 闭合上方尚未结束的括号。
    ]
    # 中文 L0078 创建列表，按给定顺序保留元素给 “output_steps” 赋值。
    output_steps = [
        # 中文 L0079 用训练统计量把动作恢复为环境所需的数值尺度。
        steps.unnormalize,
        # 中文 L0080 把预测动作移回 CPU。
        steps.to_cpu,
    # 中文 L0081 闭合上方尚未结束的括号。
    ]
    # 中文 L0082 把两组有序步骤分别包装成可调用的前后处理流水线。
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
