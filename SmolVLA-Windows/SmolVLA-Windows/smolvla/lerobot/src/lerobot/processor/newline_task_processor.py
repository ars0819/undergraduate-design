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
# 中文 L0017 从模块 lerobot.configs 导入 PipelineFeatureType、PolicyFeature。
from lerobot.configs import PipelineFeatureType, PolicyFeature

# 中文 L0019 从模块 .pipeline 导入 ComplementaryDataProcessorStep、ProcessorStepRegistry。
from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry


# NOTE: The registry name "smolvla_new_line_processor" is kept for backward compatibility
# with serialized processor configs that reference this name.
# 中文 L0024 把该类注册在旧 checkpoint 保存的名字 smolvla_new_line_processor 下。
@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
# 中文 L0025 定义类 NewLineTaskProcessorStep，括号列出基类 ComplementaryDataProcessorStep。
class NewLineTaskProcessorStep(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    # 中文 L0034 处理 transition 中独立于观测 Tensor 的附加数据。
    def complementary_data(self, complementary_data):
        # 中文 L0035 条件 "task" not in complementary_data 成立时进入下方分支。
        if "task" not in complementary_data:
            # 中文 L0036 引用已有的值 complementary_data作为当前调用的返回值。
            return complementary_data

        # 中文 L0038 用方括号按键或索引读取 complementary_data给 “task”（任务） 赋值。
        task = complementary_data["task"]
        # 中文 L0039 条件 task is None 成立时进入下方分支。
        if task is None:
            # 中文 L0040 引用已有的值 complementary_data作为当前调用的返回值。
            return complementary_data

        # 中文 L0042 调用 dict，用它的结果给 “new_complementary_data” 赋值。
        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        # 中文 L0045 条件 isinstance(task, str) 成立时进入下方分支。
        if isinstance(task, str):
            # Single string: add newline if not present
            # 中文 L0047 检查单条任务文字是否已经有末尾换行。
            if not task.endswith("\n"):
                # 中文 L0048 计算此行的右侧表达式给 “new_complementary_data['task']” 赋值。
                new_complementary_data["task"] = f"{task}\n"
        # 中文 L0049 前面的分支均未命中时，再判断 isinstance(task, list) and all(isinstance(t, str) for t in task)。
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            # 中文 L0051 如果一批中有多条任务文字，逐项保证末尾有换行。
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        # 中文 L0054 引用已有的值 new_complementary_data作为当前调用的返回值。
        return new_complementary_data

    # 中文 L0056 定义方法 transform_features。
    def transform_features(
        # 中文 L0057 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0058 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 中文 L0059 引用已有的值 features作为当前调用的返回值。
        return features
