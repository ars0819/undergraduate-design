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
# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0016 从模块 collections.abc 导入 Mapping。
from collections.abc import Mapping
# 中文 L0017 从模块 copy 导入 deepcopy。
from copy import deepcopy
# 中文 L0018 从模块 dataclasses 导入 dataclass、field。
from dataclasses import dataclass, field
# 中文 L0019 从模块 typing 导入 Any。
from typing import Any

# 中文 L0021 从模块 lerobot.configs 导入 PipelineFeatureType、PolicyFeature。
from lerobot.configs import PipelineFeatureType, PolicyFeature

# 中文 L0023 从模块 .pipeline 导入 ObservationProcessorStep、ProcessorStepRegistry。
from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


# 中文 L0026 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0027 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="rename_observations_processor")
# 中文 L0028 按配置修改观测字典中的键，例如把外部相机键映射成 checkpoint 使用的键。
class RenameObservationsProcessorStep(ObservationProcessorStep):
    """
    A processor step that renames keys in an observation dictionary.

    This step is useful for creating a standardized data interface by mapping keys
    from an environment's format to the format expected by a LeRobot policy or
    other downstream components.

    Attributes:
        rename_map: A dictionary mapping from old key names to new key names.
                    Keys present in an observation that are not in this map will
                    be kept with their original names.
    """

    # 中文 L0042 声明“rename_map”的类型为 dict[str, str]，调用 field，用它的结果作为初始值。
    rename_map: dict[str, str] = field(default_factory=dict)

    # 中文 L0044 定义方法 observation。
    def observation(self, observation):
        # 中文 L0045 把观测字典和改名映射交给通用函数。底层张量不需要重新编码。
        return rename_keys(observation, self.rename_map)

    # 中文 L0047 定义方法 get_config。
    def get_config(self) -> dict[str, Any]:
        # 中文 L0048 创建字典，用键查找对应的值作为当前调用的返回值。
        return {"rename_map": self.rename_map}

    # 中文 L0050 定义方法 transform_features。
    def transform_features(
        # 中文 L0051 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0052 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transforms:
        - Each key in the observation that appears in `rename_map` is renamed to its value.
        - Keys not in `rename_map` remain unchanged.
        """
        # 中文 L0057 声明“new_features”的类型为 dict[PipelineFeatureType, dict[str, PolicyFeature]]，调用 features.copy，用它的结果作为初始值。
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = features.copy()
        # 中文 L0058 调用 rename_keys，用它的结果给 “new_features[PipelineFeatureType.OBSERVATION]” 赋值。
        new_features[PipelineFeatureType.OBSERVATION] = rename_keys(
            # 中文 L0059 延续上面的语句，features[PipelineFeatureType.OBSERVATION], self.rename_map。
            features[PipelineFeatureType.OBSERVATION], self.rename_map
        # 中文 L0060 闭合上方尚未结束的括号。
        )
        # 中文 L0061 引用已有的值 new_features作为当前调用的返回值。
        return new_features


# 中文 L0064 定义函数 rename_stats。
def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """
    Renames the top-level keys in a statistics dictionary using a provided mapping.

    This is a helper function typically used to keep normalization statistics
    consistent with renamed observation or action features. It performs a defensive
    deep copy to avoid modifying the original `stats` dictionary.

    Args:
        stats: A nested dictionary of statistics, where top-level keys are
               feature names (e.g., `{"observation.state": {"mean": 0.5}}`).
        rename_map: A dictionary mapping old feature names to new feature names.

    Returns:
        A new statistics dictionary with its top-level keys renamed. Returns an
        empty dictionary if the input `stats` is empty.
    """
    # 中文 L0081 条件 not stats 成立时进入下方分支。
    if not stats:
        # 中文 L0082 创建字典，用键查找对应的值作为当前调用的返回值。
        return {}
    # 中文 L0083 用字典推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
    return {
        # 中文 L0084 延续上面的语句，key: deepcopy(sub_stats) if sub_stats is not None else {}。
        key: deepcopy(sub_stats) if sub_stats is not None else {}
        # 中文 L0085 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, sub_stats in rename_keys(stats, rename_map).items()
    # 中文 L0086 闭合上方尚未结束的括号。
    }


# 中文 L0089 定义函数 _rename_key_with_metadata。
def _rename_key_with_metadata(key: str, rename_map: dict[str, str]) -> str:
    # 中文 L0090 原文档中的说明或续行。
    """Rename a feature key while preserving temporal sampling metadata suffixes."""
    # 中文 L0091 条件 key in rename_map 成立时进入下方分支。
    if key in rename_map:
        # 中文 L0092 用方括号按键或索引读取 rename_map作为当前调用的返回值。
        return rename_map[key]
    # 中文 L0093 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for suffix in ("_is_pad", "_padding_mask"):
        # 中文 L0094 条件 key.endswith(suffix) 成立时进入下方分支。
        if key.endswith(suffix):
            # 中文 L0095 对 key 做切片，右端索引不包含在结果中给 “base” 赋值。
            base = key[: -len(suffix)]
            # 中文 L0096 条件 base in rename_map 成立时进入下方分支。
            if base in rename_map:
                # 中文 L0097 计算此行的右侧表达式作为当前调用的返回值。
                return f"{rename_map[base]}{suffix}"
    # 中文 L0098 引用已有的值 key作为当前调用的返回值。
    return key


# 中文 L0101 定义函数 rename_batch_keys。
def rename_batch_keys(batch: dict[str, Any], rename_map: dict[str, str] | None) -> dict[str, Any]:
    # 中文 L0102 原文档中的说明或续行。
    """Canonicalize raw dataset keys before they are grouped into a transition."""
    # 中文 L0103 条件 not rename_map 成立时进入下方分支。
    if not rename_map:
        # 中文 L0104 引用已有的值 batch作为当前调用的返回值。
        return batch
    # 中文 L0105 调用 rename_keys，用它的结果作为当前调用的返回值。
    return rename_keys(batch, rename_map)


# 中文 L0108 支持任意值类型 V 的改名函数。映射的键会被检查和更新。
def rename_keys[V](mapping: Mapping[str, V], rename_map: dict[str, str] | None) -> dict[str, V]:
    """Rename the keys of `mapping` (and their temporal padding keys) with `rename_map`.

    Raises:
        ValueError: Two keys get the same name, e.g. `{"action.cartesian": "action"}` on data that
            already has an `action`. Rename the existing key as well, e.g. `{"action": "action.joint"}`.
    """
    # 中文 L0115 声明“renamed”的类型为 dict[str, V]，创建字典，用键查找对应的值作为初始值。
    renamed: dict[str, V] = {}
    # 中文 L0116 声明“sources”的类型为 dict[str, str]，创建字典，用键查找对应的值作为初始值。
    sources: dict[str, str] = {}
    # 中文 L0117 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key, value in mapping.items():
        # 中文 L0118 调用 _rename_key_with_metadata，用它的结果给 “new_key” 赋值。
        new_key = _rename_key_with_metadata(key, rename_map or {})
        # 中文 L0119 条件 new_key in sources 成立时进入下方分支。
        if new_key in sources:
            # 中文 L0120 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0121 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Both {sources[new_key]!r} and {key!r} are named {new_key!r} after rename_map. Map one of "
                # 中文 L0122 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"them to another name as well, e.g. add {{{new_key!r}: {new_key + '.original'!r}}} to rename "
                # 中文 L0123 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"a key onto an existing {new_key!r}."
            # 中文 L0124 闭合上方尚未结束的括号。
            )
        # 中文 L0125 引用已有的值 value给 “renamed[new_key]” 赋值。
        renamed[new_key] = value
        # 中文 L0126 引用已有的值 key给 “sources[new_key]” 赋值。
        sources[new_key] = key
    # 中文 L0127 引用已有的值 renamed作为当前调用的返回值。
    return renamed
