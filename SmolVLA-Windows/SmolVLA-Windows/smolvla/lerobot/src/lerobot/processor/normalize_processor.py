#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, RobotObservation, TransitionKey

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry


@dataclass
class _NormalizationMixin:
    """
    A mixin class providing core functionality for normalization and unnormalization.

    This class manages normalization statistics (`stats`), converts them to tensors for
    efficient computation, handles device placement, and implements the logic for
    applying normalization transformations (mean/std and min/max). It is designed to
    be inherited by concrete `ProcessorStep` implementations and should not be used
    directly.

    **Stats Override Preservation:**
    When stats are explicitly provided during construction (e.g., via overrides in
    `DataProcessorPipeline.from_pretrained()`), they are preserved even when
    `load_state_dict()` is called. This allows users to override normalization
    statistics from saved models while keeping the rest of the model state intact.

    Examples:
        ```python
        # Common use case: Override with dataset stats
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset("my_dataset")
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": dataset.meta.stats}}
        )
        # dataset.meta.stats will be used, not the stats from the saved model

        # Custom stats override
        custom_stats = {"action": {"mean": [0.0], "std": [1.0]}}
        pipeline = DataProcessorPipeline.from_pretrained(
            "model_path", overrides={"normalizer_processor": {"stats": custom_stats}}
        )
        ```

    Attributes:
        features: A dictionary mapping feature names to `PolicyFeature` objects, defining
            the data structure to be processed.
        norm_map: A dictionary mapping `FeatureType` to `NormalizationMode`, specifying
            which normalization method to use for each type of feature. String keys (e.g.
            from a JSON config) are converted to `FeatureType` in `__post_init__`.
        stats: A dictionary containing the normalization statistics (e.g., mean, std,
            min, max) for each feature.
        device: The PyTorch device on which to store and perform tensor operations.
        eps: A small epsilon value to prevent division by zero in normalization
            calculations.
        normalize_observation_keys: An optional set of keys to selectively apply
            normalization to specific observation features.
        _tensor_stats: An internal dictionary holding the normalization statistics as
            PyTorch tensors.
        _stats_explicitly_provided: Internal flag tracking whether stats were explicitly
            provided during construction (used for override preservation).
    """

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode] | dict[str, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """
        Initializes the mixin after dataclass construction.

        This method handles the robust deserialization of `features` and `norm_map`
        from JSON-compatible formats (where enums become strings and tuples become
        lists) and converts the provided `stats` dictionary into a dictionary of
        tensors (`_tensor_stats`) on the specified device.
        """
        # Track if stats were explicitly provided (not None and not empty)
        self._stats_explicitly_provided = self.stats is not None and bool(self.stats)
        # Robust JSON deserialization handling (guard empty maps).
        if self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.features = reconstructed

        # if keys are strings (JSON), rebuild enum map
        if self.norm_map and all(isinstance(k, str) for k in self.norm_map):
            reconstructed = {}
            for ft_type_str, norm_mode_str in self.norm_map.items():
                reconstructed[FeatureType(ft_type_str)] = NormalizationMode(norm_mode_str)
            self.norm_map = reconstructed

        # Convert stats to tensors and move to the target device once during initialization.
        self.stats = self.stats or {}
        if self.dtype is None:
            self.dtype = torch.float32
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        self._reshape_visual_stats()

    def _reshape_visual_stats(self) -> None:
        """Reshape flat ``(C,)`` visual stats to ``(C, 1, 1)`` for image broadcasting.

        No-op for stats from :func:`~lerobot.datasets.compute_stats.compute_stats`
        (already ``(C, 1, 1)``). Needed by RL training, which can start without
        a dataset and supplies stats manually via JSON config.
        """
        for key, feature in self.features.items():
            if feature.type != FeatureType.VISUAL:
                continue
            if key not in self._tensor_stats:
                continue
            for stat_name, stat_tensor in self._tensor_stats[key].items():
                if not isinstance(stat_tensor, Tensor) or stat_tensor.ndim != 1:
                    continue
                self._tensor_stats[key][stat_name] = stat_tensor.reshape(-1, 1, 1)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _NormalizationMixin:
        """
        Moves the processor's normalization stats to the specified device.

        Args:
            device: The target PyTorch device.

        Returns:
            The instance of the class, allowing for method chaining.
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        self._reshape_visual_stats()
        return self

    def state_dict(self) -> dict[str, Tensor]:
        """
        Returns the normalization statistics as a flat state dictionary.

        All tensors are moved to the CPU before being returned, which is standard practice
        for saving state dictionaries.

        Returns:
            A flat dictionary mapping from `'feature_name.stat_name'` to the
            corresponding statistics tensor on the CPU.
        """
        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()  # Always save to CPU
        return flat

    # 中文 L0194 定义方法 load_state_dict。
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """
        Loads normalization statistics from a state dictionary.

        The loaded tensors are moved to the processor's configured device.

        **Stats Override Preservation:**
        If stats were explicitly provided during construction (e.g., via overrides in
        `DataProcessorPipeline.from_pretrained()`), they are preserved and the state
        dictionary is ignored. This allows users to override normalization statistics
        while still loading the rest of the model state.

        This behavior is crucial for scenarios where users want to adapt a pretrained
        model to a new dataset with different statistics without retraining the entire
        model.

        Args:
            state: A flat state dictionary with keys in the format
                   `'feature_name.stat_name'`.

        Note:
            When stats are preserved due to explicit provision, only the tensor
            representation is updated to ensure consistency with the current device
            and dtype settings.
        """
        # If stats were explicitly provided during construction, preserve them
        # 中文 L0220 条件 self._stats_explicitly_provided and self.stats is not None 成立时进入下方分支。
        if self._stats_explicitly_provided and self.stats is not None:
            # Don't load from state_dict, keep the explicitly provided stats
            # But ensure _tensor_stats is properly initialized
            # 中文 L0223 调用 to_tensor，用它的结果给 “self._tensor_stats” 赋值。
            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
            # 中文 L0224 调用 self._reshape_visual_stats 执行操作。
            self._reshape_visual_stats()
            # 中文 L0225 退出当前调用并返回 None。
            return

        # Normal behavior: load stats from state_dict
        # 中文 L0228 调用 self._tensor_stats.clear 执行操作。
        self._tensor_stats.clear()
        # 中文 L0229 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for flat_key, tensor in state.items():
            # 中文 L0230 调用 flat_key.rsplit，用它的结果给 “(key, stat_name)” 赋值。
            key, stat_name = flat_key.rsplit(".", 1)
            # Load to the processor's configured device.
            # 中文 L0232 调用 tensor.to，用它的结果给 “self._tensor_stats.setdefault(key, {})[stat_name]” 赋值。
            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                # 中文 L0233 为“dtype”（元素类型）提供这一项的值，注意它所在的参数表或外层表达式。
                dtype=torch.float32, device=self.device
            # 中文 L0234 闭合上方尚未结束的括号。
            )
        # 中文 L0235 调用 self._reshape_visual_stats 执行操作。
        self._reshape_visual_stats()

        # Reconstruct the original stats dict from tensor stats for compatibility with to() method
        # and other functions that rely on self.stats
        # 中文 L0239 创建字典，用键查找对应的值给 “self.stats” 赋值。
        self.stats = {}
        # 中文 L0240 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, tensor_dict in self._tensor_stats.items():
            # 中文 L0241 创建字典，用键查找对应的值给 “self.stats[key]” 赋值。
            self.stats[key] = {}
            # 中文 L0242 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for stat_name, tensor in tensor_dict.items():
                # Convert tensor back to python/numpy format
                # 中文 L0244 调用 from_tensor_to_numpy，用它的结果给 “self.stats[key][stat_name]” 赋值。
                self.stats[key][stat_name] = from_tensor_to_numpy(tensor)

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        This method is used when saving the processor to disk, ensuring that its
        configuration can be reconstructed later.

        Returns:
            A JSON-serializable dictionary containing the configuration.
        """
        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {
                FeatureType(ft_type).value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()
            },
        }
        if self.normalize_observation_keys is not None:
            config["normalize_observation_keys"] = sorted(self.normalize_observation_keys)
        return config

    # 中文 L0269 根据特征类型和配置对观测特征逐个做尺度变换。
    def _normalize_observation(self, observation: RobotObservation, inverse: bool) -> dict[str, Tensor]:
        """
        Applies (un)normalization to all relevant features in an observation dictionary.

        Args:
            observation: The observation dictionary to process.
            inverse: If `True`, applies unnormalization; otherwise, applies normalization.

        Returns:
            A new observation dictionary with the transformed tensor values.
        """
        # 中文 L0280 调用 dict，用它的结果给 “new_observation” 赋值。
        new_observation = dict(observation)
        # 中文 L0281 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, feature in self.features.items():
            # 中文 L0282 条件 self.normalize_observation_keys is not None and key not in self.normalize_observation_keys 成立时进入下方分支。
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                # 中文 L0283 跳过这一轮剩余操作，进入下一轮循环。
                continue
            # 中文 L0284 条件 feature.type != FeatureType.ACTION and key in new_observation 成立时进入下方分支。
            if feature.type != FeatureType.ACTION and key in new_observation:
                # Convert to tensor but preserve original dtype for adaptation logic
                # 中文 L0286 调用 torch.as_tensor，用它的结果给 “tensor” 赋值。
                tensor = torch.as_tensor(new_observation[key])
                # 中文 L0287 调用 self._apply_transform，用它的结果给 “new_observation[key]” 赋值。
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        # 中文 L0288 引用已有的值 new_observation作为当前调用的返回值。
        return new_observation

    # 中文 L0290 使用动作特征的专属统计量和配置处理动作。
    def _normalize_action(self, action: Tensor, inverse: bool) -> Tensor:
        # Convert to tensor but preserve original dtype for adaptation logic
        """
        Applies (un)normalization to an action tensor.

        Args:
            action: The action tensor to process.
            inverse: If `True`, applies unnormalization; otherwise, applies normalization.

        Returns:
            The transformed action tensor.
        """
        # 中文 L0302 调用 self._apply_transform，用它的结果给 “processed_action” 赋值。
        processed_action = self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=inverse)
        # 中文 L0303 引用已有的值 processed_action作为当前调用的返回值。
        return processed_action

    # 中文 L0305 在各种归一化规则中选择当前特征对应的转换，支持正向与反向。
    def _apply_transform(
        # 中文 L0306 延续上面的语句，self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False。本段处理施加由配置指定的正向或逆向尺度变换。
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    # 中文 L0307 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor:
        """
        Core logic to apply a normalization or unnormalization transformation to a tensor.

        This method selects the appropriate normalization mode based on the feature type
        and applies the corresponding mathematical operation.

        Normalization Modes:
          - MEAN_STD: Centers data around zero with unit variance.
          - MIN_MAX: Scales data to [-1, 1] range using actual min/max values.
          - QUANTILES: Scales data to [-1, 1] range using 1st and 99th percentiles (q01/q99).
          - QUANTILE10: Scales data to [-1, 1] range using 10th and 90th percentiles (q10/q90).

        Args:
            tensor: The input tensor to transform.
            key: The feature key corresponding to the tensor.
            feature_type: The `FeatureType` of the tensor.
            inverse: If `True`, applies the inverse transformation (unnormalization).

        Returns:
            The transformed tensor.

        Raises:
            ValueError: If an unsupported normalization mode is encountered.
        """
        # 中文 L0332 调用 self.norm_map.get，用它的结果给 “norm_mode” 赋值。
        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        # 中文 L0333 条件 norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats 成立时进入下方分支。
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            # 中文 L0334 引用已有的值 tensor作为当前调用的返回值。
            return tensor

        # 中文 L0336 条件 norm_mode not in ( 成立时进入下方分支。
        if norm_mode not in (
            # 中文 L0337 延续上面的语句，NormalizationMode.MEAN_STD,。本段处理施加由配置指定的正向或逆向尺度变换。
            NormalizationMode.MEAN_STD,
            # 中文 L0338 延续上面的语句，NormalizationMode.MIN_MAX,。本段处理施加由配置指定的正向或逆向尺度变换。
            NormalizationMode.MIN_MAX,
            # 中文 L0339 延续上面的语句，NormalizationMode.QUANTILES,。本段处理施加由配置指定的正向或逆向尺度变换。
            NormalizationMode.QUANTILES,
            # 中文 L0340 延续上面的语句，NormalizationMode.QUANTILE10,。本段处理施加由配置指定的正向或逆向尺度变换。
            NormalizationMode.QUANTILE10,
        # 中文 L0341 闭合上方尚未结束的括号。
        ):
            # 中文 L0342 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # For Accelerate compatibility: Ensure stats are on the same device and dtype as the input tensor
        # 中文 L0345 条件 self._tensor_stats and key in self._tensor_stats 成立时进入下方分支。
        if self._tensor_stats and key in self._tensor_stats:
            # 中文 L0346 调用 next，用它的结果给 “first_stat” 赋值。
            first_stat = next(iter(self._tensor_stats[key].values()))
            # 中文 L0347 条件 first_stat.device != tensor.device or first_stat.dtype != tensor.dtype 成立时进入下方分支。
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                # 中文 L0348 使统计 Tensor 的 device 和 dtype 跟待处理 Tensor 对齐。
                self.to(device=tensor.device, dtype=tensor.dtype)

        # 中文 L0350 用方括号按键或索引读取 self._tensor_stats给 “stats”（归一化统计量） 赋值。
        stats = self._tensor_stats[key]

        # 中文 L0352 条件 norm_mode == NormalizationMode.MEAN_STD 成立时进入下方分支。
        if norm_mode == NormalizationMode.MEAN_STD:
            # 中文 L0353 读取逐分量均值。每个状态或动作维度可有独立统计量。
            mean = stats.get("mean", None)
            # 中文 L0354 读取逐分量标准差。
            std = stats.get("std", None)
            # 中文 L0355 条件 mean is None or std is None 成立时进入下方分支。
            if mean is None or std is None:
                # 中文 L0356 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0357 字符串字面量，供上方的调用、容器或错误消息使用。
                    "MEAN_STD normalization mode requires mean and std stats, please update the dataset with the correct stats"
                # 中文 L0358 闭合上方尚未结束的括号。
                )

            # 中文 L0360 打包成元组，元素顺序由逗号分隔的位置决定给 “(mean, std)” 赋值。
            mean, std = stats["mean"], stats["std"]
            # Avoid division by zero by adding a small epsilon.
            # 中文 L0362 正向归一化的除数是标准差加 eps，避免近零标准差导致除零。
            denom = std + self.eps
            # 中文 L0363 条件 inverse 成立时进入下方分支。
            if inverse:
                # 中文 L0364 反归一化按分量执行预测值乘标准差再加均值。
                return tensor * std + mean
            # 中文 L0365 正向归一化先减均值再除以稳定化的标准差。
            return (tensor - mean) / denom

        # 中文 L0367 条件 norm_mode == NormalizationMode.MIN_MAX 成立时进入下方分支。
        if norm_mode == NormalizationMode.MIN_MAX:
            # 中文 L0368 调用 stats.get，用它的结果给 “min_val” 赋值。
            min_val = stats.get("min", None)
            # 中文 L0369 调用 stats.get，用它的结果给 “max_val” 赋值。
            max_val = stats.get("max", None)
            # 中文 L0370 条件 min_val is None or max_val is None 成立时进入下方分支。
            if min_val is None or max_val is None:
                # 中文 L0371 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0372 字符串字面量，供上方的调用、容器或错误消息使用。
                    "MIN_MAX normalization mode requires min and max stats, please update the dataset with the correct stats"
                # 中文 L0373 闭合上方尚未结束的括号。
                )

            # 中文 L0375 打包成元组，元素顺序由逗号分隔的位置决定给 “(min_val, max_val)” 赋值。
            min_val, max_val = stats["min"], stats["max"]
            # 中文 L0376 计算右侧表达式 max_val - min_val给 “denom” 赋值。
            denom = max_val - min_val
            # When min_val == max_val, substitute the denominator with a small epsilon
            # to prevent division by zero. This consistently maps an input equal to
            # min_val to -1, ensuring a stable transformation.
            # 中文 L0380 调用 torch.where，用它的结果给 “denom” 赋值。
            denom = torch.where(
                # 中文 L0381 为“denom”提供这一项的值，注意它所在的参数表或外层表达式。
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            # 中文 L0382 闭合上方尚未结束的括号。
            )
            # 中文 L0383 条件 inverse 成立时进入下方分支。
            if inverse:
                # Map from [-1, 1] back to [min, max]
                # 中文 L0385 计算右侧表达式 return (tensor + 1) / 2 * denom + min_val作为当前调用的返回值。
                return (tensor + 1) / 2 * denom + min_val
            # Map from [min, max] to [-1, 1]
            # 中文 L0387 计算右侧表达式 return 2 * (tensor - min_val) / denom - 1作为当前调用的返回值。
            return 2 * (tensor - min_val) / denom - 1

        # 中文 L0389 条件 norm_mode == NormalizationMode.QUANTILES 成立时进入下方分支。
        if norm_mode == NormalizationMode.QUANTILES:
            # 中文 L0390 调用 stats.get，用它的结果给 “q01” 赋值。
            q01 = stats.get("q01", None)
            # 中文 L0391 调用 stats.get，用它的结果给 “q99” 赋值。
            q99 = stats.get("q99", None)
            # 中文 L0392 条件 q01 is None or q99 is None 成立时进入下方分支。
            if q01 is None or q99 is None:
                # 中文 L0393 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0394 字符串字面量，供上方的调用、容器或错误消息使用。
                    "QUANTILES normalization mode requires q01 and q99 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                # 中文 L0395 闭合上方尚未结束的括号。
                )

            # 中文 L0397 计算右侧表达式 q99 - q01给 “denom” 赋值。
            denom = q99 - q01
            # Avoid division by zero by adding epsilon when quantiles are identical
            # 中文 L0399 调用 torch.where，用它的结果给 “denom” 赋值。
            denom = torch.where(
                # 中文 L0400 为“denom”提供这一项的值，注意它所在的参数表或外层表达式。
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            # 中文 L0401 闭合上方尚未结束的括号。
            )
            # 中文 L0402 条件 inverse 成立时进入下方分支。
            if inverse:
                # 中文 L0403 计算右侧表达式 return (tensor + 1.0) * denom / 2.0 + q01作为当前调用的返回值。
                return (tensor + 1.0) * denom / 2.0 + q01
            # 中文 L0404 计算右侧表达式 return 2.0 * (tensor - q01) / denom - 1.0作为当前调用的返回值。
            return 2.0 * (tensor - q01) / denom - 1.0

        # 中文 L0406 条件 norm_mode == NormalizationMode.QUANTILE10 成立时进入下方分支。
        if norm_mode == NormalizationMode.QUANTILE10:
            # 中文 L0407 调用 stats.get，用它的结果给 “q10” 赋值。
            q10 = stats.get("q10", None)
            # 中文 L0408 调用 stats.get，用它的结果给 “q90” 赋值。
            q90 = stats.get("q90", None)
            # 中文 L0409 条件 q10 is None or q90 is None 成立时进入下方分支。
            if q10 is None or q90 is None:
                # 中文 L0410 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0411 字符串字面量，供上方的调用、容器或错误消息使用。
                    "QUANTILE10 normalization mode requires q10 and q90 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                # 中文 L0412 闭合上方尚未结束的括号。
                )

            # 中文 L0414 计算右侧表达式 q90 - q10给 “denom” 赋值。
            denom = q90 - q10
            # Avoid division by zero by adding epsilon when quantiles are identical
            # 中文 L0416 调用 torch.where，用它的结果给 “denom” 赋值。
            denom = torch.where(
                # 中文 L0417 为“denom”提供这一项的值，注意它所在的参数表或外层表达式。
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            # 中文 L0418 闭合上方尚未结束的括号。
            )
            # 中文 L0419 条件 inverse 成立时进入下方分支。
            if inverse:
                # 中文 L0420 计算右侧表达式 return (tensor + 1.0) * denom / 2.0 + q10作为当前调用的返回值。
                return (tensor + 1.0) * denom / 2.0 + q10
            # 中文 L0421 计算右侧表达式 return 2.0 * (tensor - q10) / denom - 1.0作为当前调用的返回值。
            return 2.0 * (tensor - q10) / denom - 1.0

        # If necessary stats are missing, return input unchanged.
        # 中文 L0424 引用已有的值 tensor作为当前调用的返回值。
        return tensor


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0427 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0428 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="normalizer_processor")
# 中文 L0429 定义类 NormalizerProcessorStep，括号列出基类 _NormalizationMixin、ProcessorStep。
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    A processor step that applies normalization to observations and actions in a transition.

    This class uses the logic from `_NormalizationMixin` to perform forward normalization
    (e.g., scaling data to have zero mean and unit variance, or to the range [-1, 1]).
    It is typically used in the pre-processing pipeline before feeding data to a policy.
    """

    # 中文 L0438 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0439 定义方法 from_lerobot_dataset。
    def from_lerobot_dataset(
        # 中文 L0440 cls 接收当前类，类方法被调用时由 Python 自动传入。
        cls,
        # 中文 L0441 “dataset”是当前括号中的一项，逗号将它与下一项分开。
        dataset: LeRobotDataset,
        # 中文 L0442 延续上面的语句，features: dict[str, PolicyFeature],。
        features: dict[str, PolicyFeature],
        # 中文 L0443 延续上面的语句，norm_map: dict[FeatureType, NormalizationMode],。
        norm_map: dict[FeatureType, NormalizationMode],
        # 中文 L0444 后续参数必须按名字传入。
        *,
        # 中文 L0445 为“normalize_observation_keys”提供这一项的值，注意它所在的参数表或外层表达式。
        normalize_observation_keys: set[str] | None = None,
        # 中文 L0446 为“eps”提供这一项的值，注意它所在的参数表或外层表达式。
        eps: float = 1e-8,
        # 中文 L0447 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
        device: torch.device | str | None = None,
    # 中文 L0448 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> NormalizerProcessorStep:
        """
        Creates a `NormalizerProcessorStep` instance using statistics from a `LeRobotDataset`.

        Args:
            dataset: The dataset from which to extract normalization statistics.
            features: The feature definition for the processor.
            norm_map: The mapping from feature types to normalization modes.
            normalize_observation_keys: An optional set of observation keys to normalize.
            eps: A small epsilon value for numerical stability.
            device: The target device for the processor.

        Returns:
            A new instance of `NormalizerProcessorStep`.
        """
        # 中文 L0463 调用 cls，用它的结果作为当前调用的返回值。
        return cls(
            # 中文 L0464 为“features”提供这一项的值，注意它所在的参数表或外层表达式。
            features=features,
            # 中文 L0465 为“norm_map”提供这一项的值，注意它所在的参数表或外层表达式。
            norm_map=norm_map,
            # 中文 L0466 为“stats”（归一化统计量）提供这一项的值，注意它所在的参数表或外层表达式。
            stats=dataset.meta.stats,
            # 中文 L0467 为“normalize_observation_keys”提供这一项的值，注意它所在的参数表或外层表达式。
            normalize_observation_keys=normalize_observation_keys,
            # 中文 L0468 为“eps”提供这一项的值，注意它所在的参数表或外层表达式。
            eps=eps,
            # 中文 L0469 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
            device=device,
        # 中文 L0470 闭合上方尚未结束的括号。
        )

    # 中文 L0472 对观测执行正向归一化。输入中存在动作时也处理它。
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # 中文 L0473 调用 transition.copy，用它的结果给 “new_transition” 赋值。
        new_transition = transition.copy()

        # Handle observation normalization.
        # 中文 L0476 调用 new_transition.get，用它的结果给 “observation”（观测） 赋值。
        observation = new_transition.get(TransitionKey.OBSERVATION)
        # 中文 L0477 条件 observation is not None 成立时进入下方分支。
        if observation is not None:
            # 中文 L0478 调用 self._normalize_observation，用它的结果给 “new_transition[TransitionKey.OBSERVATION]” 赋值。
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                # 中文 L0479 延续上面的语句，observation, inverse=False。
                observation, inverse=False
            # 中文 L0480 闭合上方尚未结束的括号。
            )

        # Handle action normalization.
        # 中文 L0483 调用 new_transition.get，用它的结果给 “action”（单步动作） 赋值。
        action = new_transition.get(TransitionKey.ACTION)

        # 中文 L0485 条件 action is None 成立时进入下方分支。
        if action is None:
            # 中文 L0486 引用已有的值 new_transition作为当前调用的返回值。
            return new_transition

        # 中文 L0488 条件 not isinstance(action, torch.Tensor) 成立时进入下方分支。
        if not isinstance(action, torch.Tensor):
            # 中文 L0489 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # 中文 L0491 调用 self._normalize_action，用它的结果给 “new_transition[TransitionKey.ACTION]” 赋值。
        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)

        # 中文 L0493 引用已有的值 new_transition作为当前调用的返回值。
        return new_transition

    # 中文 L0495 定义方法 transform_features。
    def transform_features(
        # 中文 L0496 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0497 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 中文 L0498 引用已有的值 features作为当前调用的返回值。
        return features


# 中文 L0501 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0502 注册装饰器。模块导入时，把随后的类或方法登记到框架的查找表。
@ProcessorStepRegistry.register(name="unnormalizer_processor")
# 中文 L0503 定义类 UnnormalizerProcessorStep，括号列出基类 _NormalizationMixin、ProcessorStep。
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    """
    A processor step that applies unnormalization to observations and actions.

    This class inverts the normalization process, scaling data back to its original
    range. It is typically used in the post-processing pipeline to convert a policy's
    normalized action output into a format that can be executed by a robot or
    environment.
    """

    # 中文 L0513 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0514 定义方法 from_lerobot_dataset。
    def from_lerobot_dataset(
        # 中文 L0515 cls 接收当前类，类方法被调用时由 Python 自动传入。
        cls,
        # 中文 L0516 “dataset”是当前括号中的一项，逗号将它与下一项分开。
        dataset: LeRobotDataset,
        # 中文 L0517 延续上面的语句，features: dict[str, PolicyFeature],。
        features: dict[str, PolicyFeature],
        # 中文 L0518 延续上面的语句，norm_map: dict[FeatureType, NormalizationMode],。
        norm_map: dict[FeatureType, NormalizationMode],
        # 中文 L0519 后续参数必须按名字传入。
        *,
        # 中文 L0520 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
        device: torch.device | str | None = None,
    # 中文 L0521 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> UnnormalizerProcessorStep:
        """
        Creates an `UnnormalizerProcessorStep` using statistics from a `LeRobotDataset`.

        Args:
            dataset: The dataset from which to extract normalization statistics.
            features: The feature definition for the processor.
            norm_map: The mapping from feature types to normalization modes.
            device: The target device for the processor.

        Returns:
            A new instance of `UnnormalizerProcessorStep`.
        """
        # 中文 L0534 调用 cls，用它的结果作为当前调用的返回值。
        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats, device=device)

    # 中文 L0536 对已有输入执行反归一化。本次后处理中处理的是动作。
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # 中文 L0537 调用 transition.copy，用它的结果给 “new_transition” 赋值。
        new_transition = transition.copy()

        # Handle observation unnormalization.
        # 中文 L0540 调用 new_transition.get，用它的结果给 “observation”（观测） 赋值。
        observation = new_transition.get(TransitionKey.OBSERVATION)
        # 中文 L0541 条件 observation is not None 成立时进入下方分支。
        if observation is not None:
            # 中文 L0542 调用 self._normalize_observation，用它的结果给 “new_transition[TransitionKey.OBSERVATION]” 赋值。
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=True)

        # Handle action unnormalization.
        # 中文 L0545 调用 new_transition.get，用它的结果给 “action”（单步动作） 赋值。
        action = new_transition.get(TransitionKey.ACTION)

        # 中文 L0547 条件 action is None 成立时进入下方分支。
        if action is None:
            # 中文 L0548 引用已有的值 new_transition作为当前调用的返回值。
            return new_transition
        # 中文 L0549 条件 not isinstance(action, torch.Tensor) 成立时进入下方分支。
        if not isinstance(action, torch.Tensor):
            # 中文 L0550 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # 中文 L0552 调用 self._normalize_action，用它的结果给 “new_transition[TransitionKey.ACTION]” 赋值。
        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)

        # 中文 L0554 引用已有的值 new_transition作为当前调用的返回值。
        return new_transition

    # 中文 L0556 定义方法 transform_features。
    def transform_features(
        # 中文 L0557 延续上面的语句，self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]。
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    # 中文 L0558 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # 中文 L0559 引用已有的值 features作为当前调用的返回值。
        return features


def hotswap_stats(
    policy_processor: PolicyProcessorPipeline, stats: dict[str, dict[str, Any]]
) -> PolicyProcessorPipeline:
    """
    Replaces normalization statistics in an existing `PolicyProcessorPipeline` instance.

    This function creates a deep copy of the provided pipeline and updates the
    statistics of any `NormalizerProcessorStep` or `UnnormalizerProcessorStep` it
    contains. This is useful for adapting a trained policy to a new environment or
    dataset with different data distributions without having to reconstruct the entire
    pipeline.

    Args:
        policy_processor: The policy processor pipeline to modify.
        stats: The new dictionary of normalization statistics to apply.

    Returns:
        A new `PolicyProcessorPipeline` instance with the updated statistics.
    """
    rp = deepcopy(policy_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats
            # Re-initialize tensor_stats on the correct device.
            step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)
    return rp
