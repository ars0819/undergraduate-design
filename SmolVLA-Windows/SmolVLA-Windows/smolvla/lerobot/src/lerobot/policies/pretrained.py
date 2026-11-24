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
# 中文 L0014 从模块 __future__ 导入 annotations。
from __future__ import annotations

# 中文 L0016 导入模块 abc。
import abc
# 中文 L0017 导入模块 builtins。
import builtins
# 中文 L0018 导入模块 dataclasses。
import dataclasses
# 中文 L0019 导入模块 logging。
import logging
# 中文 L0020 导入模块 os。
import os
# 中文 L0021 导入模块 warnings。
import warnings
# 中文 L0022 从模块 pathlib 导入 Path。
from pathlib import Path
# 中文 L0023 从模块 typing 导入 TYPE_CHECKING、Any、ClassVar、TypedDict、TypeVar、Unpack。
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeVar, Unpack

# 中文 L0025 从模块 huggingface_hub 导入 hf_hub_download、save_torch_state_dict。
from huggingface_hub import hf_hub_download, save_torch_state_dict
# 中文 L0026 从模块 huggingface_hub.constants 导入 SAFETENSORS_INDEX_FILE、SAFETENSORS_SINGLE_FILE。
from huggingface_hub.constants import SAFETENSORS_INDEX_FILE, SAFETENSORS_SINGLE_FILE
# 中文 L0027 从模块 huggingface_hub.errors 导入 EntryNotFoundError。
from huggingface_hub.errors import EntryNotFoundError
# 中文 L0028 从模块 safetensors.torch 导入 load_file、load_model（本地名 load_model_as_safetensor）。
from safetensors.torch import load_file, load_model as load_model_as_safetensor
# 中文 L0029 从模块 torch 导入 Tensor、nn。
from torch import Tensor, nn

# 中文 L0031 从模块 lerobot.configs 导入 FeatureType、PreTrainedConfig。
from lerobot.configs import FeatureType, PreTrainedConfig
# 中文 L0032 从模块 lerobot.lerobot_types 导入 PolicyOutput。
from lerobot.lerobot_types import PolicyOutput
# 中文 L0033 从模块 lerobot.optim.optimizers 导入 OptimizerParams。
from lerobot.optim.optimizers import OptimizerParams
# 中文 L0034 从模块 lerobot.utils.constants 导入 ACTION。
from lerobot.utils.constants import ACTION
# 中文 L0035 从模块 lerobot.utils.device_utils 导入 resolve_safetensors_device。
from lerobot.utils.device_utils import resolve_safetensors_device
# 中文 L0036 从模块 lerobot.utils.feature_spec 导入 FeatureSpec。
from lerobot.utils.feature_spec import FeatureSpec
# 中文 L0037 从模块 lerobot.utils.hub 导入 HubMixin。
from lerobot.utils.hub import HubMixin
# 中文 L0038 从模块 lerobot.utils.import_utils 导入 _peft_available、require_package。
from lerobot.utils.import_utils import _peft_available, require_package

# 中文 L0040 从模块 .checkpoint 导入 SafetensorsCheckpoint、safetensors_shard_names。
from .checkpoint import SafetensorsCheckpoint, safetensors_shard_names
# 中文 L0041 从模块 .chunk 导入 ChunkPolicySpec、ExecutionMode。
from .chunk import ChunkPolicySpec, ExecutionMode
# 中文 L0042 从模块 .utils 导入 log_model_loading_keys、rename_legacy_vision_model_keys。
from .utils import log_model_loading_keys, rename_legacy_vision_model_keys

# 中文 L0044 条件 TYPE_CHECKING or _peft_available 成立时进入下方分支。
if TYPE_CHECKING or _peft_available:
    # 中文 L0045 从模块 peft 导入 PEFT_TYPE_TO_CONFIG_MAPPING、PeftType、get_peft_model。
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType, get_peft_model
# 中文 L0046 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
else:
    # 中文 L0047 设为空值 None给 “PEFT_TYPE_TO_CONFIG_MAPPING” 赋值。
    PEFT_TYPE_TO_CONFIG_MAPPING = None
    # 中文 L0048 设为空值 None给 “PeftType” 赋值。
    PeftType = None
    # 中文 L0049 设为空值 None给 “get_peft_model” 赋值。
    get_peft_model = None

# 中文 L0051 条件 TYPE_CHECKING 成立时进入下方分支。
if TYPE_CHECKING:
    # 中文 L0052 从模块 lerobot.configs.train 导入 TrainPipelineConfig。
    from lerobot.configs.train import TrainPipelineConfig
    # 中文 L0053 从模块 lerobot.datasets.dataset_metadata 导入 LeRobotDatasetMetadata。
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# 中文 L0055 调用 TypeVar，用它的结果给 “T” 赋值。
T = TypeVar("T", bound="PreTrainedPolicy")

# Pinned far above any policy's total size so save_torch_state_dict always emits exactly one
# `model.safetensors` (no shards, no index) — a constant, not a computed byte count.
# 中文 L0059 使用字面值 '1TB'给 “_SINGLE_FILE_SHARD_SIZE” 赋值。
_SINGLE_FILE_SHARD_SIZE = "1TB"


# 中文 L0062 定义函数 _missing_keys。
def _missing_keys(model: nn.Module, loaded_keys: set[str]) -> list[str]:
    """Return the state-dict keys of `model` that no shard filled.

    Tied tensors share storage: a key whose storage another loaded key filled is not missing.
    """
    # 中文 L0067 声明“storage_to_keys”的类型为 dict[tuple, set[str]]，创建字典，用键查找对应的值作为初始值。
    storage_to_keys: dict[tuple, set[str]] = {}

    # 中文 L0069 定义函数 storage_key。
    def storage_key(tensor: Tensor) -> tuple:
        # 中文 L0070 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return (tensor.device, tensor.dtype, tensor.data_ptr(), tensor.shape, tensor.stride())

    # 中文 L0072 调用 model.state_dict，用它的结果给 “state_dict” 赋值。
    state_dict = model.state_dict()
    # 中文 L0073 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key, tensor in state_dict.items():
        # 中文 L0074 条件 tensor.device.type != "meta" and tensor.numel() > 0 成立时进入下方分支。
        if tensor.device.type != "meta" and tensor.numel() > 0:
            # 中文 L0075 调用 add 执行操作。
            storage_to_keys.setdefault(storage_key(tensor), set()).add(key)
    # 中文 L0076 调用 set，用它的结果给 “filled” 赋值。
    filled = set(loaded_keys)
    # 中文 L0077 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key in loaded_keys:
        # 中文 L0078 调用 state_dict.get，用它的结果给 “tensor” 赋值。
        tensor = state_dict.get(key)
        # 中文 L0079 条件 tensor is not None and tensor.device.type != "meta" and tensor.numel() > 0 成立时进入下方分支。
        if tensor is not None and tensor.device.type != "meta" and tensor.numel() > 0:
            # 中文 L0080 用增量赋值更新 filled，运算逻辑见左侧运算符。
            filled |= storage_to_keys.get(storage_key(tensor), set())
    # 中文 L0081 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
    return [key for key in state_dict if key not in filled]


# 中文 L0084 定义类 ActionSelectKwargs，括号列出基类 TypedDict。
class ActionSelectKwargs(TypedDict, total=False):
    # 中文 L0085 声明“noise”（采样起点噪声）的类型为 Tensor | None。
    noise: Tensor | None


# 中文 L0088 定义类 RTCActionSelectKwargs，括号列出基类 TypedDict。
class RTCActionSelectKwargs(TypedDict, total=False):
    # 中文 L0089 原文档中的说明或续行。
    """Keyword arguments Real-Time Chunking passes to `predict_action_chunk`."""

    # 中文 L0091 声明“inference_delay”的类型为 int | None。
    inference_delay: int | None
    # 中文 L0092 声明“prev_chunk_left_over”的类型为 Tensor | None。
    prev_chunk_left_over: Tensor | None
    # 中文 L0093 声明“execution_horizon”的类型为 int | None。
    execution_horizon: int | None


# 中文 L0096 通用策略基类同时拥有 PyTorch 模块行为、Hub 读写能力与抽象接口。
class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    # 中文 L0101 声明“config_class”的类型为 ClassVar[type[PreTrainedConfig]]。
    config_class: ClassVar[type[PreTrainedConfig]]
    # 中文 L0102 声明“name”的类型为 ClassVar[str]。
    name: ClassVar[str]

    # declarative parallelism/acceleration surface
    # Module CLASS names forming the FSDP2 wrap units (and, once wired, the activation-
    # checkpointing units). Resolved onto the accelerate plugin right before
    # `accelerator.prepare()` by `lerobot.distributed.set_fsdp_wrap_modules`; sharded training
    # with no wrap source anywhere fails loudly instead of silently wrapping only the root.
    # 中文 L0109 声明“_fsdp_wrap_modules”的类型为 ClassVar[list[str] | None]，设为空值 None作为初始值。
    _fsdp_wrap_modules: ClassVar[list[str] | None] = None
    # Non-`forward` entry points that must trigger FSDP2 unshard/reshard hooks when called on a
    # sharded policy (registered post-prepare via `torch.distributed.fsdp
    # .register_fsdp_forward_method`); calling them unregistered crashes on mixed Tensor/DTensor.
    # 中文 L0113 声明“_fsdp_forward_methods”的类型为 ClassVar[tuple[str, ...]]，打包成元组，元素顺序由逗号分隔的位置决定作为初始值。
    _fsdp_forward_methods: ClassVar[tuple[str, ...]] = ("select_action", "predict_action_chunk")
    # Capability gate for the (future) activation-checkpointing wiring.
    # 中文 L0115 声明“supports_gradient_checkpointing”的类型为 ClassVar[bool]，使用布尔值 False作为初始值。
    supports_gradient_checkpointing: ClassVar[bool] = False
    # Declarative context-parallel plan (diffusers `ContextParallelModelPlan` semantics:
    # module FQN -> sequence split/gather spec). Reserved for the CP engine round.
    # 中文 L0118 声明“_cp_plan”的类型为 ClassVar[dict[str, Any] | None]，设为空值 None作为初始值。
    _cp_plan: ClassVar[dict[str, Any] | None] = None
    # Attribute names `drop_queued_actions` clears: a `populate_queues`-style `_queues` dict
    # and a bare `_action_queue` deque. A chunking policy using another name must extend this
    # ClassVar (or override the method), otherwise dropping the queue silently does nothing.
    # 中文 L0122 声明“_action_queue_attrs”的类型为 ClassVar[tuple[str, ...]]，打包成元组，元素顺序由逗号分隔的位置决定作为初始值。
    _action_queue_attrs: ClassVar[tuple[str, ...]] = ("_queues", "_action_queue")

    # 中文 L0124 定义方法 __init__。
    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        # 中文 L0125 调用 __init__ 执行操作。
        super().__init__()
        # 中文 L0126 条件 not isinstance(config, PreTrainedConfig) 成立时进入下方分支。
        if not isinstance(config, PreTrainedConfig):
            # 中文 L0127 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0128 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                # 中文 L0129 字符串字面量，供上方的调用、容器或错误消息使用。
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                # 中文 L0130 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            # 中文 L0131 闭合上方尚未结束的括号。
            )
        # 中文 L0132 引用已有的值 config给 “self.config” 赋值。
        self.config = config
        # 中文 L0133 调用 self.register_load_state_dict_pre_hook 执行操作。
        self.register_load_state_dict_pre_hook(rename_legacy_vision_model_keys)

    # 中文 L0135 定义方法 __init_subclass__。
    def __init_subclass__(cls, **kwargs):
        # 中文 L0136 调用 __init_subclass__ 执行操作。
        super().__init_subclass__(**kwargs)
        # 中文 L0137 条件 not getattr(cls, "config_class", None) 成立时进入下方分支。
        if not getattr(cls, "config_class", None):
            # 中文 L0138 抛出 TypeError，中断当前路径并交给上层的异常处理。
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        # 中文 L0139 条件 not getattr(cls, "name", None) 成立时进入下方分支。
        if not getattr(cls, "name", None):
            # 中文 L0140 抛出 TypeError，中断当前路径并交给上层的异常处理。
            raise TypeError(f"Class {cls.__name__} must define 'name'")
        # The rollout stack gates text queries on supports_text_generation(), so a
        # generate_text() override without it is unreachable. Compared through the MRO, so an
        # override inherited from a conforming parent counts.
        # 中文 L0144 条件 ( 成立时进入下方分支。
        if (
            # 中文 L0145 延续上面的语句，cls.generate_text is not PreTrainedPolicy.generate_text。
            cls.generate_text is not PreTrainedPolicy.generate_text
            # 中文 L0146 续写上一行的布尔表达式，按 and / or 的短路规则判断。
            and cls.supports_text_generation is PreTrainedPolicy.supports_text_generation
        # 中文 L0147 闭合上方尚未结束的括号。
        ):
            # 中文 L0148 抛出 TypeError，中断当前路径并交给上层的异常处理。
            raise TypeError(
                # 中文 L0149 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"{cls.__name__} provides generate_text() but not supports_text_generation(). "
                # 中文 L0150 字符串字面量，供上方的调用、容器或错误消息使用。
                "Override supports_text_generation() too (returning True, or a "
                # 中文 L0151 字符串字面量，供上方的调用、容器或错误消息使用。
                "checkpoint-conditional value), otherwise the text head is never used."
            # 中文 L0152 闭合上方尚未结束的括号。
            )

    # Largest weights file that `save_pretrained` writes. The default keeps every policy one
    # `model.safetensors`. A policy whose weights exceed the Hub's 50 GB file limit sets a smaller
    # value (transformers shards at 50 GB since v5.5, 5 GB before) and is written as
    # `model-XXXXX-of-NNNNN.safetensors` shards
    # plus `model.safetensors.index.json`. Only enable this for policies whose loader reaches
    # `_resolve_weight_files`; independent loaders (PI0/PI05/PI0-FAST, XVLA, WALL-X, GR00T)
    # still require their own sharded-checkpoint support. `dcp_to_safetensors` always writes a
    # single `model.safetensors`.
    # 中文 L0162 声明“_max_shard_size”的类型为 ClassVar[int | str]，引用已有的值 _SINGLE_FILE_SHARD_SIZE作为初始值。
    _max_shard_size: ClassVar[int | str] = _SINGLE_FILE_SHARD_SIZE

    # 中文 L0164 定义方法 _save_pretrained。
    def _save_pretrained(self, save_directory: Path) -> None:
        """Serialize this policy's parameters (and config) into `save_directory`.

        Sharding is handled internally: under FSDP2 the full state dict is gathered through a
        COLLECTIVE, so when the policy is sharded this method (via `save_pretrained`) must be
        called on EVERY rank — a rank-0-gated call deadlocks. File writes happen on the main
        process only, in all layouts (single, DDP, sharded).

        The weights go to one `model.safetensors`, or, when they exceed `_max_shard_size`, to
        `model-XXXXX-of-NNNNN.safetensors` shards with a `model.safetensors.index.json` (the
        transformers layout).

        Args:
            save_directory (Path): Target directory for the policy config (`config.json`) and the
                safetensors weight file(s).
        """
        # Lazy imports: the persistence layer pulls in lerobot.distributed only when saving.
        # 中文 L0181 从模块 lerobot.distributed.checkpoint 导入 full_model_state_dict、is_sharded_module。
        from lerobot.distributed.checkpoint import full_model_state_dict, is_sharded_module
        # 中文 L0182 从模块 lerobot.distributed.utils 导入 is_main_process。
        from lerobot.distributed.utils import is_main_process

        # 中文 L0184 用条件表达式在 if 前与 else 后的两个值之间选择给 “model_to_save” 赋值。
        model_to_save = self.module if hasattr(self, "module") else self
        # 中文 L0185 条件 is_sharded_module(model_to_save) 成立时进入下方分支。
        if is_sharded_module(model_to_save):
            # 中文 L0186 调用 logging.info 执行操作。
            logging.info("Gathering the full state dict from all ranks (sharded policy).")
        # 中文 L0187 调用 full_model_state_dict，用它的结果给 “state_dict” 赋值。
        state_dict = full_model_state_dict(model_to_save)  # collective when sharded; {} off-main
        # 中文 L0188 条件 not state_dict or not is_main_process() 成立时进入下方分支。
        if not state_dict or not is_main_process():
            # Sharded: the gather materializes on the main rank only (emptiness check).
            # Non-sharded multi-rank (DDP): every rank holds a full dict — the explicit rank
            # gate prevents N ranks racing on the same files. Single process: never taken.
            # 中文 L0192 退出当前调用并返回 None。
            return
        # 中文 L0193 调用 self.config._save_pretrained 执行操作。
        self.config._save_pretrained(save_directory)
        # 中文 L0194 调用 save_torch_state_dict 执行操作。
        save_torch_state_dict(state_dict, str(save_directory), max_shard_size=self._max_shard_size)

    # 中文 L0196 @classmethod 将类本身传入 cls。用 SmolVLAPolicy 调用时，cls 是这个子类。
    @classmethod
    # 中文 L0197 提供通用的 checkpoint 加载入口，返回构造并加载了权重的策略实例。
    def from_pretrained(
        # 中文 L0198 “cls”（当前类）是当前括号中的一项，逗号将它与下一项分开。
        cls: builtins.type[T],
        # 中文 L0199 “pretrained_name_or_path”是当前括号中的一项，逗号将它与下一项分开。
        pretrained_name_or_path: str | Path,
        # 中文 L0200 后续参数必须按名字传入。
        *,
        # 中文 L0201 为“config”（配置对象）提供这一项的值，注意它所在的参数表或外层表达式。
        config: PreTrainedConfig | None = None,
        # 中文 L0202 为“force_download”提供这一项的值，注意它所在的参数表或外层表达式。
        force_download: bool = False,
        # 中文 L0203 为“resume_download”提供这一项的值，注意它所在的参数表或外层表达式。
        resume_download: bool | None = None,
        # 中文 L0204 为“proxies”提供这一项的值，注意它所在的参数表或外层表达式。
        proxies: dict | None = None,
        # 中文 L0205 为“token”提供这一项的值，注意它所在的参数表或外层表达式。
        token: str | bool | None = None,
        # 中文 L0206 为“cache_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        cache_dir: str | Path | None = None,
        # 中文 L0207 为“local_files_only”提供这一项的值，注意它所在的参数表或外层表达式。
        local_files_only: bool = False,
        # 中文 L0208 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
        revision: str | None = None,
        # 中文 L0209 为“strict”提供这一项的值，注意它所在的参数表或外层表达式。
        strict: bool = False,
        # 中文 L0210 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **kwargs,
    # 中文 L0211 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.

        The weights are read from `model.safetensors`, or from the shards of
        `model.safetensors.index.json` when the checkpoint is sharded, in a local directory or a
        Hub repo.
        """
        # 中文 L0220 如果上游未提供已解析配置，才在这里单独读取模型配置。评测工厂已解析并传入 config。
        if config is None:
            # 中文 L0221 调用 PreTrainedConfig.from_pretrained，用它的结果给 “config”（配置对象） 赋值。
            config = PreTrainedConfig.from_pretrained(
                # 中文 L0222 为“pretrained_name_or_path”提供这一项的值，注意它所在的参数表或外层表达式。
                pretrained_name_or_path=pretrained_name_or_path,
                # 中文 L0223 为“force_download”提供这一项的值，注意它所在的参数表或外层表达式。
                force_download=force_download,
                # 中文 L0224 为“resume_download”提供这一项的值，注意它所在的参数表或外层表达式。
                resume_download=resume_download,
                # 中文 L0225 为“proxies”提供这一项的值，注意它所在的参数表或外层表达式。
                proxies=proxies,
                # 中文 L0226 为“token”提供这一项的值，注意它所在的参数表或外层表达式。
                token=token,
                # 中文 L0227 为“cache_dir”提供这一项的值，注意它所在的参数表或外层表达式。
                cache_dir=cache_dir,
                # 中文 L0228 为“local_files_only”提供这一项的值，注意它所在的参数表或外层表达式。
                local_files_only=local_files_only,
                # 中文 L0229 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
                revision=revision,
                # 中文 L0230 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
                **kwargs,
            # 中文 L0231 闭合上方尚未结束的括号。
            )
        # 中文 L0232 调用 str，用它的结果给 “model_id” 赋值。
        model_id = str(pretrained_name_or_path)
        # 中文 L0233 cls(config, **kwargs) 实例化具体策略，从而执行 SmolVLAPolicy.__init__。
        instance = cls(config, **kwargs)
        # 中文 L0234 读取属性 config.device给 “device”（运算设备） 赋值。
        device = config.device
        # 中文 L0235 条件 device is None 成立时进入下方分支。
        if device is None:
            # PreTrainedConfig.__post_init__ always resolves a device; None here is a programming error.
            # 中文 L0237 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"{type(config).__name__}.device is unset; cannot load the weights")
        # 中文 L0238 按模型来源和 revision 找到单文件权重或分片清单。
        weight_files = cls._resolve_weight_files(
            # 中文 L0239 “model_id”是当前括号中的一项，逗号将它与下一项分开。
            model_id,
            # 中文 L0240 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
            revision=revision,
            # 中文 L0241 为“cache_dir”提供这一项的值，注意它所在的参数表或外层表达式。
            cache_dir=cache_dir,
            # 中文 L0242 为“force_download”提供这一项的值，注意它所在的参数表或外层表达式。
            force_download=force_download,
            # 中文 L0243 为“token”提供这一项的值，注意它所在的参数表或外层表达式。
            token=token,
            # 中文 L0244 为“local_files_only”提供这一项的值，注意它所在的参数表或外层表达式。
            local_files_only=local_files_only,
        # 中文 L0245 闭合上方尚未结束的括号。
        )
        # 中文 L0246 将已定位的 safetensors 权重张量赋给刚构造的策略实例。
        policy = cls._load_weight_files(instance, weight_files, device, strict)

        # 中文 L0248 把策略模块参数和缓冲区放到配置的运行设备。
        policy.to(device)
        # 中文 L0249 切换到评测模式，改变 Dropout 等模块的行为。
        policy.eval()
        # 中文 L0250 将配置完成的策略实例返回给工厂。
        return policy

    # 中文 L0252 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0253 定义方法 _resolve_weight_files。作用是找到单文件权重或分片文件。
    def _resolve_weight_files(cls, model_id: str, **hub_kwargs: Any) -> list[Path]:
        """Return the safetensors file(s) of a checkpoint, downloading them from the Hub if needed.

        Args:
            model_id (str): A local directory or a Hub repo id.
            **hub_kwargs: Download options of `hf_hub_download` (`revision`, `cache_dir`, `token`, ...).

        Returns:
            list[Path]: One `model.safetensors`, or the shards that `model.safetensors.index.json`
            lists, in sorted order.

        Raises:
            FileNotFoundError: Neither file is present.
            HfHubHTTPError: A Hub request failed for a reason other than a missing file.
        """
        # 中文 L0268 条件 os.path.isdir(model_id) 成立时进入下方分支。
        if os.path.isdir(model_id):
            # 中文 L0269 调用 Path，用它的结果给 “root” 赋值。
            root = Path(model_id)
            # 中文 L0270 条件 (root / SAFETENSORS_SINGLE_FILE).is_file() 成立时进入下方分支。
            if (root / SAFETENSORS_SINGLE_FILE).is_file():
                # 中文 L0271 调用 logging.info 执行操作。
                logging.info("Loading weights from local directory")
                # 中文 L0272 创建列表，按给定顺序保留元素作为当前调用的返回值。
                return [root / SAFETENSORS_SINGLE_FILE]
            # 中文 L0273 条件 (root / SAFETENSORS_INDEX_FILE).is_file() 成立时进入下方分支。
            if (root / SAFETENSORS_INDEX_FILE).is_file():
                # 中文 L0274 调用 logging.info 执行操作。
                logging.info("Loading sharded weights from local directory")
                # 中文 L0275 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
                return [root / name for name in safetensors_shard_names(root / SAFETENSORS_INDEX_FILE)]
            # 中文 L0276 抛出 FileNotFoundError，中断当前路径并交给上层的异常处理。
            raise FileNotFoundError(
                # 中文 L0277 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"{root} holds neither {SAFETENSORS_SINGLE_FILE} nor {SAFETENSORS_INDEX_FILE}"
            # 中文 L0278 闭合上方尚未结束的括号。
            )
        # 中文 L0279 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0280 用 Hub 解析单文件权重路径。离线配置下原评测复用本地缓存。
            return [Path(hf_hub_download(repo_id=model_id, filename=SAFETENSORS_SINGLE_FILE, **hub_kwargs))]
        # 中文 L0281 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except EntryNotFoundError:
            # 中文 L0282 空语句，占住当前代码块的位置。
            pass
        # 中文 L0283 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0284 调用 Path，用它的结果给 “index” 赋值。
            index = Path(hf_hub_download(repo_id=model_id, filename=SAFETENSORS_INDEX_FILE, **hub_kwargs))
        # 中文 L0285 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except EntryNotFoundError as e:
            # 中文 L0286 抛出 FileNotFoundError，中断当前路径并交给上层的异常处理。
            raise FileNotFoundError(
                # 中文 L0287 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Neither {SAFETENSORS_SINGLE_FILE} nor {SAFETENSORS_INDEX_FILE} found on the "
                # 中文 L0288 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"HuggingFace Hub in {model_id}"
            # 中文 L0289 延续上面的语句，) from e。本段处理找到单文件权重或分片文件。
            ) from e
        # 中文 L0290 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
        return [
            # 中文 L0291 延续上面的语句，Path(hf_hub_download(repo_id=model_id, filename=name, **hub_kwargs))。本段处理找到单文件权重或分片文件。
            Path(hf_hub_download(repo_id=model_id, filename=name, **hub_kwargs))
            # 中文 L0292 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for name in safetensors_shard_names(index)
        # 中文 L0293 闭合上方尚未结束的括号。
        ]

    # 中文 L0295 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0296 定义方法 _load_weight_files。
    def _load_weight_files(cls, model: T, files: list[Path], map_location: str, strict: bool) -> T:
        """Dispatch a checkpoint to the policy's loader without writing a merged copy.

        Override `_load_checkpoint` to filter, rename or assign tensors across all shards.
        Legacy single-file hooks still work for single-file checkpoints; they must adopt the
        checkpoint hook before loading shards.
        """
        # 中文 L0303 条件 not files 成立时进入下方分支。
        if not files:
            # 中文 L0304 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("A checkpoint must contain at least one weights file")
        # 中文 L0305 单文件 checkpoint 走直接加载路径。本次缓存包含一个 model.safetensors。
        if len(files) == 1:
            # 中文 L0306 把权重文件交给 safetensors 加载接口，并传递设备和严格性参数。
            return cls._load_as_safetensor(model, str(files[0]), map_location, strict)
        # A legacy single-file hook can stay. Shards are refused only when `_load_checkpoint`
        # is still the base implementation.
        # 中文 L0309 计算此行的右侧表达式给 “legacy_hook” 赋值。
        legacy_hook = cls._load_as_safetensor.__func__ is not PreTrainedPolicy._load_as_safetensor.__func__  # type: ignore[attr-defined]
        # 中文 L0310 计算此行的右侧表达式给 “base_checkpoint_hook” 赋值。
        base_checkpoint_hook = (
            # 中文 L0311 延续上面的语句，cls._load_checkpoint.__func__ is PreTrainedPolicy._load_checkpoint.__func__ # type: ignore[attr-defined]。
            cls._load_checkpoint.__func__ is PreTrainedPolicy._load_checkpoint.__func__  # type: ignore[attr-defined]
        # 中文 L0312 闭合上方尚未结束的括号。
        )
        # 中文 L0313 条件 legacy_hook and base_checkpoint_hook 成立时进入下方分支。
        if legacy_hook and base_checkpoint_hook:
            # 中文 L0314 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
            raise NotImplementedError(
                # 中文 L0315 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"{cls.__name__} must implement _load_checkpoint to load sharded weights"
            # 中文 L0316 闭合上方尚未结束的括号。
            )
        # 中文 L0317 调用 cls._load_checkpoint，用它的结果作为当前调用的返回值。
        return cls._load_checkpoint(model, SafetensorsCheckpoint(files), map_location, strict)

    # 中文 L0319 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0320 定义方法 _load_checkpoint。作用是逐个加载权重张量并检查模型状态。
    def _load_checkpoint(
        # 中文 L0321 延续上面的语句，cls, model: T, checkpoint: SafetensorsCheckpoint, map_location: str, strict: bool。本段处理逐个加载权重张量并检查模型状态。
        cls, model: T, checkpoint: SafetensorsCheckpoint, map_location: str, strict: bool
    # 中文 L0322 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> T:
        """Load a checkpoint, checking keys after all shards and load-state hooks have run.

        The default streams one shard at a time. Custom loaders can inspect the whole
        checkpoint through its keys, shapes and tensors without creating a temporary file.
        """
        # 中文 L0328 调用 resolve_safetensors_device，用它的结果给 “device”（运算设备） 赋值。
        device = resolve_safetensors_device(map_location)
        # 中文 L0329 条件 len(checkpoint.files) == 1 成立时进入下方分支。
        if len(checkpoint.files) == 1:
            # 中文 L0330 调用 load_model_as_safetensor，用它的结果给 “(missing_keys, unexpected_keys)” 赋值。
            missing_keys, unexpected_keys = load_model_as_safetensor(
                # 中文 L0331 延续上面的语句，model, str(checkpoint.files[0]), strict=strict, device=device。本段处理逐个加载权重张量并检查模型状态。
                model, str(checkpoint.files[0]), strict=strict, device=device
            # 中文 L0332 闭合上方尚未结束的括号。
            )
        # 中文 L0333 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0334 调用 set，用它的结果给 “expected” 赋值。
            expected = set(model.state_dict())
            # 中文 L0335 调用 set，用它的结果给 “missing” 赋值。
            missing = set(expected)
            # 中文 L0336 声明“unexpected”的类型为 set[str]，调用 set，用它的结果作为初始值。
            unexpected: set[str] = set()
            # 中文 L0337 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for file in checkpoint.files:
                # 中文 L0338 调用 load_file，用它的结果给 “state”（机器人状态或当前步骤的状态） 赋值。
                state = load_file(str(file), device=device)
                # 中文 L0339 调用 model.load_state_dict，用它的结果给 “result” 赋值。
                result = model.load_state_dict(state, strict=False)
                # 中文 L0340 调用 missing.intersection_update 执行操作。
                missing.intersection_update(result.missing_keys)
                # 中文 L0341 调用 unexpected.update，把新条目写入已有容器。
                unexpected.update(result.unexpected_keys)
                # 中文 L0342 延续上面的语句，del state。本段处理逐个加载权重张量并检查模型状态。
                del state
            # 中文 L0343 调用 _missing_keys，用它的结果给 “missing_keys” 赋值。
            missing_keys = _missing_keys(model, expected - missing)
            # 中文 L0344 调用 sorted，用它的结果给 “unexpected_keys” 赋值。
            unexpected_keys = sorted(unexpected)
            # 中文 L0345 条件 strict and (missing_keys or unexpected_keys) 成立时进入下方分支。
            if strict and (missing_keys or unexpected_keys):
                # 中文 L0346 抛出 RuntimeError，中断当前路径并交给上层的异常处理。
                raise RuntimeError(
                    # 中文 L0347 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"Error(s) in loading state_dict for {type(model).__name__}: "
                    # 中文 L0348 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"missing keys {missing_keys}, unexpected keys {unexpected_keys}"
                # 中文 L0349 闭合上方尚未结束的括号。
                )
        # 中文 L0350 调用 log_model_loading_keys 执行操作。
        log_model_loading_keys(missing_keys, unexpected_keys)
        # 中文 L0351 引用已有的值 model作为当前调用的返回值。
        return model

    # 中文 L0353 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0354 定义方法 _load_as_safetensor。
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        # 中文 L0355 原文档中的说明或续行。
        """Compatibility entry point for a single weights file."""
        # 中文 L0356 调用 cls._load_checkpoint，用它的结果作为当前调用的返回值。
        return cls._load_checkpoint(model, SafetensorsCheckpoint([Path(model_file)]), map_location, strict)

    # 中文 L0358 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0359 定义方法 get_optim_params。
    def get_optim_params(self) -> OptimizerParams:
        """
        Returns the policy-specific parameters (parameters or param groups) to be passed on to the optimizer.
        """
        # 中文 L0363 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0365 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0366 定义方法 reset。
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        # 中文 L0371 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0373 定义方法 drop_queued_actions。
    def drop_queued_actions(self) -> None:
        """Discard actions precomputed by earlier ``select_action`` calls.

        Forces a fresh forward pass on the next ``select_action`` so a mid-episode conditioning
        change (e.g. a new instruction) takes effect at once instead of after the queue drains.
        Unlike :meth:`reset`, the rest of the episode state is kept.  Call it from the thread
        that calls ``select_action``.  Clears the queues named in :attr:`_action_queue_attrs`;
        a policy that keeps no action queue inherits a no-op.
        """
        # 中文 L0382 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for attr in self._action_queue_attrs:
            # 中文 L0383 调用 getattr，用它的结果给 “queue” 赋值。
            queue = getattr(self, attr, None)
            # 中文 L0384 条件 isinstance(queue, dict) 成立时进入下方分支。
            if isinstance(queue, dict):
                # populate_queues-style dict: clear only ACTION, not the observation history.
                # 中文 L0386 条件 ACTION in queue 成立时进入下方分支。
                if ACTION in queue:
                    # 中文 L0387 调用 clear 执行操作。
                    queue[ACTION].clear()
            # 中文 L0388 前面的分支均未命中时，再判断 queue is not None。
            elif queue is not None:
                # 中文 L0389 调用 queue.clear 执行操作。
                queue.clear()

    # 中文 L0391 定义方法 count_queued_actions。
    def count_queued_actions(self) -> int:
        """Number of actions already computed and waiting to be served.

        0 for a policy that keeps no action queue (never chunks) or whose queue is
        currently empty (about to compute a fresh chunk). Reads the same queues
        :meth:`drop_queued_actions` clears, named in :attr:`_action_queue_attrs`.
        """
        # 中文 L0398 使用字面值 0给 “total” 赋值。
        total = 0
        # 中文 L0399 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for attr in self._action_queue_attrs:
            # 中文 L0400 调用 getattr，用它的结果给 “queue” 赋值。
            queue = getattr(self, attr, None)
            # 中文 L0401 条件 isinstance(queue, dict) 成立时进入下方分支。
            if isinstance(queue, dict):
                # 中文 L0402 调用 queue.get，用它的结果给 “action_queue” 赋值。
                action_queue = queue.get(ACTION)
                # 中文 L0403 用增量赋值更新 total，运算逻辑见左侧运算符。
                total += len(action_queue) if action_queue is not None else 0
            # 中文 L0404 前面的分支均未命中时，再判断 queue is not None。
            elif queue is not None:
                # 中文 L0405 用增量赋值更新 total，运算逻辑见左侧运算符。
                total += len(queue)
        # 中文 L0406 引用已有的值 total作为当前调用的返回值。
        return total

    # 中文 L0408 定义方法 supports_rtc。
    def supports_rtc(self) -> bool:
        # 中文 L0409 原文档中的说明或续行。
        """Whether this policy implements Real-Time Chunking inference semantics."""
        # 中文 L0410 使用布尔值 False作为当前调用的返回值。
        return False

    # 中文 L0412 定义方法 validate_chunk_input_features。
    def validate_chunk_input_features(self, features: tuple[FeatureSpec, ...]) -> None:
        """Validate canonical inputs before serving, with exact shapes by default.

        Policies that explicitly resize or mask missing inputs may override this
        contract. Such exceptions belong with the policy, never the transport.
        The negotiated wire schema still fixes every supplied feature's shape.
        """
        # 中文 L0419 组合多个布尔判断，按 Python 的短路规则求值给 “expected” 赋值。
        expected = self.config.input_features or {}
        # 中文 L0420 用字典推导式从可迭代对象逐项取值并构造结果给 “supplied” 赋值。
        supplied = {feature.name: feature for feature in features}
        # 中文 L0421 条件 supplied.keys() != expected.keys() 成立时进入下方分支。
        if supplied.keys() != expected.keys():
            # 中文 L0422 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("Canonical feature names must exactly match policy input_features.")
        # 中文 L0423 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for name, policy_feature in expected.items():
            # 中文 L0424 用方括号按键或索引读取 supplied给 “feature”（输入或输出特征描述） 赋值。
            feature = supplied[name]
            # 中文 L0425 条件 policy_feature.type == FeatureType.VISUAL 成立时进入下方分支。
            if policy_feature.type == FeatureType.VISUAL:
                # 中文 L0426 读取属性 policy_feature.shape给 “shape”（各轴的长度） 赋值。
                shape = policy_feature.shape
                # 中文 L0427 条件 ( 成立时进入下方分支。
                if (
                    # 中文 L0428 延续上面的语句，len(shape) != 3。
                    len(shape) != 3
                    # 中文 L0429 续写上一行的布尔表达式，按 and / or 的短路规则判断。
                    or feature.kind != "rgb"
                    # 中文 L0430 续写上一行的布尔表达式，按 and / or 的短路规则判断。
                    or feature.shape != (shape[1], shape[2], shape[0])
                # 中文 L0431 闭合上方尚未结束的括号。
                ):
                    # 中文 L0432 抛出 ValueError，中断当前路径并交给上层的异常处理。
                    raise ValueError(f"RGB feature shape differs from checkpoint: {name}.")
            # 中文 L0433 前面的分支均未命中时，再判断 feature.kind != "tensor" or feature.shape != tuple(policy_feature.shape)。
            elif feature.kind != "tensor" or feature.shape != tuple(policy_feature.shape):
                # 中文 L0434 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(f"Tensor feature shape differs from checkpoint: {name}.")

    # 中文 L0436 定义方法 chunk_inference_spec。
    def chunk_inference_spec(self) -> ChunkPolicySpec:
        """Declare the default current-observation chunk-serving author contract.

        Inheriting this implementation declares that ``predict_action_chunk`` does
        every preparation needed by direct callers, accepts the complete current
        observation, and returns ``[B, chunk_size, A]`` in model coordinates. Its
        canonical processor pair must support whole chunks. ``n_action_steps`` is
        the ordinary playback slice; RTC retains the full prediction horizon.

        A policy requiring select_action-only preparation, other action layouts,
        or temporal sampling must override this method and reject unsupported
        configurations or provide a custom runner. Method presence or a successful
        warmup is not evidence for this contract. Policy/processor state is allowed
        only within an exclusive session and reset() must clear it for a new run.
        """
        # 中文 L0451 读取属性 self.config给 “config”（配置对象） 赋值。
        config = self.config
        # 中文 L0452 条件 config.n_obs_steps != 1 成立时进入下方分支。
        if config.n_obs_steps != 1:
            # 中文 L0453 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("Chunk inference requires n_obs_steps=1; observation history is unsupported.")
        # 中文 L0454 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for field_name in (
            # 中文 L0455 字符串字面量，供上方的调用、容器或错误消息使用。
            "observation_delta_indices",
            # 中文 L0456 字符串字面量，供上方的调用、容器或错误消息使用。
            "image_observation_delta_indices",
            # 中文 L0457 字符串字面量，供上方的调用、容器或错误消息使用。
            "state_observation_delta_indices",
        # 中文 L0458 闭合上方尚未结束的括号。
        ):
            # 中文 L0459 调用 getattr，用它的结果给 “indices” 赋值。
            indices = getattr(config, field_name, None)
            # 中文 L0460 条件 indices is not None and list(indices) != [0] 成立时进入下方分支。
            if indices is not None and list(indices) != [0]:
                # 中文 L0461 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(f"Chunk inference cannot sample temporal history ({field_name}).")
        # 中文 L0462 条件 getattr(config, "temporal_ensemble_coeff", None) is not None 成立时进入下方分支。
        if getattr(config, "temporal_ensemble_coeff", None) is not None:
            # 中文 L0463 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("Chunk inference does not support temporal ensembling.")
        # 中文 L0464 条件 getattr(config, "use_visual_memory", False) or getattr(config, "use_proprioceptive_memory", False) 成立时进入下方分支。
        if getattr(config, "use_visual_memory", False) or getattr(config, "use_proprioceptive_memory", False):
            # 中文 L0465 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0466 字符串字面量，供上方的调用、容器或错误消息使用。
                "Chunk inference cannot sample observation memory at the required control cadence."
            # 中文 L0467 闭合上方尚未结束的括号。
            )
        # 中文 L0468 调用 getattr，用它的结果给 “prediction_steps” 赋值。
        prediction_steps = getattr(config, "chunk_size", None)
        # 中文 L0469 条件 not isinstance(prediction_steps, int) or prediction_steps <= 0 成立时进入下方分支。
        if not isinstance(prediction_steps, int) or prediction_steps <= 0:
            # 中文 L0470 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0471 字符串字面量，供上方的调用、容器或错误消息使用。
                "The default chunk runner requires a positive chunk_size; provide a custom runner."
            # 中文 L0472 闭合上方尚未结束的括号。
            )
        # 中文 L0473 调用 getattr，用它的结果给 “execution_steps” 赋值。
        execution_steps = getattr(config, "n_action_steps", prediction_steps)
        # 中文 L0474 创建列表，按给定顺序保留元素给 “modes” 赋值。
        modes = [ExecutionMode.CHUNK]
        # 中文 L0475 调用 int，用它的结果给 “training_max_delay” 赋值。
        training_max_delay = int(getattr(config, "rtc_training_max_delay", 0))
        # 中文 L0476 条件 self.supports_rtc() 成立时进入下方分支。
        if self.supports_rtc():
            # 中文 L0477 调用 modes.append，在列表末尾追加一个元素。
            modes.append(ExecutionMode.RTC_GUIDED)
            # 中文 L0478 条件 training_max_delay > 0 成立时进入下方分支。
            if training_max_delay > 0:
                # 中文 L0479 调用 modes.append，在列表末尾追加一个元素。
                modes.append(ExecutionMode.RTC_TRAINED)
        # 中文 L0480 调用 ChunkPolicySpec，用它的结果作为当前调用的返回值。
        return ChunkPolicySpec(
            # 中文 L0481 为“prediction_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            prediction_steps=prediction_steps,
            # 中文 L0482 为“execution_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            execution_steps=execution_steps,
            # 中文 L0483 为“modes”提供这一项的值，注意它所在的参数表或外层表达式。
            modes=tuple(modes),
            # 中文 L0484 为“training_max_delay”提供这一项的值，注意它所在的参数表或外层表达式。
            training_max_delay=training_max_delay,
        # 中文 L0485 闭合上方尚未结束的括号。
        )

    # 中文 L0487 定义方法 supports_text_generation。
    def supports_text_generation(self) -> bool:
        # 中文 L0488 原文档中的说明或续行。
        """Whether this policy implements :meth:`generate_text` (override both together)."""
        # 中文 L0489 使用布尔值 False作为当前调用的返回值。
        return False

    # 中文 L0491 定义方法 generate_text。
    def generate_text(self, batch: dict[str, Any]) -> str:
        """Run the policy's text head on a preprocessed observation batch.

        The request rides on ``batch`` as complementary data (:data:`~lerobot.utils.constants.QUERY_KIND`
        / ``QUERY_TEXT``); a ``next_subtask`` reply is fed straight into ``set_task``, so it must be
        exactly one subtask, not a plan or a numbered list.  Returns the generated text, and must not
        mutate action-producing state (queues, observation history).
        """
        # 中文 L0499 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError(
            # 中文 L0500 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"{type(self).__name__} has no text head — it cannot answer questions or plan subtasks."
        # 中文 L0501 闭合上方尚未结束的括号。
        )

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    # 中文 L0504 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0505 定义方法 forward。
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        # 中文 L0515 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0517 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0518 定义方法 predict_action_chunk。
    def predict_action_chunk(
        # 中文 L0519 延续上面的语句，self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]。
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    # 中文 L0520 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor | PolicyOutput:
        """Returns the action chunk (for action chunking policies) for a given observation, potentially in batch mode.

        Child classes using action chunking should use this method within `select_action` to form the action chunk
        cached for selection.

        A policy that supports a `noise` keyword starts its sampling from that tensor, shaped as the policy would
        draw it. When `noise` is None, the policy draws it as usual.

        May return a `PolicyOutput` carrying the chunk and its prediction, as for `select_action`.
        """
        # 中文 L0531 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0533 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0534 定义方法 select_action。
    def select_action(
        # 中文 L0535 延续上面的语句，self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]。
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    # 中文 L0536 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor | PolicyOutput:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.

        Returns a bare action tensor, or a `PolicyOutput`: a batch, keyed like the input batch, holding
        the action under ``"action"`` and what the policy expects or believes besides it under
        ``"prediction"`` (a `PolicyPrediction`). The prediction is batched like the action, one value
        per environment, and each entry is keyed by what it refers to: predicted frames by the
        observation key they predict, language by annotation style ("subtask", "plan", "memory"),
        boxes (annotation VQA schema) by the camera they are drawn on. Callers normalize the two
        return types with `lerobot.processor.policy_output_to_transition`; `--display_data` shows
        the prediction of the first environment.

        A prediction holds values for this step: one spanning a chunk (e.g. an imagined clip) is
        returned piece by piece, with the action each piece belongs to. Each entry stays displayed
        until it is returned again. Predicting is opt-in through a config field of the policy's
        choosing, and must not change the actions; text that should steer the policy goes
        through `generate_text`.
        """
        # 中文 L0557 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0559 定义方法 push_model_to_hub。
    def push_model_to_hub(
        # 中文 L0560 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0561 “cfg”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
        cfg: TrainPipelineConfig,
        # 中文 L0562 为“peft_model”提供这一项的值，注意它所在的参数表或外层表达式。
        peft_model=None,
        # 中文 L0563 为“state_dict”提供这一项的值，注意它所在的参数表或外层表达式。
        state_dict: dict[str, Tensor] | None = None,
        # 中文 L0564 为“dataset_meta”提供这一项的值，注意它所在的参数表或外层表达式。
        dataset_meta: LeRobotDatasetMetadata | None = None,
    # 中文 L0565 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> None:
        """Publish this policy to the Hub.

        Deprecated: use :func:`lerobot.common.train_utils.publish_trained_model` instead, which
        also publishes the pre/post-processors alongside the model.

        Args:
            cfg (TrainPipelineConfig): The training config; saved as `train_config.json` and
                used to render the model card.
            peft_model: The PEFT wrapper when training adapters, whose weights replace the full
                model weights in the published repo. Defaults to None.
            state_dict (dict[str, Tensor] | None): Ignored; weights are now gathered internally
                when the policy is sharded. Defaults to None.
            dataset_meta (LeRobotDatasetMetadata | None): Dataset metadata for the model card,
                if available. Defaults to None.
        """
        # 中文 L0581 从模块 lerobot.common.train_utils 导入 publish_trained_model。
        from lerobot.common.train_utils import publish_trained_model

        # 中文 L0583 调用 warnings.warn 执行操作。
        warnings.warn(
            # 中文 L0584 字符串字面量，供上方的调用、容器或错误消息使用。
            "PreTrainedPolicy.push_model_to_hub is deprecated and will be removed in a future "
            # 中文 L0585 字符串字面量，供上方的调用、容器或错误消息使用。
            "version. Use lerobot.common.train_utils.publish_trained_model(cfg, model, "
            # 中文 L0586 字符串字面量，供上方的调用、容器或错误消息使用。
            "preprocessor, postprocessor, dataset_meta) instead.",
            # 中文 L0587 “FutureWarning”是当前括号中的一项，逗号将它与下一项分开。
            FutureWarning,
            # 中文 L0588 为“stacklevel”提供这一项的值，注意它所在的参数表或外层表达式。
            stacklevel=2,
        # 中文 L0589 闭合上方尚未结束的括号。
        )
        # 中文 L0590 条件 state_dict is not None 成立时进入下方分支。
        if state_dict is not None:
            # 中文 L0591 调用 warnings.warn 执行操作。
            warnings.warn(
                # 中文 L0592 字符串字面量，供上方的调用、容器或错误消息使用。
                "The `state_dict` argument is ignored: sharded weights are gathered internally "
                # 中文 L0593 字符串字面量，供上方的调用、容器或错误消息使用。
                "when the policy is saved.",
                # 中文 L0594 “FutureWarning”是当前括号中的一项，逗号将它与下一项分开。
                FutureWarning,
                # 中文 L0595 为“stacklevel”提供这一项的值，注意它所在的参数表或外层表达式。
                stacklevel=2,
            # 中文 L0596 闭合上方尚未结束的括号。
            )
        # 中文 L0597 调用 publish_trained_model 执行操作。
        publish_trained_model(cfg, self, None, None, dataset_meta, peft_model=peft_model)

    # 中文 L0599 定义方法 wrap_with_peft。
    def wrap_with_peft(
        # 中文 L0600 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0601 为“peft_config”提供这一项的值，注意它所在的参数表或外层表达式。
        peft_config=None,
        # 中文 L0602 为“peft_cli_overrides”提供这一项的值，注意它所在的参数表或外层表达式。
        peft_cli_overrides: dict | None = None,
    # 中文 L0603 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> PreTrainedPolicy:
        """
        Wrap this policy with PEFT adapters for parameter-efficient fine-tuning.

        This method is the single entry point for PEFT integration. Subclasses should
        override `_get_default_peft_targets()` to provide default target modules, and
        `_validate_peft_config()` for policy-specific validation.

        Args:
            peft_config: Optional PEFT adapter configuration (e.g., LoraConfig).
                If provided, used directly (with CLI overrides applied).
            peft_cli_overrides: Optional dict of CLI overrides (method_type, target_modules, r, etc.)
                These are merged with policy defaults to build the final config.
        """
        # 中文 L0617 调用 require_package 执行操作。
        require_package("peft", extra="peft")

        # If user provided a complete config, use it directly (with overrides)
        # 中文 L0620 条件 peft_config is not None 成立时进入下方分支。
        if peft_config is not None:
            # 中文 L0621 引用已有的值 peft_config给 “final_config” 赋值。
            final_config = peft_config
            # 中文 L0622 条件 peft_cli_overrides 成立时进入下方分支。
            if peft_cli_overrides:
                # 中文 L0623 调用 self._apply_peft_cli_overrides，用它的结果给 “final_config” 赋值。
                final_config = self._apply_peft_cli_overrides(final_config, peft_cli_overrides)
        # 中文 L0624 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # Build config from defaults + CLI overrides
            # 中文 L0626 调用 self._build_peft_config，用它的结果给 “final_config” 赋值。
            final_config = self._build_peft_config(peft_cli_overrides or {})

        # Validate the configuration
        # 中文 L0629 调用 self._validate_peft_config 执行操作。
        self._validate_peft_config(final_config)

        # Freeze base parameters, only adapter params will be trained
        # 中文 L0632 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for p in self.parameters():
            # 中文 L0633 调用 p.requires_grad_ 执行操作。
            p.requires_grad_(False)

        # Store pretrained path for PEFT's base_model_name_or_path
        # 中文 L0636 条件 self.config.pretrained_path 成立时进入下方分支。
        if self.config.pretrained_path:
            # 中文 L0637 调用 str，用它的结果给 “self.name_or_path” 赋值。
            self.name_or_path = str(self.config.pretrained_path)

        # Wrap with PEFT
        # 中文 L0640 调用 get_peft_model，用它的结果给 “peft_model” 赋值。
        peft_model = get_peft_model(self, final_config, revision=self.config.pretrained_revision)

        # Mark config as using PEFT for proper loading later
        # 中文 L0643 使用布尔值 True给 “peft_model.config.use_peft” 赋值。
        peft_model.config.use_peft = True

        # 中文 L0645 调用 logging.info 执行操作。
        logging.info(f"Wrapped {self.name} with PEFT ({type(final_config).__name__})")
        # 中文 L0646 引用已有的值 peft_model作为当前调用的返回值。
        return peft_model

    # 中文 L0648 定义方法 _get_default_peft_targets。
    def _get_default_peft_targets(self) -> dict[str, Any] | None:
        """
        Return default PEFT target modules for this policy.

        Override this in subclasses to provide policy-specific defaults. These defaults
        are PEFT-method agnostic - they only specify which modules to target.

        """
        # 中文 L0656 设为空值 None作为当前调用的返回值。
        return None

    # 中文 L0658 定义方法 _validate_peft_config。
    def _validate_peft_config(self, peft_config) -> None:
        """
        Validate the PEFT configuration for this policy.

        Override this in subclasses to add policy-specific validation or warnings.
        The default implementation checks that a pretrained_path exists.

        Args:
            peft_config: The PEFT configuration to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # 中文 L0671 条件 not self.config.pretrained_path 成立时进入下方分支。
        if not self.config.pretrained_path:
            # 中文 L0672 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0673 字符串字面量，供上方的调用、容器或错误消息使用。
                "Training from scratch using PEFT is unlikely to yield good results. "
                # 中文 L0674 字符串字面量，供上方的调用、容器或错误消息使用。
                "Supply a `policy.pretrained_path` to fine-tune an existing model."
            # 中文 L0675 闭合上方尚未结束的括号。
            )

    # 中文 L0677 定义方法 _preprocess_peft_cli_overrides。
    def _preprocess_peft_cli_overrides(self, cli_overrides: dict, peft_method_type) -> dict:
        """
        Preprocess CLI overrides: rename keys and handle method-specific init_type.

        Args:
            cli_overrides: Dict of CLI options (will be copied, not mutated).
            peft_method_type: The PeftType enum value for the PEFT method.

        Returns:
            Preprocessed dict with renamed keys and init_type mapped to method-specific key.
        """
        # 中文 L0688 调用 require_package 执行操作。
        require_package("peft", extra="peft")

        # 中文 L0690 调用 cli_overrides.copy，用它的结果给 “cli_overrides” 赋值。
        cli_overrides = cli_overrides.copy()

        # Handle the full_training_modules -> modules_to_save rename
        # 中文 L0693 条件 "full_training_modules" in cli_overrides 成立时进入下方分支。
        if "full_training_modules" in cli_overrides:
            # 中文 L0694 调用 cli_overrides.pop，用它的结果给 “cli_overrides['modules_to_save']” 赋值。
            cli_overrides["modules_to_save"] = cli_overrides.pop("full_training_modules")

        # Remove method_type as it's handled separately
        # 中文 L0697 调用 cli_overrides.pop 执行操作。
        cli_overrides.pop("method_type", None)

        # Handle init_type specially based on PEFT method
        # 中文 L0700 调用 cli_overrides.pop，用它的结果给 “init_type” 赋值。
        init_type = cli_overrides.pop("init_type", None)
        # 中文 L0701 条件 init_type is not None 成立时进入下方分支。
        if init_type is not None:
            # 中文 L0702 条件 peft_method_type == PeftType.LORA 成立时进入下方分支。
            if peft_method_type == PeftType.LORA:
                # 中文 L0703 引用已有的值 init_type给 “cli_overrides['init_lora_weights']” 赋值。
                cli_overrides["init_lora_weights"] = init_type
            # 中文 L0704 前面的分支均未命中时，再判断 peft_method_type == PeftType.MISS。
            elif peft_method_type == PeftType.MISS:
                # 中文 L0705 引用已有的值 init_type给 “cli_overrides['init_weights']” 赋值。
                cli_overrides["init_weights"] = init_type
            # 中文 L0706 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0707 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(f"Init type '{init_type}' unknown for PEFT method {peft_method_type}.")

        # 中文 L0709 引用已有的值 cli_overrides作为当前调用的返回值。
        return cli_overrides

    # 中文 L0711 定义方法 _build_peft_config。
    def _build_peft_config(self, cli_overrides: dict):
        # 中文 L0712 原文档中的说明或续行。
        """Build a PEFT config from policy defaults and CLI overrides."""
        # 中文 L0713 调用 require_package 执行操作。
        require_package("peft", extra="peft")

        # Determine PEFT method type (default to LORA)
        # 中文 L0716 组合多个布尔判断，按 Python 的短路规则求值给 “method_type_str” 赋值。
        method_type_str = cli_overrides.get("method_type") or "lora"
        # 中文 L0717 用方括号按键或索引读取 PeftType给 “peft_method_type” 赋值。
        peft_method_type = PeftType[method_type_str.upper()]
        # 中文 L0718 用方括号按键或索引读取 PEFT_TYPE_TO_CONFIG_MAPPING给 “peft_config_cls” 赋值。
        peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

        # Preprocess CLI overrides
        # 中文 L0721 调用 self._preprocess_peft_cli_overrides，用它的结果给 “cli_overrides” 赋值。
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # Start with policy defaults, apply CLI overrides
        # 中文 L0724 调用 dict，用它的结果给 “config_dict” 赋值。
        config_dict = dict(self._get_default_peft_targets() or {})
        # 中文 L0725 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, value in cli_overrides.items():
            # 中文 L0726 条件 value is not None 成立时进入下方分支。
            if value is not None:
                # 中文 L0727 引用已有的值 value给 “config_dict[key]” 赋值。
                config_dict[key] = value

        # Ensure we have target_modules
        # 中文 L0730 条件 not config_dict.get("target_modules") 成立时进入下方分支。
        if not config_dict.get("target_modules"):
            # 中文 L0731 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0732 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Policy '{self.name}' does not define default target_modules. "
                # 中文 L0733 字符串字面量，供上方的调用、容器或错误消息使用。
                "Please pass --peft.target_modules explicitly."
            # 中文 L0734 闭合上方尚未结束的括号。
            )

        # 中文 L0736 调用 peft_config_cls，用它的结果作为当前调用的返回值。
        return peft_config_cls(**config_dict)

    # 中文 L0738 定义方法 _apply_peft_cli_overrides。
    def _apply_peft_cli_overrides(self, peft_config, cli_overrides: dict):
        # 中文 L0739 原文档中的说明或续行。
        """Apply CLI overrides to an existing PEFT config."""
        # 中文 L0740 调用 require_package 执行操作。
        require_package("peft", extra="peft")

        # Get method type from existing config or CLI override
        # 中文 L0743 调用 cli_overrides.get，用它的结果给 “method_type_str” 赋值。
        method_type_str = cli_overrides.get("method_type")
        # 中文 L0744 条件 method_type_str 成立时进入下方分支。
        if method_type_str:
            # 中文 L0745 用方括号按键或索引读取 PeftType给 “peft_method_type” 赋值。
            peft_method_type = PeftType[method_type_str.upper()]
            # 中文 L0746 用方括号按键或索引读取 PEFT_TYPE_TO_CONFIG_MAPPING给 “peft_config_cls” 赋值。
            peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]
        # 中文 L0747 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0748 调用 PeftType，用它的结果给 “peft_method_type” 赋值。
            peft_method_type = PeftType(peft_config.peft_type)
            # 中文 L0749 调用 type，用它的结果给 “peft_config_cls” 赋值。
            peft_config_cls = type(peft_config)

        # Preprocess CLI overrides
        # 中文 L0752 调用 self._preprocess_peft_cli_overrides，用它的结果给 “cli_overrides” 赋值。
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # Start with existing config, apply CLI overrides
        # 中文 L0755 用字典推导式从可迭代对象逐项取值并构造结果给 “config_dict” 赋值。
        config_dict = {k: v for k, v in dataclasses.asdict(peft_config).items() if not k.startswith("_")}
        # 中文 L0756 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, value in cli_overrides.items():
            # 中文 L0757 条件 value is not None 成立时进入下方分支。
            if value is not None:
                # 中文 L0758 引用已有的值 value给 “config_dict[key]” 赋值。
                config_dict[key] = value

        # 中文 L0760 调用 peft_config_cls，用它的结果作为当前调用的返回值。
        return peft_config_cls(**config_dict)
