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
# 中文 L0014 导入模块 abc。
import abc
# 中文 L0015 导入模块 builtins。
import builtins
# 中文 L0016 导入模块 json。
import json
# 中文 L0017 导入模块 os。
import os
# 中文 L0018 导入模块 tempfile。
import tempfile
# 中文 L0019 从模块 dataclasses 导入 dataclass、field。
from dataclasses import dataclass, field
# 中文 L0020 从模块 logging 导入 getLogger。
from logging import getLogger
# 中文 L0021 从模块 pathlib 导入 Path。
from pathlib import Path
# 中文 L0022 从模块 typing 导入 Any、TypeVar。
from typing import Any, TypeVar

# 中文 L0024 导入模块 draccus。
import draccus
# 中文 L0025 导入模块 torch。
import torch
# 中文 L0026 从模块 huggingface_hub 导入 hf_hub_download。
from huggingface_hub import hf_hub_download
# 中文 L0027 从模块 huggingface_hub.constants 导入 CONFIG_NAME。
from huggingface_hub.constants import CONFIG_NAME
# 中文 L0028 从模块 huggingface_hub.errors 导入 HfHubHTTPError。
from huggingface_hub.errors import HfHubHTTPError

# 中文 L0030 从模块 lerobot.optim 导入 LRSchedulerConfig、OptimizerConfig。
from lerobot.optim import LRSchedulerConfig, OptimizerConfig
# 中文 L0031 从模块 lerobot.utils.constants 导入 ACTION、OBS_STATE。
from lerobot.utils.constants import ACTION, OBS_STATE
# 中文 L0032 从模块 lerobot.utils.device_utils 导入 auto_select_torch_device、is_amp_available、is_torch_device_available。
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
# 中文 L0033 从模块 lerobot.utils.hub 导入 HubMixin。
from lerobot.utils.hub import HubMixin

# 中文 L0035 从模块 .types 导入 FeatureType、PolicyFeature。
from .types import FeatureType, PolicyFeature

# 中文 L0037 调用 TypeVar，用它的结果给 “T” 赋值。
T = TypeVar("T", bound="PreTrainedConfig")
# 中文 L0038 调用 getLogger，用它的结果给 “logger” 赋值。
logger = getLogger(__name__)


# 中文 L0041 数据类装饰器。Python 根据字段生成初始化等方法。
@dataclass
# 中文 L0042 通用策略配置基类，继承配置类型注册机制、Hub 接口和抽象基类接口。
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
    """

    # 中文 L0057 声明“n_obs_steps”的类型为 int，使用字面值 1作为初始值。
    n_obs_steps: int = 1

    # `input_features` can be set to None/null in order to infer those values from the dataset.
    # 中文 L0060 声明“input_features”的类型为 dict[str, PolicyFeature] | None，调用 field，用它的结果作为初始值。
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    # 中文 L0061 声明“output_features”的类型为 dict[str, PolicyFeature] | None，调用 field，用它的结果作为初始值。
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    # 中文 L0063 声明“device”（运算设备）的类型为 str | None，设为空值 None作为初始值。
    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # Parameter storage dtype. None uses PyTorch's default when constructing a policy.
    # 中文 L0065 声明“dtype”（元素类型）的类型为 torch.dtype | None，设为空值 None作为初始值。
    dtype: torch.dtype | None = None  # e.g. torch.float32, torch.bfloat16
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    # 中文 L0068 声明“use_amp”的类型为 bool，使用布尔值 False作为初始值。
    use_amp: bool = False

    # Whether the policy employed PEFT for training.
    # 中文 L0071 声明“use_peft”的类型为 bool，使用布尔值 False作为初始值。
    use_peft: bool = False

    # 中文 L0073 声明“push_to_hub”的类型为 bool，使用布尔值 True作为初始值。
    push_to_hub: bool = True  # TODO: rename; shadows HubMixin.push_to_hub()
    # 中文 L0074 声明“repo_id”（Hub 仓库标识）的类型为 str | None，设为空值 None作为初始值。
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    # 中文 L0077 声明“private”的类型为 bool | None，设为空值 None作为初始值。
    private: bool | None = None
    # Add tags to your policy on the hub.
    # 中文 L0079 声明“tags”的类型为 list[str] | None，设为空值 None作为初始值。
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    # 中文 L0081 声明“license”的类型为 str | None，设为空值 None作为初始值。
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    # 中文 L0084 声明“pretrained_path”的类型为 Path | None，设为空值 None作为初始值。
    pretrained_path: Path | None = None
    # Optional Hub revision (commit hash, branch, or tag) to pin the pretrained model version.
    # 中文 L0086 声明“pretrained_revision”的类型为 str | None，设为空值 None作为初始值。
    pretrained_revision: str | None = None

    # 中文 L0088 定义方法 __post_init__。
    def __post_init__(self) -> None:
        # 中文 L0089 条件 self.dtype is not None and not isinstance(self.dtype, torch.dtype) 成立时进入下方分支。
        if self.dtype is not None and not isinstance(self.dtype, torch.dtype):
            # 中文 L0090 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"config.dtype must be a torch.dtype or None, got {self.dtype!r}.")

        # 中文 L0092 条件 not self.device or not is_torch_device_available(self.device) 成立时进入下方分支。
        if not self.device or not is_torch_device_available(self.device):
            # 中文 L0093 调用 auto_select_torch_device，用它的结果给 “auto_device” 赋值。
            auto_device = auto_select_torch_device()
            # 中文 L0094 调用 logger.warning 执行操作。
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            # 中文 L0095 读取属性 auto_device.type给 “self.device” 赋值。
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        # 中文 L0098 条件 self.use_amp and not is_amp_available(self.device) 成立时进入下方分支。
        if self.use_amp and not is_amp_available(self.device):
            # 中文 L0099 调用 logger.warning 执行操作。
            logger.warning(
                # 中文 L0100 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            # 中文 L0101 闭合上方尚未结束的括号。
            )
            # 中文 L0102 使用布尔值 False给 “self.use_amp” 赋值。
            self.use_amp = False

    # 中文 L0104 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0105 定义方法 type。
    def type(self) -> str:
        # 中文 L0106 调用 self.get_choice_name，用它的结果给 “choice_name” 赋值。
        choice_name = self.get_choice_name(self.__class__)
        # 中文 L0107 条件 not isinstance(choice_name, str) 成立时进入下方分支。
        if not isinstance(choice_name, str):
            # 中文 L0108 抛出 TypeError，中断当前路径并交给上层的异常处理。
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        # 中文 L0109 引用已有的值 choice_name作为当前调用的返回值。
        return choice_name

    # 中文 L0111 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0112 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0113 定义方法 observation_delta_indices。
    def observation_delta_indices(self) -> list[int] | None:
        # 中文 L0114 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0116 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0117 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0118 定义方法 action_delta_indices。
    def action_delta_indices(self) -> list[int] | None:
        # 中文 L0119 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0121 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0122 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0123 定义方法 reward_delta_indices。
    def reward_delta_indices(self) -> list[int] | None:
        # 中文 L0124 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0126 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0127 定义方法 get_optimizer_preset。
    def get_optimizer_preset(self) -> OptimizerConfig:
        # 中文 L0128 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0130 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0131 定义方法 get_scheduler_preset。
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        # 中文 L0132 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0134 应用装饰器 abc.abstractmethod，它会处理紧接着的函数或类。
    @abc.abstractmethod
    # 中文 L0135 定义方法 validate_features。
    def validate_features(self) -> None:
        # 中文 L0136 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
        raise NotImplementedError

    # 中文 L0138 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0139 定义方法 robot_state_feature。
    def robot_state_feature(self) -> PolicyFeature | None:
        # 中文 L0140 条件 not self.input_features 成立时进入下方分支。
        if not self.input_features:
            # 中文 L0141 设为空值 None作为当前调用的返回值。
            return None
        # 中文 L0142 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for ft_name, ft in self.input_features.items():
            # 中文 L0143 条件 ft.type is FeatureType.STATE and ft_name == OBS_STATE 成立时进入下方分支。
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                # 中文 L0144 引用已有的值 ft作为当前调用的返回值。
                return ft
        # 中文 L0145 设为空值 None作为当前调用的返回值。
        return None

    # 中文 L0147 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0148 定义方法 env_state_feature。
    def env_state_feature(self) -> PolicyFeature | None:
        # 中文 L0149 条件 not self.input_features 成立时进入下方分支。
        if not self.input_features:
            # 中文 L0150 设为空值 None作为当前调用的返回值。
            return None
        # 中文 L0151 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for _, ft in self.input_features.items():
            # 中文 L0152 条件 ft.type is FeatureType.ENV 成立时进入下方分支。
            if ft.type is FeatureType.ENV:
                # 中文 L0153 引用已有的值 ft作为当前调用的返回值。
                return ft
        # 中文 L0154 设为空值 None作为当前调用的返回值。
        return None

    # 中文 L0156 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0157 定义方法 image_features。
    def image_features(self) -> dict[str, PolicyFeature]:
        # 中文 L0158 条件 not self.input_features 成立时进入下方分支。
        if not self.input_features:
            # 中文 L0159 创建字典，用键查找对应的值作为当前调用的返回值。
            return {}
        # 中文 L0160 用字典推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    # 中文 L0162 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0163 定义方法 action_feature。
    def action_feature(self) -> PolicyFeature | None:
        # 中文 L0164 条件 not self.output_features 成立时进入下方分支。
        if not self.output_features:
            # 中文 L0165 设为空值 None作为当前调用的返回值。
            return None
        # 中文 L0166 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for ft_name, ft in self.output_features.items():
            # 中文 L0167 条件 ft.type is FeatureType.ACTION and ft_name == ACTION 成立时进入下方分支。
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                # 中文 L0168 引用已有的值 ft作为当前调用的返回值。
                return ft
        # 中文 L0169 设为空值 None作为当前调用的返回值。
        return None

    # 中文 L0171 定义方法 _save_pretrained。
    def _save_pretrained(self, save_directory: Path) -> None:
        # Encode against the base class so draccus includes the choice "type" key,
        # which `from_pretrained` needs to resolve the concrete subclass.
        # 中文 L0174 进入上下文管理器，缩进块结束时由它处理退出动作。
        with open(save_directory / CONFIG_NAME, "w") as f:
            # 中文 L0175 调用 json.dump 执行操作。
            json.dump(draccus.encode(self, PreTrainedConfig), f, indent=4)

    # 中文 L0177 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0178 定义方法 from_pretrained。作用是按 checkpoint 的 type 和覆盖参数恢复具体策略配置。
    def from_pretrained(
        # 中文 L0179 “cls”（当前类）是当前括号中的一项，逗号将它与下一项分开。
        cls: builtins.type[T],
        # 中文 L0180 “pretrained_name_or_path”是当前括号中的一项，逗号将它与下一项分开。
        pretrained_name_or_path: str | Path,
        # 中文 L0181 后续参数必须按名字传入。
        *,
        # 中文 L0182 为“force_download”提供这一项的值，注意它所在的参数表或外层表达式。
        force_download: bool = False,
        # 中文 L0183 为“resume_download”提供这一项的值，注意它所在的参数表或外层表达式。
        resume_download: bool | None = None,
        # 中文 L0184 为“proxies”提供这一项的值，注意它所在的参数表或外层表达式。
        proxies: dict[Any, Any] | None = None,
        # 中文 L0185 为“token”提供这一项的值，注意它所在的参数表或外层表达式。
        token: str | bool | None = None,
        # 中文 L0186 为“cache_dir”提供这一项的值，注意它所在的参数表或外层表达式。
        cache_dir: str | Path | None = None,
        # 中文 L0187 为“local_files_only”提供这一项的值，注意它所在的参数表或外层表达式。
        local_files_only: bool = False,
        # 中文 L0188 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
        revision: str | None = None,
        # 中文 L0189 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **policy_kwargs: Any,
    # 中文 L0190 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> T:
        # 中文 L0191 把路径或 Hub 标识统一转成字符串，供后续判断和读取使用。
        model_id = str(pretrained_name_or_path)
        # 中文 L0192 声明“config_file”的类型为 str | None，设为空值 None作为初始值。
        config_file: str | None = None
        # 中文 L0193 如果标识在本机就是目录，优先在这个目录找配置文件。
        if Path(model_id).is_dir():
            # 中文 L0194 条件 CONFIG_NAME in os.listdir(model_id) 成立时进入下方分支。
            if CONFIG_NAME in os.listdir(model_id):
                # 中文 L0195 调用 os.path.join，用它的结果给 “config_file” 赋值。
                config_file = os.path.join(model_id, CONFIG_NAME)
            # 中文 L0196 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0197 调用 logger.error 执行操作。
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        # 中文 L0198 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0199 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
            try:
                # 中文 L0200 由 Hub 客户端按仓库和 revision 解析配置文件。原评测启用离线模式，复用本地缓存。
                config_file = hf_hub_download(
                    # 中文 L0201 为“repo_id”（Hub 仓库标识）提供这一项的值，注意它所在的参数表或外层表达式。
                    repo_id=model_id,
                    # 中文 L0202 为“filename”（文件名）提供这一项的值，注意它所在的参数表或外层表达式。
                    filename=CONFIG_NAME,
                    # 中文 L0203 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
                    revision=revision,
                    # 中文 L0204 为“cache_dir”提供这一项的值，注意它所在的参数表或外层表达式。
                    cache_dir=cache_dir,
                    # 中文 L0205 为“force_download”提供这一项的值，注意它所在的参数表或外层表达式。
                    force_download=force_download,
                    # 中文 L0206 为“token”提供这一项的值，注意它所在的参数表或外层表达式。
                    token=token,
                    # 中文 L0207 为“local_files_only”提供这一项的值，注意它所在的参数表或外层表达式。
                    local_files_only=local_files_only,
                # 中文 L0208 闭合上方尚未结束的括号。
                )
            # 中文 L0209 捕获这里列出的异常类型，然后执行下方的处理逻辑。
            except HfHubHTTPError as e:
                # 中文 L0210 抛出 FileNotFoundError，中断当前路径并交给上层的异常处理。
                raise FileNotFoundError(
                    # 中文 L0211 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                # 中文 L0212 延续上面的语句，) from e。本段处理按 checkpoint 的 type 和覆盖参数恢复具体策略配置。
                ) from e

        # 中文 L0214 条件 config_file is None 成立时进入下方分支。
        if config_file is None:
            # 中文 L0215 抛出 FileNotFoundError，中断当前路径并交给上层的异常处理。
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        # 中文 L0217 进入上下文管理器，缩进块结束时由它处理退出动作。
        with open(config_file) as f:
            # 中文 L0218 解析 JSON，得到普通字典，还需解析为具体配置类型。
            config = json.load(f)

        # Resolve the concrete config subclass from the serialized "type" tag, then parse
        # the config (with CLI overrides) directly for that class. The "type" key is
        # stripped because draccus only consumes it when parsing the registry base class.
        # 中文 L0223 取出并删除序列化的 type。本次值为 smolvla。
        policy_type = config.pop("type", None)
        # 中文 L0224 条件 policy_type is None 成立时进入下方分支。
        if policy_type is None:
            # 中文 L0225 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Missing 'type' field in {CONFIG_NAME} of {model_id}")
        # 中文 L0226 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0227 按注册名找到配置类，本次得到 SmolVLAConfig。
            config_cls = cls.get_choice_class(policy_type)
        # 中文 L0228 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except Exception as e:
            # 中文 L0229 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0230 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Policy type '{policy_type}' (from {CONFIG_NAME} of {model_id}) is not registered. "
                # 中文 L0231 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Available policy types: {cls.get_known_choices()}"
            # 中文 L0232 延续上面的语句，) from e。本段处理按 checkpoint 的 type 和覆盖参数恢复具体策略配置。
            ) from e

        # 中文 L0234 进入上下文管理器，缩进块结束时由它处理退出动作。
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            # 中文 L0235 调用 json.dump 执行操作。
            json.dump(config, f)
            # 中文 L0236 读取属性 f.name给 “config_file” 赋值。
            config_file = f.name

        # 中文 L0238 调用 policy_kwargs.pop，用它的结果给 “cli_overrides” 赋值。
        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        # 中文 L0239 进入上下文管理器，缩进块结束时由它处理退出动作。
        with draccus.config_type("json"):
            # 中文 L0240 按具体数据类解析配置值，用调用方传入的覆盖项更新它。
            config = draccus.parse(config_cls, config_file, args=cli_overrides)

        # The serialized config.json does not carry the source path, so parsing
        # yields pretrained_path=None. Backfill it: downstream code (e.g.
        # make_policy / make_pre_post_processors) silently builds an untrained
        # processor pipeline without it, instead of loading the pretrained stats.
        # 中文 L0246 把权重文件所在的目录或 Hub 标识一起记到配置实例。
        config.pretrained_path = Path(model_id)
        # 中文 L0247 记录固定的模型版本，后续加载权重和处理器时使用。
        config.pretrained_revision = revision
        # 中文 L0248 引用已有的值 config作为当前调用的返回值。
        return config
