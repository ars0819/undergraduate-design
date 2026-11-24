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
# 中文 L0015 导入模块 datetime，简称 dt。
import datetime as dt
# 中文 L0016 从模块 dataclasses 导入 dataclass、field。
from dataclasses import dataclass, field
# 中文 L0017 从模块 logging 导入 getLogger。
from logging import getLogger
# 中文 L0018 从模块 pathlib 导入 Path。
from pathlib import Path

# 中文 L0020 从模块 lerobot 导入 envs、policies。
from lerobot import envs, policies  # noqa: F401

# 中文 L0022 从模块 . 导入 parser。
from . import parser
# 中文 L0023 从模块 .default 导入 EvalConfig。
from .default import EvalConfig
# 中文 L0024 从模块 .policies 导入 PreTrainedConfig。
from .policies import PreTrainedConfig

# 中文 L0026 调用 getLogger，用它的结果给 “logger” 赋值。
logger = getLogger(__name__)


# 中文 L0029 数据类装饰器根据下列字段生成初始化方法，自动在初始化完成后调用 __post_init__。
@dataclass
# 中文 L0030 评测用的顶层数据类，把环境、策略、评测参数和输出位置装在一个实例里。
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    # 中文 L0034 env 存环境配置实例，LIBERO 运行解析成 LiberoEnvConfig。
    env: envs.EnvConfig
    # 中文 L0035 用 default_factory 每次创建一个新的 EvalConfig，避免在多个配置实例间共享同一个可变对象。
    eval: EvalConfig = field(default_factory=EvalConfig)
    # 中文 L0036 policy 可以是某个策略配置子类的实例，也允许在解析中先留空。
    policy: PreTrainedConfig | None = None
    # 中文 L0037 声明“output_dir”的类型为 Path | None，设为空值 None作为初始值。
    output_dir: Path | None = None
    # 中文 L0038 声明“job_name”的类型为 str | None，设为空值 None作为初始值。
    job_name: str | None = None
    # 中文 L0039 声明“seed”（随机种子）的类型为 int | None，使用字面值 1000作为初始值。
    seed: int | None = 1000
    # TF32 in float32 matmuls and cuDNN convolutions on Ampere and newer GPUs.
    # 中文 L0041 声明“tf32”的类型为 bool | None，使用布尔值 True作为初始值。
    tf32: bool | None = True
    # Rename map for the observation to override the image and state keys
    # 中文 L0043 相机键替换映射，字典的键和值均为字符串。
    rename_map: dict[str, str] = field(default_factory=dict)
    # Removed: the env gives its own consent, with --env.trust_remote_code. Kept to say so when passed.
    # 中文 L0045 声明“trust_remote_code”的类型为 bool，使用布尔值 False作为初始值。
    trust_remote_code: bool = False

    # 中文 L0047 数据类初始化字段后调用这一方法完成配置的剩余工作。
    def __post_init__(self) -> None:
        # 中文 L0048 条件 self.trust_remote_code 成立时进入下方分支。
        if self.trust_remote_code:
            # 中文 L0049 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0050 字符串字面量，供上方的调用、容器或错误消息使用。
                "lerobot-eval --trust_remote_code was removed: pass --env.trust_remote_code=true, the consent "
                # 中文 L0051 字符串字面量，供上方的调用、容器或错误消息使用。
                "of the EnvHub environment that runs the code"
            # 中文 L0052 闭合上方尚未结束的括号。
            )
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        # 中文 L0054 提取 --policy.path 对应的 checkpoint 名称或目录。
        policy_path = parser.get_path_arg("policy")
        # 中文 L0055 条件 policy_path 成立时进入下方分支。
        if policy_path:
            # 中文 L0056 取来自 YAML 或 JSON 顶层配置的策略覆盖值。
            yaml_overrides = parser.get_yaml_overrides("policy")
            # 中文 L0057 取 --policy.* 命令行覆盖项，找不到时用空列表。
            cli_overrides = parser.get_cli_overrides("policy") or []
            # 中文 L0058 根据 checkpoint 的 config.json 恢复具体策略的配置和参数。
            self.policy = PreTrainedConfig.from_pretrained(
                # 中文 L0059 “policy_path”是当前括号中的一项，逗号将它与下一项分开。
                policy_path,
                # 中文 L0060 为“revision”（固定的 checkpoint 版本）提供这一项的值，注意它所在的参数表或外层表达式。
                revision=parser.get_pretrained_revision("policy"),
                # 中文 L0061 按顺序拼接配置文件覆盖项和命令行覆盖项，后者位于列表末尾。
                cli_overrides=yaml_overrides + cli_overrides,
            # 中文 L0062 闭合上方尚未结束的括号。
            )
            # 中文 L0063 把 checkpoint 来源记到策略配置，后续权重和处理器工厂需要它。
            self.policy.pretrained_path = Path(policy_path)

        # 中文 L0065 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0066 调用 logger.warning 执行操作。
            logger.warning(
                # 中文 L0067 字符串字面量，供上方的调用、容器或错误消息使用。
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            # 中文 L0068 闭合上方尚未结束的括号。
            )

        # 中文 L0070 条件 not self.job_name 成立时进入下方分支。
        if not self.job_name:
            # 中文 L0071 条件 self.env is None 成立时进入下方分支。
            if self.env is None:
                # 中文 L0072 计算此行的右侧表达式给 “self.job_name” 赋值。
                self.job_name = f"{self.policy.type if self.policy is not None else 'scratch'}"
            # 中文 L0073 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0074 计算此行的右侧表达式给 “self.job_name” 赋值。
                self.job_name = (
                    # 中文 L0075 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"{self.env.type}_{self.policy.type if self.policy is not None else 'scratch'}"
                # 中文 L0076 闭合上方尚未结束的括号。
                )

            # 中文 L0078 调用 logger.warning 执行操作。
            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        # 中文 L0080 条件 not self.output_dir 成立时进入下方分支。
        if not self.output_dir:
            # 中文 L0081 调用 dt.datetime.now，用它的结果给 “now” 赋值。
            now = dt.datetime.now()
            # 中文 L0082 计算此行的右侧表达式给 “eval_dir” 赋值。
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            # 中文 L0083 计算右侧表达式 Path("outputs/eval") / eval_dir给 “self.output_dir” 赋值。
            self.output_dir = Path("outputs/eval") / eval_dir

    # 中文 L0085 类方法装饰器。调用时首个参数 cls 是类本身。
    @classmethod
    # 中文 L0086 声明 policy 是需要特殊处理的路径字段。
    def __get_path_fields__(cls) -> list[str]:
        # 中文 L0087 原文档中的说明或续行。
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        # 中文 L0088 创建列表，按给定顺序保留元素作为当前调用的返回值。
        return ["policy"]
