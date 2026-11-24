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

# NOTE: gymnasium is currently a core dependency but is a candidate for moving to an
# optional extra in the future. When that transition happens, uncomment the guard below
# and update the extra name to the one that will contain gymnasium.
# from lerobot.utils.import_utils import require_package
# require_package("gymnasium", extra="<update_extra>", import_name="gymnasium")

# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0021 从模块 .configs 导入 AlohaEnvConfig、EnvClientConfig、EnvConfig、HILSerlRobotEnvConfig、HubEnvConfig、PushtEnvConfig。
from .configs import (
    # 中文 L0022 “AlohaEnvConfig”是当前括号中的一项，逗号将它与下一项分开。
    AlohaEnvConfig,
    # 中文 L0023 “EnvClientConfig”是当前括号中的一项，逗号将它与下一项分开。
    EnvClientConfig,
    # 中文 L0024 “EnvConfig”是当前括号中的一项，逗号将它与下一项分开。
    EnvConfig,
    # 中文 L0025 “HILSerlRobotEnvConfig”是当前括号中的一项，逗号将它与下一项分开。
    HILSerlRobotEnvConfig,
    # 中文 L0026 “HubEnvConfig”是当前括号中的一项，逗号将它与下一项分开。
    HubEnvConfig,
    # 中文 L0027 “PushtEnvConfig”是当前括号中的一项，逗号将它与下一项分开。
    PushtEnvConfig,
# 中文 L0028 闭合上方尚未结束的括号。
)
# 中文 L0029 从同包内的 factory.py 导入环境工厂，并使这些名字可以从 lerobot.envs 获取。
from .factory import make_env, make_env_config, make_env_pre_post_processors, make_task_env
# 中文 L0030 把同包内 utils.py 里的观测和环境工具放到 lerobot.envs 包的命名空间。
from .utils import check_env_attributes_and_types, close_envs, env_to_policy_features, preprocess_observation

# 中文 L0032 __all__ 列出包公开的名字，主要用于控制 from package import *。上面的导入语句把对象绑定到包。
__all__ = [
    # 中文 L0033 字符串字面量，供上方的调用、容器或错误消息使用。
    "AlohaEnvConfig",
    # 中文 L0034 字符串字面量，供上方的调用、容器或错误消息使用。
    "EnvClientConfig",
    # 中文 L0035 字符串字面量，供上方的调用、容器或错误消息使用。
    "EnvConfig",
    # 中文 L0036 字符串字面量，供上方的调用、容器或错误消息使用。
    "HILSerlRobotEnvConfig",
    # 中文 L0037 字符串字面量，供上方的调用、容器或错误消息使用。
    "HubEnvConfig",
    # 中文 L0038 字符串字面量，供上方的调用、容器或错误消息使用。
    "PushtEnvConfig",
    # 中文 L0039 字符串字面量，供上方的调用、容器或错误消息使用。
    "check_env_attributes_and_types",
    # 中文 L0040 字符串字面量，供上方的调用、容器或错误消息使用。
    "close_envs",
    # 中文 L0041 字符串字面量，供上方的调用、容器或错误消息使用。
    "env_to_policy_features",
    # 中文 L0042 字符串字面量，供上方的调用、容器或错误消息使用。
    "make_env",
    # 中文 L0043 字符串字面量，供上方的调用、容器或错误消息使用。
    "make_env_config",
    # 中文 L0044 字符串字面量，供上方的调用、容器或错误消息使用。
    "make_env_pre_post_processors",
    # 中文 L0045 字符串字面量，供上方的调用、容器或错误消息使用。
    "make_task_env",
    # 中文 L0046 字符串字面量，供上方的调用、容器或错误消息使用。
    "preprocess_observation",
# 中文 L0047 闭合上方尚未结束的括号。
]
