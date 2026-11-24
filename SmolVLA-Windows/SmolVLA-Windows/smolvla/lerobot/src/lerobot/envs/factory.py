#!/usr/bin/env python

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
# 中文 L0016 从模块 __future__ 导入 annotations。
from __future__ import annotations

# 中文 L0018 导入模块 logging。
import logging
# 中文 L0019 从模块 collections.abc 导入 Callable。
from collections.abc import Callable
# 中文 L0020 从模块 functools 导入 partial。
from functools import partial
# 中文 L0021 从模块 typing 导入 TYPE_CHECKING、Any。
from typing import TYPE_CHECKING, Any

# 中文 L0023 导入模块 gymnasium，简称 gym。
import gymnasium as gym

# 中文 L0025 从模块 lerobot.configs 导入 FeatureType、PolicyFeature。
from lerobot.configs import FeatureType, PolicyFeature
# 中文 L0026 从模块 lerobot.sim 导入 EnvDescriptor、LocalVectorEnv。
from lerobot.sim import EnvDescriptor, LocalVectorEnv
# 中文 L0027 从模块 lerobot.sim.integrations 导入 envhub。
from lerobot.sim.integrations import envhub
# 中文 L0028 从模块 lerobot.utils.constants 导入 ACTION、OBS_ENV_STATE。
from lerobot.utils.constants import ACTION, OBS_ENV_STATE

# 中文 L0030 从模块 .configs 导入 EnvConfig、HubEnvConfig、LiberoEnvConfig。
from .configs import EnvConfig, HubEnvConfig, LiberoEnvConfig

# 中文 L0032 条件 TYPE_CHECKING 成立时进入下方分支。
if TYPE_CHECKING:
    # 中文 L0033 从模块 lerobot.env_server 导入 RemoteVectorEnv。
    from lerobot.env_server import RemoteVectorEnv

# 中文 L0035 调用 logging.getLogger，用它的结果给 “logger” 赋值。
logger = logging.getLogger(__name__)

# Timeout of each request to a served env.
# 中文 L0038 使用字面值 120.0给 “SERVED_TIMEOUT_S” 赋值。
SERVED_TIMEOUT_S = 120.0


# 中文 L0041 定义函数 make_env_config。
def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    # 中文 L0042 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0043 调用 EnvConfig.get_choice_class，用它的结果给 “cls”（当前类） 赋值。
        cls = EnvConfig.get_choice_class(env_type)
    # 中文 L0044 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except KeyError as err:
        # 中文 L0045 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0046 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Environment type '{env_type}' is not registered. "
            # 中文 L0047 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Available: {list(EnvConfig.get_known_choices().keys())}"
        # 中文 L0048 延续上面的语句，) from err。
        ) from err
    # 中文 L0049 调用 cls，用它的结果作为当前调用的返回值。
    return cls(**kwargs)


# 中文 L0052 选择环境侧的前后处理接口。它与 checkpoint 的策略处理器是两组独立接口。
def make_env_pre_post_processors(
    # 中文 L0053 “env_cfg”是当前括号中的一项，逗号将它与下一项分开。
    env_cfg: EnvConfig,
    # 中文 L0054 “policy_cfg”是当前括号中的一项，逗号将它与下一项分开。
    policy_cfg: Any,
# 中文 L0055 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> tuple[Any, Any]:
    """
    Create preprocessor and postprocessor pipelines for environment observations.

    Returns a tuple of (preprocessor, postprocessor). By default, delegates to
    ``env_cfg.get_env_processors()``.  The XVLAConfig and Cosmos3Config overrides for LIBERO environments
    stay here because they depend on the *policy* config, not the env config.
    """
    # 中文 L0063 从模块 lerobot.policies.cosmos3.configuration_cosmos3 导入 Cosmos3Config。
    from lerobot.policies.cosmos3.configuration_cosmos3 import Cosmos3Config
    # 中文 L0064 从模块 lerobot.policies.xvla.configuration_xvla 导入 XVLAConfig。
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    # 中文 L0066 条件 isinstance(policy_cfg, XVLAConfig) and isinstance(env_cfg, LiberoEnvConfig) 成立时进入下方分支。
    if isinstance(policy_cfg, XVLAConfig) and isinstance(env_cfg, LiberoEnvConfig):
        # 中文 L0067 从模块 lerobot.policies.xvla.processor_xvla 导入 make_xvla_libero_pre_post_processors。
        from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors

        # 中文 L0069 调用 make_xvla_libero_pre_post_processors，用它的结果作为当前调用的返回值。
        return make_xvla_libero_pre_post_processors()

    # 中文 L0071 条件 isinstance(policy_cfg, Cosmos3Config) and isinstance(env_cfg, LiberoEnvConfig) 成立时进入下方分支。
    if isinstance(policy_cfg, Cosmos3Config) and isinstance(env_cfg, LiberoEnvConfig):
        # 中文 L0072 从模块 lerobot.policies.cosmos3.processor_cosmos3 导入 make_cosmos3_libero_pre_post_processors。
        from lerobot.policies.cosmos3.processor_cosmos3 import make_cosmos3_libero_pre_post_processors

        # 中文 L0074 调用 make_cosmos3_libero_pre_post_processors，用它的结果作为当前调用的返回值。
        return make_cosmos3_libero_pre_post_processors()

    # 中文 L0076 SmolVLA 走环境配置的默认处理器，此处返回两条空步骤流水线。
    return env_cfg.get_env_processors()


# 中文 L0079 创建按套件和任务 ID 索引的环境字典，每个条目包装一批环境。
def make_env(
    # 中文 L0080 “cfg”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    cfg: EnvConfig,
    # 中文 L0081 为“n_envs”（环境实例数）提供这一项的值，注意它所在的参数表或外层表达式。
    n_envs: int = 1,
    # 中文 L0082 为“use_async_envs”提供这一项的值，注意它所在的参数表或外层表达式。
    use_async_envs: bool = False,
# 中文 L0083 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Build the vector envs of a config.

    Args:
        cfg (`EnvConfig`):
            The env config. An EnvHub repository is `HubEnvConfig(hub_path="user/repo", trust_remote_code=True)`.
        n_envs (`int`, *optional*, defaults to `1`):
            Number of envs in each vector env.
        use_async_envs (`bool`, *optional*, defaults to `False`):
            Run each env in its own worker process.

    Returns:
        `dict[str, dict[int, gym.vector.VectorEnv]]`: The vector envs, as `{suite: {task_id: env}}`. A
        single-task env has one suite, with task ID 0.

    Raises:
        ValueError: If `n_envs` is below 1.
        RuntimeError: If an EnvHub repo's code would run without its config's `trust_remote_code`.
        ModuleNotFoundError: If the simulator, or a package the repo's `env.py` needs, is not installed.
    """
    # 中文 L0103 条件 n_envs < 1 成立时进入下方分支。
    if n_envs < 1:
        # 中文 L0104 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("`n_envs` must be at least 1")
    # 中文 L0105 条件 isinstance(cfg, str) 成立时进入下方分支。
    if isinstance(cfg, str):
        # 中文 L0106 抛出 TypeError，中断当前路径并交给上层的异常处理。
        raise TypeError(
            # 中文 L0107 字符串字面量，供上方的调用、容器或错误消息使用。
            "make_env() no longer takes a Hub repo ID. The equivalent is "
            # 中文 L0108 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f'make_env(HubEnvConfig(hub_path="{cfg}", trust_remote_code=True, use_native_batch=True), n_envs=...). '
            # 中文 L0109 字符串字面量，供上方的调用、容器或错误消息使用。
            "Without use_native_batch, LeRobot builds the envs itself, as lerobot-eval and the env server do."
        # 中文 L0110 闭合上方尚未结束的括号。
        )
    # 中文 L0111 条件 type(cfg).create_envs is not EnvConfig.create_envs 成立时进入下方分支。
    if type(cfg).create_envs is not EnvConfig.create_envs:
        # 中文 L0112 抛出 TypeError，中断当前路径并交给上层的异常处理。
        raise TypeError(
            # 中文 L0113 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"{type(cfg).__name__} overrides EnvConfig.create_envs(), which make_env no longer calls: make_env "
            # 中文 L0114 字符串字面量，供上方的调用、容器或错误消息使用。
            "builds every env from its config's sim_settings(). Return the simulator and its options from "
            # 中文 L0115 字符串字面量，供上方的调用、容器或错误消息使用。
            "sim_settings() instead, as described in https://huggingface.co/docs/lerobot/adding_benchmarks"
        # 中文 L0116 闭合上方尚未结束的括号。
        )
    # 中文 L0117 条件 isinstance(cfg, HubEnvConfig) and cfg.use_native_batch 成立时进入下方分支。
    if isinstance(cfg, HubEnvConfig) and cfg.use_native_batch:
        # 中文 L0118 调用 _make_native_batch，用它的结果作为当前调用的返回值。
        return _make_native_batch(cfg, n_envs, use_async_envs)
    # 中文 L0119 获取环境描述符和可创建指定套件任务的闭包。
    descriptor, make = _env_factory(cfg, n_envs, use_async_envs)
    # 中文 L0120 用条件表达式在 if 前与 else 后的两个值之间选择给 “groups” 赋值。
    groups = [task.strip() for task in cfg.task.split(",")] if cfg.task else list(descriptor.tasks)
    # 中文 L0121 用字典推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
    return {
        # 中文 L0122 用字典推导式组合任务套件与环境实例。
        group: {task_id: make(group, task_id) for task_id in _suite(descriptor, group)} for group in groups
    # 中文 L0123 闭合上方尚未结束的括号。
    }


# 中文 L0126 定义函数 make_task_env。
def make_task_env(
    # 中文 L0127 “cfg”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    cfg: EnvConfig,
    # 中文 L0128 为“task_group”（任务套件名）提供这一项的值，注意它所在的参数表或外层表达式。
    task_group: str | None = None,
    # 中文 L0129 为“task_id”（套件内的任务编号）提供这一项的值，注意它所在的参数表或外层表达式。
    task_id: int | None = None,
    # 中文 L0130 为“n_envs”（环境实例数）提供这一项的值，注意它所在的参数表或外层表达式。
    n_envs: int = 1,
    # 中文 L0131 为“use_async_envs”提供这一项的值，注意它所在的参数表或外层表达式。
    use_async_envs: bool = False,
# 中文 L0132 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> LocalVectorEnv | RemoteVectorEnv:
    # 中文 L0133 原文档中的说明或续行。
    """Build the local or served vector env of one task; no `task_id` means the suite's first task."""
    # 中文 L0134 调用 _env_factory，用它的结果给 “(descriptor, make)” 赋值。
    descriptor, make = _env_factory(cfg, n_envs, use_async_envs, task_group, task_id)
    # 中文 L0135 组合多个布尔判断，按 Python 的短路规则求值给 “group” 赋值。
    group = task_group or next(iter(descriptor.tasks))
    # 中文 L0136 调用 _suite，用它的结果给 “task_ids” 赋值。
    task_ids = _suite(descriptor, group)
    # 中文 L0137 条件 task_id is None 成立时进入下方分支。
    if task_id is None:
        # 中文 L0138 用方括号按键或索引读取 task_ids给 “task_id”（套件内的任务编号） 赋值。
        task_id = task_ids[0]
    # 中文 L0139 条件 task_id not in task_ids 成立时进入下方分支。
    if task_id not in task_ids:
        # 中文 L0140 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Unknown task ID {task_id} for {group!r}; this suite has {sorted(task_ids)}")
    # 中文 L0141 调用 make，用它的结果作为当前调用的返回值。
    return make(group, task_id)


# 中文 L0144 定义函数 _make_native_batch。
def _make_native_batch(
    # 中文 L0145 延续上面的语句，cfg: HubEnvConfig, n_envs: int, use_async_envs: bool。
    cfg: HubEnvConfig, n_envs: int, use_async_envs: bool
# 中文 L0146 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    # 中文 L0147 原文档中的说明或续行。
    """Return an EnvHub repo's own vector envs (e.g. GPU batches) unchanged."""
    # 中文 L0148 条件 cfg.server is not None 成立时进入下方分支。
    if cfg.server is not None:
        # 中文 L0149 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("use_native_batch runs locally; it cannot use an env server")
    # 中文 L0150 条件 not cfg.hub_path 成立时进入下方分支。
    if not cfg.hub_path:
        # 中文 L0151 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("--env.hub_path is required for an EnvHub environment")
    # Provider configs (Isaac Lab Arena, Unitree G1) subclass HubEnvConfig and also pass themselves as `cfg`,
    # since their options are config fields.
    # 中文 L0154 用条件表达式在 if 前与 else 后的两个值之间选择给 “provider” 赋值。
    provider = None if type(cfg) is HubEnvConfig else cfg
    # 中文 L0155 调用 envhub.make_envs，用它的结果作为当前调用的返回值。
    return envhub.make_envs(
        # 中文 L0156 延续上面的语句，cfg.hub_path, n_envs, use_async_envs, cfg.trust_remote_code, cfg=provider, **cfg.make_env_kwargs。
        cfg.hub_path, n_envs, use_async_envs, cfg.trust_remote_code, cfg=provider, **cfg.make_env_kwargs
    # 中文 L0157 闭合上方尚未结束的括号。
    )


# 中文 L0160 统一选择环境构建路径。原评测使用本机仿真。
def _env_factory(
    # 中文 L0161 “cfg”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
    cfg: EnvConfig,
    # 中文 L0162 “n_envs”（环境实例数）是当前括号中的一项，逗号将它与下一项分开。
    n_envs: int,
    # 中文 L0163 “use_async_envs”是当前括号中的一项，逗号将它与下一项分开。
    use_async_envs: bool,
    # 中文 L0164 为“task_group”（任务套件名）提供这一项的值，注意它所在的参数表或外层表达式。
    task_group: str | None = None,
    # 中文 L0165 为“task_id”（套件内的任务编号）提供这一项的值，注意它所在的参数表或外层表达式。
    task_id: int | None = None,
# 中文 L0166 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> tuple[EnvDescriptor, Callable[[str, int], LocalVectorEnv | RemoteVectorEnv]]:
    # 中文 L0167 原文档概述本段功能，选择本地或远程环境创建方式并生成任务构造器。
    """Return the env's descriptor, and a function building the vector env of one of its tasks."""
    # 中文 L0168 向环境配置读取具体仿真设置。
    settings = cfg.sim_settings()
    # 中文 L0169 读取属性 cfg.server给 “server” 赋值。
    server = cfg.server
    # 中文 L0170 条件 server is not None 成立时进入下方分支。
    if server is not None:
        # 中文 L0171 从模块 lerobot.env_server 导入 EnvClient、RemoteVectorEnv。
        from lerobot.env_server import EnvClient, RemoteVectorEnv

        # 中文 L0173 条件 cfg.max_parallel_tasks != 1 成立时进入下方分支。
        if cfg.max_parallel_tasks != 1:
            # 中文 L0174 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("A served environment runs one task at a time")
        # 中文 L0175 调用 partial，用它的结果给 “connect” 赋值。
        connect = partial(
            # 中文 L0176 “EnvClient”是当前括号中的一项，逗号将它与下一项分开。
            EnvClient,
            # 中文 L0177 延续上面的语句，server.endpoint,。本段处理选择本地或远程环境创建方式并生成任务构造器。
            server.endpoint,
            # 中文 L0178 延续上面的语句，server.deployment or cfg.type,。本段处理选择本地或远程环境创建方式并生成任务构造器。
            server.deployment or cfg.type,
            # 中文 L0179 “SERVED_TIMEOUT_S”是当前括号中的一项，逗号将它与下一项分开。
            SERVED_TIMEOUT_S,
            # 中文 L0180 为“handshake_timeout_s”提供这一项的值，注意它所在的参数表或外层表达式。
            handshake_timeout_s=server.handshake_timeout_s,
            # 中文 L0181 为“zenoh_config_path”提供这一项的值，注意它所在的参数表或外层表达式。
            zenoh_config_path=server.zenoh_config_path,
            # 中文 L0182 为“max_payload_bytes”提供这一项的值，注意它所在的参数表或外层表达式。
            max_payload_bytes=server.max_payload_bytes,
        # 中文 L0183 闭合上方尚未结束的括号。
        )
        # 中文 L0184 调用 connect，用它的结果给 “probe” 赋值。
        probe = connect()
        # 中文 L0185 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0186 调用 probe.describe，用它的结果给 “descriptor” 赋值。
            descriptor = probe.describe()
        # 中文 L0187 离开 try 时执行这里的清理，即使前面的路径发生异常。
        finally:
            # 中文 L0188 调用 probe.close，关闭对象或释放关联资源。
            probe.close()
        # 中文 L0189 调用 check_served 执行操作。
        check_served(descriptor.config, settings, server.endpoint)

        # 中文 L0191 定义函数 make。
        def make(group: str, task_id: int) -> LocalVectorEnv | RemoteVectorEnv:
            # 中文 L0192 调用 RemoteVectorEnv，用它的结果作为当前调用的返回值。
            return RemoteVectorEnv(
                # 中文 L0193 延续上面的语句，connect(encoding=server.encoding, jpeg_quality=server.jpeg_quality),。
                connect(encoding=server.encoding, jpeg_quality=server.jpeg_quality),
                # 中文 L0194 “n_envs”（环境实例数）是当前括号中的一项，逗号将它与下一项分开。
                n_envs,
                # 中文 L0195 “group”是当前括号中的一项，逗号将它与下一项分开。
                group,
                # 中文 L0196 “task_id”（套件内的任务编号）是当前括号中的一项，逗号将它与下一项分开。
                task_id,
                # 中文 L0197 延续上面的语句，use_async_envs and n_envs > 1,。
                use_async_envs and n_envs > 1,
                # 中文 L0198 “descriptor”是当前括号中的一项，逗号将它与下一项分开。
                descriptor,
            # 中文 L0199 闭合上方尚未结束的括号。
            )

        # 中文 L0201 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return descriptor, make

    # 中文 L0203 把配置解析为 SimConfig，以便动态定位 LIBERO 接入模块。
    sim = cfg.sim_config()
    # 中文 L0204 仅在显式选择异步且环境多于一个时使用异步路径。
    parallel = use_async_envs and n_envs > 1
    # Reuse the env built to read the descriptor for its own task instead of building it twice.
    # 中文 L0206 构造第一个 LocalVectorEnv，以获取任务和特征描述符并尽量复用环境。
    first = [LocalVectorEnv(sim, n_envs, task_group, task_id, parallel)]
    # 中文 L0207 读取属性 descriptor给 “descriptor” 赋值。
    descriptor = first[0].descriptor

    # 中文 L0209 定义函数 make_local。
    def make_local(group: str, task_id: int) -> LocalVectorEnv | RemoteVectorEnv:
        # 中文 L0210 条件 first 成立时进入下方分支。
        if first:
            # 中文 L0211 调用 first.pop，用它的结果给 “env” 赋值。
            env = first.pop()
            # 中文 L0212 条件 (group, task_id) == (env.task_group, env.task_id) 成立时进入下方分支。
            if (group, task_id) == (env.task_group, env.task_id):
                # 中文 L0213 引用已有的值 env作为当前调用的返回值。
                return env
            # 中文 L0214 调用 env.close，关闭对象或释放关联资源。
            env.close()
        # With the descriptor given, the other tasks build their sims on first use.
        # 中文 L0216 调用 LocalVectorEnv，用它的结果作为当前调用的返回值。
        return LocalVectorEnv(sim, n_envs, group, task_id, parallel, descriptor)

    # 中文 L0218 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
    return descriptor, make_local


# 中文 L0221 定义函数 _suite。
def _suite(descriptor: EnvDescriptor, group: str) -> list[int]:
    # 中文 L0222 原文档中的说明或续行。
    """Return the task IDs of one suite of the env, or raise naming the suites it has."""
    # 中文 L0223 条件 group not in descriptor.tasks 成立时进入下方分支。
    if group not in descriptor.tasks:
        # 中文 L0224 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"Unknown task {group!r}; this environment has {sorted(descriptor.tasks)}")
    # 中文 L0225 用方括号按键或索引读取 descriptor.tasks作为当前调用的返回值。
    return descriptor.tasks[group]


# 中文 L0228 将仿真描述符的输入输出维度转换成策略工厂认识的特征定义。
def descriptor_features(descriptor: EnvDescriptor) -> dict[str, PolicyFeature]:
    # 中文 L0229 原文档中的说明或续行。
    """Return an env's observation and action features, typed as for a dataset; images stay channel last."""
    # 中文 L0230 创建字典，用键查找对应的值给 “features” 赋值。
    features = {}
    # 中文 L0231 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for spec in descriptor.features:
        # 中文 L0232 条件 spec.kind == "rgb" 成立时进入下方分支。
        if spec.kind == "rgb":
            # 中文 L0233 读取属性 FeatureType.VISUAL给 “kind” 赋值。
            kind = FeatureType.VISUAL
        # 中文 L0234 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0235 用条件表达式在 if 前与 else 后的两个值之间选择给 “kind” 赋值。
            kind = FeatureType.ENV if spec.name == OBS_ENV_STATE else FeatureType.STATE
        # 中文 L0236 调用 PolicyFeature，用它的结果给 “features[spec.name]” 赋值。
        features[spec.name] = PolicyFeature(type=kind, shape=spec.shape)
    # 中文 L0237 调用 PolicyFeature，用它的结果给 “features[ACTION]” 赋值。
    features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=descriptor.action_feature.shape)
    # 中文 L0238 引用已有的值 features作为当前调用的返回值。
    return features


# 中文 L0241 定义函数 check_served。
def check_served(served: dict[str, Any], expected: dict[str, Any], endpoint: str) -> None:
    # 中文 L0242 原文档中的说明或续行。
    """Raise if the server runs a different env than ``expected``; warn if only its kwargs differ."""

    # 中文 L0244 定义函数 plain。
    def plain(value: Any) -> Any:
        # Make YAML, wire and dataclass values comparable: tuples become lists, recursively.
        # 中文 L0246 条件 isinstance(value, dict) 成立时进入下方分支。
        if isinstance(value, dict):
            # 中文 L0247 用字典推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
            return {key: plain(item) for key, item in value.items()}
        # 中文 L0248 条件 isinstance(value, (list, tuple)) 成立时进入下方分支。
        if isinstance(value, (list, tuple)):
            # 中文 L0249 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
            return [plain(item) for item in value]
        # 中文 L0250 引用已有的值 value作为当前调用的返回值。
        return value

    # A server opens any of the tasks it configures (all of them for an empty list), so the requested tasks
    # are checked against the descriptor's task list instead.
    # 中文 L0254 用列表推导式从可迭代对象逐项取值并构造结果给 “errors” 赋值。
    errors = [
        # 中文 L0255 f 字符串片段，花括号中的表达式会被求值并插入文本。
        f"{key}: server {served.get(key)!r}, expected {value!r}"
        # 中文 L0256 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, value in expected.items()
        # 中文 L0257 条件 key not in ("kwargs", "task") and value is not None and plain(value) != plain(served.get(key)) 成立时进入下方分支。
        if key not in ("kwargs", "task") and value is not None and plain(value) != plain(served.get(key))
    # 中文 L0258 闭合上方尚未结束的括号。
    ]
    # 中文 L0259 组合多个布尔判断，按 Python 的短路规则求值给 “served_kwargs” 赋值。
    served_kwargs = served.get("kwargs") or {}
    # hub_path and trust_remote_code may differ: the env server loads EnvHub repos from its own image.
    # 中文 L0261 用列表推导式从可迭代对象逐项取值并构造结果给 “differences” 赋值。
    differences = [
        # 中文 L0262 f 字符串片段，花括号中的表达式会被求值并插入文本。
        f"kwargs.{key}: server {served_kwargs.get(key)!r}, expected {value!r}"
        # 中文 L0263 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, value in (expected.get("kwargs") or {}).items()
        # 中文 L0264 条件 key not in ("hub_path", "trust_remote_code") and plain(value) != plain(served_kwargs.get(key)) 成立时进入下方分支。
        if key not in ("hub_path", "trust_remote_code") and plain(value) != plain(served_kwargs.get(key))
    # 中文 L0265 闭合上方尚未结束的括号。
    ]
    # 中文 L0266 条件 errors 成立时进入下方分支。
    if errors:
        # 中文 L0267 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(
            # 中文 L0268 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"The env server at {endpoint} runs another environment than this command describes "
            # 中文 L0269 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"({'; '.join(errors)}). Pass the same --env options as the server's configuration."
        # 中文 L0270 闭合上方尚未结束的括号。
        )
    # 中文 L0271 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for difference in differences:
        # 中文 L0272 调用 logger.warning 执行操作。
        logger.warning("The env server at %s differs in %s", endpoint, difference)
