# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
# 中文 L0015 原文档中的说明或续行。
"""Vector envs for simulators, with no autoreset: a finished env stays finished until `reset`."""

# 中文 L0017 导入模块 contextlib。
import contextlib
# 中文 L0018 导入模块 importlib。
import importlib
# 中文 L0019 导入模块 logging。
import logging
# 中文 L0020 导入模块 multiprocessing。
import multiprocessing
# 中文 L0021 导入模块 traceback。
import traceback
# 中文 L0022 从模块 collections.abc 导入 Callable、Mapping。
from collections.abc import Callable, Mapping
# 中文 L0023 从模块 dataclasses 导入 asdict、dataclass、field、replace。
from dataclasses import asdict, dataclass, field, replace
# 中文 L0024 从模块 functools 导入 partial。
from functools import partial
# 中文 L0025 从模块 multiprocessing.connection 导入 Connection。
from multiprocessing.connection import Connection
# 中文 L0026 从模块 typing 导入 Any。
from typing import Any

# 中文 L0028 导入模块 gymnasium，简称 gym。
import gymnasium as gym
# 中文 L0029 导入模块 numpy，简称 np。
import numpy as np
# 中文 L0030 从模块 gymnasium 导入 spaces。
from gymnasium import spaces
# 中文 L0031 从模块 gymnasium.vector.utils 导入 CloudpickleWrapper、batch_space。
from gymnasium.vector.utils import CloudpickleWrapper, batch_space

# 中文 L0033 从模块 lerobot.sim.config 导入 SimConfig。
from lerobot.sim.config import SimConfig
# 中文 L0034 从模块 lerobot.sim.integrations 导入 simulator_required。
from lerobot.sim.integrations import simulator_required
# 中文 L0035 从模块 lerobot.sim.types 导入 EnvDescriptor、StepResult。
from lerobot.sim.types import EnvDescriptor, StepResult
# 中文 L0036 从模块 lerobot.utils.constants 导入 ACTION、OBS_STATE。
from lerobot.utils.constants import ACTION, OBS_STATE
# 中文 L0037 从模块 lerobot.utils.feature_spec 导入 FeatureSpec、validate_array。
from lerobot.utils.feature_spec import FeatureSpec, validate_array

# 中文 L0039 调用 logging.getLogger，用它的结果给 “logger” 赋值。
logger = logging.getLogger(__name__)

# Env attributes copied from a worker process after it builds the env and after each reset.
# 中文 L0042 打包成元组，元素顺序由逗号分隔的位置决定给 “_ENV_ATTRS” 赋值。
_ENV_ATTRS = ("task_description", "task", "num_envs")
# Episode limit for an env that declares none.
# 中文 L0044 使用字面值 500给 “_DEFAULT_MAX_EPISODE_STEPS” 赋值。
_DEFAULT_MAX_EPISODE_STEPS = 500
# Gymnasium 0.29, which some simulators pin (RoboMME's ManiSkill), has no AutoresetMode; there a vector env always
# autoresets, so none from vector_env_fns is accepted.
# 中文 L0047 调用 getattr，用它的结果给 “_NO_AUTORESET” 赋值。
_NO_AUTORESET = getattr(getattr(gym.vector, "AutoresetMode", None), "DISABLED", "disabled")


# 中文 L0050 应用装饰器 dataclass(frozen=True)，它会处理紧接着的函数或类。
@dataclass(frozen=True)
# 中文 L0051 定义类 _LastStep。
class _LastStep:
    # 中文 L0052 原文档中的说明或续行。
    """One env's latest observation, reward, episode flags and metrics."""

    # 中文 L0054 声明“obs”的类型为 dict[str, np.ndarray]。
    obs: dict[str, np.ndarray]
    # 中文 L0055 声明“reward”（单步奖励）的类型为 float，使用字面值 0.0作为初始值。
    reward: float = 0.0
    # 中文 L0056 声明“terminated”（任务终止标志）的类型为 bool，使用布尔值 False作为初始值。
    terminated: bool = False
    # 中文 L0057 声明“truncated”（截断标志）的类型为 bool，使用布尔值 False作为初始值。
    truncated: bool = False
    # 中文 L0058 声明“is_success”（仿真的成功标志）的类型为 bool，使用布尔值 False作为初始值。
    is_success: bool = False
    # The latest value of each metric the env reported in this episode.
    # 中文 L0060 声明“metrics”的类型为 dict[str, float]，调用 field，用它的结果作为初始值。
    metrics: dict[str, float] = field(default_factory=dict)

    # 中文 L0062 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0063 定义方法 done。
    def done(self) -> bool:
        # 中文 L0064 原文档中的说明或续行。
        """Whether the episode has ended."""
        # 中文 L0065 组合多个布尔判断，按 Python 的短路规则求值作为当前调用的返回值。
        return self.terminated or self.truncated


# 中文 L0068 定义函数 _max_episode_steps。
def _max_episode_steps(env: Any) -> int | None:
    # 中文 L0069 原文档中的说明或续行。
    """Return the env's episode limit, looked up through Gymnasium wrappers, or None if it has none."""
    # 中文 L0070 调用 getattr，用它的结果给 “get_wrapper_attr” 赋值。
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    # 中文 L0071 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0072 用条件表达式在 if 前与 else 后的两个值之间选择给 “limit” 赋值。
        limit = get_wrapper_attr("_max_episode_steps") if get_wrapper_attr else env._max_episode_steps
    # 中文 L0073 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except AttributeError:
        # 中文 L0074 设为空值 None作为当前调用的返回值。
        return None
    # 中文 L0075 用条件表达式在 if 前与 else 后的两个值之间选择作为当前调用的返回值。
    return None if limit is None else int(limit)


# 中文 L0078 定义函数 _spaces。
def _spaces(env: Any) -> tuple[spaces.Space, spaces.Space]:
    # 中文 L0079 原文档中的说明或续行。
    """Return the observation and action spaces of one env; for a vector env, those of each of its envs."""
    # 中文 L0080 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
    return (
        # 中文 L0081 延续上面的语句，getattr(env, "single_observation_space", env.observation_space),。
        getattr(env, "single_observation_space", env.observation_space),
        # 中文 L0082 延续上面的语句，getattr(env, "single_action_space", env.action_space),。
        getattr(env, "single_action_space", env.action_space),
    # 中文 L0083 闭合上方尚未结束的括号。
    )


# 中文 L0086 定义函数 _row。
def _row(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    # 中文 L0087 原文档中的说明或续行。
    """Return env `index`'s observation from a vector env's batched, possibly nested, observation."""
    # 中文 L0088 用字典推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
    return {
        # 中文 L0089 延续上面的语句，key: _row(value, index) if isinstance(value, Mapping) else value[index]。
        key: _row(value, index) if isinstance(value, Mapping) else value[index]
        # 中文 L0090 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key, value in batch.items()
    # 中文 L0091 闭合上方尚未结束的括号。
    }


# 中文 L0094 定义函数 _build_env。
def _build_env(sim_type: str, make: Callable[[], Any]) -> Any:
    # 中文 L0095 原文档中的说明或续行。
    """Build one env, or one vector env, with `make`; reject a vector env that autoresets."""
    # 中文 L0096 进入上下文管理器，缩进块结束时由它处理退出动作。
    with simulator_required(sim_type):
        # 中文 L0097 调用 make，用它的结果给 “env” 赋值。
        env = make()
    # 中文 L0098 条件 isinstance(env, gym.vector.VectorEnv) 成立时进入下方分支。
    if isinstance(env, gym.vector.VectorEnv):
        # 中文 L0099 调用 env.metadata.get，用它的结果给 “mode” 赋值。
        mode = env.metadata.get("autoreset_mode")
        # 中文 L0100 条件 mode is not _NO_AUTORESET 成立时进入下方分支。
        if mode is not _NO_AUTORESET:
            # 中文 L0101 调用 env.close，关闭对象或释放关联资源。
            env.close()
            # 中文 L0102 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0103 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"A vector env from vector_env_fns must not autoreset; set metadata['autoreset_mode'] to "
                # 中文 L0104 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"AutoresetMode.DISABLED (got {mode})"
            # 中文 L0105 闭合上方尚未结束的括号。
            )
    # 中文 L0106 引用已有的值 env作为当前调用的返回值。
    return env


# 中文 L0109 定义函数 _env_worker。
def _env_worker(pipe: Connection, make: CloudpickleWrapper) -> None:
    # 中文 L0110 原文档中的说明或续行。
    """Worker process: build one env, then run the commands sent on `pipe` until `close`."""
    # 中文 L0111 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0112 调用 make，用它的结果给 “env” 赋值。
        env = make()
    # 中文 L0113 捕获这里列出的异常类型，然后执行下方的处理逻辑。
    except BaseException:
        # 中文 L0114 调用 pipe.send 执行操作。
        pipe.send((False, traceback.format_exc()))
        # 中文 L0115 退出当前调用并返回 None。
        return
    # 中文 L0116 调用 pipe.send 执行操作。
    pipe.send((True, None))
    # 中文 L0117 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
    try:
        # 中文 L0118 持续执行下面的循环体，每轮开始都重新检查条件。
        while True:
            # 中文 L0119 调用 pipe.recv，用它的结果给 “(command, argument)” 赋值。
            command, argument = pipe.recv()
            # 中文 L0120 条件 command == "close" 成立时进入下方分支。
            if command == "close":
                # 中文 L0121 结束所在的最近一层循环。
                break
            # 中文 L0122 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
            try:
                # 中文 L0123 条件 command == "reset" 成立时进入下方分支。
                if command == "reset":
                    # 中文 L0124 声明“result”的类型为 Any，调用 env.reset，用它的结果作为初始值。
                    result: Any = env.reset(seed=argument)
                # 中文 L0125 前面的分支均未命中时，再判断 command == "step"。
                elif command == "step":
                    # 中文 L0126 调用 env.step，用它的结果给 “result” 赋值。
                    result = env.step(argument)
                # 中文 L0127 前面的分支均未命中时，再判断 command == "spaces"。
                elif command == "spaces":
                    # 中文 L0128 调用 _spaces，用它的结果给 “result” 赋值。
                    result = _spaces(env)
                # 中文 L0129 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
                else:  # "attrs"
                    # 中文 L0130 用字典推导式从可迭代对象逐项取值并构造结果给 “result” 赋值。
                    result = {name: getattr(env, name) for name in _ENV_ATTRS if hasattr(env, name)}
                    # 中文 L0131 调用 _max_episode_steps，用它的结果给 “result['_max_episode_steps']” 赋值。
                    result["_max_episode_steps"] = _max_episode_steps(env)
                # 中文 L0132 调用 pipe.send 执行操作。
                pipe.send((True, result))
            # 中文 L0133 捕获这里列出的异常类型，然后执行下方的处理逻辑。
            except Exception:
                # 中文 L0134 调用 pipe.send 执行操作。
                pipe.send((False, traceback.format_exc()))
    # 中文 L0135 离开 try 时执行这里的清理，即使前面的路径发生异常。
    finally:
        # 中文 L0136 调用 env.close，关闭对象或释放关联资源。
        env.close()
        # 中文 L0137 调用 pipe.close，关闭对象或释放关联资源。
        pipe.close()


# 中文 L0140 定义类 _ProcessEnv。
class _ProcessEnv:
    """One environment in its own process. A batch steps in parallel: send all, then receive all.

    Not Gymnasium's `AsyncVectorEnv`, which also steps finished environments (Gymnasium leaves that
    undefined) and builds an extra one in this process to read the spaces.
    """

    # 中文 L0147 定义方法 __init__。
    def __init__(self, make: Callable[[], Any]):
        # 中文 L0148 用条件表达式在 if 前与 else 后的两个值之间选择给 “start_method” 赋值。
        start_method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        # 中文 L0149 声明“context”的类型为 Any，调用 multiprocessing.get_context，用它的结果作为初始值。
        context: Any = multiprocessing.get_context(start_method)
        # 中文 L0150 调用 context.Pipe，用它的结果给 “(self._pipe, child)” 赋值。
        self._pipe, child = context.Pipe()
        # 中文 L0151 调用 context.Process，用它的结果给 “self._process” 赋值。
        self._process = context.Process(
            # 中文 L0152 为“target”提供这一项的值，注意它所在的参数表或外层表达式。
            target=_env_worker, args=(child, CloudpickleWrapper(make)), daemon=True
        # 中文 L0153 闭合上方尚未结束的括号。
        )
        # 中文 L0154 调用 self._process.start，启动线程或进程。
        self._process.start()
        # 中文 L0155 调用 child.close，关闭对象或释放关联资源。
        child.close()
        # 中文 L0156 使用布尔值 False给 “self._close_requested” 赋值。
        self._close_requested = False

    # 中文 L0158 定义方法 ready。
    def ready(self) -> None:
        # 中文 L0159 原文档中的说明或续行。
        """Wait until the environment is built, then read its spaces and attributes."""
        # 中文 L0160 调用 self._receive 执行操作。
        self._receive()
        # 中文 L0161 调用 self._call，用它的结果给 “(self.observation_space, self.action_space)” 赋值。
        self.observation_space, self.action_space = self._call("spaces")
        # 中文 L0162 调用 self._refresh 执行操作。
        self._refresh()

    # 中文 L0164 定义方法 _exit_error。
    def _exit_error(self) -> RuntimeError:
        # 中文 L0165 调用 self._process.join，等待对应线程或进程结束。
        self._process.join(1)
        # 中文 L0166 调用 RuntimeError，用它的结果作为当前调用的返回值。
        return RuntimeError(
            # 中文 L0167 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Environment process exited with code {self._process.exitcode}. If its simulator cannot run in worker "
            # 中文 L0168 字符串字面量，供上方的调用、容器或错误消息使用。
            "processes on this platform, run it in the main process (use_async_envs=false, e.g. "
            # 中文 L0169 字符串字面量，供上方的调用、容器或错误消息使用。
            "--eval.use_async_envs=false)"
        # 中文 L0170 闭合上方尚未结束的括号。
        )

    # 中文 L0172 定义方法 _receive。
    def _receive(self) -> Any:
        # 中文 L0173 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0174 调用 self._pipe.recv，用它的结果给 “(ok, value)” 赋值。
            ok, value = self._pipe.recv()
        # 中文 L0175 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except EOFError:
            # 中文 L0176 抛出 self._exit_error，中断当前路径并交给上层的异常处理。
            raise self._exit_error() from None
        # 中文 L0177 条件 not ok 成立时进入下方分支。
        if not ok:
            # 中文 L0178 抛出 RuntimeError，中断当前路径并交给上层的异常处理。
            raise RuntimeError(f"Environment process failed: {value}")
        # 中文 L0179 引用已有的值 value作为当前调用的返回值。
        return value

    # 中文 L0181 定义方法 _call。
    def _call(self, command: str, argument: Any = None) -> Any:
        # 中文 L0182 调用 self.send 执行操作。
        self.send(command, argument)
        # 中文 L0183 调用 self._receive，用它的结果作为当前调用的返回值。
        return self._receive()

    # 中文 L0185 定义方法 _refresh。
    def _refresh(self) -> None:
        # 中文 L0186 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for name, value in self._call("attrs").items():
            # 中文 L0187 调用 setattr 执行操作。
            setattr(self, name, value)

    # 中文 L0189 定义方法 send。
    def send(self, command: str, argument: Any = None) -> None:
        # 中文 L0190 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0191 调用 self._pipe.send 执行操作。
            self._pipe.send((command, argument))
        # 中文 L0192 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except BrokenPipeError:
            # 中文 L0193 抛出 self._exit_error，中断当前路径并交给上层的异常处理。
            raise self._exit_error() from None

    # 中文 L0195 定义方法 receive。
    def receive(self, refresh: bool = False) -> Any:
        # 中文 L0196 调用 self._receive，用它的结果给 “result” 赋值。
        result = self._receive()
        # 中文 L0197 条件 refresh 成立时进入下方分支。
        if refresh:
            # 中文 L0198 调用 self._refresh 执行操作。
            self._refresh()
        # 中文 L0199 引用已有的值 result作为当前调用的返回值。
        return result

    # 中文 L0201 定义方法 reset。
    def reset(self, seed: int | list[int | None] | None = None) -> Any:
        # 中文 L0202 调用 self.send 执行操作。
        self.send("reset", seed)
        # 中文 L0203 调用 self.receive，用它的结果作为当前调用的返回值。
        return self.receive(refresh=True)

    # 中文 L0205 定义方法 step。
    def step(self, action: np.ndarray) -> Any:
        # 中文 L0206 调用 self.send 执行操作。
        self.send("step", action)
        # 中文 L0207 调用 self.receive，用它的结果作为当前调用的返回值。
        return self.receive()

    # 中文 L0209 定义方法 request_close。
    def request_close(self) -> None:
        # 中文 L0210 原文档中的说明或续行。
        """Ask the process to close its environment and exit, without waiting. Asks only once."""
        # 中文 L0211 条件 self._close_requested 成立时进入下方分支。
        if self._close_requested:
            # 中文 L0212 退出当前调用并返回 None。
            return
        # 中文 L0213 使用布尔值 True给 “self._close_requested” 赋值。
        self._close_requested = True
        # 中文 L0214 进入上下文管理器，缩进块结束时由它处理退出动作。
        with contextlib.suppress(OSError):
            # 中文 L0215 调用 self._pipe.send 执行操作。
            self._pipe.send(("close", None))

    # 中文 L0217 定义方法 close。
    def close(self) -> None:
        # 中文 L0218 原文档中的说明或续行。
        """Close the environment and its process, terminating the process if it still runs after 10 s."""
        # 中文 L0219 调用 self.request_close 执行操作。
        self.request_close()
        # 中文 L0220 调用 self._process.join，等待对应线程或进程结束。
        self._process.join(10)
        # 中文 L0221 条件 self._process.is_alive() 成立时进入下方分支。
        if self._process.is_alive():
            # 中文 L0222 调用 self._process.terminate 执行操作。
            self._process.terminate()
        # 中文 L0223 调用 self._pipe.close，关闭对象或释放关联资源。
        self._pipe.close()


# 中文 L0226 用统一的批量接口包装同一任务的一组仿真环境。
class LocalVectorEnv(gym.vector.VectorEnv):
    """`num_envs` envs of one simulator task, with no autoreset.

    A finished env is not stepped until `reset`: it keeps its last observation and gets zero reward. With
    `use_async_envs`, each env runs in its own worker process.

    An integration that defines `vector_env_fns` builds the whole batch as one Gymnasium vector env instead
    (a GPU simulator stepping every env at once), unless it returns None for this config, as EnvHub does for a
    repo without one. That env must not autoreset; it is stepped as a whole, and
    the rows of finished envs are ignored, so they stay finished as above. With `use_async_envs`, it runs in
    one worker process.
    """

    # 中文 L0239 使用布尔值 False给 “closed” 赋值。
    closed = False

    # 中文 L0241 定义方法 __init__。作用是加载仿真接入并记录环境组配置和特征。
    def __init__(
        # 中文 L0242 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0243 “config”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
        config: SimConfig,
        # 中文 L0244 为“num_envs”（环境实例数）提供这一项的值，注意它所在的参数表或外层表达式。
        num_envs: int = 1,
        # 中文 L0245 为“task_group”（任务套件名）提供这一项的值，注意它所在的参数表或外层表达式。
        task_group: str | None = None,
        # 中文 L0246 为“task_id”（套件内的任务编号）提供这一项的值，注意它所在的参数表或外层表达式。
        task_id: int | None = None,
        # 中文 L0247 为“use_async_envs”提供这一项的值，注意它所在的参数表或外层表达式。
        use_async_envs: bool = False,
        # 中文 L0248 为“descriptor”提供这一项的值，注意它所在的参数表或外层表达式。
        descriptor: EnvDescriptor | None = None,
    # 中文 L0249 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> None:
        """Select the task; given a `descriptor`, build the simulators on first use instead of now.

        Args:
            config (`SimConfig`):
                The simulator to run.
            num_envs (`int`, *optional*, defaults to `1`):
                Number of envs in the batch.
            task_group (`str`, *optional*):
                Suite to run, e.g. `libero_spatial`. Defaults to the integration's first one.
            task_id (`int`, *optional*):
                Task within the suite. Defaults to its first task.
            use_async_envs (`bool`, *optional*, defaults to `False`):
                Run each env in its own worker process.
            descriptor (`EnvDescriptor`, *optional*):
                The env's descriptor, if already known, as for another task of the same simulator.
        """
        # 中文 L0266 进入上下文管理器，缩进块结束时由它处理退出动作。
        with simulator_required(config.type):
            # 中文 L0267 从仿真类型字符串动态找到实现模块，本次是 LIBERO 接入文件。
            integration = importlib.import_module(f"lerobot.sim.integrations.{config.type}")
            # 中文 L0268 调用 getattr，用它的结果给 “vector_env_fns” 赋值。
            vector_env_fns = getattr(integration, "vector_env_fns", None)
            # None from vector_env_fns: this config has no batched sim (e.g. an EnvHub repo without one).
            # 中文 L0270 设为空值 None给 “factories” 赋值。
            factories = None
            # 中文 L0271 条件 vector_env_fns is not None 成立时进入下方分支。
            if vector_env_fns is not None:
                # 中文 L0272 调用 vector_env_fns，用它的结果给 “factories” 赋值。
                factories = vector_env_fns(task=config.task, n_envs=num_envs, **config.kwargs)
            # One vector env for the whole batch, instead of one env per batch entry.
            # 中文 L0274 计算此行的右侧表达式给 “self._batched” 赋值。
            self._batched = factories is not None
            # 中文 L0275 条件 factories is None 成立时进入下方分支。
            if factories is None:
                # 中文 L0276 在 LIBERO 分支向插件索取每个任务的环境构造函数。
                factories = integration.env_fns(task=config.task, n_envs=num_envs, **config.kwargs)
        # 中文 L0277 保存该仿真特有的观测转换函数，在 reset 和 step 之后会调用。
        self._canonical_observation = integration.canonical_observation
        # 中文 L0278 引用已有的值 config给 “self.config” 赋值。
        self.config = config
        # 中文 L0279 引用已有的值 num_envs给 “self.num_envs” 赋值。
        self.num_envs = num_envs
        # 中文 L0280 引用已有的值 use_async_envs给 “self.use_async_envs” 赋值。
        self.use_async_envs = use_async_envs
        # 中文 L0281 声明“self.tasks”的类型为 dict[str, list[int]]，用字典推导式从可迭代对象逐项取值并构造结果作为初始值。
        self.tasks: dict[str, list[int]] = {group: list(tasks) for group, tasks in factories.items()}
        # 中文 L0282 组合多个布尔判断，按 Python 的短路规则求值给 “self.task_group” 赋值。
        self.task_group = task_group or next(iter(factories))
        # 中文 L0283 调用 factories.get，用它的结果给 “task_factories” 赋值。
        task_factories = factories.get(self.task_group, {})
        # Default to the suite's first task id, not 0: a suite can be filtered (e.g. LIBERO's task_ids).
        # 中文 L0285 声明“self.task_id”的类型为 int，用条件表达式在 if 前与 else 后的两个值之间选择作为初始值。
        self.task_id: int = next(iter(task_factories), 0) if task_id is None else task_id
        # 中文 L0286 条件 self.task_id not in task_factories 成立时进入下方分支。
        if self.task_id not in task_factories:
            # 中文 L0287 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"Unknown task: {self.task_group}/{self.task_id}")
        # 中文 L0288 用条件表达式在 if 前与 else 后的两个值之间选择给 “fns” 赋值。
        fns = [task_factories[self.task_id]] if self._batched else task_factories[self.task_id]
        # 中文 L0289 用 partial 为各环境工厂绑定仿真类型，稍后再创建实例。
        self._env_fns: list[Callable[[], Any]] = [partial(_build_env, config.type, fn) for fn in fns]
        # 中文 L0290 声明“self._envs”的类型为 list[Any]，创建列表，按给定顺序保留元素作为初始值。
        self._envs: list[Any] = []
        # 中文 L0291 声明“self._last”的类型为 list[_LastStep]，创建列表，按给定顺序保留元素作为初始值。
        self._last: list[_LastStep] = []
        # 中文 L0292 声明“self._elapsed_steps”的类型为 np.ndarray，调用 np.zeros，用它的结果作为初始值。
        self._elapsed_steps: np.ndarray = np.zeros(num_envs, dtype=np.int64)
        # 中文 L0293 声明“self._max_steps”的类型为 list[int]，创建列表，按给定顺序保留元素作为初始值。
        self._max_steps: list[int] = []
        # 中文 L0294 声明“self._latest”的类型为 StepResult | None，设为空值 None作为初始值。
        self._latest: StepResult | None = None
        # 中文 L0295 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0296 组合多个布尔判断，按 Python 的短路规则求值给 “self.descriptor” 赋值。
            self.descriptor = descriptor or self._describe()
        # 中文 L0297 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except BaseException:
            # 中文 L0298 调用 self._close_envs 执行操作。
            self._close_envs()
            # 中文 L0299 抛出 异常，中断当前路径并交给上层的异常处理。
            raise
        # 中文 L0300 调用 self.descriptor.observation_space，用它的结果给 “self.single_observation_space” 赋值。
        self.single_observation_space = self.descriptor.observation_space()
        # 中文 L0301 调用 self.descriptor.action_space，用它的结果给 “self.single_action_space” 赋值。
        self.single_action_space = self.descriptor.action_space()
        # 中文 L0302 从单环境观测空间得到批量观测空间，最前面新增环境轴。
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        # 中文 L0303 同样将单环境动作形状 [7] 扩展为批量动作形状 [B,7]。
        self.action_space = batch_space(self.single_action_space, num_envs)
        # 中文 L0304 创建字典，用键查找对应的值给 “self.metadata” 赋值。
        self.metadata = {
            # 中文 L0305 字典项“render_fps”，冒号右边给出它的值。
            "render_fps": config.fps,
            # 中文 L0306 字典项“autoreset_mode”，冒号右边给出它的值。
            "autoreset_mode": _NO_AUTORESET,
            # 中文 L0307 声明输出观测已经使用 LeRobot 标准键，rollout 根据它选择转换函数。
            "canonical_observations": True,
        # 中文 L0308 闭合上方尚未结束的括号。
        }
        # 中文 L0309 使用字面值 'rgb_array'给 “self.render_mode” 赋值。
        self.render_mode = "rgb_array"

    # 中文 L0311 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0312 定义方法 latest。
    def latest(self) -> StepResult:
        # 中文 L0313 原文档中的说明或续行。
        """The result of the last reset or step."""
        # 中文 L0314 条件 self._latest is None 成立时进入下方分支。
        if self._latest is None:
            # 中文 L0315 抛出 RuntimeError，中断当前路径并交给上层的异常处理。
            raise RuntimeError("Reset before reading the environment")
        # 中文 L0316 读取属性 self._latest作为当前调用的返回值。
        return self._latest

    # 中文 L0318 重置每个环境，返回一批初始观测和信息。
    def reset(
        # 中文 L0319 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0320 后续参数必须按名字传入。
        *,
        # 中文 L0321 为“seed”（随机种子）提供这一项的值，注意它所在的参数表或外层表达式。
        seed: int | list[int | None] | None = None,
        # 中文 L0322 为“options”提供这一项的值，注意它所在的参数表或外层表达式。
        options: dict[str, Any] | None = None,
    # 中文 L0323 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # 中文 L0324 原文档概述本段功能，执行批量重置并收集初始观测。
        """Reset every env; an int `seed` gives env `i` the seed `seed + i`."""
        # 中文 L0325 条件 options 成立时进入下方分支。
        if options:
            # 中文 L0326 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("This environment does not define reset options")
        # 中文 L0327 条件 isinstance(seed, int) 成立时进入下方分支。
        if isinstance(seed, int):
            # 中文 L0328 若只得到一个整数，则为每个环境分配 seed 加其下标。
            seeds: list[int | None] = [seed + index for index in range(self.num_envs)]
        # 中文 L0329 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0330 用条件表达式在 if 前与 else 后的两个值之间选择给 “seeds” 赋值。
            seeds = list(seed) if seed is not None else [None] * self.num_envs
        # 中文 L0331 条件 len(seeds) != self.num_envs 成立时进入下方分支。
        if len(seeds) != self.num_envs:
            # 中文 L0332 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("Seed count must match the batch size")
        # 中文 L0333 调用 self._build 执行操作。
        self._build()
        # 中文 L0334 先得到底层各环境的 NumPy 观测，再逐个整理为 LeRobot 键并缓存。
        self._last = [_LastStep(self._canonical_observation(obs)) for obs in self._reset_envs(seeds)]
        # 中文 L0335 使用字面值 0给 “self._elapsed_steps[:]” 赋值。
        self._elapsed_steps[:] = 0
        # 中文 L0336 调用 self._snapshot，用它的结果给 “result” 赋值。
        result = self._snapshot()
        # 中文 L0337 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return result.obs, result.info()

    # 中文 L0339 让仍在运行的各个仿真实例各前进一步。
    def step(
        # 中文 L0340 延续上面的语句，self, actions: np.ndarray。本段处理批量推进仿真并整理奖励、终止和成功标志。
        self, actions: np.ndarray
    # 中文 L0341 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        # 中文 L0342 原文档概述本段功能，批量推进仿真并整理奖励、终止和成功标志。
        """Step every unfinished env once; finished envs are not stepped until `reset`."""
        # 中文 L0343 条件 not self._last 成立时进入下方分支。
        if not self._last:
            # 中文 L0344 抛出 RuntimeError，中断当前路径并交给上层的异常处理。
            raise RuntimeError("Reset before stepping")
        # 中文 L0345 把所有动作转为 NumPy float32。本次形状是 [1,7]。
        actions = np.asarray(actions, dtype=np.float32)
        # 中文 L0346 检查交给向量环境的动作带有与实例数量一致的批次轴。
        validate_array(actions, self.descriptor.action_feature, leading_shape=(self.num_envs,))
        # Finished envs keep their terminal observation and flags, with zero reward.
        # 中文 L0348 已结束的环境保留终止观测和标记，本步将奖励置零以便统一批量返回。
        self._last = [replace(last, reward=0.0) if last.done else last for last in self._last]
        # 中文 L0349 记下还未结束的环境编号，本步只给它们应用动作。
        running = [i for i, last in enumerate(self._last) if not last.done]
        # 中文 L0350 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for i, obs, reward, terminated, truncated, success, metrics in self._step_envs(actions, running):
            # 中文 L0351 用增量赋值更新 self._elapsed_steps[i]，运算逻辑见左侧运算符。
            self._elapsed_steps[i] += 1
            # 中文 L0352 调用 bool，用它的结果给 “truncated”（截断标志） 赋值。
            truncated = bool(truncated or self._elapsed_steps[i] >= self._max_steps[i])
            # 中文 L0353 将本步仿真观测整理为可用于策略推理的标准 NumPy 观测。
            observation = self._canonical_observation(obs)
            # 中文 L0354 条件 self._batched and not self.use_async_envs and (terminated or truncated) 成立时进入下方分支。
            if self._batched and not self.use_async_envs and (terminated or truncated):
                # A native vector env can overwrite its observation buffers while stepping other rows.
                # 中文 L0356 用字典推导式从可迭代对象逐项取值并构造结果给 “observation”（观测） 赋值。
                observation = {name: value.copy() for name, value in observation.items()}
            # 中文 L0357 调用 _LastStep，用它的结果给 “self._last[i]” 赋值。
            self._last[i] = _LastStep(
                # 中文 L0358 为“obs”提供这一项的值，注意它所在的参数表或外层表达式。
                obs=observation,
                # 中文 L0359 为“reward”（单步奖励）提供这一项的值，注意它所在的参数表或外层表达式。
                reward=float(reward),
                # 中文 L0360 为“terminated”（任务终止标志）提供这一项的值，注意它所在的参数表或外层表达式。
                terminated=bool(terminated),
                # 中文 L0361 为“truncated”（截断标志）提供这一项的值，注意它所在的参数表或外层表达式。
                truncated=truncated,
                # 中文 L0362 把仿真成功判定保存为普通 Python 布尔值。
                is_success=bool(success),
                # 中文 L0363 为“metrics”提供这一项的值，注意它所在的参数表或外层表达式。
                metrics={**self._last[i].metrics, **{name: float(value) for name, value in metrics.items()}},
            # 中文 L0364 闭合上方尚未结束的括号。
            )
        # 中文 L0365 调用 self._snapshot，用它的结果给 “result” 赋值。
        result = self._snapshot()
        # 中文 L0366 返回批量观测、奖励、终止、截断与包含成功标志的信息。
        return result.obs, result.reward, result.terminated, result.truncated, result.info()

    # 中文 L0368 定义方法 call。
    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        # 中文 L0369 原文档中的说明或续行。
        """Return a supported per-environment attribute."""
        # 中文 L0370 条件 args or kwargs 成立时进入下方分支。
        if args or kwargs:
            # 中文 L0371 抛出 TypeError，中断当前路径并交给上层的异常处理。
            raise TypeError("Environment attributes do not take arguments")
        # 中文 L0372 条件 name == "_max_episode_steps" 成立时进入下方分支。
        if name == "_max_episode_steps":
            # The descriptor may come from another task, whose episode limit can differ: read this task's own.
            # 中文 L0374 调用 self._build 执行操作。
            self._build()
            # 中文 L0375 计算右侧表达式 return (self._max_steps[0],) * self.num_envs作为当前调用的返回值。
            return (self._max_steps[0],) * self.num_envs
        # 中文 L0376 条件 name == "task" 成立时进入下方分支。
        if name == "task":
            # 中文 L0377 计算右侧表达式 return (self.task_group,) * self.num_envs作为当前调用的返回值。
            return (self.task_group,) * self.num_envs
        # 中文 L0378 条件 name == "task_description" 成立时进入下方分支。
        if name == "task_description":
            # 中文 L0379 读取属性 self.latest.task作为当前调用的返回值。
            return self.latest.task
        # 中文 L0380 条件 name == "render" 成立时进入下方分支。
        if name == "render":
            # 中文 L0381 调用 self.render，用它的结果作为当前调用的返回值。
            return self.render()
        # 中文 L0382 抛出 AttributeError，中断当前路径并交给上层的异常处理。
        raise AttributeError(f"Environment attribute is not exposed: {name}")

    # 中文 L0384 定义方法 get_attr。
    def get_attr(self, name: str) -> tuple[Any, ...]:
        # 中文 L0385 原文档中的说明或续行。
        """Return a supported per-environment attribute."""
        # 中文 L0386 调用 self.call，用它的结果作为当前调用的返回值。
        return self.call(name)

    # 中文 L0388 定义方法 render。
    def render(self) -> tuple[np.ndarray, ...]:
        # 中文 L0389 原文档中的说明或续行。
        """Return each env's latest RGB frame, taken from the last observation instead of a new render."""
        # 中文 L0390 调用 self.latest.frames，用它的结果作为当前调用的返回值。
        return self.latest.frames()

    # 中文 L0392 定义方法 close。
    def close(self, **kwargs: Any) -> None:
        """Release simulator resources, once.

        Defined here rather than through Gymnasium's `close_extras`: Gymnasium 0.29's `close` reads attributes
        its `VectorEnv.__init__` sets, which isn't called here.
        """
        # 中文 L0398 条件 not self.closed 成立时进入下方分支。
        if not self.closed:
            # 中文 L0399 调用 self._close_envs 执行操作。
            self._close_envs()
            # 中文 L0400 使用布尔值 True给 “self.closed” 赋值。
            self.closed = True

    # 中文 L0402 定义方法 _build。
    def _build(self) -> None:
        # 中文 L0403 原文档中的说明或续行。
        """Build the simulators if not built yet, so each task builds on first use."""
        # 中文 L0404 条件 self._envs 成立时进入下方分支。
        if self._envs:
            # 中文 L0405 退出当前调用并返回 None。
            return
        # 中文 L0406 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
        try:
            # 中文 L0407 条件 self.use_async_envs 成立时进入下方分支。
            if self.use_async_envs:
                # Start every process before waiting, so the environments build in parallel.
                # 中文 L0409 调用 self._envs.extend，将可迭代对象展开后追加到末尾。
                self._envs.extend(_ProcessEnv(fn) for fn in self._env_fns)
                # 中文 L0410 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for env in self._envs:
                    # 中文 L0411 调用 env.ready 执行操作。
                    env.ready()
            # 中文 L0412 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0413 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for fn in self._env_fns:
                    # 中文 L0414 调用 self._envs.append，在列表末尾追加一个元素。
                    self._envs.append(fn())
            # 中文 L0415 条件 self._batched and getattr(self._envs[0], "num_envs", None) != self.num_envs 成立时进入下方分支。
            if self._batched and getattr(self._envs[0], "num_envs", None) != self.num_envs:
                # 中文 L0416 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0417 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"vector_env_fns of {self.config.type} built {getattr(self._envs[0], 'num_envs', None)} "
                    # 中文 L0418 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"envs, expected {self.num_envs}"
                # 中文 L0419 闭合上方尚未结束的括号。
                )
        # 中文 L0420 捕获这里列出的异常类型，然后执行下方的处理逻辑。
        except BaseException:
            # 中文 L0421 调用 self._close_envs 执行操作。
            self._close_envs()
            # 中文 L0422 抛出 异常，中断当前路径并交给上层的异常处理。
            raise
        # A vector env has one limit, for all of its envs.
        # 中文 L0424 计算右侧表达式 [_max_episode_steps(env) for env in self._envs] * (self.num_envs if self._batched else 1)给 “limits” 赋值。
        limits = [_max_episode_steps(env) for env in self._envs] * (self.num_envs if self._batched else 1)
        # 中文 L0425 条件 None in limits 成立时进入下方分支。
        if None in limits:
            # 中文 L0426 调用 logger.warning 执行操作。
            logger.warning(
                # 中文 L0427 字符串字面量，供上方的调用、容器或错误消息使用。
                "%s declares no episode limit; episodes are truncated after %d steps",
                # 中文 L0428 延续上面的语句，self.config.type,。
                self.config.type,
                # 中文 L0429 “_DEFAULT_MAX_EPISODE_STEPS”是当前括号中的一项，逗号将它与下一项分开。
                _DEFAULT_MAX_EPISODE_STEPS,
            # 中文 L0430 闭合上方尚未结束的括号。
            )
        # 中文 L0431 用列表推导式从可迭代对象逐项取值并构造结果给 “self._max_steps” 赋值。
        self._max_steps = [limit or _DEFAULT_MAX_EPISODE_STEPS for limit in limits]

    # 中文 L0433 定义方法 _describe。
    def _describe(self) -> EnvDescriptor:
        # 中文 L0434 原文档中的说明或续行。
        """Build the envs and describe the first one's observations (with LeRobot keys) and actions."""
        # 中文 L0435 调用 self._build 执行操作。
        self._build()
        # 中文 L0436 调用 _spaces，用它的结果给 “(observation_space, action_space)” 赋值。
        observation_space, action_space = _spaces(self._envs[0])
        # 中文 L0437 条件 not isinstance(observation_space, spaces.Dict) 成立时进入下方分支。
        if not isinstance(observation_space, spaces.Dict):
            # 中文 L0438 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0439 字符串字面量，供上方的调用、容器或错误消息使用。
                "LeRobot simulation needs dict observations "
                # 中文 L0440 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"(pixels, agent_pos, environment_state), got {observation_space}"
            # 中文 L0441 闭合上方尚未结束的括号。
            )
        # 中文 L0442 条件 not isinstance(action_space, spaces.Box) or len(action_space.shape) != 1 成立时进入下方分支。
        if not isinstance(action_space, spaces.Box) or len(action_space.shape) != 1:
            # 中文 L0443 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"LeRobot simulation needs a 1-D Box action space, got {action_space}")
        # 中文 L0444 读取属性 self.config给 “config”（配置对象） 赋值。
        config = self.config
        # 中文 L0445 创建列表，按给定顺序保留元素给 “features” 赋值。
        features = []
        # 中文 L0446 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for name, value in self._canonical_observation(observation_space.sample()).items():
            # 中文 L0447 用条件表达式在 if 前与 else 后的两个值之间选择给 “names” 赋值。
            names = config.state_names if name == OBS_STATE else ()
            # 中文 L0448 条件 not names and value.ndim == 1 成立时进入下方分支。
            if not names and value.ndim == 1:
                # 中文 L0449 调用 tuple，用它的结果给 “names” 赋值。
                names = tuple(f"state_{index}" for index in range(value.shape[0]))
            # 中文 L0450 用条件表达式在 if 前与 else 后的两个值之间选择给 “kind” 赋值。
            kind = "rgb" if value.dtype == np.uint8 else "tensor"
            # 中文 L0451 调用 features.append，在列表末尾追加一个元素。
            features.append(FeatureSpec(name, value.shape, value.dtype.name, kind, names, config.semantics))
        # 中文 L0452 用方括号按键或索引读取 action_space.shape给 “dim” 赋值。
        dim = action_space.shape[0]
        # 中文 L0453 组合多个布尔判断，按 Python 的短路规则求值给 “action_names” 赋值。
        action_names = config.action_names or tuple(f"command_{index}" for index in range(dim))
        # 中文 L0454 调用 FeatureSpec，用它的结果给 “action”（单步动作） 赋值。
        action = FeatureSpec(ACTION, (dim,), "float32", names=action_names, semantics=config.semantics)
        # 中文 L0455 调用 EnvDescriptor，用它的结果作为当前调用的返回值。
        return EnvDescriptor(
            # 中文 L0456 为“sim_type”提供这一项的值，注意它所在的参数表或外层表达式。
            sim_type=config.type,
            # 中文 L0457 为“semantics”提供这一项的值，注意它所在的参数表或外层表达式。
            semantics=config.semantics,
            # 中文 L0458 为“features”提供这一项的值，注意它所在的参数表或外层表达式。
            features=tuple(features),
            # 中文 L0459 为“action_feature”提供这一项的值，注意它所在的参数表或外层表达式。
            action_feature=action,
            # 中文 L0460 为“control_mode”提供这一项的值，注意它所在的参数表或外层表达式。
            control_mode=config.control_mode,
            # 中文 L0461 为“fps”提供这一项的值，注意它所在的参数表或外层表达式。
            fps=config.fps,
            # 中文 L0462 为“tasks”提供这一项的值，注意它所在的参数表或外层表达式。
            tasks=self.tasks,
            # 中文 L0463 为“max_episode_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            max_episode_steps=self._max_steps[0],
            # 中文 L0464 为“robot_type”提供这一项的值，注意它所在的参数表或外层表达式。
            robot_type=config.robot_type,
            # 中文 L0465 为“config”（配置对象）提供这一项的值，注意它所在的参数表或外层表达式。
            config=asdict(config),
        # 中文 L0466 闭合上方尚未结束的括号。
        )

    # 中文 L0468 定义方法 _reset_envs。
    def _reset_envs(self, seeds: list[int | None]) -> list[Any]:
        # 中文 L0469 原文档中的说明或续行。
        """Reset the simulators, one seed per env, and return each env's raw observation."""
        # 中文 L0470 条件 self._batched 成立时进入下方分支。
        if self._batched:
            # 中文 L0471 调用 reset，用它的结果给 “(batch, _)” 赋值。
            batch, _ = self._envs[0].reset(seed=seeds)
            # 中文 L0472 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
            return [_row(batch, i) for i in range(self.num_envs)]
        # 中文 L0473 条件 self.use_async_envs 成立时进入下方分支。
        if self.use_async_envs:
            # Send every reset before waiting, so the processes reset in parallel.
            # 中文 L0475 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for env, env_seed in zip(self._envs, seeds, strict=True):
                # 中文 L0476 调用 env.send 执行操作。
                env.send("reset", env_seed)
            # 中文 L0477 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
            return [env.receive(refresh=True)[0] for env in self._envs]
        # 中文 L0478 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
        return [env.reset(seed=env_seed)[0] for env, env_seed in zip(self._envs, seeds, strict=True)]

    # 中文 L0480 按环境索引分发动作，并整合每个环境报告的本步结果。
    def _step_envs(self, actions: np.ndarray, running: list[int]) -> list[tuple[Any, ...]]:
        """Step the envs in `running`.

        Return `(i, obs, reward, terminated, truncated, success, metrics)` for each.
        """
        # 中文 L0485 条件 self._batched 成立时进入下方分支。
        if self._batched:
            # 中文 L0486 条件 not running 成立时进入下方分支。
            if not running:
                # 中文 L0487 创建列表，按给定顺序保留元素作为当前调用的返回值。
                return []
            # One vector env steps every env; the rows of finished envs are dropped.
            # 中文 L0489 调用 step，用它的结果给 “(batch, rewards, terminated, truncated, info)” 赋值。
            batch, rewards, terminated, truncated, info = self._envs[0].step(actions)
            # 中文 L0490 调用 info.get，用它的结果给 “success” 赋值。
            success = info.get("is_success", info.get("success", np.zeros(self.num_envs, dtype=bool)))
            # 中文 L0491 组合多个布尔判断，按 Python 的短路规则求值给 “metrics” 赋值。
            metrics = info.get("metrics") or {}
            # 中文 L0492 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
            return [
                # 中文 L0493 延续上面的语句，(i, _row(batch, i), rewards[i], terminated[i], truncated[i], success[i], _row(metrics, i))。本段处理把批量动作分给对应仿真实例。
                (i, _row(batch, i), rewards[i], terminated[i], truncated[i], success[i], _row(metrics, i))
                # 中文 L0494 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for i in running
            # 中文 L0495 闭合上方尚未结束的括号。
            ]
        # 中文 L0496 条件 self.use_async_envs 成立时进入下方分支。
        if self.use_async_envs:
            # Send every step before waiting, so the processes step in parallel.
            # 中文 L0498 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for i in running:
                # 中文 L0499 调用 send 执行操作。
                self._envs[i].send("step", actions[i])
            # 中文 L0500 用列表推导式从可迭代对象逐项取值并构造结果给 “results” 赋值。
            results = [self._envs[i].receive() for i in running]
        # 中文 L0501 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0502 原评测走这个同步分支，actions[i] 去掉环境轴，传给单个 LiberoEnv 的动作形状为 [7]。
            results = [self._envs[i].step(actions[i]) for i in running]
        # 中文 L0503 用列表推导式从可迭代对象逐项取值并构造结果作为当前调用的返回值。
        return [
            # 中文 L0504 延续上面的语句，(。本段处理把批量动作分给对应仿真实例。
            (
                # 中文 L0505 “i”是当前括号中的一项，逗号将它与下一项分开。
                i,
                # 中文 L0506 “obs”是当前括号中的一项，逗号将它与下一项分开。
                obs,
                # 中文 L0507 “reward”（单步奖励）是当前括号中的一项，逗号将它与下一项分开。
                reward,
                # 中文 L0508 “terminated”（任务终止标志）是当前括号中的一项，逗号将它与下一项分开。
                terminated,
                # 中文 L0509 “truncated”（截断标志）是当前括号中的一项，逗号将它与下一项分开。
                truncated,
                # 中文 L0510 先读取仿真报告的 is_success，也兼容名为 success 的字段。
                info.get("is_success", info.get("success", False)),
                # 中文 L0511 延续上面的语句，info.get("metrics") or {},。本段处理把批量动作分给对应仿真实例。
                info.get("metrics") or {},
            # 中文 L0512 闭合上方尚未结束的括号。
            )
            # 中文 L0513 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for i, (obs, reward, terminated, truncated, info) in zip(running, results, strict=True)
        # 中文 L0514 闭合上方尚未结束的括号。
        ]

    # 中文 L0516 定义方法 _snapshot。作用是将每个环境状态整理成批量结果。
    def _snapshot(self) -> StepResult:
        # 中文 L0517 原文档概述本段功能，将每个环境状态整理成批量结果。
        """Return the batch's latest result, and keep it as `latest`."""
        # 中文 L0518 对每个观测键把所有实例的数组叠成一批。相机从 HWC 变成 BHWC。
        obs = {key: np.stack([last.obs[key] for last in self._last]) for key in self._last[0].obs}
        # A metric an env has not reported in this episode reads as zero.
        # 中文 L0520 调用 dict.fromkeys，用它的结果给 “names” 赋值。
        names = dict.fromkeys(name for last in self._last for name in last.metrics)
        # 中文 L0521 调用 tuple，用它的结果给 “task”（任务） 赋值。
        task = tuple(
            # 中文 L0522 延续上面的语句，str(getattr(env, "task_description", getattr(env, "task", self.task_group))) for env in self._envs。本段处理将每个环境状态整理成批量结果。
            str(getattr(env, "task_description", getattr(env, "task", self.task_group))) for env in self._envs
        # 中文 L0523 闭合上方尚未结束的括号。
        )
        # 中文 L0524 调用 StepResult，用它的结果给 “self._latest” 赋值。
        self._latest = StepResult(
            # 中文 L0525 为“obs”提供这一项的值，注意它所在的参数表或外层表达式。
            obs=obs,
            # A vector env has one task description, for all of its envs.
            # 中文 L0527 为“task”（任务）提供这一项的值，注意它所在的参数表或外层表达式。
            task=task * self.num_envs if self._batched else task,
            # 中文 L0528 为“reward”（单步奖励）提供这一项的值，注意它所在的参数表或外层表达式。
            reward=np.array([last.reward for last in self._last], dtype=np.float32),
            # 中文 L0529 为“terminated”（任务终止标志）提供这一项的值，注意它所在的参数表或外层表达式。
            terminated=np.array([last.terminated for last in self._last]),
            # 中文 L0530 为“truncated”（截断标志）提供这一项的值，注意它所在的参数表或外层表达式。
            truncated=np.array([last.truncated for last in self._last]),
            # 中文 L0531 把每个环境的成功标志堆为一个大小为 B 的 NumPy 数组。
            is_success=np.array([last.is_success for last in self._last]),
            # 中文 L0532 为“elapsed_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            elapsed_steps=self._elapsed_steps.copy(),
            # 中文 L0533 为“metrics”提供这一项的值，注意它所在的参数表或外层表达式。
            metrics={
                # 中文 L0534 为“name”提供这一项的值，注意它所在的参数表或外层表达式。
                name: np.array([last.metrics.get(name, 0.0) for last in self._last], dtype=np.float32)
                # 中文 L0535 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for name in names
            # 中文 L0536 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
            },
        # 中文 L0537 闭合上方尚未结束的括号。
        )
        # 中文 L0538 读取属性 self._latest作为当前调用的返回值。
        return self._latest

    # 中文 L0540 定义方法 _close_envs。
    def _close_envs(self) -> None:
        # 中文 L0541 原文档中的说明或续行。
        """Close every env, even if one fails to close."""
        # 中文 L0542 打包成元组，元素顺序由逗号分隔的位置决定给 “(envs, self._envs)” 赋值。
        envs, self._envs = self._envs, []
        # 中文 L0543 条件 self.use_async_envs 成立时进入下方分支。
        if self.use_async_envs:
            # Ask every process first, so they shut down in parallel instead of one after another.
            # 中文 L0545 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for env in envs:
                # 中文 L0546 调用 env.request_close 执行操作。
                env.request_close()
        # 中文 L0547 进入上下文管理器，缩进块结束时由它处理退出动作。
        with contextlib.ExitStack() as stack:
            # 中文 L0548 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for env in envs:
                # 中文 L0549 调用 stack.callback 执行操作。
                stack.callback(env.close)
