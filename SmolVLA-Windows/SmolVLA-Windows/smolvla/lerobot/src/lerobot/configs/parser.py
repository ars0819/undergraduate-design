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
import importlib
import inspect
import json
import pkgutil
import sys
import tempfile
from argparse import ArgumentError
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from pathlib import Path
from pkgutil import ModuleInfo
from types import ModuleType
from typing import Any, TypeVar, cast

import draccus
import torch
import yaml
from draccus.help_formatter import SimpleHelpFormatter
from draccus.utils import DecodingError
from draccus.wrappers import DataclassWrapper
from draccus.wrappers.choice_wrapper import ChoiceWrapper, UnionWrapper
from draccus.wrappers.field_wrapper import FieldWrapper
from draccus.wrappers.suppressing_argparse import SuppressingArgumentParser
from draccus.wrappers.wrapper import AggregateWrapper, Wrapper

from lerobot.utils.utils import has_method

F = TypeVar("F", bound=Callable[..., object])

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"

# Storage for path args extracted from YAML/JSON config files, so that
# get_path_arg() can find them even when they weren't passed via CLI.
_config_path_args: dict[str, str] = {}

# Storage for non-path YAML overrides so validate() can pass them to from_pretrained.
_config_yaml_overrides: dict[str, list[str]] = {}


# Register draccus encode/decode functions for the torch.dtype values held by dtype fields.
@draccus.decode.register(torch.dtype)
def _decode_dtype(value: Any, path: Sequence[str] = ()) -> torch.dtype:
    """Resolve a dtype name coming from config.json or the command line.

    Args:
        value: The raw serialized value, e.g. "bfloat16".
        path: The field breadcrumb draccus prefixes onto error messages, e.g. ("policy", "dtype").
    """
    dtype = getattr(torch, str(value).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid dtype {value!r}: expected a torch dtype name.")
    return dtype


@draccus.encode.register(torch.dtype)
def _encode_dtype(value: torch.dtype, declared_type: type | None = None) -> str:
    """Serialize a dtype to its bare name, e.g. torch.bfloat16 -> "bfloat16"."""
    return str(value).removeprefix("torch.")


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0075 定义函数 _flatten_to_cli_args。
def _flatten_to_cli_args(d: dict, prefix: str = "") -> list[str]:
    # 中文 L0076 原文档中的说明或续行。
    """Recursively flatten a nested dict to CLI-style args (e.g. {"lr": 1e-4} -> ["--lr=0.0001"])."""
    # 中文 L0077 创建列表，按给定顺序保留元素给 “args”（位置参数序列） 赋值。
    args = []
    # 中文 L0078 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for key, value in d.items():
        # 中文 L0079 条件 key in (PATH_KEY, draccus.CHOICE_TYPE_KEY) 成立时进入下方分支。
        if key in (PATH_KEY, draccus.CHOICE_TYPE_KEY):
            # 中文 L0080 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0081 用条件表达式在 if 前与 else 后的两个值之间选择给 “full_key” 赋值。
        full_key = f"{prefix}.{key}" if prefix else key
        # 中文 L0082 条件 isinstance(value, bool) 成立时进入下方分支。
        if isinstance(value, bool):
            # 中文 L0083 调用 lower，用它的结果给 “value” 赋值。
            value = str(value).lower()
        # 中文 L0084 条件 isinstance(value, dict) 成立时进入下方分支。
        if isinstance(value, dict):
            # 中文 L0085 调用 args.extend，将可迭代对象展开后追加到末尾。
            args.extend(_flatten_to_cli_args(value, full_key))
        # 中文 L0086 前面的分支均未命中时，再判断 isinstance(value, list)。
        elif isinstance(value, list):
            # 中文 L0087 调用 args.append，在列表末尾追加一个元素。
            args.append(f"--{full_key}={json.dumps(value)}")
        # 中文 L0088 前面的分支均未命中时，再判断 value is not None。
        elif value is not None:
            # 中文 L0089 调用 args.append，在列表末尾追加一个元素。
            args.append(f"--{full_key}={value}")
    # 中文 L0090 引用已有的值 args作为当前调用的返回值。
    return args


# 中文 L0093 定义函数 get_cli_overrides。作用是从命令行提取指定配置分组的覆盖设置。
def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Parses arguments from cli at a given nested attribute level.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    # 中文 L0102 条件 args is None 成立时进入下方分支。
    if args is None:
        # 中文 L0103 对 sys.argv 做切片，右端索引不包含在结果中给 “args”（位置参数序列） 赋值。
        args = sys.argv[1:]
    # 中文 L0104 创建列表，按给定顺序保留元素给 “attr_level_args” 赋值。
    attr_level_args = []
    # 中文 L0105 计算此行的右侧表达式给 “detect_string” 赋值。
    detect_string = f"--{field_name}."
    # 中文 L0106 打包成元组，元素顺序由逗号分隔的位置决定给 “excluded_names” 赋值。
    excluded_names = (draccus.CHOICE_TYPE_KEY, PATH_KEY)
    # 中文 L0107 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for index, arg in enumerate(args):
        # 中文 L0108 条件 not arg.startswith(detect_string) 成立时进入下方分支。
        if not arg.startswith(detect_string):
            # 中文 L0109 跳过这一轮剩余操作，进入下一轮循环。
            continue

        # 中文 L0111 调用 arg.removeprefix，用它的结果给 “denested_arg” 赋值。
        denested_arg = arg.removeprefix(detect_string)
        # 中文 L0112 条件 denested_arg.split("=", maxsplit=1)[0] in excluded_names 成立时进入下方分支。
        if denested_arg.split("=", maxsplit=1)[0] in excluded_names:
            # 中文 L0113 跳过这一轮剩余操作，进入下一轮循环。
            continue

        # 中文 L0115 调用 attr_level_args.append，在列表末尾追加一个元素。
        attr_level_args.append(f"--{denested_arg}")
        # 中文 L0116 条件 "=" not in arg and index + 1 < len(args) and not args[index + 1].startswith("--") 成立时进入下方分支。
        if "=" not in arg and index + 1 < len(args) and not args[index + 1].startswith("--"):
            # 中文 L0117 调用 attr_level_args.append，在列表末尾追加一个元素。
            attr_level_args.append(args[index + 1])

    # 中文 L0119 引用已有的值 attr_level_args作为当前调用的返回值。
    return attr_level_args


# 中文 L0122 定义函数 parse_arg。
def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    # 中文 L0123 条件 args is None 成立时进入下方分支。
    if args is None:
        # 中文 L0124 对 sys.argv 做切片，右端索引不包含在结果中给 “args”（位置参数序列） 赋值。
        args = sys.argv[1:]
    # 中文 L0125 计算此行的右侧表达式给 “option” 赋值。
    option = f"--{arg_name}"
    # 中文 L0126 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for index, arg in enumerate(args):
        # 中文 L0127 条件 arg.startswith(f"{option}=") 成立时进入下方分支。
        if arg.startswith(f"{option}="):
            # 中文 L0128 调用 arg.removeprefix，用它的结果作为当前调用的返回值。
            return arg.removeprefix(f"{option}=")
        # 中文 L0129 条件 arg == option and index + 1 < len(args) and not args[index + 1].startswith("--") 成立时进入下方分支。
        if arg == option and index + 1 < len(args) and not args[index + 1].startswith("--"):
            # 中文 L0130 用方括号按键或索引读取 args作为当前调用的返回值。
            return args[index + 1]
    # 中文 L0131 设为空值 None作为当前调用的返回值。
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict[str, str]:
    """Parse plugin-related arguments from command-line arguments.

    This function extracts arguments from command-line arguments that match a specified suffix pattern.
    It accepts arguments in the formats '--key=value' and '--key value' and returns them as a dictionary.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        cli_args (Sequence[str]): A sequence of command-line arguments to parse.

    Returns:
        dict: A dictionary containing the parsed plugin arguments where:
            - Keys are the argument names (with '--' prefix removed if present)
            - Values are the corresponding argument values

    Example:
        >>> args = ["--env.discover_packages_path=my_package", "--other_arg=value"]
        >>> parse_plugin_args("discover_packages_path", args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for index, arg in enumerate(args):
        if not arg.startswith("--"):
            continue

        key, separator, value = arg[2:].partition("=")
        if plugin_arg_suffix not in key:
            continue
        if not separator:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                continue
            value = args[index + 1]
        plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path.

    This function attempts to load a plugin by importing its package and any submodules.
    Plugin registration is expected to happen during package initialization, i.e. when
    the package is imported the gym environment should be registered and the config classes
    registered with their parents using the `register_subclass` decorator.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g. "mypackage.plugins.myplugin")

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or if the package path is invalid.

    Examples:
        >>> load_plugin("external_plugin.core")  # Loads plugin from external package

    Notes:
        - The plugin package should handle its own registration during import
        - All submodules in the plugin package will be imported
        - Implementation follows the plugin discovery pattern from Python packaging guidelines

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg: ModuleType) -> Iterable[ModuleInfo]:
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


# 中文 L0218 定义函数 get_path_arg。
def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    # 中文 L0219 调用 parse_arg，用它的结果给 “result” 赋值。
    result = parse_arg(f"{field_name}.{PATH_KEY}", args)
    # 中文 L0220 条件 result is None 成立时进入下方分支。
    if result is None:
        # 中文 L0221 调用 _config_path_args.get，用它的结果给 “result” 赋值。
        result = _config_path_args.get(field_name)
    # 中文 L0222 引用已有的值 result作为当前调用的返回值。
    return result


# 中文 L0225 定义函数 get_yaml_overrides。
def get_yaml_overrides(field_name: str) -> list[str]:
    # 中文 L0226 调用 _config_yaml_overrides.get，用它的结果作为当前调用的返回值。
    return _config_yaml_overrides.get(field_name, [])


# 中文 L0229 定义函数 get_pretrained_revision。
def get_pretrained_revision(field_name: str) -> str | None:
    # 中文 L0230 原文档中的说明或续行。
    """Return `--<field_name>.pretrained_revision` from the CLI, else from the `--config_path` file."""
    # 中文 L0231 调用 parse_arg，用它的结果给 “revision”（固定的 checkpoint 版本） 赋值。
    revision = parse_arg("pretrained_revision", get_cli_overrides(field_name) or [])
    # 中文 L0232 条件 revision is None 成立时进入下方分支。
    if revision is None:
        # 中文 L0233 调用 parse_arg，用它的结果给 “revision”（固定的 checkpoint 版本） 赋值。
        revision = parse_arg("pretrained_revision", get_yaml_overrides(field_name))
    # 中文 L0234 引用已有的值 revision作为当前调用的返回值。
    return revision


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def _register_scoped_actions(
    wrapper: Wrapper, parser: SuppressingArgumentParser, cli_args: Sequence[str]
) -> None:
    """Like draccus's own Wrapper.register_actions, but for a ChoiceType field only recurses into
    the already-selected subclass (per CLI `.type` args), instead of every registered choice.

    This mirrors draccus 0.11.x's internal wrapper traversal because its public parser eagerly registers
    every choice before parsing the command line. Keep this in sync when updating draccus.
    """
    if isinstance(wrapper, ChoiceWrapper):
        group = parser.add_argument_group(title=wrapper.title, description=wrapper.description)
        children = wrapper._children
        arg_name = f"{wrapper.dest}.{draccus.CHOICE_TYPE_KEY}" if wrapper.dest else draccus.CHOICE_TYPE_KEY
        group.add_argument(
            f"--{arg_name}",
            choices=list(children.keys()),
            help=f"Which type of {wrapper.title} to use",
            required=wrapper.required,
        )
        selected = get_type_arg(wrapper.dest, cli_args) if wrapper.dest else None
        if selected in children:
            _register_scoped_actions(children[selected], parser, cli_args)
    elif isinstance(wrapper, DataclassWrapper):
        group = parser.add_argument_group(title=wrapper.title, description=wrapper.description)
        for child in wrapper._children:
            if isinstance(child, AggregateWrapper):
                parser.add_argument(
                    f"--{child.name}", type=str, required=False, help=f"Config file for {child.name}"
                )
                _register_scoped_actions(child, parser, cli_args)
            elif isinstance(child, FieldWrapper):
                child.add_action(group)
    elif isinstance(wrapper, UnionWrapper):
        group = parser.add_argument_group(title=wrapper.title, description=wrapper.description)
        has_field_wrapper = False
        for child in wrapper._children:
            if isinstance(child, (DataclassWrapper, ChoiceWrapper)):
                _register_scoped_actions(child, parser, cli_args)
            elif isinstance(child, FieldWrapper):
                has_field_wrapper = True
        if has_field_wrapper:
            group.add_argument(f"--{wrapper.dest}", required=False)
    else:
        wrapper.register_actions(parser)


def print_scoped_help(config_class: type, cli_args: Sequence[str]) -> None:
    """Prints --help output scoped to the choices already resolved on the CLI (e.g. --env.type=pusht),
    instead of draccus's default of expanding every registered subclass of every ChoiceType field."""
    parser = SuppressingArgumentParser(formatter_class=SimpleHelpFormatter)
    parser.add_argument(
        f"--{draccus.utils.CONFIG_ARG}", type=str, help="Path for a config file to parse with draccus"
    )
    _register_scoped_actions(DataclassWrapper(config_class), parser, cli_args)
    parser.print_help()


# 中文 L0298 定义函数 filter_arg。
def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    # 中文 L0299 条件 args is None 成立时进入下方分支。
    if args is None:
        # 中文 L0300 创建列表，按给定顺序保留元素作为当前调用的返回值。
        return []
    # 中文 L0301 计算此行的右侧表达式给 “option” 赋值。
    option = f"--{field_to_filter}"
    # 中文 L0302 创建列表，按给定顺序保留元素给 “filtered_args” 赋值。
    filtered_args = []
    # 中文 L0303 使用字面值 0给 “index” 赋值。
    index = 0
    # 中文 L0304 持续执行下面的循环体，每轮开始都重新检查条件。
    while index < len(args):
        # 中文 L0305 用方括号按键或索引读取 args给 “arg” 赋值。
        arg = args[index]
        # 中文 L0306 条件 arg == option 成立时进入下方分支。
        if arg == option:
            # 中文 L0307 用增量赋值更新 index，运算逻辑见左侧运算符。
            index += 1
            # 中文 L0308 条件 index < len(args) and not args[index].startswith("--") 成立时进入下方分支。
            if index < len(args) and not args[index].startswith("--"):
                # 中文 L0309 用增量赋值更新 index，运算逻辑见左侧运算符。
                index += 1
            # 中文 L0310 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0311 条件 arg.startswith(f"{option}=") 成立时进入下方分支。
        if arg.startswith(f"{option}="):
            # 中文 L0312 用增量赋值更新 index，运算逻辑见左侧运算符。
            index += 1
            # 中文 L0313 跳过这一轮剩余操作，进入下一轮循环。
            continue
        # 中文 L0314 调用 filtered_args.append，在列表末尾追加一个元素。
        filtered_args.append(arg)
        # 中文 L0315 用增量赋值更新 index，运算逻辑见左侧运算符。
        index += 1
    # 中文 L0316 引用已有的值 filtered_args作为当前调用的返回值。
    return filtered_args


# 中文 L0319 定义函数 filter_path_args。
def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    Filters command-line arguments related to fields with specific path arguments.

    Args:
        fields_to_filter (str | list[str]): A single str or a list of str whose arguments need to be filtered.
        args (Sequence[str] | None): The sequence of command-line arguments to be filtered.
            Defaults to None.

    Returns:
        list[str]: A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a type
            argument (e.g., `--field_name.type`) are specified for the same field.
    """
    # 中文 L0336 条件 isinstance(fields_to_filter, str) 成立时进入下方分支。
    if isinstance(fields_to_filter, str):
        # 中文 L0337 创建列表，按给定顺序保留元素给 “fields_to_filter” 赋值。
        fields_to_filter = [fields_to_filter]

    # 中文 L0339 用条件表达式在 if 前与 else 后的两个值之间选择给 “filtered_args” 赋值。
    filtered_args = [] if args is None else list(args)

    # 中文 L0341 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
    for field in fields_to_filter:
        # 中文 L0342 条件 get_path_arg(field, args) 成立时进入下方分支。
        if get_path_arg(field, args):
            # 中文 L0343 条件 get_type_arg(field, args) 成立时进入下方分支。
            if get_type_arg(field, args):
                # 中文 L0344 抛出 ArgumentError，中断当前路径并交给上层的异常处理。
                raise ArgumentError(
                    # 中文 L0345 为“argument”提供这一项的值，注意它所在的参数表或外层表达式。
                    argument=None,
                    # 中文 L0346 为“message”提供这一项的值，注意它所在的参数表或外层表达式。
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                # 中文 L0347 闭合上方尚未结束的括号。
                )
            # 中文 L0348 计算此行的右侧表达式给 “option_prefix” 赋值。
            option_prefix = f"--{field}."
            # 中文 L0349 创建列表，按给定顺序保留元素给 “retained_args” 赋值。
            retained_args = []
            # 中文 L0350 使用字面值 0给 “index” 赋值。
            index = 0
            # 中文 L0351 持续执行下面的循环体，每轮开始都重新检查条件。
            while index < len(filtered_args):
                # 中文 L0352 用方括号按键或索引读取 filtered_args给 “arg” 赋值。
                arg = filtered_args[index]
                # 中文 L0353 条件 arg.startswith(option_prefix) 成立时进入下方分支。
                if arg.startswith(option_prefix):
                    # 中文 L0354 用增量赋值更新 index，运算逻辑见左侧运算符。
                    index += 1
                    # 中文 L0355 条件 ( 成立时进入下方分支。
                    if (
                        # 中文 L0356 字符串字面量，供上方的调用、容器或错误消息使用。
                        "=" not in arg
                        # 中文 L0357 续写上一行的布尔表达式，按 and / or 的短路规则判断。
                        and index < len(filtered_args)
                        # 中文 L0358 续写上一行的布尔表达式，按 and / or 的短路规则判断。
                        and not filtered_args[index].startswith("--")
                    # 中文 L0359 闭合上方尚未结束的括号。
                    ):
                        # 中文 L0360 用增量赋值更新 index，运算逻辑见左侧运算符。
                        index += 1
                    # 中文 L0361 跳过这一轮剩余操作，进入下一轮循环。
                    continue
                # 中文 L0362 调用 retained_args.append，在列表末尾追加一个元素。
                retained_args.append(arg)
                # 中文 L0363 用增量赋值更新 index，运算逻辑见左侧运算符。
                index += 1
            # 中文 L0364 引用已有的值 retained_args给 “filtered_args” 赋值。
            filtered_args = retained_args

    # 中文 L0366 引用已有的值 filtered_args作为当前调用的返回值。
    return filtered_args


def extract_path_fields_from_config(config_path: str, path_fields: list[str]) -> str:
    """Extract `path` fields from a YAML/JSON config before draccus processes it.

    When a user specifies e.g. ``policy.path: lerobot/smolvla_base`` in a YAML config,
    draccus will fail because ``path`` is not a valid field on policy config classes.
    This function extracts those path values, stores them in ``_config_path_args`` for
    later retrieval by ``get_path_arg()``, and returns a cleaned temp config file path.
    """
    config_file = Path(config_path)
    suffix = config_file.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_file) as f:
            config_data = json.load(f)
    else:
        return config_path

    if not isinstance(config_data, dict):
        return config_path

    modified = False
    for field in path_fields:
        if field in config_data and isinstance(config_data[field], dict) and PATH_KEY in config_data[field]:
            _config_path_args[field] = str(config_data[field].pop(PATH_KEY))
            remaining = config_data[field]
            if remaining:
                _config_yaml_overrides[field] = _flatten_to_cli_args(remaining)
            del config_data[field]
            modified = True

    if not modified:
        return config_path

    # Write cleaned config to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        if suffix in (".yaml", ".yml"):
            yaml.dump(config_data, tmp, default_flow_style=False)
        else:
            json.dump(config_data, tmp, indent=2)
    return tmp.name


# 中文 L0414 wrap 是装饰器工厂，调用 wrap() 得到一个接收目标函数的装饰器。
def wrap(config_path: Path | None = None) -> Callable[[F], F]:
    """
    HACK: Similar to draccus.wrap but does three additional things:
        - Will remove '.path' arguments from CLI in order to process them later on.
        - If a 'config_path' is passed and the main config class has a 'from_pretrained' method, will
          initialize it from there to allow to fetch configs from the hub directly
        - Will load plugins specified in the CLI arguments. These plugins will typically register
            their own subclasses of config classes, so that draccus can find the right class to instantiate
            from the CLI '.type' arguments
    """

    # 中文 L0425 接收需要包装的真实函数，例如 eval_main。
    def wrapper_outer(fn: F) -> F:
        # 中文 L0426 应用装饰器 wraps(fn)，它会处理紧接着的函数或类。
        @wraps(fn)
        # 中文 L0427 定义最终被调用的外层函数，先准备 cfg 再执行真实函数体。
        def wrapper_inner(*args: Any, **kwargs: Any) -> Any:
            # 中文 L0428 调用 inspect.getfullargspec，用它的结果给 “argspec” 赋值。
            argspec = inspect.getfullargspec(fn)
            # 中文 L0429 用方括号按键或索引读取 argspec.annotations给 “argtype” 赋值。
            argtype = argspec.annotations[argspec.args[0]]
            # 中文 L0430 条件 len(args) > 0 and type(args[0]) is argtype 成立时进入下方分支。
            if len(args) > 0 and type(args[0]) is argtype:
                # 中文 L0431 用方括号按键或索引读取 args给 “cfg”（配置对象） 赋值。
                cfg = args[0]
                # 中文 L0432 对 args 做切片，右端索引不包含在结果中给 “args”（位置参数序列） 赋值。
                args = args[1:]
            # 中文 L0433 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0434 直接读取终端参数，sys.argv 的第一项是程序名，切片从后面的参数开始。
                cli_args = sys.argv[1:]
                # 中文 L0435 调用 parse_plugin_args，用它的结果给 “plugin_args” 赋值。
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                # 中文 L0436 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    # 中文 L0437 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
                    try:
                        # 中文 L0438 调用 load_plugin 执行操作。
                        load_plugin(plugin_path)
                    # 中文 L0439 捕获这里列出的异常类型，然后执行下方的处理逻辑。
                    except PluginLoadError as e:
                        # add the relevant CLI arg to the error message
                        # 中文 L0441 抛出 PluginLoadError，中断当前路径并交给上层的异常处理。
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    # 中文 L0442 调用 filter_arg，用它的结果给 “cli_args” 赋值。
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                # 中文 L0443 条件 "--help" in cli_args or "-h" in cli_args 成立时进入下方分支。
                if "--help" in cli_args or "-h" in cli_args:
                    # 中文 L0444 调用 print_scoped_help 执行操作。
                    print_scoped_help(argtype, cli_args)
                    # 中文 L0445 调用 sys.exit 执行操作。
                    sys.exit(0)
                # 中文 L0446 调用 parse_arg，用它的结果给 “config_path_cli” 赋值。
                config_path_cli = parse_arg("config_path", cli_args)
                # 中文 L0447 条件 has_method(argtype, "__get_path_fields__") 成立时进入下方分支。
                if has_method(argtype, "__get_path_fields__"):
                    # 中文 L0448 调用 argtype.__get_path_fields__，用它的结果给 “path_fields” 赋值。
                    path_fields = argtype.__get_path_fields__()
                    # 中文 L0449 按路径字段规则过滤交给通用配置解析器的参数。
                    cli_args = filter_path_args(path_fields, cli_args)
                    # Also extract path fields from the YAML/JSON config file
                    # 中文 L0451 条件 config_path_cli 成立时进入下方分支。
                    if config_path_cli:
                        # 中文 L0452 调用 extract_path_fields_from_config，用它的结果给 “config_path_cli” 赋值。
                        config_path_cli = extract_path_fields_from_config(config_path_cli, path_fields)
                # 中文 L0453 开始执行可能抛出异常的代码，后续 except 或 finally 提供处理。
                try:
                    # 中文 L0454 条件 has_method(argtype, "from_pretrained") and config_path_cli 成立时进入下方分支。
                    if has_method(argtype, "from_pretrained") and config_path_cli:
                        # 中文 L0455 调用 filter_arg，用它的结果给 “cli_args” 赋值。
                        cli_args = filter_arg("config_path", cli_args)
                        # 中文 L0456 调用 argtype.from_pretrained，用它的结果给 “cfg”（配置对象） 赋值。
                        cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                    # 中文 L0457 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
                    else:
                        # 中文 L0458 条件 config_path_cli 成立时进入下方分支。
                        if config_path_cli:
                            # 中文 L0459 调用 filter_arg，用它的结果给 “cli_args” 赋值。
                            cli_args = filter_arg("config_path", cli_args)
                        # 中文 L0460 用 draccus 根据数据类类型和终端参数生成 EvalPipelineConfig 实例。
                        cfg = draccus.parse(
                            # 中文 L0461 为“config_class”提供这一项的值，注意它所在的参数表或外层表达式。
                            config_class=argtype,
                            # 中文 L0462 为“config_path”提供这一项的值，注意它所在的参数表或外层表达式。
                            config_path=config_path_cli or config_path,
                            # 中文 L0463 为“args”（位置参数序列）提供这一项的值，注意它所在的参数表或外层表达式。
                            args=cli_args,
                        # 中文 L0464 闭合上方尚未结束的括号。
                        )
                # 中文 L0465 捕获这里列出的异常类型，然后执行下方的处理逻辑。
                except DecodingError as e:
                    # 中文 L0466 调用 print 执行操作。
                    print(f"error: {e}", file=sys.stderr)
                    # 中文 L0467 调用 sys.exit 执行操作。
                    sys.exit(1)
            # 中文 L0468 把解析得到的 cfg 传给真实函数，并保留其他显式传入的参数。
            response = fn(cfg, *args, **kwargs)
            # 中文 L0469 引用已有的值 response作为当前调用的返回值。
            return response

        # 中文 L0471 调用 cast，用它的结果作为当前调用的返回值。
        return cast(F, wrapper_inner)

    # 中文 L0473 调用 cast，用它的结果作为当前调用的返回值。
    return cast(Callable[[F], F], wrapper_outer)
