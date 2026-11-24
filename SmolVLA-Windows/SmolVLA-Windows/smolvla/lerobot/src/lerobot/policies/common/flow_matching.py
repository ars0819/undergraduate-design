#!/usr/bin/env python

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

"""Flow-matching sampling primitives shared across policies.

Canonical versions of the beta-distributed timestep sampler, the noise sampler, the
training-input construction and the forward-Euler denoising loop (with its real-time-chunking
hook) that the openpi-derived policies (pi0, pi05, smolvla, eo1) and the forward-convention
policies (evo1, groot, wall_x) historically each carried a copy of.

The samplers deliberately return *untransformed* draws: every adopter's distribution,
dtype and order of casting versus transformation is decided at its own call site, because
those choices are baked into released checkpoints. All functions are stateless; adopting
them does not affect checkpoints.

``FlowConvention`` spells out the direction of time for both halves of a policy's flow
matching: ``make_flow_matching_inputs`` at training time and ``euler_integrate`` at
inference time. It is an argument to each of them rather than a property of the policy, so
passing the same one to both remains the caller's responsibility.
"""

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

BetaSampler = Callable[[float, float, int], Tensor]
"""Draws ``bsize`` raw ``Beta(alpha, beta)`` samples as ``(alpha, beta, bsize) -> (bsize,)``."""


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0049 定义类 FlowConvention，括号列出基类 str、enum.Enum。
class FlowConvention(str, enum.Enum):
    """Which end of the ``[0, 1]`` schedule holds the noise.

    Both conventions describe the same family of probability paths, they only disagree on
    the direction of time (and therefore on the sign of the velocity field). A checkpoint is
    tied to the convention it was trained with, so this is a property of the policy, never a
    tunable.
    """

    # 中文 L0058 使用字面值 'noise_at_one'给 “NOISE_AT_ONE” 赋值。
    NOISE_AT_ONE = "noise_at_one"
    """openpi (pi0, pi05, smolvla, eo1): ``x_1`` is noise, integrate ``t: 1 -> 0``.

    ``dt = -1/num_steps``, ``time = 1.0 + step * dt``, and the clean sample is
    ``x_0 = x_t - time * v_t``.
    """

    # 中文 L0065 使用字面值 'noise_at_zero'给 “NOISE_AT_ZERO” 赋值。
    NOISE_AT_ZERO = "noise_at_zero"
    """groot / evo1 / wall_x: ``x_0`` is noise, integrate ``t: 0 -> 1``.

    ``dt = +1/num_steps``, ``time = step * dt``, and the clean sample is
    ``x_1 = x_t + (1 - time) * v_t``.
    """


def _beta_distribution(alpha: float, beta: float) -> "torch.distributions.Beta":
    """``Beta(alpha, beta)`` whose concentrations are pinned to CPU.

    Beta sampling goes through ``_sample_dirichlet``, which is unimplemented on MPS and meta,
    so the draw happens on CPU and the result is moved afterwards. The concentrations are
    built with an explicit ``device="cpu"`` rather than the ambient default device, so a
    call inside a ``torch.device(...)`` context still samples on CPU.
    """
    alpha_t = torch.tensor(alpha, device="cpu", dtype=torch.float32)
    beta_t = torch.tensor(beta, device="cpu", dtype=torch.float32)
    return torch.distributions.Beta(alpha_t, beta_t, validate_args=False)


def device_beta_sampler(device) -> BetaSampler:
    """Build a ``sample_beta`` hook that draws from ``device``'s RNG instead of the CPU one.

    Only for callers whose released behavior depends on the device-side RNG stream (wall_x).
    Drawing on CPU and moving the result yields different numbers for the same seed, so such
    callers cannot use the default CPU path without changing what their checkpoints see.
    """

    def sample(alpha: float, beta: float, bsize: int) -> Tensor:
        dist = torch.distributions.Beta(
            torch.tensor(alpha, dtype=torch.float32, device=device),
            torch.tensor(beta, dtype=torch.float32, device=device),
            validate_args=False,
        )
        return dist.sample([bsize])

    return sample


def sample_beta(
    alpha: float,
    beta: float,
    bsize: int,
    device,
    *,
    dtype: torch.dtype | None = None,
    sampler: BetaSampler | None = None,
) -> Tensor:  # see openpi
    """Draw raw ``Beta(alpha, beta)`` samples of shape ``(bsize,)``.

    No affine transform is applied. Callers whose recipe transforms the sample (complement,
    scale, clamp) apply it themselves: the order of those operations relative to the dtype
    cast is policy-specific and numerically load-bearing, so it cannot be folded in here.

    Args:
        device: Device the sample is moved to.
        dtype: Cast applied together with the device move. ``None`` (default) keeps the
            sampler's float32, the openpi convention. Pass the action dtype to reproduce a
            ``.to(device, dtype=...)`` that precedes the caller's own transform (groot).
        sampler: Optional hook replacing the CPU distribution, for callers that must
            preserve a different RNG stream (see ``device_beta_sampler``).
    """
    if sampler is not None:
        raw = sampler(alpha, beta, bsize)
    else:
        raw = _beta_distribution(alpha, beta).sample((bsize,))
    if dtype is None:
        return raw.to(device)
    return raw.to(device, dtype=dtype)


# 中文 L0137 构造采样初始噪声，默认分布为标准正态，默认 dtype 为 float32。
def sample_noise(
    # 中文 L0138 “shape”（各轴的长度）是当前括号中的一项，逗号将它与下一项分开。
    shape,
    # 中文 L0139 “device”（运算设备）是当前括号中的一项，逗号将它与下一项分开。
    device,
    # 中文 L0140 后续参数必须按名字传入。
    *,
    # 中文 L0141 为“dtype”（元素类型）提供这一项的值，注意它所在的参数表或外层表达式。
    dtype: torch.dtype = torch.float32,
    # 中文 L0142 为“distribution”提供这一项的值，注意它所在的参数表或外层表达式。
    distribution: Literal["normal", "uniform"] = "normal",
# 中文 L0143 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> Tensor:
    """The flow-matching ``x_1`` noise sample.

    Args:
        dtype: Defaults to float32, so existing openpi-derived callers
            (pi0/pi05/smolvla/eo1, wall_x) are unchanged. Pass the action dtype when the
            policy samples directly in it (groot, evo1) instead of casting afterwards.
        distribution: ``"normal"`` (default) draws standard-normal noise, matching
            ``torch.randn``. ``"uniform"`` draws from ``[-1, 1)`` as ``rand * 2 - 1``
            (evo1).
    """
    # 中文 L0154 条件 distribution == "normal" 成立时进入下方分支。
    if distribution == "normal":
        # 中文 L0155 按给定 shape、dtype 和 device 创建正态随机 Tensor。
        return torch.normal(
            # 中文 L0156 为“mean”（均值）提供这一项的值，注意它所在的参数表或外层表达式。
            mean=0.0,
            # 中文 L0157 为“std”（标准差）提供这一项的值，注意它所在的参数表或外层表达式。
            std=1.0,
            # 中文 L0158 为“size”提供这一项的值，注意它所在的参数表或外层表达式。
            size=shape,
            # 中文 L0159 为“dtype”（元素类型）提供这一项的值，注意它所在的参数表或外层表达式。
            dtype=dtype,
            # 中文 L0160 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
            device=device,
        # 中文 L0161 闭合上方尚未结束的括号。
        )
    # 中文 L0162 条件 distribution == "uniform" 成立时进入下方分支。
    if distribution == "uniform":
        # 中文 L0163 计算右侧表达式 dtype, device=device) * 2 - 1作为当前调用的返回值。
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    # 中文 L0164 抛出 ValueError，中断当前路径并交给上层的异常处理。
    raise ValueError(f"Unknown noise distribution {distribution!r}, expected 'normal' or 'uniform'")


def sample_time_beta(
    bsize: int,
    device,
    *,
    alpha: float,
    beta: float,
    scale: float = 1.0,
    offset: float = 0.0,
) -> Tensor:
    """Beta-distributed flow-matching timesteps: ``Beta(alpha, beta) * scale + offset``.

    ``scale`` and ``offset`` default to the identity transform (``scale=1.0, offset=0.0``),
    i.e. the raw Beta sample in float32. The ``scale=0.999, offset=0.001`` endpoint offsets
    are the pi-family recipe, which pi0/pi05/smolvla/eo1 pass in from their configs; they
    are not a default of this helper.
    """
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


def make_flow_matching_inputs(
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    *,
    convention: FlowConvention,
    prefix_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the noised actions, velocity target and model timesteps for a training step.

    The interpolation is linear between the two endpoints of the probability path, and
    ``convention`` (see :class:`FlowConvention`) decides which endpoint the noise sits at,
    and therefore both the interpolation and the sign of the velocity target:

    * ``NOISE_AT_ONE``: ``x_t = t * noise + (1 - t) * actions`` and ``v = noise - actions``.
    * ``NOISE_AT_ZERO``: ``x_t = (1 - t) * noise + t * actions`` and ``v = actions - noise``.

    This is the training-time half of the contract that :func:`euler_integrate` implements at
    inference time. Nothing here checks that the two agree: a policy that passes different
    conventions to each learns a velocity pointing the wrong way along its own schedule.

    The returned target is computed in the dtype of ``actions`` and ``noise``. A caller that
    regresses in a different precision than it interpolates in (evo1) has to build its own.

    Args:
        actions: ``(batch, horizon, action_dim)`` clean action chunk.
        noise: Noise sample broadcastable to ``actions``.
        time: ``(batch,)`` timesteps.
        convention: Which end of the schedule holds the noise.
        prefix_mask: Optional ``(batch, horizon)`` boolean mask marking clean action-prefix
            positions (real-time chunking). Masked positions are given the model time at the
            clean end of the schedule, which makes the interpolation return the clean action
            there, so they are never noised. The inference-time counterpart is
            ``euler_integrate``'s ``hard_prefix`` / ``hard_prefix_mask``.

    Returns:
        ``(x_t, velocity_target, model_time)``: the noised actions fed to the network, the
        regression target for the predicted velocity, and the timesteps fed to the network.
        ``model_time`` is ``(batch,)`` without a prefix and ``(batch, horizon)`` with one.
    """
    if not isinstance(convention, FlowConvention):
        raise TypeError(f"convention must be a FlowConvention, got {convention!r}")
    if time.shape != actions.shape[:1]:
        raise ValueError(f"time must have shape (batch,), got {tuple(time.shape)}")
    if prefix_mask is not None and prefix_mask.shape != actions.shape[:2]:
        raise ValueError(f"prefix_mask must have shape (batch, horizon), got {tuple(prefix_mask.shape)}")
    noise_at_one = convention is FlowConvention.NOISE_AT_ONE

    if prefix_mask is None:
        model_time = time
        expanded_time = time[:, None, None]
    else:
        # Clean tokens sit at the far end of the schedule from the noise.
        clean_time = 0.0 if noise_at_one else 1.0
        model_time = time[:, None].expand_as(prefix_mask)
        model_time = torch.where(prefix_mask, torch.full_like(model_time, clean_time), model_time)
        expanded_time = model_time.unsqueeze(-1)

    if noise_at_one:
        x_t = expanded_time * noise + (1 - expanded_time) * actions
        velocity_target = noise - actions
    else:
        x_t = (1 - expanded_time) * noise + expanded_time * actions
        velocity_target = actions - noise

    return x_t, velocity_target, model_time


# 中文 L0256 通用显式 Euler 采样。参数 denoise_fn 是根据当前 x_t 和 t 返回速度的可调用对象。
def euler_integrate(
    # 中文 L0257 延续上面的语句，denoise_fn: Callable[..., Tensor],。本段处理沿估计速度场执行多次显式 Euler 更新。
    denoise_fn: Callable[..., Tensor],
    # 中文 L0258 “noise”（采样起点噪声）是当前括号中的一项，逗号将它与下一项分开。
    noise: Tensor,
    # 中文 L0259 为“num_steps”（每块动作生成的积分步数）提供这一项的值，注意它所在的参数表或外层表达式。
    num_steps: int | None = None,
    # 中文 L0260 后续参数必须按名字传入。
    *,
    # 中文 L0261 为“convention”提供这一项的值，注意它所在的参数表或外层表达式。
    convention: FlowConvention = FlowConvention.NOISE_AT_ONE,
    # 中文 L0262 为“step_aware”提供这一项的值，注意它所在的参数表或外层表达式。
    step_aware: bool = False,
    # 中文 L0263 为“time_grid”提供这一项的值，注意它所在的参数表或外层表达式。
    time_grid: Tensor | None = None,
    # 中文 L0264 为“velocity_scale”提供这一项的值，注意它所在的参数表或外层表达式。
    velocity_scale: Tensor | None = None,
    # 中文 L0265 为“rtc_processor”提供这一项的值，注意它所在的参数表或外层表达式。
    rtc_processor: "RTCProcessor | None" = None,
    # 中文 L0266 为“rtc_enabled”提供这一项的值，注意它所在的参数表或外层表达式。
    rtc_enabled: bool = False,
    # 中文 L0267 为“inference_delay”提供这一项的值，注意它所在的参数表或外层表达式。
    inference_delay: int | None = None,
    # 中文 L0268 为“prev_chunk_left_over”提供这一项的值，注意它所在的参数表或外层表达式。
    prev_chunk_left_over: Tensor | None = None,
    # 中文 L0269 为“execution_horizon”提供这一项的值，注意它所在的参数表或外层表达式。
    execution_horizon: int | None = None,
    # 中文 L0270 为“hard_prefix”提供这一项的值，注意它所在的参数表或外层表达式。
    hard_prefix: Tensor | None = None,
    # 中文 L0271 为“hard_prefix_mask”提供这一项的值，注意它所在的参数表或外层表达式。
    hard_prefix_mask: Tensor | None = None,
    # 中文 L0272 为“precompute_times”提供这一项的值，注意它所在的参数表或外层表达式。
    precompute_times: bool = False,
# 中文 L0273 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> Tensor:
    """Forward-Euler integration of a flow-matching velocity field between noise and actions.

    The loop is ``x_t <- x_t + dt * v_t`` over a uniform schedule whose direction is set by
    ``convention`` (see :class:`FlowConvention`), with an optional real-time-chunking (RTC)
    guidance hook wrapping the velocity computation and debug tracking after each step.

    RTC conventions: :class:`~lerobot.policies.rtc.modeling_rtc.RTCProcessor` is written
    against ``NOISE_AT_ONE``. Under ``NOISE_AT_ZERO`` this function hands it ``1 - time`` and
    negates the velocity both entering and leaving the processor, so the guided step is
    correct in the caller's own convention. ``rtc_processor.track`` therefore always reports
    ``time`` and ``v_t`` in the ``NOISE_AT_ONE`` convention, whatever the caller's, which
    keeps the entries logged here consistent with those logged inside ``denoise_step``.

    Args:
        denoise_fn: Computes the velocity ``v_t`` from ``(x_t, time_tensor)`` where
            ``time_tensor`` is a float32 tensor of shape ``(batch_size,)``. The returned
            velocity must have the same shape and dtype as ``x_t``.
        noise: Initial sample of shape ``(batch_size, ...)``: ``x_1`` under ``NOISE_AT_ONE``,
            ``x_0`` under ``NOISE_AT_ZERO``.
        num_steps: Number of Euler steps. Required unless ``time_grid`` is given.
        convention: Which end of the schedule holds the noise.
        step_aware: Call ``denoise_fn`` as ``denoise_fn(x_t, time_tensor, step)`` with the
            integer index of the current Euler step. Policies whose network input is a discrete
            function of that index (groot's timestep buckets, evo1's positional-encoding lookup)
            take it from here; recovering it from ``time_tensor`` would both round-trip through
            the host once per step and risk landing in a different bucket.
        time_grid: Optional explicit schedule of ``num_steps + 1`` sample times, replacing the
            uniform one. Step ``k`` evaluates the velocity at ``time_grid[k]`` and advances by
            ``time_grid[k + 1] - time_grid[k]``, so non-uniform schedules are supported.
        velocity_scale: Optional per-element weighting of the update, applied as
            ``x_t + (dt * v_t) * velocity_scale``. Used to freeze or ramp part of the chunk.
        rtc_processor: Optional RTC processor. Debug tracking fires whenever it is set and
            has debugging enabled, even if RTC guidance itself is disabled (this mirrors
            the historical per-policy loops).
        rtc_enabled: Whether to route the velocity computation through
            ``rtc_processor.denoise_step`` (requires ``rtc_processor``).
        inference_delay: RTC guidance parameter, forwarded verbatim.
        prev_chunk_left_over: RTC guidance parameter, forwarded verbatim.
        execution_horizon: RTC guidance parameter, forwarded verbatim.
        hard_prefix: Optional clean action prefix to clamp throughout denoising.
        hard_prefix_mask: Boolean mask selecting the values clamped from ``hard_prefix``.
        precompute_times: Create the float32 timestep schedule on the target device once,
            avoiding a scalar tensor allocation/host transfer in each denoising step.
            Uses the same Python scalar arithmetic as the historical loop.
    """
    # 中文 L0319 用方括号按键或索引读取 noise.shape给 “bsize”（批大小） 赋值。
    bsize = noise.shape[0]
    # 中文 L0320 读取属性 noise.device给 “device”（运算设备） 赋值。
    device = noise.device
    # 中文 L0321 条件 not isinstance(convention, FlowConvention) 成立时进入下方分支。
    if not isinstance(convention, FlowConvention):
        # 中文 L0322 抛出 TypeError，中断当前路径并交给上层的异常处理。
        raise TypeError(f"convention must be a FlowConvention, got {convention!r}")
    # 中文 L0323 这里判断采样时间的方向。SmolVLA 使用默认的噪声在 t=1 约定。
    noise_at_one = convention is FlowConvention.NOISE_AT_ONE

    # 中文 L0325 条件 time_grid is not None 成立时进入下方分支。
    if time_grid is not None:
        # 中文 L0326 条件 time_grid.ndim != 1 or time_grid.shape[0] < 2 成立时进入下方分支。
        if time_grid.ndim != 1 or time_grid.shape[0] < 2:
            # 中文 L0327 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(f"time_grid must be 1-D with at least 2 entries, got {tuple(time_grid.shape)}")
        # 中文 L0328 条件 num_steps is not None and num_steps != time_grid.shape[0] - 1 成立时进入下方分支。
        if num_steps is not None and num_steps != time_grid.shape[0] - 1:
            # 中文 L0329 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0330 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"num_steps={num_steps} conflicts with a time_grid of {time_grid.shape[0]} entries"
            # 中文 L0331 闭合上方尚未结束的括号。
            )
        # 中文 L0332 计算右侧表达式 time_grid.shape[0] - 1给 “num_steps”（每块动作生成的积分步数） 赋值。
        num_steps = time_grid.shape[0] - 1
    # 中文 L0333 前面的分支均未命中时，再判断 num_steps is None。
    elif num_steps is None:
        # 中文 L0334 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("euler_integrate requires either num_steps or time_grid")

    # Clean tokens sit at the far end of the schedule from the noise.
    # 中文 L0337 在 SmolVLA 使用的时间约定下，干净动作终点位于 t=0。
    clean_time = 0.0 if noise_at_one else 1.0
    # 中文 L0338 将时间跨度均分成 num_steps 段。原评测每步时间增量约为 -0.1。
    uniform_dt = (-1.0 if noise_at_one else 1.0) / num_steps

    # 中文 L0340 定义函数 velocity。
    def velocity(x_t: Tensor, time_tensor: Tensor, step: int) -> Tensor:
        # 中文 L0341 用条件表达式在 if 前与 else 后的两个值之间选择作为当前调用的返回值。
        return denoise_fn(x_t, time_tensor, step) if step_aware else denoise_fn(x_t, time_tensor)

    # 中文 L0343 用条件表达式在 if 前与 else 后的两个值之间选择给 “guidance” 赋值。
    guidance = rtc_processor if rtc_enabled else None
    # 中文 L0344 条件 rtc_enabled and guidance is None 成立时进入下方分支。
    if rtc_enabled and guidance is None:
        # 中文 L0345 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("rtc_processor is required when rtc_enabled is True")

    # 中文 L0347 定义函数 uniform_time。
    def uniform_time(step: int) -> float:
        # 中文 L0348 根据循环下标计算本次速度评估的时间，SmolVLA 从 1 开始。
        return (1.0 + step * uniform_dt) if noise_at_one else step * uniform_dt

    # 中文 L0350 用条件表达式在 if 前与 else 后的两个值之间选择给 “times” 赋值。
    times = (
        # 中文 L0351 延续上面的语句，torch.tensor([uniform_time(step) for step in range(num_steps)], dtype=torch.float32, device=device)。本段处理沿估计速度场执行多次显式 Euler 更新。
        torch.tensor([uniform_time(step) for step in range(num_steps)], dtype=torch.float32, device=device)
        # 中文 L0352 条件 precompute_times and time_grid is None 成立时进入下方分支。
        if precompute_times and time_grid is None
        # 中文 L0353 补充上方跨行表达式的分支值或条件。
        else None
    # 中文 L0354 闭合上方尚未结束的括号。
    )
    # 中文 L0355 将初始噪声作为第一个动作块估计。
    x_t = noise
    # 中文 L0356 执行配置指定的积分轮数。每轮都处理整个动作块。
    for step in range(num_steps):
        # 中文 L0357 条件 time_grid is None 成立时进入下方分支。
        if time_grid is None:
            # 中文 L0358 引用已有的值 uniform_dt给 “dt” 赋值。
            dt = uniform_dt
            # 中文 L0359 用条件表达式在 if 前与 else 后的两个值之间选择给 “time_tensor” 赋值。
            time_tensor = (
                # 中文 L0360 延续上面的语句，times[step].expand(bsize)。本段处理沿估计速度场执行多次显式 Euler 更新。
                times[step].expand(bsize)
                # 中文 L0361 条件 times is not None 成立时进入下方分支。
                if times is not None
                # 中文 L0362 创建形状 [B] 的 float32 时间 Tensor，device 与动作噪声一致。
                else torch.full((bsize,), uniform_time(step), dtype=torch.float32, device=device)
            # 中文 L0363 闭合上方尚未结束的括号。
            )
        # 中文 L0364 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # Slice the grid rather than reading it back: `float(time_grid[step])` would
            # synchronise with the accelerator once per step.
            # 中文 L0367 计算右侧表达式 time_grid[step + 1] - time_grid[step]给 “dt” 赋值。
            dt = time_grid[step + 1] - time_grid[step]
            # 中文 L0368 调用 expand，用它的结果给 “time_tensor” 赋值。
            time_tensor = time_grid[step].to(dtype=torch.float32, device=device).expand(bsize)

        # 中文 L0370 条件 hard_prefix is not None 成立时进入下方分支。
        if hard_prefix is not None:
            # 中文 L0371 条件 hard_prefix_mask is None 成立时进入下方分支。
            if hard_prefix_mask is None:
                # 中文 L0372 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError("hard_prefix_mask is required when hard_prefix is provided")
            # 中文 L0373 调用 torch.where，用它的结果给 “x_t”（采样时间 t 的当前动作块估计） 赋值。
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)
            # 中文 L0374 调用 clone，用它的结果给 “time_tensor” 赋值。
            time_tensor = time_tensor[:, None].expand(bsize, x_t.shape[1]).clone()
            # 中文 L0375 引用已有的值 clean_time给 “time_tensor[hard_prefix_mask[..., 0]]” 赋值。
            time_tensor[hard_prefix_mask[..., 0]] = clean_time

        # 中文 L0377 定义函数 denoise_step_partial_call。
        def denoise_step_partial_call(
            # 中文 L0378 为“input_x_t”提供这一项的值，注意它所在的参数表或外层表达式。
            input_x_t: Tensor, current_timestep: Tensor = time_tensor, current_step: int = step
        # 中文 L0379 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
        ) -> Tensor:
            # 中文 L0380 调用 velocity，用它的结果作为当前调用的返回值。
            return velocity(input_x_t, current_timestep, current_step)

        # 中文 L0382 定义函数 flipped_denoise_step_partial_call。
        def flipped_denoise_step_partial_call(
            # 中文 L0383 为“input_x_t”提供这一项的值，注意它所在的参数表或外层表达式。
            input_x_t: Tensor, current_timestep: Tensor = time_tensor, current_step: int = step
        # 中文 L0384 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
        ) -> Tensor:
            # 中文 L0385 计算此行的右侧表达式作为当前调用的返回值。
            return -velocity(input_x_t, current_timestep, current_step)

        # 中文 L0387 组合多个布尔判断，按 Python 的短路规则求值给 “needs_rtc_time” 赋值。
        needs_rtc_time = rtc_enabled or (rtc_processor is not None and rtc_processor.is_debug_enabled())
        # 中文 L0388 条件 needs_rtc_time 成立时进入下方分支。
        if needs_rtc_time:
            # RTCProcessor takes a plain float in the NOISE_AT_ONE convention. On the explicit-grid
            # path this is the one place the schedule has to come back to the host.
            # 中文 L0391 用条件表达式在 if 前与 else 后的两个值之间选择给 “host_time” 赋值。
            host_time = uniform_time(step) if time_grid is None else float(time_grid[step])
            # 中文 L0392 用条件表达式在 if 前与 else 后的两个值之间选择给 “rtc_time” 赋值。
            rtc_time = host_time if noise_at_one else 1.0 - host_time

        # 中文 L0394 条件 guidance is not None 成立时进入下方分支。
        if guidance is not None:
            # 中文 L0395 调用 guidance.denoise_step，用它的结果给 “v_t”（当前速度场输出） 赋值。
            v_t = guidance.denoise_step(
                # 中文 L0396 为“x_t”（采样时间 t 的当前动作块估计）提供这一项的值，注意它所在的参数表或外层表达式。
                x_t=x_t,
                # 中文 L0397 为“prev_chunk_left_over”提供这一项的值，注意它所在的参数表或外层表达式。
                prev_chunk_left_over=prev_chunk_left_over,
                # 中文 L0398 为“inference_delay”提供这一项的值，注意它所在的参数表或外层表达式。
                inference_delay=inference_delay,
                # 中文 L0399 为“time”提供这一项的值，注意它所在的参数表或外层表达式。
                time=rtc_time,
                # 中文 L0400 为“original_denoise_step_partial”提供这一项的值，注意它所在的参数表或外层表达式。
                original_denoise_step_partial=(
                    # 中文 L0401 延续上面的语句，denoise_step_partial_call if noise_at_one else flipped_denoise_step_partial_call。本段处理沿估计速度场执行多次显式 Euler 更新。
                    denoise_step_partial_call if noise_at_one else flipped_denoise_step_partial_call
                # 中文 L0402 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
                ),
                # 中文 L0403 为“execution_horizon”提供这一项的值，注意它所在的参数表或外层表达式。
                execution_horizon=execution_horizon,
            # 中文 L0404 闭合上方尚未结束的括号。
            )
            # 中文 L0405 条件 not noise_at_one 成立时进入下方分支。
            if not noise_at_one:
                # 中文 L0406 计算此行的右侧表达式给 “v_t”（当前速度场输出） 赋值。
                v_t = -v_t
        # 中文 L0407 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0408 标准 SmolVLA 评测在这里调用模型计算速度，形状与当前动作块对齐。
            v_t = denoise_step_partial_call(x_t)

        # 中文 L0410 速度乘小时间增量，得到沿速度方向的一次动作估计变化。
        update = dt * v_t
        # 中文 L0411 条件 velocity_scale is not None 成立时进入下方分支。
        if velocity_scale is not None:
            # 中文 L0412 计算右侧表达式 update * velocity_scale给 “update” 赋值。
            update = update * velocity_scale
        # 中文 L0413 将变化量加到当前估计上。向量或动作块的各个元素分别相加。
        x_t = x_t + update

        # 中文 L0415 条件 hard_prefix is not None 成立时进入下方分支。
        if hard_prefix is not None:
            # 中文 L0416 调用 torch.where，用它的结果给 “x_t”（采样时间 t 的当前动作块估计） 赋值。
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)

        # 中文 L0418 条件 rtc_processor is not None and rtc_processor.is_debug_enabled() 成立时进入下方分支。
        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            # 中文 L0419 调用 rtc_processor.track 执行操作。
            rtc_processor.track(time=rtc_time, x_t=x_t, v_t=v_t if noise_at_one else -v_t)

    # 中文 L0421 返回最后一轮的动作块，后续由策略裁去填充的动作分量。
    return x_t
