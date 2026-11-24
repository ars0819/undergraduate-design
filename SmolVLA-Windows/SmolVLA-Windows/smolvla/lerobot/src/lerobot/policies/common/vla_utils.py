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

"""Helpers shared by the openpi-derived VLA policies (pi0, pi05, pi0_fast, smolvla, eo1, xvla).

These are the canonical versions of functions that historically were copy-pasted per
policy. They are pure (no parameters, no module state), so importing them from here
instead of a policy-local copy has no effect on checkpoints.
"""

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache
else:
    DynamicCache = None


# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0041 用不同频率的正弦和余弦给一个标量时间生成向量。
def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    # 中文 L0042 为“time”提供这一项的值，注意它所在的参数表或外层表达式。
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
# 中文 L0043 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
) -> Tensor:
    # 中文 L0044 原文档中的说明或续行。
    """Compute sine-cosine embeddings for scalar or per-action positions."""
    # 中文 L0045 输出宽度必须是偶数，正弦和余弦各占一半。
    if dimension % 2 != 0:
        # 中文 L0046 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    # 中文 L0048 条件 time.ndim not in (1, 2) 成立时进入下方分支。
    if time.ndim not in (1, 2):
        # 中文 L0049 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError("The time tensor must have shape (batch_size,) or (batch_size, action_horizon).")

    # 中文 L0051 选择目标设备可用的计算类型，后续调用方再转到动作嵌入 dtype。
    dtype = get_safe_dtype(torch.float64, device.type)
    # 中文 L0052 调用 torch.linspace，用它的结果给 “fraction” 赋值。
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    # 中文 L0053 计算右侧表达式 min_period * (max_period / min_period) ** fraction给 “period” 赋值。
    period = min_period * (max_period / min_period) ** fraction

    # 中文 L0055 计算右侧表达式 1.0 / period * 2 * math.pi给 “scaling_factor” 赋值。
    scaling_factor = 1.0 / period * 2 * math.pi
    # 中文 L0056 插入末维并广播，每个时间乘以全部频率。
    sin_input = time[..., None] * scaling_factor
    # 中文 L0057 把正弦和余弦结果沿最后一轴拼在一起。
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


# 中文 L0060 把一维的有效位和分块标记转换为二维注意力矩阵。
def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:  # see openpi (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    # 中文 L0081 条件 att_masks.ndim != 2 成立时进入下方分支。
    if att_masks.ndim != 2:
        # 中文 L0082 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(att_masks.ndim)
    # 中文 L0083 条件 pad_masks.ndim != 2 成立时进入下方分支。
    if pad_masks.ndim != 2:
        # 中文 L0084 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(pad_masks.ndim)

    # 中文 L0086 从块边界做累计和，得到每个 token 的块编号。
    cumsum = torch.cumsum(att_masks, dim=1)
    # 中文 L0087 通过广播比较键与查询的块编号。查询能看见同块和更早的块。
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    # 中文 L0088 对每个查询和键组合检查两者是否都是有效 token。
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    # 中文 L0089 同时满足块顺序和有效位时，才允许该注意力连接。
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Expand boolean 2D attention masks to the additive 4D layout expected by transformers.

    Valid positions become 0.0 and masked positions the large negative openpi constant.
    """
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def clone_past_key_values(past_key_values):
    """Clone the DynamicCache returned by prefix prefill for compiled denoising."""
    if DynamicCache is None:
        require_package("transformers", extra="transformers-dep")

    return DynamicCache(
        tuple(
            (keys.clone(), values.clone(), sliding_window) for keys, values, sliding_window in past_key_values
        )
    )


# 中文 L0116 在最后的特征轴尾部补零，保留批次轴和已有的时间轴。
def pad_vector(vector: Tensor, new_dim: int, *, truncate: bool = False) -> Tensor:
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)

    With ``truncate=False`` (openpi behavior), vectors whose last dimension is already
    >= new_dim are returned unchanged. With ``truncate=True`` (xVLA behavior), the last
    dimension is truncated to exactly ``new_dim`` (which may be 0).
    """
    # 中文 L0126 条件 vector.shape[-1] == new_dim 成立时进入下方分支。
    if vector.shape[-1] == new_dim:
        # 中文 L0127 引用已有的值 vector作为当前调用的返回值。
        return vector
    # 中文 L0128 条件 not truncate 成立时进入下方分支。
    if not truncate:
        # 中文 L0129 条件 vector.shape[-1] >= new_dim 成立时进入下方分支。
        if vector.shape[-1] >= new_dim:
            # 中文 L0130 引用已有的值 vector作为当前调用的返回值。
            return vector
        # 中文 L0131 对末维左侧补 0 个、右侧补差值个零。
        return F.pad(vector, (0, new_dim - vector.shape[-1]))
    # 中文 L0132 调用 list，用它的结果给 “shape”（各轴的长度） 赋值。
    shape = list(vector.shape)
    # 中文 L0133 用方括号按键或索引读取 shape给 “current_dim” 赋值。
    current_dim = shape[-1]
    # 中文 L0134 引用已有的值 new_dim给 “shape[-1]” 赋值。
    shape[-1] = new_dim
    # 中文 L0135 调用 vector.new_zeros，用它的结果给 “new_vector” 赋值。
    new_vector = vector.new_zeros(*shape)
    # 中文 L0136 调用 min，用它的结果给 “length” 赋值。
    length = min(current_dim, new_dim)
    # 中文 L0137 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “new_vector[..., :length]” 赋值。
    new_vector[..., :length] = vector[..., :length]
    # 中文 L0138 引用已有的值 new_vector作为当前调用的返回值。
    return new_vector


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Padding is centered (openpi convention). For the top-left-padding variant used by
    smolvla/xvla, see :func:`resize_with_pad`.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# 中文 L0218 四轴图像调整到指定高宽。保持宽高比，若需要则在上方和左方填充。
def resize_with_pad(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
    """Resize a (b, c, h, w) image without distortion, padding on the LEFT and TOP.

    This is the smolvla/xvla convention. For the centered-padding openpi variant, see
    :func:`resize_with_pad_torch`. ``pad_value`` is keyword-only on purpose: callers
    historically used different values (0, -1) and must state their choice explicitly.
    """
    # 中文 L0225 条件 img.ndim != 4 成立时进入下方分支。
    if img.ndim != 4:
        # 中文 L0226 抛出 ValueError，中断当前路径并交给上层的异常处理。
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    # 中文 L0228 对 img.shape 做切片，右端索引不包含在结果中给 “(current_height, current_width)” 赋值。
    current_height, current_width = img.shape[2:]
    # 中文 L0229 条件 current_height == height and current_width == width 成立时进入下方分支。
    if current_height == height and current_width == width:
        # 中文 L0230 引用已有的值 img作为当前调用的返回值。
        return img

    # 中文 L0232 找高和宽更紧的比例约束，保证缩放结果不超出目标大小。
    ratio = max(current_width / width, current_height / height)
    # 中文 L0233 调用 int，用它的结果给 “resized_height” 赋值。
    resized_height = int(current_height / ratio)
    # 中文 L0234 调用 int，用它的结果给 “resized_width” 赋值。
    resized_width = int(current_width / ratio)
    # 中文 L0235 用双线性插值调整像素网格。
    resized_img = F.interpolate(
        # 中文 L0236 延续上面的语句，img, size=(resized_height, resized_width), mode="bilinear", align_corners=False。
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    # 中文 L0237 闭合上方尚未结束的括号。
    )

    # 中文 L0239 调用 max，用它的结果给 “pad_height” 赋值。
    pad_height = max(0, height - resized_height)
    # 中文 L0240 调用 max，用它的结果给 “pad_width” 赋值。
    pad_width = max(0, width - resized_width)
    # 中文 L0241 按 PyTorch 的参数顺序填左、右、上、下。这个调用只可能填左和上。
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    # 中文 L0242 引用已有的值 padded_img作为当前调用的返回值。
    return padded_img
