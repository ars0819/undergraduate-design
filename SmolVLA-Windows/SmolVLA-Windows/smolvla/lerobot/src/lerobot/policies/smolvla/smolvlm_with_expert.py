# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# 中文 L0015 导入模块 copy。
import copy
# 中文 L0016 从模块 typing 导入 TYPE_CHECKING。
from typing import TYPE_CHECKING

# 中文 L0018 导入模块 torch。
import torch
# 中文 L0019 从模块 torch 导入 nn。
from torch import nn

# 中文 L0021 从模块 lerobot.utils.import_utils 导入 _transformers_available、require_package。
from lerobot.utils.import_utils import _transformers_available, require_package

# 中文 L0023 条件 TYPE_CHECKING or _transformers_available 成立时进入下方分支。
if TYPE_CHECKING or _transformers_available:
    # 中文 L0024 从模块 transformers 导入下方列出的多个名字。
    from transformers import (
        # 中文 L0025 “AutoConfig”是当前括号中的一项，逗号将它与下一项分开。
        AutoConfig,
        # 中文 L0026 “AutoModel”是当前括号中的一项，逗号将它与下一项分开。
        AutoModel,
        # 中文 L0027 “AutoModelForImageTextToText”是当前括号中的一项，逗号将它与下一项分开。
        AutoModelForImageTextToText,
        # 中文 L0028 “AutoProcessor”是当前括号中的一项，逗号将它与下一项分开。
        AutoProcessor,
        # 中文 L0029 “DynamicCache”是当前括号中的一项，逗号将它与下一项分开。
        DynamicCache,
        # 中文 L0030 “SmolVLMForConditionalGeneration”是当前括号中的一项，逗号将它与下一项分开。
        SmolVLMForConditionalGeneration,
    # 中文 L0031 闭合上方尚未结束的括号。
    )
# 中文 L0032 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
else:
    # 中文 L0033 设为空值 None给 “AutoConfig” 赋值。
    AutoConfig = None
    # 中文 L0034 设为空值 None给 “AutoModel” 赋值。
    AutoModel = None
    # 中文 L0035 设为空值 None给 “AutoModelForImageTextToText” 赋值。
    AutoModelForImageTextToText = None
    # 中文 L0036 设为空值 None给 “AutoProcessor” 赋值。
    AutoProcessor = None
    # 中文 L0037 设为空值 None给 “DynamicCache” 赋值。
    DynamicCache = None
    # 中文 L0038 设为空值 None给 “SmolVLMForConditionalGeneration” 赋值。
    SmolVLMForConditionalGeneration = None


# 中文 L0041 对 Q 或 K 的每对特征按 token 位置旋转，给注意力提供位置关系。
def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    # 中文 L0045 计算右侧表达式 x.shape[-1] // 2给 “d_half” 赋值。
    d_half = x.shape[-1] // 2
    # 中文 L0046 读取属性 x.device给 “device”（运算设备） 赋值。
    device = x.device
    # 中文 L0047 读取属性 x.dtype给 “dtype”（元素类型） 赋值。
    dtype = x.dtype
    # 中文 L0048 调用 x.to，用它的结果给 “x” 赋值。
    x = x.to(torch.float32)

    # 中文 L0050 计算右侧表达式 (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)给 “freq_exponents” 赋值。
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    # 中文 L0051 计算右侧表达式 max_wavelength**freq_exponents给 “timescale” 赋值。
    timescale = max_wavelength**freq_exponents
    # 中文 L0052 计算右侧表达式 positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)给 “radians” 赋值。
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    # 中文 L0054 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “radians” 赋值。
    radians = radians[..., None, :]

    # 中文 L0056 调用 torch.sin，用它的结果给 “sin” 赋值。
    sin = torch.sin(radians)  # .to(dtype=dtype)
    # 中文 L0057 调用 torch.cos，用它的结果给 “cos” 赋值。
    cos = torch.cos(radians)  # .to(dtype=dtype)

    # 中文 L0059 调用 x.split，用它的结果给 “(x1, x2)” 赋值。
    x1, x2 = x.split(d_half, dim=-1)
    # 中文 L0060 调用 torch.empty_like，用它的结果给 “res” 赋值。
    res = torch.empty_like(x)
    # 中文 L0061 计算右侧表达式 x1 * cos - x2 * sin给 “res[..., :d_half]” 赋值。
    res[..., :d_half] = x1 * cos - x2 * sin
    # 中文 L0062 计算右侧表达式 x2 * cos + x1 * sin给 “res[..., d_half:]” 赋值。
    res[..., d_half:] = x2 * cos + x1 * sin

    # 中文 L0064 调用 res.to，用它的结果作为当前调用的返回值。
    return res.to(dtype)


# 中文 L0067 定义函数 get_intermediate_size。
def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    # 中文 L0068 调用 int，用它的结果给 “hidden_dim” 赋值。
    hidden_dim = int(2 * hidden_dim / 3)
    # 中文 L0069 调用 int，用它的结果给 “hidden_dim” 赋值。
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    # 中文 L0070 计算右侧表达式 multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)给 “hidden_dim” 赋值。
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    # 中文 L0071 引用已有的值 hidden_dim作为当前调用的返回值。
    return hidden_dim


# 中文 L0074 用一个 PyTorch 模块组合视觉语言骨干和用于动作的语言模型式专家。
class SmolVLMWithExpertModel(nn.Module):
    # 中文 L0075 定义方法 __init__。作用是加载视觉语言骨干并构造较窄的动作专家。
    def __init__(
        # 中文 L0076 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0077 为“model_id”提供这一项的值，注意它所在的参数表或外层表达式。
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        # 中文 L0078 为“load_vlm_weights”提供这一项的值，注意它所在的参数表或外层表达式。
        load_vlm_weights: bool = True,
        # 中文 L0079 为“train_expert_only”提供这一项的值，注意它所在的参数表或外层表达式。
        train_expert_only: bool = True,
        # 中文 L0080 为“freeze_vision_encoder”提供这一项的值，注意它所在的参数表或外层表达式。
        freeze_vision_encoder: bool = False,
        # 中文 L0081 为“attention_mode”提供这一项的值，注意它所在的参数表或外层表达式。
        attention_mode: str = "self_attn",
        # 中文 L0082 为“num_expert_layers”提供这一项的值，注意它所在的参数表或外层表达式。
        num_expert_layers: int = -1,
        # 中文 L0083 为“num_vlm_layers”提供这一项的值，注意它所在的参数表或外层表达式。
        num_vlm_layers: int = -1,
        # 中文 L0084 为“self_attn_every_n_layers”提供这一项的值，注意它所在的参数表或外层表达式。
        self_attn_every_n_layers: int = -1,
        # 中文 L0085 为“expert_width_multiplier”提供这一项的值，注意它所在的参数表或外层表达式。
        expert_width_multiplier: float = 0.5,
        # 中文 L0086 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
        device: str = "auto",
    # 中文 L0087 闭合上方尚未结束的括号。
    ):
        # 中文 L0088 调用 __init__ 执行操作。
        super().__init__()
        # 中文 L0089 调用 require_package 执行操作。
        require_package("transformers", extra="smolvla")
        # 中文 L0090 该开关控制加载预训练视觉语言骨干，本次 checkpoint 设为 True。
        if load_vlm_weights:
            # 中文 L0091 调用 print 执行操作。
            print(f"Loading  {model_id} weights ...")
            # 中文 L0092 用 Transformers 创建骨干并读取已缓存的预训练参数。
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                # 中文 L0093 “model_id”是当前括号中的一项，逗号将它与下一项分开。
                model_id,
                # 中文 L0094 构造这条路径时请求以 bfloat16 加载视觉语言骨干。运行时应逐个算子看 dtype。
                torch_dtype="bfloat16",
                # 中文 L0095 为“low_cpu_mem_usage”提供这一项的值，注意它所在的参数表或外层表达式。
                low_cpu_mem_usage=True,
            # 中文 L0096 闭合上方尚未结束的括号。
            )
            # 中文 L0097 读取属性 self.vlm.config给 “config”（配置对象） 赋值。
            config = self.vlm.config
        # 中文 L0098 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0099 调用 AutoConfig.from_pretrained，用它的结果给 “config”（配置对象） 赋值。
            config = AutoConfig.from_pretrained(model_id)
            # 中文 L0100 调用 SmolVLMForConditionalGeneration，用它的结果给 “self.vlm” 赋值。
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        # 中文 L0101 加载与骨干配套的处理器，模型会读取它的 tokenizer 特殊 token ID。
        self.processor = AutoProcessor.from_pretrained(model_id)
        # 中文 L0102 正数表示保留前若干层。本次评测使用前 16 层。
        if num_vlm_layers > 0:
            # 中文 L0103 调用 print 执行操作。
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            # 中文 L0104 对 text_model.layers 做切片，右端索引不包含在结果中给 “self.get_vlm_model().text_model.layers” 赋值。
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        # 中文 L0105 保存实际使用的语言骨干层数。
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        # 中文 L0106 引用已有的值 config给 “self.config” 赋值。
        self.config = config
        # Smaller lm expert
        # 中文 L0108 深复制骨干文本配置。修改专家宽度时保留骨干原来的配置。
        lm_expert_config = copy.deepcopy(config.text_config)
        # 中文 L0109 读取属性 lm_expert_config.hidden_size给 “hidden_size” 赋值。
        hidden_size = lm_expert_config.hidden_size
        # 中文 L0110 用配置的比例缩小专家的隐藏宽度，再转为整数。
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        # 中文 L0111 调用 get_intermediate_size，用它的结果给 “lm_expert_config.intermediate_size” 赋值。
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        # 中文 L0112 读取属性 self.num_vlm_layers给 “lm_expert_config.num_hidden_layers” 赋值。
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        # 中文 L0113 条件 num_expert_layers > 0 成立时进入下方分支。
        if num_expert_layers > 0:
            # 中文 L0114 检查此条件，条件不成立时抛出 AssertionError。
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                # 中文 L0115 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            # 中文 L0116 闭合上方尚未结束的括号。
            )
            # 中文 L0117 显式设置专家层数时走这个分支。本次 0 让专家初始化继承前面设定的骨干层数。
            lm_expert_config.num_hidden_layers = num_expert_layers
        # 中文 L0118 根据调整后的配置创建动作专家。完整策略 checkpoint 稍后恢复相关权重。
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        # 中文 L0120 调用 len，用它的结果给 “self.num_expert_layers” 赋值。
        self.num_expert_layers = len(self.lm_expert.layers)
        # 中文 L0121 引用已有的值 self_attn_every_n_layers给 “self.self_attn_every_n_layers” 赋值。
        self.self_attn_every_n_layers = self_attn_every_n_layers
        # 中文 L0122 交叉注意力模式需要让一部分专家层接受来自视觉语言骨干的 K/V 宽度。
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            # 中文 L0124 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for layer_idx in range(len(self.lm_expert.layers)):
                # 中文 L0125 按层号取余。选作自注意力的层保留原 K/V 投影。
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    # 中文 L0126 跳过这一轮剩余操作，进入下一轮循环。
                    continue
                # 中文 L0127 给交叉注意力层重建 K 投影，让它接收骨干的 K 向量。
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    # 中文 L0128 延续上面的语句，config.text_config.num_key_value_heads * config.text_config.head_dim,。本段处理加载视觉语言骨干并构造较窄的动作专家。
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    # 中文 L0129 延续上面的语句，lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,。本段处理加载视觉语言骨干并构造较窄的动作专家。
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    # 中文 L0130 为“bias”提供这一项的值，注意它所在的参数表或外层表达式。
                    bias=lm_expert_config.attention_bias,
                # 中文 L0131 闭合上方尚未结束的括号。
                )
                # 中文 L0132 给交叉注意力层重建 V 投影，让它接收骨干的 V 向量。
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    # 中文 L0133 延续上面的语句，config.text_config.num_key_value_heads * config.text_config.head_dim,。本段处理加载视觉语言骨干并构造较窄的动作专家。
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    # 中文 L0134 延续上面的语句，lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,。本段处理加载视觉语言骨干并构造较窄的动作专家。
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    # 中文 L0135 为“bias”提供这一项的值，注意它所在的参数表或外层表达式。
                    bias=lm_expert_config.attention_bias,
                # 中文 L0136 闭合上方尚未结束的括号。
                )
        # Remove unused embed_tokens
        # 中文 L0138 移除专家自带的词嵌入表。动作嵌入由策略的动作投影生成。
        self.lm_expert.embed_tokens = None

        # 中文 L0140 读取属性 self.config.text_config.num_attention_heads给 “self.num_attention_heads” 赋值。
        self.num_attention_heads = self.config.text_config.num_attention_heads
        # 中文 L0141 读取属性 self.config.text_config.num_key_value_heads给 “self.num_key_value_heads” 赋值。
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        # 中文 L0143 引用已有的值 freeze_vision_encoder给 “self.freeze_vision_encoder” 赋值。
        self.freeze_vision_encoder = freeze_vision_encoder
        # 中文 L0144 引用已有的值 train_expert_only给 “self.train_expert_only” 赋值。
        self.train_expert_only = train_expert_only
        # 中文 L0145 引用已有的值 attention_mode给 “self.attention_mode” 赋值。
        self.attention_mode = attention_mode
        # 中文 L0146 读取属性 lm_expert_config.hidden_size给 “self.expert_hidden_size” 赋值。
        self.expert_hidden_size = lm_expert_config.hidden_size
        # 中文 L0147 调用 self.set_requires_grad 执行操作。
        self.set_requires_grad()

    # 中文 L0149 定义方法 get_vlm_model。
    def get_vlm_model(self):
        # 中文 L0150 读取属性 self.vlm.model作为当前调用的返回值。
        return self.vlm.model

    # 中文 L0152 定义方法 set_requires_grad。
    def set_requires_grad(self):
        # 中文 L0153 条件 self.freeze_vision_encoder 成立时进入下方分支。
        if self.freeze_vision_encoder:
            # 中文 L0154 调用 vision_model.eval，切换 PyTorch 模块到评测模式。
            self.get_vlm_model().vision_model.eval()
            # 中文 L0155 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for params in self.get_vlm_model().vision_model.parameters():
                # 中文 L0156 使用布尔值 False给 “params.requires_grad” 赋值。
                params.requires_grad = False
        # 中文 L0157 条件 self.train_expert_only 成立时进入下方分支。
        if self.train_expert_only:
            # 中文 L0158 调用 self.vlm.eval，切换 PyTorch 模块到评测模式。
            self.vlm.eval()
            # 中文 L0159 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for params in self.vlm.parameters():
                # 中文 L0160 使用布尔值 False给 “params.requires_grad” 赋值。
                params.requires_grad = False
        # 中文 L0161 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # To avoid unused params issue with distributed training
            # 中文 L0163 创建列表，按给定顺序保留元素给 “last_layers” 赋值。
            last_layers = [self.num_vlm_layers - 1]
            # 中文 L0164 条件 ( 成立时进入下方分支。
            if (
                # 中文 L0165 延续上面的语句，self.num_vlm_layers != self.num_expert_layers。
                self.num_vlm_layers != self.num_expert_layers
                # 中文 L0166 续写上一行的布尔表达式，按 and / or 的短路规则判断。
                and self.num_vlm_layers % self.num_expert_layers == 0
            # 中文 L0167 闭合上方尚未结束的括号。
            ):
                # 中文 L0168 调用 last_layers.append，在列表末尾追加一个元素。
                last_layers.append(self.num_vlm_layers - 2)
            # 中文 L0169 创建列表，按给定顺序保留元素给 “frozen_layers” 赋值。
            frozen_layers = [
                # 中文 L0170 字符串字面量，供上方的调用、容器或错误消息使用。
                "lm_head",
                # 中文 L0171 字符串字面量，供上方的调用、容器或错误消息使用。
                "text_model.norm.weight",
            # 中文 L0172 闭合上方尚未结束的括号。
            ]
            # 中文 L0173 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for layer in last_layers:
                # 中文 L0174 调用 frozen_layers.append，在列表末尾追加一个元素。
                frozen_layers.append(f"text_model.layers.{layer}.")

            # 中文 L0176 调用 set，用它的结果给 “unmatched_patterns” 赋值。
            unmatched_patterns = set(frozen_layers)
            # 中文 L0177 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for name, params in self.vlm.named_parameters():
                # 中文 L0178 用列表推导式从可迭代对象逐项取值并构造结果给 “matched_patterns” 赋值。
                matched_patterns = [k for k in frozen_layers if k in name]
                # 中文 L0179 条件 matched_patterns 成立时进入下方分支。
                if matched_patterns:
                    # 中文 L0180 使用布尔值 False给 “params.requires_grad” 赋值。
                    params.requires_grad = False
                    # 中文 L0181 调用 unmatched_patterns.difference_update 执行操作。
                    unmatched_patterns.difference_update(matched_patterns)
            # 中文 L0182 条件 unmatched_patterns 成立时进入下方分支。
            if unmatched_patterns:
                # 中文 L0183 抛出 RuntimeError，中断当前路径并交给上层的异常处理。
                raise RuntimeError(
                    # 中文 L0184 字符串字面量，供上方的调用、容器或错误消息使用。
                    "Some frozen layer patterns matched no VLM parameters, so the corresponding layers "
                    # 中文 L0185 字符串字面量，供上方的调用、容器或错误消息使用。
                    "would silently remain trainable (parameter naming may have changed in transformers): "
                    # 中文 L0186 f 字符串片段，花括号中的表达式会被求值并插入文本。
                    f"{sorted(unmatched_patterns)}"
                # 中文 L0187 闭合上方尚未结束的括号。
                )
        # To avoid unused params issue with distributed training
        # 中文 L0189 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for name, params in self.lm_expert.named_parameters():
            # 中文 L0190 条件 "lm_head" in name 成立时进入下方分支。
            if "lm_head" in name:
                # 中文 L0191 使用布尔值 False给 “params.requires_grad” 赋值。
                params.requires_grad = False

    # 中文 L0193 定义方法 train。
    def train(self, mode: bool = True):
        # 中文 L0194 调用 train 执行操作。
        super().train(mode)

        # 中文 L0196 条件 self.freeze_vision_encoder 成立时进入下方分支。
        if self.freeze_vision_encoder:
            # 中文 L0197 调用 vision_model.eval，切换 PyTorch 模块到评测模式。
            self.get_vlm_model().vision_model.eval()

        # 中文 L0199 条件 self.train_expert_only 成立时进入下方分支。
        if self.train_expert_only:
            # 中文 L0200 调用 self.vlm.eval，切换 PyTorch 模块到评测模式。
            self.vlm.eval()

    # 中文 L0202 输入是一批图像，输出是供语言侧使用的一串图像 token 嵌入。
    def embed_image(self, image: torch.Tensor):
        # 中文 L0203 设为空值 None给 “patch_attention_mask” 赋值。
        patch_attention_mask = None
        # Get sequence from the vision encoder
        # 中文 L0205 读取属性 last_hidden_state给 “image_hidden_states” 赋值。
        image_hidden_states = (
            # 中文 L0206 延续上面的语句，self.get_vlm_model()。
            self.get_vlm_model()
            # 中文 L0207 在骨干内部调用视觉编码器，对像素提取特征序列。
            .vision_model(
                # 中文 L0208 把像素转成视觉编码器使用的元素类型，设备由上游流水线决定。
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                # 中文 L0209 为“patch_attention_mask”提供这一项的值，注意它所在的参数表或外层表达式。
                patch_attention_mask=patch_attention_mask,
            # 中文 L0210 闭合上方尚未结束的括号。
            )
            # 中文 L0211 取出视觉编码器最后一层的隐藏状态。
            .last_hidden_state
        # 中文 L0212 闭合上方尚未结束的括号。
        )
        # Modality projection & resampling
        # 中文 L0214 通过视觉到语言的连接器投影和重采样图像特征。
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        # 中文 L0215 引用已有的值 image_hidden_states作为当前调用的返回值。
        return image_hidden_states

    # 中文 L0217 定义语言 token 查表接口，接收整数 ID 张量。
    def embed_language_tokens(self, tokens: torch.Tensor):
        # 中文 L0218 取文本模型的输入嵌入表并调用它。输入 [B,L] 变成 [B,L,H]。
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    # 中文 L0220 对存在的前缀或后缀序列执行自注意力，并在启用时读写 K/V 缓存。
    def forward_attn_layer(
        # 中文 L0221 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0222 “model_layers”是当前括号中的一项，逗号将它与下一项分开。
        model_layers,
        # 中文 L0223 “inputs_embeds”是当前括号中的一项，逗号将它与下一项分开。
        inputs_embeds,
        # 中文 L0224 “layer_idx”是当前括号中的一项，逗号将它与下一项分开。
        layer_idx,
        # 中文 L0225 “position_ids”是当前括号中的一项，逗号将它与下一项分开。
        position_ids,
        # 中文 L0226 “attention_mask”是当前括号中的一项，逗号将它与下一项分开。
        attention_mask,
        # 中文 L0227 “batch_size”（批大小）是当前括号中的一项，逗号将它与下一项分开。
        batch_size,
        # 中文 L0228 “head_dim”是当前括号中的一项，逗号将它与下一项分开。
        head_dim,
        # 中文 L0229 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
        use_cache: bool = True,
        # 中文 L0230 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
        past_key_values: "DynamicCache | None" = None,
    # 中文 L0231 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> "tuple[list[torch.Tensor], DynamicCache | None]":
        # 中文 L0232 创建列表，按给定顺序保留元素给 “query_states”（查询向量 Q） 赋值。
        query_states = []
        # 中文 L0233 创建列表，按给定顺序保留元素给 “key_states”（键向量 K） 赋值。
        key_states = []
        # 中文 L0234 创建列表，按给定顺序保留元素给 “value_states”（值向量 V） 赋值。
        value_states = []
        # 中文 L0235 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for i, hidden_states in enumerate(inputs_embeds):
            # 中文 L0236 用方括号按键或索引读取 容器给 “layer” 赋值。
            layer = model_layers[i][layer_idx]
            # 中文 L0237 条件 hidden_states is None or layer is None 成立时进入下方分支。
            if hidden_states is None or layer is None:
                # 中文 L0238 跳过这一轮剩余操作，进入下一轮循环。
                continue
            # 中文 L0239 先规范化本层的隐藏状态。
            hidden_states = layer.input_layernorm(hidden_states)

            # 中文 L0241 对 hidden_states.shape 做切片，右端索引不包含在结果中给 “input_shape” 赋值。
            input_shape = hidden_states.shape[:-1]
            # 中文 L0242 保留序列等前导轴，把投影结果的最后一轴拆成多个头以及每头宽度。
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            # 中文 L0244 在投影前使输入 dtype 与投影权重一致。
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            # 中文 L0245 对 token 做 Q 投影，问题从查询这一侧提出。
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            # 中文 L0246 对 token 做 K 投影，用来和查询计算匹配程度。
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            # 中文 L0247 对 token 做 V 投影，真正沿注意力权重混合的是这组向量。
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # 中文 L0249 调用 query_states.append，在列表末尾追加一个元素。
            query_states.append(query_state)
            # 中文 L0250 调用 key_states.append，在列表末尾追加一个元素。
            key_states.append(key_state)
            # 中文 L0251 调用 value_states.append，在列表末尾追加一个元素。
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        # 中文 L0255 沿序列轴拼接已有分支的查询，内部布局保持 [B,L,头数,头宽]。
        query_states = torch.cat(query_states, dim=1)
        # 中文 L0256 调用 torch.cat，用它的结果给 “key_states”（键向量 K） 赋值。
        key_states = torch.cat(key_states, dim=1)
        # 中文 L0257 调用 torch.cat，用它的结果给 “value_states”（值向量 V） 赋值。
        value_states = torch.cat(value_states, dim=1)
        # 中文 L0258 用方括号按键或索引读取 query_states.shape给 “seq_len” 赋值。
        seq_len = query_states.shape[1]
        # 中文 L0259 条件 seq_len < position_ids.shape[1] 成立时进入下方分支。
        if seq_len < position_ids.shape[1]:
            # 中文 L0260 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “_position_ids” 赋值。
            _position_ids = position_ids[:, :seq_len]
            # 中文 L0261 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “_attention_mask” 赋值。
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        # 中文 L0262 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0263 引用已有的值 position_ids给 “_position_ids” 赋值。
            _position_ids = position_ids
            # 中文 L0264 引用已有的值 attention_mask给 “_attention_mask” 赋值。
            _attention_mask = attention_mask

        # 中文 L0266 引用已有的值 _attention_mask给 “attention_mask_” 赋值。
        attention_mask_ = _attention_mask
        # 中文 L0267 引用已有的值 _position_ids给 “position_ids_” 赋值。
        position_ids_ = _position_ids

        # 中文 L0269 把位置信息写进 Q 旋转。
        query_states = apply_rope(query_states, position_ids_)
        # 中文 L0270 对 K 做同样的按位置旋转。
        key_states = apply_rope(key_states, position_ids_)

        # 中文 L0272 条件 use_cache 成立时进入下方分支。
        if use_cache:
            # 中文 L0273 条件 past_key_values is None 成立时进入下方分支。
            if past_key_values is None:
                # 中文 L0274 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(
                    # 中文 L0275 字符串字面量，供上方的调用、容器或错误消息使用。
                    "`use_cache=True` requires a `past_key_values` cache (forward() creates one on prefill)."
                # 中文 L0276 闭合上方尚未结束的括号。
                )
            # `DynamicCache` stores tensors as [batch, heads, seq, head_dim]; this module works with
            # [batch, seq, heads, head_dim]. During prefix prefill this stores the (post-RoPE) K/V and
            # returns them unchanged; during denoising it appends the suffix K/V and returns
            # [prefix; suffix], exactly like the previous hand-rolled dict cache.
            # 中文 L0281 更新本层 K/V。前缀预填充阶段保存条件，后续自注意力可暂时追加本轮动作 token。
            key_states, value_states = past_key_values.update(
                # 中文 L0282 交换序列轴和头轴以符合 DynamicCache 的布局 [B,头数,L,头宽]。
                key_states.transpose(1, 2), value_states.transpose(1, 2), layer_idx
            # 中文 L0283 闭合上方尚未结束的括号。
            )
            # 中文 L0284 把 K 的布局恢复为本模块使用的 [B,L,头数,头宽]。
            key_states = key_states.transpose(1, 2)
            # 中文 L0285 把 V 的布局恢复为本模块使用的 [B,L,头数,头宽]。
            value_states = value_states.transpose(1, 2)

        # 中文 L0287 调用 self.get_attention_interface，用它的结果给 “attention_interface” 赋值。
        attention_interface = self.get_attention_interface()

        # 中文 L0289 调用 attention_interface，用它的结果给 “att_output” 赋值。
        att_output = attention_interface(
            # 中文 L0290 延续上面的语句，attention_mask_, batch_size, head_dim, query_states, key_states, value_states。本段处理对一层做自注意力，并在需要时更新 K/V 缓存。
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        # 中文 L0291 闭合上方尚未结束的括号。
        )
        # 中文 L0292 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return [att_output], past_key_values

    # 中文 L0294 交叉注意力路径让动作专家提出查询，并从观测前缀获得 K/V。
    def forward_cross_attn_layer(
        # 中文 L0295 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0296 “model_layers”是当前括号中的一项，逗号将它与下一项分开。
        model_layers,
        # 中文 L0297 “inputs_embeds”是当前括号中的一项，逗号将它与下一项分开。
        inputs_embeds,
        # 中文 L0298 “layer_idx”是当前括号中的一项，逗号将它与下一项分开。
        layer_idx,
        # 中文 L0299 “position_ids”是当前括号中的一项，逗号将它与下一项分开。
        position_ids,
        # 中文 L0300 “attention_mask”是当前括号中的一项，逗号将它与下一项分开。
        attention_mask,
        # 中文 L0301 “batch_size”（批大小）是当前括号中的一项，逗号将它与下一项分开。
        batch_size,
        # 中文 L0302 “head_dim”是当前括号中的一项，逗号将它与下一项分开。
        head_dim,
        # 中文 L0303 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
        use_cache: bool = True,
        # 中文 L0304 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
        past_key_values: "DynamicCache | None" = None,
    # 中文 L0305 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> "tuple[list[torch.Tensor], DynamicCache | None]":
        # 中文 L0306 调用 self.get_attention_interface，用它的结果给 “attention_interface” 赋值。
        attention_interface = self.get_attention_interface()

        # 中文 L0308 创建列表，按给定顺序保留元素给 “att_outputs” 赋值。
        att_outputs = []
        # 中文 L0309 检查此条件，条件不成立时抛出 AssertionError。
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None), (
            # 中文 L0310 f 字符串片段，花括号中的表达式会被求值并插入文本。
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        # 中文 L0311 闭合上方尚未结束的括号。
        )

        # 中文 L0313 条件 len(inputs_embeds) == 2 and not past_key_values 成立时进入下方分支。
        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            # 中文 L0315 用方括号按键或索引读取 shape给 “seq_len” 赋值。
            seq_len = inputs_embeds[0].shape[1]
            # 中文 L0316 打包成元组，元素顺序由逗号分隔的位置决定给 “(position_id, expert_position_id)” 赋值。
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            # 中文 L0317 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “prefix_attention_mask” 赋值。
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            # 中文 L0319 用方括号按键或索引读取 容器给 “layer” 赋值。
            layer = model_layers[0][layer_idx]

            # 中文 L0321 调用 layer.input_layernorm，用它的结果给 “hidden_states” 赋值。
            hidden_states = layer.input_layernorm(inputs_embeds[0])

            # 中文 L0323 对 hidden_states.shape 做切片，右端索引不包含在结果中给 “input_shape” 赋值。
            input_shape = hidden_states.shape[:-1]
            # 中文 L0324 打包成元组，元素顺序由逗号分隔的位置决定给 “hidden_shape” 赋值。
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            # 中文 L0326 调用 hidden_states.to，用它的结果给 “hidden_states” 赋值。
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            # 中文 L0327 调用 view，用它的结果给 “query_state” 赋值。
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            # 中文 L0328 调用 view，用它的结果给 “key_state” 赋值。
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            # 中文 L0329 调用 view，用它的结果给 “value_states”（值向量 V） 赋值。
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # 中文 L0332 调用 apply_rope，用它的结果给 “query_states”（查询向量 Q） 赋值。
            query_states = apply_rope(query_state, position_id)
            # 中文 L0333 调用 apply_rope，用它的结果给 “key_states”（键向量 K） 赋值。
            key_states = apply_rope(key_state, position_id)

            # 中文 L0335 调用 attention_interface，用它的结果给 “att_output” 赋值。
            att_output = attention_interface(
                # 中文 L0336 延续上面的语句，prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states。本段处理让动作专家使用观测前缀 K/V 执行交叉注意力。
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            # 中文 L0337 闭合上方尚未结束的括号。
            )
            # 中文 L0338 调用 att_outputs.append，在列表末尾追加一个元素。
            att_outputs.append(att_output)
        # 中文 L0339 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0340 引用已有的值 position_ids给 “expert_position_id” 赋值。
            expert_position_id = position_ids

        # 中文 L0342 条件 use_cache and past_key_values is not None 成立时进入下方分支。
        if use_cache and past_key_values is not None:
            # Cross-attention layers never fill the cache themselves: during the prefix prefill every
            # layer goes through `forward_attn_layer`, which stores the (post-RoPE) VLM K/V for this
            # layer index. Here we only read them back (no concatenation: the expert cross-attends to
            # the fixed prefix). `DynamicCache` stores [batch, heads, seq, head_dim]; transpose back.
            # 中文 L0347 从当前层缓存取出前缀的 K，再转回本模块布局。
            key_states = past_key_values.layers[layer_idx].keys.transpose(1, 2)
            # 中文 L0348 从当前层缓存取出前缀的 V，再转回本模块布局。
            value_states = past_key_values.layers[layer_idx].values.transpose(1, 2)

        # Expert
        # 中文 L0351 用方括号按键或索引读取 容器给 “expert_layer” 赋值。
        expert_layer = model_layers[1][layer_idx]
        # 中文 L0352 条件 expert_layer is not None 成立时进入下方分支。
        if expert_layer is not None:
            # 中文 L0353 调用 expert_layer.input_layernorm，用它的结果给 “expert_hidden_states” 赋值。
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            # 中文 L0355 对 expert_hidden_states.shape 做切片，右端索引不包含在结果中给 “expert_input_shape” 赋值。
            expert_input_shape = expert_hidden_states.shape[:-1]
            # 中文 L0356 打包成元组，元素顺序由逗号分隔的位置决定给 “expert_hidden_shape” 赋值。
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            # 中文 L0358 调用 expert_hidden_states.to，用它的结果给 “expert_hidden_states” 赋值。
            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            # 中文 L0359 把动作后缀隐藏状态投影为专家的 Q。
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            # reshape (not view): K/V read back from the cache are transposed, hence non-contiguous
            # 中文 L0362 合并相邻的 K 头维度，以便专家的 K 线性层重新投影。
            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).reshape(
                # 中文 L0363 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
                *key_states.shape[:2], -1
            # 中文 L0364 闭合上方尚未结束的括号。
            )
            # 中文 L0365 调用 view，用它的结果给 “expert_key_states” 赋值。
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                # 中文 L0366 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            # 中文 L0367 闭合上方尚未结束的括号。
            )  # k_proj should have same dim as kv

            # 中文 L0369 对 V 做同类布局变换，使它能进入专家的 V 线性层。
            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).reshape(
                # 中文 L0370 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
                *value_states.shape[:2], -1
            # 中文 L0371 闭合上方尚未结束的括号。
            )
            # 中文 L0372 调用 view，用它的结果给 “expert_value_states” 赋值。
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                # 中文 L0373 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            # 中文 L0374 闭合上方尚未结束的括号。
            )

            # 中文 L0376 计算右侧表达式 (给 “expert_position_id” 赋值。
            expert_position_id = (
                # 中文 L0377 延续上面的语句，expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values。本段处理让动作专家使用观测前缀 K/V 执行交叉注意力。
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            # 中文 L0378 闭合上方尚未结束的括号。
            )  # start from 0
            # 中文 L0379 从总掩码选择后缀查询和前缀键对应的矩形区域。
            expert_attention_mask = attention_mask[
                # 中文 L0380 延续上面的语句，:, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :。本段处理让动作专家使用观测前缀 K/V 执行交叉注意力。
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            # 中文 L0381 闭合上方尚未结束的括号。
            ]  # take into account kv

            # 中文 L0383 对专家 Q 应用位置旋转。
            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            # 中文 L0385 调用 attention_interface，用它的结果给 “att_output” 赋值。
            att_output = attention_interface(
                # 中文 L0386 “expert_attention_mask”是当前括号中的一项，逗号将它与下一项分开。
                expert_attention_mask,
                # 中文 L0387 “batch_size”（批大小）是当前括号中的一项，逗号将它与下一项分开。
                batch_size,
                # 中文 L0388 “head_dim”是当前括号中的一项，逗号将它与下一项分开。
                head_dim,
                # 中文 L0389 “expert_query_states”是当前括号中的一项，逗号将它与下一项分开。
                expert_query_states,
                # 中文 L0390 “expert_key_states”是当前括号中的一项，逗号将它与下一项分开。
                expert_key_states,
                # 中文 L0391 “expert_value_states”是当前括号中的一项，逗号将它与下一项分开。
                expert_value_states,
            # 中文 L0392 闭合上方尚未结束的括号。
            )
            # 中文 L0393 调用 att_outputs.append，在列表末尾追加一个元素。
            att_outputs.append(att_output)
        # 中文 L0394 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0395 调用 att_outputs.append，在列表末尾追加一个元素。
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        # 中文 L0398 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return att_outputs, past_key_values

    # 中文 L0400 定义方法 get_model_layers。
    def get_model_layers(self, models: list) -> list:
        # 中文 L0401 创建列表，按给定顺序保留元素给 “vlm_layers” 赋值。
        vlm_layers = []
        # 中文 L0402 创建列表，按给定顺序保留元素给 “expert_layers” 赋值。
        expert_layers = []
        # 中文 L0403 计算右侧表达式 self.num_vlm_layers // self.num_expert_layers给 “multiple_of” 赋值。
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        # 中文 L0404 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for i in range(self.num_vlm_layers):
            # 中文 L0405 条件 multiple_of > 0 and i > 0 and i % multiple_of != 0 成立时进入下方分支。
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                # 中文 L0406 设为空值 None给 “expert_layer” 赋值。
                expert_layer = None
            # 中文 L0407 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0408 用条件表达式在 if 前与 else 后的两个值之间选择给 “expert_layer_index” 赋值。
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                # 中文 L0409 用方括号按键或索引读取 layers给 “expert_layer” 赋值。
                expert_layer = models[1].layers[expert_layer_index]
            # 中文 L0410 调用 vlm_layers.append，在列表末尾追加一个元素。
            vlm_layers.append(models[0].layers[i])
            # 中文 L0411 调用 expert_layers.append，在列表末尾追加一个元素。
            expert_layers.append(expert_layer)
        # 中文 L0412 创建列表，按给定顺序保留元素作为当前调用的返回值。
        return [vlm_layers, expert_layers]

    # 中文 L0414 逐层处理视觉语言前缀或动作后缀，并在需要时构建或复用缓存。
    def forward(
        # 中文 L0415 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0416 为“attention_mask”提供这一项的值，注意它所在的参数表或外层表达式。
        attention_mask: torch.Tensor | None = None,
        # 中文 L0417 为“position_ids”提供这一项的值，注意它所在的参数表或外层表达式。
        position_ids: torch.LongTensor | None = None,
        # 中文 L0418 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
        past_key_values: "DynamicCache | None" = None,
        # 中文 L0419 为“inputs_embeds”提供这一项的值，注意它所在的参数表或外层表达式。
        inputs_embeds: list[torch.Tensor | None] | None = None,
        # 中文 L0420 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
        use_cache: bool | None = None,
    # 中文 L0421 闭合上方尚未结束的括号。
    ):
        # 中文 L0422 条件 inputs_embeds is None 成立时进入下方分支。
        if inputs_embeds is None:
            # 中文 L0423 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0424 字符串字面量，供上方的调用、容器或错误消息使用。
                "`inputs_embeds` is required: pass the [prefix, suffix] embeddings (either may be None)."
            # 中文 L0425 闭合上方尚未结束的括号。
            )
        # `None` means "no cache", like `False`.
        # 中文 L0427 把可选的缓存开关转成明确的布尔值。
        use_cache = bool(use_cache)
        # 中文 L0428 创建列表，按给定顺序保留元素给 “models” 赋值。
        models = [self.get_vlm_model().text_model, self.lm_expert]
        # 中文 L0429 调用 self.get_model_layers，用它的结果给 “model_layers” 赋值。
        model_layers = self.get_model_layers(models)
        # 中文 L0430 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            # 中文 L0434 条件 hidden_states is None 成立时进入下方分支。
            if hidden_states is None:
                # 中文 L0435 跳过这一轮剩余操作，进入下一轮循环。
                continue
            # 中文 L0436 用方括号按键或索引读取 hidden_states.shape给 “batch_size”（批大小） 赋值。
            batch_size = hidden_states.shape[0]

        # Prefix prefill: no cache was passed, so create one and fill it (every layer runs
        # self-attention over the prefix). When a filled cache is passed (denoising), layers
        # read from it instead.
        # 中文 L0441 启用缓存且当前还未传入缓存对象时，视为前缀预填充。
        fill_kv_cache = use_cache and past_key_values is None
        # 中文 L0442 条件 fill_kv_cache 成立时进入下方分支。
        if fill_kv_cache:
            # 中文 L0443 为前缀预计算创建一个空的逐层 K/V 缓存。
            past_key_values = DynamicCache()

        # RMSNorm
        # 中文 L0446 读取属性 self.num_vlm_layers给 “num_layers” 赋值。
        num_layers = self.num_vlm_layers
        # 中文 L0447 读取属性 self.vlm.config.text_config.head_dim给 “head_dim” 赋值。
        head_dim = self.vlm.config.text_config.head_dim
        # 中文 L0448 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for layer_idx in range(num_layers):
            # 中文 L0449 条件 ( 成立时进入下方分支。
            if (
                # 中文 L0450 前缀预填充优先走自注意力来建立缓存。
                fill_kv_cache
                # 中文 L0451 续写上一行的布尔表达式，按 and / or 的短路规则判断。
                or "cross" not in self.attention_mode
                # 中文 L0452 在后缀推理阶段，配置指定的层间隔仍采用自注意力。
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            # 中文 L0453 闭合上方尚未结束的括号。
            ):
                # 中文 L0454 将这一层交给自注意力实现。
                att_outputs, past_key_values = self.forward_attn_layer(
                    # 中文 L0455 “model_layers”是当前括号中的一项，逗号将它与下一项分开。
                    model_layers,
                    # 中文 L0456 “inputs_embeds”是当前括号中的一项，逗号将它与下一项分开。
                    inputs_embeds,
                    # 中文 L0457 “layer_idx”是当前括号中的一项，逗号将它与下一项分开。
                    layer_idx,
                    # 中文 L0458 “position_ids”是当前括号中的一项，逗号将它与下一项分开。
                    position_ids,
                    # 中文 L0459 “attention_mask”是当前括号中的一项，逗号将它与下一项分开。
                    attention_mask,
                    # 中文 L0460 “batch_size”（批大小）是当前括号中的一项，逗号将它与下一项分开。
                    batch_size,
                    # 中文 L0461 “head_dim”是当前括号中的一项，逗号将它与下一项分开。
                    head_dim,
                    # 中文 L0462 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
                    use_cache=use_cache,
                    # 中文 L0463 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
                    past_key_values=past_key_values,
                # 中文 L0464 闭合上方尚未结束的括号。
                )
            # 中文 L0465 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0466 剩下的交叉注意力层交给专门实现读取前缀条件。
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    # 中文 L0467 “model_layers”是当前括号中的一项，逗号将它与下一项分开。
                    model_layers,
                    # 中文 L0468 “inputs_embeds”是当前括号中的一项，逗号将它与下一项分开。
                    inputs_embeds,
                    # 中文 L0469 “layer_idx”是当前括号中的一项，逗号将它与下一项分开。
                    layer_idx,
                    # 中文 L0470 “position_ids”是当前括号中的一项，逗号将它与下一项分开。
                    position_ids,
                    # 中文 L0471 “attention_mask”是当前括号中的一项，逗号将它与下一项分开。
                    attention_mask,
                    # 中文 L0472 “batch_size”（批大小）是当前括号中的一项，逗号将它与下一项分开。
                    batch_size,
                    # 中文 L0473 “head_dim”是当前括号中的一项，逗号将它与下一项分开。
                    head_dim,
                    # 中文 L0474 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
                    use_cache=use_cache,
                    # 中文 L0475 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
                    past_key_values=past_key_values,
                # 中文 L0476 闭合上方尚未结束的括号。
                )
            # 中文 L0477 创建列表，按给定顺序保留元素给 “outputs_embeds” 赋值。
            outputs_embeds = []
            # 中文 L0478 使用字面值 0给 “start” 赋值。
            start = 0
            # 中文 L0479 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for i, hidden_states in enumerate(inputs_embeds):
                # 中文 L0480 用方括号按键或索引读取 容器给 “layer” 赋值。
                layer = model_layers[i][layer_idx]
                # 中文 L0481 用条件表达式在 if 前与 else 后的两个值之间选择给 “att_output” 赋值。
                att_output = (
                    # 中文 L0482 延续上面的语句，att_outputs[i] if i < len(att_outputs) else att_outputs[0]。本段处理逐层执行所选注意力再做残差和前馈变换。
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                # 中文 L0483 闭合上方尚未结束的括号。
                )  # in case of self_attn
                # 中文 L0484 条件 hidden_states is not None 成立时进入下方分支。
                if hidden_states is not None:
                    # 中文 L0485 条件 layer is None 成立时进入下方分支。
                    if layer is None:
                        # 中文 L0486 调用 outputs_embeds.append，在列表末尾追加一个元素。
                        outputs_embeds.append(hidden_states)
                        # 中文 L0487 跳过这一轮剩余操作，进入下一轮循环。
                        continue
                    # 中文 L0488 计算右侧表达式 start + hidden_states.shape[1]给 “end” 赋值。
                    end = start + hidden_states.shape[1]

                    # 中文 L0490 条件 att_output.dtype != layer.self_attn.o_proj.weight.dtype 成立时进入下方分支。
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        # 中文 L0491 调用 att_output.to，用它的结果给 “att_output” 赋值。
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    # 中文 L0492 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “att_out” 赋值。
                    att_out = att_output[:, start:end]
                    # 中文 L0493 将各注意力头合并后的输出投影回当前支路的隐藏空间。
                    out_emb = layer.self_attn.o_proj(att_out)

                    # 中文 L0495 将输入隐藏状态加回，构成第一个残差连接。
                    out_emb += hidden_states
                    # 中文 L0496 调用 out_emb.clone，用它的结果给 “after_first_residual” 赋值。
                    after_first_residual = out_emb.clone()

                    # 中文 L0498 调用 layer.post_attention_layernorm，用它的结果给 “out_emb” 赋值。
                    out_emb = layer.post_attention_layernorm(out_emb)
                    # 中文 L0499 经过非线性前馈网络以更新 token 的特征。
                    out_emb = layer.mlp(out_emb)

                    # 中文 L0501 将注意力子层输出加回，构成第二个残差连接。
                    out_emb += after_first_residual

                    # 中文 L0503 调用 outputs_embeds.append，在列表末尾追加一个元素。
                    outputs_embeds.append(out_emb)

                    # 中文 L0505 用条件表达式在 if 前与 else 后的两个值之间选择给 “start” 赋值。
                    start = end if len(att_outputs) == 1 else 0
                # 中文 L0506 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
                else:
                    # 中文 L0507 调用 outputs_embeds.append，在列表末尾追加一个元素。
                    outputs_embeds.append(None)

            # 中文 L0509 引用已有的值 outputs_embeds给 “inputs_embeds” 赋值。
            inputs_embeds = outputs_embeds

        # final norm
        # 中文 L0512 创建列表，按给定顺序保留元素给 “outputs_embeds” 赋值。
        outputs_embeds = []
        # 中文 L0513 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for i, hidden_states in enumerate(inputs_embeds):
            # 中文 L0514 条件 hidden_states is not None 成立时进入下方分支。
            if hidden_states is not None:
                # 中文 L0515 对每条有效支路的末层隐藏状态做最终归一化。
                out_emb = models[i].norm(hidden_states)
                # 中文 L0516 调用 outputs_embeds.append，在列表末尾追加一个元素。
                outputs_embeds.append(out_emb)
            # 中文 L0517 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0518 调用 outputs_embeds.append，在列表末尾追加一个元素。
                outputs_embeds.append(None)
        # 中文 L0519 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return outputs_embeds, past_key_values

    # 中文 L0521 定义方法 get_attention_interface。
    def get_attention_interface(self):
        # 中文 L0522 读取属性 self.eager_attention_forward给 “attention_interface” 赋值。
        attention_interface = self.eager_attention_forward
        # 中文 L0523 引用已有的值 attention_interface作为当前调用的返回值。
        return attention_interface

    # 中文 L0525 实际调用的普通 PyTorch 注意力实现，适合看清各个张量运算。
    def eager_attention_forward(
        # 中文 L0526 延续上面的语句，self, attention_mask, batch_size, head_dim, query_states, key_states, value_states。本段处理用矩阵乘法、掩码和 softmax 计算注意力输出。
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    # 中文 L0527 闭合上方尚未结束的括号。
    ):
        # 中文 L0528 读取属性 self.num_attention_heads给 “num_att_heads” 赋值。
        num_att_heads = self.num_attention_heads
        # 中文 L0529 读取属性 self.num_key_value_heads给 “num_key_value_heads” 赋值。
        num_key_value_heads = self.num_key_value_heads
        # 中文 L0530 多个查询头可以共享一个 K/V 头。该比例决定需重复的组数。
        num_key_value_groups = num_att_heads // num_key_value_heads

        # 中文 L0532 用方括号按键或索引读取 key_states.shape给 “sequence_length” 赋值。
        sequence_length = key_states.shape[1]

        # 中文 L0534 调用 expand，用它的结果给 “key_states”（键向量 K） 赋值。
        key_states = key_states[:, :, :, None, :].expand(
            # 中文 L0535 延续上面的语句，batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim。本段处理用矩阵乘法、掩码和 softmax 计算注意力输出。
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        # 中文 L0536 闭合上方尚未结束的括号。
        )
        # 中文 L0537 调用 key_states.reshape，用它的结果给 “key_states”（键向量 K） 赋值。
        key_states = key_states.reshape(
            # 中文 L0538 延续上面的语句，batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim。本段处理用矩阵乘法、掩码和 softmax 计算注意力输出。
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        # 中文 L0539 闭合上方尚未结束的括号。
        )

        # 中文 L0541 调用 expand，用它的结果给 “value_states”（值向量 V） 赋值。
        value_states = value_states[:, :, :, None, :].expand(
            # 中文 L0542 延续上面的语句，batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim。本段处理用矩阵乘法、掩码和 softmax 计算注意力输出。
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        # 中文 L0543 闭合上方尚未结束的括号。
        )
        # 中文 L0544 调用 value_states.reshape，用它的结果给 “value_states”（值向量 V） 赋值。
        value_states = value_states.reshape(
            # 中文 L0545 延续上面的语句，batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim。本段处理用矩阵乘法、掩码和 softmax 计算注意力输出。
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        # 中文 L0546 闭合上方尚未结束的括号。
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        # 中文 L0549 将 Q 提升为 float32 后计算点积，提高分数计算精度。
        query_states = query_states.to(dtype=torch.float32)
        # 中文 L0550 将 K 也转成 float32，与 Q 一起计算。
        key_states = key_states.to(dtype=torch.float32)

        # 中文 L0552 调用 query_states.transpose，用它的结果给 “query_states”（查询向量 Q） 赋值。
        query_states = query_states.transpose(1, 2)
        # 中文 L0553 调用 key_states.transpose，用它的结果给 “key_states”（键向量 K） 赋值。
        key_states = key_states.transpose(1, 2)

        # 中文 L0555 矩阵乘法计算每个查询与每个键的点积，得到 [B,头数,L查询,L键]。
        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        # 中文 L0556 用头宽的平方根缩放点积，使分数尺度更稳定。
        att_weights *= head_dim**-0.5

        # 中文 L0558 调用 att_weights.to，用它的结果给 “att_weights” 赋值。
        att_weights = att_weights.to(dtype=torch.float32)
        # 中文 L0559 读取属性 min给 “big_neg” 赋值。
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        # 中文 L0560 把不可见键对应分数填为极小值。掩码在头轴广播。
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        # 中文 L0561 沿键位置轴用 softmax 归一化，每个查询得到一组对键的权重。
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        # 中文 L0562 将注意力概率转回 V 的 dtype 再做加权求和。
        probs = probs.to(dtype=value_states.dtype)

        # 中文 L0564 概率矩阵和 V 相乘，对可见键对应的值向量加权求和。
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        # 中文 L0566 从 [B,头数,L查询,头宽] 变成 [B,L查询,头数,头宽]。
        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        # 中文 L0568 合并头数与头宽，得到 [B,L查询,隐藏维度]。
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        # 中文 L0570 引用已有的值 att_output作为当前调用的返回值。
        return att_output
