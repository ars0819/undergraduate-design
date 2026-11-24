#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

# 本文件已在 LeRobot 原版的基础上加入中文教学注释。
# 中文 L0055 导入模块 math。
import math
# 中文 L0056 从模块 collections 导入 deque。
from collections import deque
# 中文 L0057 从模块 typing 导入 Any、Unpack。
from typing import Any, Unpack

# 中文 L0059 导入模块 torch。
import torch
# 中文 L0060 导入模块 torch.nn.functional，简称 F。
import torch.nn.functional as F  # noqa: N812
# 中文 L0061 从模块 torch 导入 Tensor、nn。
from torch import Tensor, nn

# 中文 L0063 从模块 lerobot.configs 导入 FeatureType。
from lerobot.configs import FeatureType
# 中文 L0064 从模块 lerobot.utils.constants 导入 ACTION、OBS_LANGUAGE_ATTENTION_MASK、OBS_LANGUAGE_TOKENS、OBS_STATE。
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
# 中文 L0065 从模块 lerobot.utils.feature_spec 导入 FeatureSpec。
from lerobot.utils.feature_spec import FeatureSpec
# 中文 L0066 从模块 lerobot.utils.import_utils 导入 require_package。
from lerobot.utils.import_utils import require_package

# 中文 L0068 从模块 ..common.flow_matching 导入 FlowConvention、euler_integrate、make_flow_matching_inputs、sample_noise、sample_time_beta。
from ..common.flow_matching import (
    # 中文 L0069 “FlowConvention”是当前括号中的一项，逗号将它与下一项分开。
    FlowConvention,
    # 中文 L0070 “euler_integrate”是当前括号中的一项，逗号将它与下一项分开。
    euler_integrate,
    # 中文 L0071 “make_flow_matching_inputs”是当前括号中的一项，逗号将它与下一项分开。
    make_flow_matching_inputs,
    # 中文 L0072 “sample_noise”是当前括号中的一项，逗号将它与下一项分开。
    sample_noise,
    # 中文 L0073 “sample_time_beta”是当前括号中的一项，逗号将它与下一项分开。
    sample_time_beta,
# 中文 L0074 闭合上方尚未结束的括号。
)
# 中文 L0075 从模块 ..common.vla_utils 导入 create_sinusoidal_pos_embedding、make_att_2d_masks、pad_vector、resize_with_pad。
from ..common.vla_utils import (
    # 中文 L0076 “create_sinusoidal_pos_embedding”是当前括号中的一项，逗号将它与下一项分开。
    create_sinusoidal_pos_embedding,
    # 中文 L0077 “make_att_2d_masks”是当前括号中的一项，逗号将它与下一项分开。
    make_att_2d_masks,
    # 中文 L0078 “pad_vector”是当前括号中的一项，逗号将它与下一项分开。
    pad_vector,
    # 中文 L0079 “resize_with_pad”是当前括号中的一项，逗号将它与下一项分开。
    resize_with_pad,
# 中文 L0080 闭合上方尚未结束的括号。
)
# 中文 L0081 从模块 ..pretrained 导入 PreTrainedPolicy、RTCActionSelectKwargs。
from ..pretrained import PreTrainedPolicy, RTCActionSelectKwargs
# 中文 L0082 从模块 ..rtc.modeling_rtc 导入 RTCProcessor。
from ..rtc.modeling_rtc import RTCProcessor
# 中文 L0083 从模块 ..utils 导入 populate_queues。
from ..utils import (
    # 中文 L0084 “populate_queues”是当前括号中的一项，逗号将它与下一项分开。
    populate_queues,
# 中文 L0085 闭合上方尚未结束的括号。
)
# 中文 L0086 从模块 .configuration_smolvla 导入 SmolVLAConfig。
from .configuration_smolvla import SmolVLAConfig
# 中文 L0087 从模块 .smolvlm_with_expert 导入 SmolVLMWithExpertModel。
from .smolvlm_with_expert import SmolVLMWithExpertModel


# 中文 L0090 定义函数 normalize。
def normalize(x, min_val, max_val):
    # 中文 L0091 计算右侧表达式 return (x - min_val) / (max_val - min_val)作为当前调用的返回值。
    return (x - min_val) / (max_val - min_val)


# 中文 L0094 定义函数 unnormalize。
def unnormalize(x, min_val, max_val):
    # 中文 L0095 计算右侧表达式 return x * (max_val - min_val) + min_val作为当前调用的返回值。
    return x * (max_val - min_val) + min_val


# 中文 L0098 定义函数 safe_arcsin。
def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    # 中文 L0101 调用 torch.arcsin，用它的结果作为当前调用的返回值。
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


# 中文 L0104 定义函数 aloha_gripper_to_angular。
def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    # 中文 L0111 调用 unnormalize，用它的结果给 “value” 赋值。
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    # 中文 L0114 定义函数 linear_to_radian。
    def linear_to_radian(linear_position, arm_length, horn_radius):
        # 中文 L0115 计算右侧表达式 (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)给 “value” 赋值。
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        # 中文 L0116 调用 safe_arcsin，用它的结果作为当前调用的返回值。
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    # 中文 L0119 调用 linear_to_radian，用它的结果给 “value” 赋值。
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    # 中文 L0123 调用 normalize，用它的结果作为当前调用的返回值。
    return normalize(value, min_val=0.4, max_val=1.5)


# 中文 L0126 定义函数 aloha_gripper_from_angular。
def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    # 中文 L0131 调用 unnormalize，用它的结果给 “value” 赋值。
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    # 中文 L0135 调用 normalize，用它的结果作为当前调用的返回值。
    return normalize(value, min_val=-0.6213, max_val=1.4910)


# 中文 L0138 定义函数 aloha_gripper_from_angular_inv。
def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    # 中文 L0140 调用 unnormalize，用它的结果给 “value” 赋值。
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    # 中文 L0141 调用 normalize，用它的结果作为当前调用的返回值。
    return normalize(value, min_val=0.4, max_val=1.5)


# 中文 L0144 定义面向评测器的策略类，继承 PreTrainedPolicy 提供的权重加载和统一接口。
class SmolVLAPolicy(PreTrainedPolicy):
    # 中文 L0145 原文档概述本段功能，承接评测器调用并管理动作队列。
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    # 中文 L0147 类属性指明与该策略对应的配置类。
    config_class = SmolVLAConfig
    # 中文 L0148 类属性给策略设置注册名称。
    name = "smolvla"

    # 中文 L0150 定义方法 supports_rtc。
    def supports_rtc(self) -> bool:
        # 中文 L0151 使用布尔值 True作为当前调用的返回值。
        return True

    # 中文 L0153 策略构造方法，接收解析完成的 SmolVLAConfig 实例。
    def __init__(
        # 中文 L0154 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0155 “config”（配置对象）是当前括号中的一项，逗号将它与下一项分开。
        config: SmolVLAConfig,
        # 中文 L0156 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **kwargs,
    # 中文 L0157 闭合上方尚未结束的括号。
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        # 中文 L0164 调用 require_package 执行操作。
        require_package("transformers", extra="smolvla")
        # 中文 L0165 调用父类初始化，从而完成 PyTorch nn.Module 和策略基类的初始化。
        super().__init__(config)
        # 中文 L0166 现在检查输入特征。特征定义可以来自 checkpoint 或调用方传入的描述。
        config.validate_features()
        # 中文 L0167 用 self.config 保存实例属性，其他方法可通过 self 访问。
        self.config = config
        # 中文 L0168 调用 self.init_rtc_processor 执行操作。
        self.init_rtc_processor()
        # 中文 L0169 构造内部生成模型。self.model 由这里持有，策略的动作队列在外面管理。
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        # 中文 L0170 创建初始的空动作队列。
        self.reset()

    # 中文 L0172 定义方法 reset。
    def reset(self):
        # 中文 L0173 原文档中的说明或续行。
        """This should be called whenever the environment is reset."""
        # 中文 L0174 每次 reset 都创建新的队列字典，避免上回合余下的动作进入新回合。
        self._queues = {
            # 中文 L0175 用 deque 实现先进先出的动作队列。它的容量由 n_action_steps 决定。
            ACTION: deque(maxlen=self.config.n_action_steps),
        # 中文 L0176 闭合上方尚未结束的括号。
        }

    # 中文 L0178 定义方法 init_rtc_processor。
    def init_rtc_processor(self):
        # 中文 L0179 原文档中的说明或续行。
        """Initialize RTC processor if RTC is enabled in config."""
        # 中文 L0180 设为空值 None给 “self.rtc_processor” 赋值。
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        # 中文 L0184 条件 self.config.rtc_config is not None 成立时进入下方分支。
        if self.config.rtc_config is not None:
            # 中文 L0185 调用 RTCProcessor，用它的结果给 “self.rtc_processor” 赋值。
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            # 中文 L0190 调用 getattr，用它的结果给 “model_value” 赋值。
            model_value = getattr(self, "model", None)
            # 中文 L0191 条件 model_value is not None 成立时进入下方分支。
            if model_value is not None:
                # 中文 L0192 读取属性 self.rtc_processor给 “model_value.rtc_processor” 赋值。
                model_value.rtc_processor = self.rtc_processor

    # 中文 L0194 定义方法 get_optim_params。
    def get_optim_params(self) -> dict:
        # 中文 L0195 调用 self.parameters，用它的结果作为当前调用的返回值。
        return self.parameters()

    # 中文 L0197 把一批图像、状态和文字送入生成模型，并将内部动作宽度裁成对外宽度。
    def _get_action_chunk(
        # 中文 L0198 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0199 延续上面的语句，batch: dict[str, Tensor],。本段处理准备模型输入，采样动作块并裁出实际动作维度。
        batch: dict[str, Tensor],
        # 中文 L0200 后续参数必须按名字传入。
        *,
        # 中文 L0201 为“noise”（采样起点噪声）提供这一项的值，注意它所在的参数表或外层表达式。
        noise: Tensor | None = None,
        # 中文 L0202 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **kwargs: Unpack[RTCActionSelectKwargs],
    # 中文 L0203 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        # 中文 L0209 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for k in batch:
            # 中文 L0210 只为已有历史队列的观测键组装时间维。ACTION 的队列另有入队逻辑。
            if k in self._queues and k != ACTION:
                # 中文 L0211 沿第 1 轴堆叠历史，形成 [批次,观测时间,…] 的布局。
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        # 中文 L0213 按 checkpoint 期望的相机键取出并处理图像，返回图像和相机有效掩码。
        images, img_masks = self.prepare_images(batch)
        # 中文 L0214 处理机器人状态的观测时间轴和末维，使状态符合内部投影层输入宽度。
        state = self.prepare_state(batch)
        # 中文 L0215 读取前处理分词得到的语言 token ID 张量。
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        # 中文 L0216 读取语言掩码，用来区分有效 token 和填充 token。
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # 中文 L0218 调用模型生成一整个动作块，形状是 [B,chunk_size,max_action_dim]。
        actions = self.model.sample_actions(
            # 中文 L0219 延续上面的语句，images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs。本段处理准备模型输入，采样动作块并裁出实际动作维度。
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        # 中文 L0220 闭合上方尚未结束的括号。
        )

        # Unpad actions
        # 中文 L0223 读取 checkpoint 声明的输出动作特征。
        action_feature = self.config.action_feature
        # 中文 L0224 条件 action_feature is None 成立时进入下方分支。
        if action_feature is None:
            # 中文 L0225 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("SmolVLA requires an action output feature.")
        # 中文 L0226 保留真实动作宽度。实际评测将 [1,50,32] 裁成 [1,50,7]。
        actions = actions[:, :, : action_feature.shape[0]]

        # 中文 L0228 条件 self.config.adapt_to_pi_aloha 成立时进入下方分支。
        if self.config.adapt_to_pi_aloha:
            # 中文 L0229 调用 self._pi_aloha_encode_actions，用它的结果给 “actions”（动作批或动作序列） 赋值。
            actions = self._pi_aloha_encode_actions(actions)

        # 中文 L0231 引用已有的值 actions作为当前调用的返回值。
        return actions

    # 中文 L0233 定义方法 _prepare_batch。
    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # 中文 L0234 进入 Aloha 专用兼容转换的条件。本次 LIBERO 配置将其关闭。
        if self.config.adapt_to_pi_aloha:
            # 中文 L0235 调用 self._pi_aloha_decode_state，用它的结果给 “batch[OBS_STATE]” 赋值。
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        # 中文 L0237 引用已有的值 batch作为当前调用的返回值。
        return batch

    # 中文 L0239 包装整块预测方法，执行推理时关闭梯度记录。
    @torch.no_grad()
    # 中文 L0240 定义方法 predict_action_chunk。
    def predict_action_chunk(
        # 中文 L0241 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0242 延续上面的语句，batch: dict[str, Tensor],。
        batch: dict[str, Tensor],
        # 中文 L0243 后续参数必须按名字传入。
        *,
        # 中文 L0244 为“noise”（采样起点噪声）提供这一项的值，注意它所在的参数表或外层表达式。
        noise: Tensor | None = None,
        # 中文 L0245 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **kwargs: Unpack[RTCActionSelectKwargs],
    # 中文 L0246 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor:
        # 中文 L0247 设置为评测态，改变 Dropout 等训练相关模块的行为。
        self.eval()

        # 中文 L0249 调用 self._prepare_batch，用它的结果给 “batch”（批量输入字典） 赋值。
        batch = self._prepare_batch(batch)
        # 中文 L0250 调用 populate_queues，用它的结果给 “self._queues” 赋值。
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # 中文 L0252 这条整块预测接口直接算动作序列，调用方可自行执行或调度。
        actions = self._get_action_chunk(batch, noise=noise, **kwargs)
        # 中文 L0253 引用已有的值 actions作为当前调用的返回值。
        return actions

    # 中文 L0255 在逐步动作选择方法中关闭梯度记录。
    @torch.no_grad()
    # 中文 L0256 一次返回一个控制步的动作。动作队列空时才预测新的动作块。
    def select_action(
        # 中文 L0257 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0258 延续上面的语句，batch: dict[str, Tensor],。本段处理队列空时生成动作块，每控制步返回一个动作。
        batch: dict[str, Tensor],
        # 中文 L0259 后续参数必须按名字传入。
        *,
        # 中文 L0260 为“noise”（采样起点噪声）提供这一项的值，注意它所在的参数表或外层表达式。
        noise: Tensor | None = None,
        # 中文 L0261 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **kwargs: Unpack[RTCActionSelectKwargs],
    # 中文 L0262 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        # 中文 L0270 检查逐步接口支持的模式。需要启用 RTC 时要走整块接口。
        assert not self._rtc_enabled(), (
            # 中文 L0271 字符串字面量，供上方的调用、容器或错误消息使用。
            "RTC is not supported for select_action, use it with predict_action_chunk"
        # 中文 L0272 闭合上方尚未结束的括号。
        )

        # 中文 L0274 调用 self.eval，切换 PyTorch 模块到评测模式。
        self.eval()
        # 中文 L0275 必要时先转换输入 batch 的兼容格式。
        batch = self._prepare_batch(batch)
        # 中文 L0276 更新配置中的观测队列，排除用于出队的 ACTION 队列。
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # 中文 L0278 检查动作队列是否见底。队列还有动作时直接跳到第 285 行。
        if self._check_get_actions_condition():
            # 中文 L0279 用当前观测执行一次完整模型采样，得到动作块。
            actions = self._get_action_chunk(batch, noise=noise)

            # 中文 L0281 注意这里注释中的时间轴名称。模型首先产生 chunk_size 个位置，实际截取的数量由下一条语句决定。
            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # 中文 L0282 队列按时间逐项入队，需要把 [B,T,A] 调整为 [T,B,A]。
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            # 中文 L0283 交换批次轴和时间轴，并截取前 n_action_steps 个时间位置。每个入队元素是 [B,A]。
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        # 中文 L0285 从队首取出并移除一个动作。原评测返回 [1,7]，批次轴仍保留。
        return self._queues[ACTION].popleft()

    # 中文 L0287 定义方法 _check_get_actions_condition。
    def _check_get_actions_condition(self) -> bool:
        # 中文 L0288 用队列长度为零作为重新生成动作的条件。
        return len(self._queues[ACTION]) == 0

    # 中文 L0290 定义方法 _rtc_enabled。
    def _rtc_enabled(self) -> bool:
        # 中文 L0291 组合多个布尔判断，按 Python 的短路规则求值作为当前调用的返回值。
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # 中文 L0293 定义方法 forward。作用是计算训练损失。
    def forward(
        # 中文 L0294 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0295 延续上面的语句，batch: dict[str, Tensor],。本段处理计算训练损失。
        batch: dict[str, Tensor],
        # 中文 L0296 为“noise”（采样起点噪声）提供这一项的值，注意它所在的参数表或外层表达式。
        noise: Tensor | None = None,
        # 中文 L0297 为“time”提供这一项的值，注意它所在的参数表或外层表达式。
        time: Tensor | None = None,
        # 中文 L0298 为“reduction”提供这一项的值，注意它所在的参数表或外层表达式。
        reduction: str = "mean",
    # 中文 L0299 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> tuple[Tensor, dict[str, float]]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # 中文 L0310 条件 self.config.adapt_to_pi_aloha 成立时进入下方分支。
        if self.config.adapt_to_pi_aloha:
            # 中文 L0311 调用 self._pi_aloha_decode_state，用它的结果给 “batch[OBS_STATE]” 赋值。
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            # 中文 L0312 调用 self._pi_aloha_encode_actions_inv，用它的结果给 “batch[ACTION]” 赋值。
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        # 中文 L0314 调用 self.prepare_images，用它的结果给 “(images, img_masks)” 赋值。
        images, img_masks = self.prepare_images(batch)
        # 中文 L0315 调用 self.prepare_state，用它的结果给 “state”（机器人状态或当前步骤的状态） 赋值。
        state = self.prepare_state(batch)
        # 中文 L0316 用方括号按键或索引读取 batch给 “lang_tokens”（任务文字的 token ID） 赋值。
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        # 中文 L0317 用方括号按键或索引读取 batch给 “lang_masks”（文字 token 的有效位） 赋值。
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        # 中文 L0318 调用 self.prepare_action，用它的结果给 “actions”（动作批或动作序列） 赋值。
        actions = self.prepare_action(batch)
        # 中文 L0319 调用 batch.get，用它的结果给 “actions_is_pad” 赋值。
        actions_is_pad = batch.get("action_is_pad")
        # 中文 L0320 声明“loss_dict”的类型为 dict[str, float]，创建字典，用键查找对应的值作为初始值。
        loss_dict: dict[str, float] = {}
        # 中文 L0321 调用 self.model.forward，用它的结果给 “losses” 赋值。
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        # 中文 L0322 读取属性 self.config.action_feature给 “action_feature” 赋值。
        action_feature = self.config.action_feature
        # 中文 L0323 条件 action_feature is None 成立时进入下方分支。
        if action_feature is None:
            # 中文 L0324 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("SmolVLA requires an action output feature.")
        # 中文 L0325 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “losses” 赋值。
        losses = losses[:, :, : action_feature.shape[0]]
        # 中文 L0326 调用 item，用它的结果给 “loss_dict['losses_after_forward']” 赋值。
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        # 中文 L0328 条件 actions_is_pad is not None 成立时进入下方分支。
        if actions_is_pad is not None:
            # 中文 L0329 计算此行的右侧表达式给 “in_episode_bound” 赋值。
            in_episode_bound = ~actions_is_pad
            # 中文 L0330 计算右侧表达式 losses * in_episode_bound.unsqueeze(-1)给 “losses” 赋值。
            losses = losses * in_episode_bound.unsqueeze(-1)
            # 中文 L0331 调用 item，用它的结果给 “loss_dict['losses_after_in_ep_bound']” 赋值。
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        # 中文 L0334 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “losses” 赋值。
        losses = losses[:, :, : self.config.max_action_dim]
        # 中文 L0335 调用 item，用它的结果给 “loss_dict['losses_after_rm_padding']” 赋值。
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        # 中文 L0337 条件 reduction == "none" 成立时进入下方分支。
        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            # 中文 L0339 条件 actions_is_pad is None 成立时进入下方分支。
            if actions_is_pad is None:
                # 中文 L0340 调用 losses.mean，用它的结果给 “per_sample_loss” 赋值。
                per_sample_loss = losses.mean(dim=(1, 2))
            # 中文 L0341 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0342 调用 clamp_min，用它的结果给 “num_valid” 赋值。
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                # 中文 L0343 计算右侧表达式 losses.sum(dim=(1, 2)) / num_valid给 “per_sample_loss” 赋值。
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            # 中文 L0344 调用 item，用它的结果给 “loss_dict['loss']” 赋值。
            loss_dict["loss"] = per_sample_loss.mean().item()
            # 中文 L0345 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
            return per_sample_loss, loss_dict
        # 中文 L0346 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            # 中文 L0348 条件 actions_is_pad is None 成立时进入下方分支。
            if actions_is_pad is None:
                # 中文 L0349 调用 losses.mean，用它的结果给 “loss” 赋值。
                loss = losses.mean()
            # 中文 L0350 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0351 调用 clamp_min，用它的结果给 “num_valid” 赋值。
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                # 中文 L0352 计算右侧表达式 losses.sum() / num_valid给 “loss” 赋值。
                loss = losses.sum() / num_valid
            # 中文 L0353 调用 loss.item，用它的结果给 “loss_dict['loss']” 赋值。
            loss_dict["loss"] = loss.item()
            # 中文 L0354 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
            return loss, loss_dict

    # 中文 L0356 定义方法 validate_chunk_input_features。
    def validate_chunk_input_features(self, features: tuple[FeatureSpec, ...]) -> None:
        """Preserve prepare_images' camera subset, resizing and padding semantics.

        The deployment explicitly selects physical cameras. Missing visual slots
        remain missing; prepare_images applies the checkpoint's empty-camera limit
        and masks exactly as in local inference. Never synthesize unmasked cameras.
        """
        # 中文 L0363 组合多个布尔判断，按 Python 的短路规则求值给 “expected” 赋值。
        expected = self.config.input_features or {}
        # 中文 L0364 用字典推导式从可迭代对象逐项取值并构造结果给 “supplied” 赋值。
        supplied = {feature.name: feature for feature in features}
        # 中文 L0365 用集合推导式从可迭代对象逐项取值并构造结果给 “required” 赋值。
        required = {name for name, feature in expected.items() if feature.type != FeatureType.VISUAL}
        # 中文 L0366 条件 not required <= supplied.keys() or not supplied.keys() <= expected.keys() 成立时进入下方分支。
        if not required <= supplied.keys() or not supplied.keys() <= expected.keys():
            # 中文 L0367 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("SmolVLA requires all nonvisual inputs and only known camera features.")
        # 中文 L0368 条件 not any(expected[name].type == FeatureType.VISUAL for name in supplied) 成立时进入下方分支。
        if not any(expected[name].type == FeatureType.VISUAL for name in supplied):
            # 中文 L0369 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("SmolVLA requires at least one camera.")
        # 中文 L0370 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for name, feature in supplied.items():
            # 中文 L0371 用方括号按键或索引读取 expected给 “policy_feature” 赋值。
            policy_feature = expected[name]
            # 中文 L0372 条件 policy_feature.type == FeatureType.VISUAL 成立时进入下方分支。
            if policy_feature.type == FeatureType.VISUAL:
                # 中文 L0373 读取属性 policy_feature.shape给 “shape”（各轴的长度） 赋值。
                shape = policy_feature.shape
                # 中文 L0374 条件 len(shape) != 3 or shape[0] != 3 or feature.kind != "rgb" 成立时进入下方分支。
                if len(shape) != 3 or shape[0] != 3 or feature.kind != "rgb":
                    # 中文 L0375 抛出 ValueError，中断当前路径并交给上层的异常处理。
                    raise ValueError(f"SmolVLA requires RGB camera inputs: {name}.")
                # 中文 L0376 条件 self.config.resize_imgs_with_padding is None and feature.shape != ( 成立时进入下方分支。
                if self.config.resize_imgs_with_padding is None and feature.shape != (
                    # 中文 L0377 延续上面的语句，shape[1],。
                    shape[1],
                    # 中文 L0378 延续上面的语句，shape[2],。
                    shape[2],
                    # 中文 L0379 延续上面的语句，shape[0],。
                    shape[0],
                # 中文 L0380 闭合上方尚未结束的括号。
                ):
                    # 中文 L0381 抛出 ValueError，中断当前路径并交给上层的异常处理。
                    raise ValueError(f"RGB feature shape differs without policy resizing: {name}.")
            # 中文 L0382 前面的分支均未命中时，再判断 feature.kind != "tensor" or feature.shape != tuple(policy_feature.shape)。
            elif feature.kind != "tensor" or feature.shape != tuple(policy_feature.shape):
                # 中文 L0383 抛出 ValueError，中断当前路径并交给上层的异常处理。
                raise ValueError(f"Tensor feature shape differs from checkpoint: {name}.")

    # 中文 L0385 将前处理给出的相机张量整理为视觉编码器输入，并为每路相机保留有效位。
    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        # 中文 L0389 创建列表，按给定顺序保留元素给 “images” 赋值。
        images = []
        # 中文 L0390 创建列表，按给定顺序保留元素给 “img_masks”（相机输入有效位） 赋值。
        img_masks = []
        # 中文 L0391 按配置列出的相机键顺序挑出 batch 中已存在的键。
        present_img_keys = [key for key in self.config.image_features if key in batch]
        # 中文 L0392 用列表推导式从可迭代对象逐项取值并构造结果给 “missing_img_keys” 赋值。
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        # 中文 L0394 条件 len(present_img_keys) == 0 成立时进入下方分支。
        if len(present_img_keys) == 0:
            # 中文 L0395 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0396 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            # 中文 L0397 闭合上方尚未结束的括号。
            )
        # Preprocess image features present in the batch
        # 中文 L0399 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for key in present_img_keys:
            # 中文 L0400 五维图像包含观测时间轴，选最后一帧。四维 [B,C,H,W] 图像直接使用。
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            # 中文 L0401 只有 uint8 像素需要这里的除以 255。本次仿真观测此前已转换为 float32。
            if img.dtype == torch.uint8:
                # 中文 L0402 计算右侧表达式 img.to(torch.float32) / 255.0给 “img” 赋值。
                img = img.to(torch.float32) / 255.0
            # 中文 L0403 条件 self.config.resize_imgs_with_padding is not None 成立时进入下方分支。
            if self.config.resize_imgs_with_padding is not None:
                # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
                # 中文 L0405 按原始宽高比缩放、补齐到目标尺寸。本次每张变成 [1,3,512,512]。
                img = resize_with_pad(
                    # 中文 L0406 “img”是当前括号中的一项，逗号将它与下一项分开。
                    img,
                    # 中文 L0407 延续上面的语句，self.config.resize_imgs_with_padding[1],。本段处理整理相机输入的形状、尺度和有效位。
                    self.config.resize_imgs_with_padding[1],
                    # 中文 L0408 延续上面的语句，self.config.resize_imgs_with_padding[0],。本段处理整理相机输入的形状、尺度和有效位。
                    self.config.resize_imgs_with_padding[0],
                    # 中文 L0409 为“pad_value”提供这一项的值，注意它所在的参数表或外层表达式。
                    pad_value=0,
                # 中文 L0410 闭合上方尚未结束的括号。
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            # 中文 L0413 逐像素施加线性变换，把 [0,1] 的值映射到 [-1,1]。
            img = img * 2.0 - 1.0

            # 中文 L0415 用方括号按键或索引读取 img.shape给 “bsize”（批大小） 赋值。
            bsize = img.shape[0]
            # 中文 L0416 从图像取得当前设备，后面创建的掩码要在同一设备。
            device = img.device
            # 中文 L0417 条件 f"{key}_padding_mask" in batch 成立时进入下方分支。
            if f"{key}_padding_mask" in batch:
                # 中文 L0418 调用 bool，用它的结果给 “mask” 赋值。
                mask = batch[f"{key}_padding_mask"].bool()
            # 中文 L0419 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
            else:
                # 中文 L0420 给没有显式掩码的相机创建 [B] 的布尔有效位，与当前图像同设备。
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            # 中文 L0421 调用 images.append，在列表末尾追加一个元素。
            images.append(img)
            # 中文 L0422 调用 img_masks.append，在列表末尾追加一个元素。
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        # 中文 L0426 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for num_empty_cameras in range(len(missing_img_keys)):
            # 中文 L0427 条件 num_empty_cameras >= self.config.empty_cameras 成立时进入下方分支。
            if num_empty_cameras >= self.config.empty_cameras:
                # 中文 L0428 结束所在的最近一层循环。
                break
            # 中文 L0429 对配置允许的空相机创建全 -1 图像作为占位。
            img = torch.ones_like(img) * -1
            # 中文 L0430 创建全假掩码，提示注意力忽略这个占位相机。
            mask = torch.zeros_like(mask)
            # 中文 L0431 调用 images.append，在列表末尾追加一个元素。
            images.append(img)
            # 中文 L0432 调用 img_masks.append，在列表末尾追加一个元素。
            img_masks.append(mask)
        # 中文 L0433 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return images, img_masks

    # 中文 L0435 定义方法 _pi_aloha_decode_state。
    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        # 中文 L0437 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for motor_idx in [1, 2, 8, 9]:
            # 中文 L0438 用增量赋值更新 state[:, motor_idx]，运算逻辑见左侧运算符。
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        # 中文 L0440 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for motor_idx in [6, 13]:
            # 中文 L0441 调用 aloha_gripper_to_angular，用它的结果给 “state[:, motor_idx]” 赋值。
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        # 中文 L0442 引用已有的值 state作为当前调用的返回值。
        return state

    # 中文 L0444 定义方法 _pi_aloha_encode_actions。
    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        # 中文 L0446 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for motor_idx in [1, 2, 8, 9]:
            # 中文 L0447 用增量赋值更新 actions[:, :, motor_idx]，运算逻辑见左侧运算符。
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        # 中文 L0449 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for motor_idx in [6, 13]:
            # 中文 L0450 调用 aloha_gripper_from_angular，用它的结果给 “actions[:, :, motor_idx]” 赋值。
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        # 中文 L0451 引用已有的值 actions作为当前调用的返回值。
        return actions

    # 中文 L0453 定义方法 _pi_aloha_encode_actions_inv。
    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        # 中文 L0455 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for motor_idx in [1, 2, 8, 9]:
            # 中文 L0456 用增量赋值更新 actions[:, :, motor_idx]，运算逻辑见左侧运算符。
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        # 中文 L0458 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for motor_idx in [6, 13]:
            # 中文 L0459 调用 aloha_gripper_from_angular_inv，用它的结果给 “actions[:, :, motor_idx]” 赋值。
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        # 中文 L0460 引用已有的值 actions作为当前调用的返回值。
        return actions

    # 中文 L0462 定义方法 prepare_state。作用是提取最新机器人状态并补齐到内部宽度。
    def prepare_state(self, batch):
        # 中文 L0463 原文档概述本段功能，提取最新机器人状态并补齐到内部宽度。
        """Pad state"""
        # 中文 L0464 有观测时间轴时取最后一条状态，没有时间轴时直接取状态张量。
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        # 中文 L0465 沿最后一轴补零到 max_state_dim，实际评测此处由 [1,8] 变为 [1,32]。
        state = pad_vector(state, self.config.max_state_dim)
        # 中文 L0466 引用已有的值 state作为当前调用的返回值。
        return state

    # 中文 L0468 定义方法 prepare_action。
    def prepare_action(self, batch):
        # 中文 L0469 原文档中的说明或续行。
        """Pad action"""
        # 中文 L0470 调用 pad_vector，用它的结果给 “actions”（动作批或动作序列） 赋值。
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        # 中文 L0471 引用已有的值 actions作为当前调用的返回值。
        return actions

    # 中文 L0473 定义方法 _get_default_peft_targets。
    def _get_default_peft_targets(self) -> dict[str, Any]:
        # 中文 L0474 原文档中的说明或续行。
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        # 中文 L0475 使用字面值 'state_proj|action_in_proj|action_out_proj|action_time_mlp_in|ac…给 “common_projections” 赋值。
        common_projections = (
            # 中文 L0476 字符串字面量，供上方的调用、容器或错误消息使用。
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        # 中文 L0477 闭合上方尚未结束的括号。
        )
        # 中文 L0478 计算此行的右侧表达式给 “target_modules” 赋值。
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        # 中文 L0479 创建字典，用键查找对应的值作为当前调用的返回值。
        return {
            # 中文 L0480 字典项“target_modules”，冒号右边给出它的值。
            "target_modules": target_modules,
            # 中文 L0481 字典项“modules_to_save”，冒号右边给出它的值。
            "modules_to_save": [],
        # 中文 L0482 闭合上方尚未结束的括号。
        }

    # 中文 L0484 定义方法 _validate_peft_config。
    def _validate_peft_config(self, peft_config) -> None:
        # 中文 L0485 原文档中的说明或续行。
        """Validate PEFT configuration for SmolVLA."""
        # 中文 L0486 调用 _validate_peft_config 执行操作。
        super()._validate_peft_config(peft_config)
        # 中文 L0487 条件 not self.config.load_vlm_weights 成立时进入下方分支。
        if not self.config.load_vlm_weights:
            # 中文 L0488 导入模块 logging。
            import logging

            # 中文 L0490 调用 logging.warning 执行操作。
            logging.warning(
                # 中文 L0491 字符串字面量，供上方的调用、容器或错误消息使用。
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                # 中文 L0492 字符串字面量，供上方的调用、容器或错误消息使用。
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            # 中文 L0493 闭合上方尚未结束的括号。
            )


# 中文 L0496 定义函数 pad_tensor。
def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    # 中文 L0508 对 tensor.shape 做切片，右端索引不包含在结果中给 “(b, d)” 赋值。
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    # 中文 L0511 调用 torch.full，用它的结果给 “padded_tensor” 赋值。
    padded_tensor = torch.full(
        # 中文 L0512 延续上面的语句，(b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device。
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    # 中文 L0513 闭合上方尚未结束的括号。
    )
    # 中文 L0514 引用已有的值 tensor给 “padded_tensor[:, :d]” 赋值。
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    # 中文 L0516 引用已有的值 padded_tensor作为当前调用的返回值。
    return padded_tensor


# 中文 L0519 定义类 VLAFlowMatching，括号列出基类 nn.Module。作用是根据视觉、语言和状态条件生成连续动作序列。
class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    # 中文 L0545 构造执行动作采样的 PyTorch 模块，初始化视觉语言骨干、专家和投影层。
    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        # 中文 L0546 调用 __init__ 执行操作。
        super().__init__()
        # 中文 L0547 引用已有的值 config给 “self.config” 赋值。
        self.config = config

        # 中文 L0549 组合视觉语言模型和动作专家，模型构造参数由同一个配置实例传入。
        self.vlm_with_expert = SmolVLMWithExpertModel(
            # 中文 L0550 为“model_id”提供这一项的值，注意它所在的参数表或外层表达式。
            model_id=self.config.vlm_model_name,
            # 中文 L0551 为“freeze_vision_encoder”提供这一项的值，注意它所在的参数表或外层表达式。
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            # 中文 L0552 为“train_expert_only”提供这一项的值，注意它所在的参数表或外层表达式。
            train_expert_only=self.config.train_expert_only,
            # 中文 L0553 为“load_vlm_weights”提供这一项的值，注意它所在的参数表或外层表达式。
            load_vlm_weights=self.config.load_vlm_weights,
            # 中文 L0554 为“attention_mode”提供这一项的值，注意它所在的参数表或外层表达式。
            attention_mode=self.config.attention_mode,
            # 中文 L0555 为“num_expert_layers”提供这一项的值，注意它所在的参数表或外层表达式。
            num_expert_layers=self.config.num_expert_layers,
            # 中文 L0556 为“num_vlm_layers”提供这一项的值，注意它所在的参数表或外层表达式。
            num_vlm_layers=self.config.num_vlm_layers,
            # 中文 L0557 为“self_attn_every_n_layers”提供这一项的值，注意它所在的参数表或外层表达式。
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            # 中文 L0558 为“expert_width_multiplier”提供这一项的值，注意它所在的参数表或外层表达式。
            expert_width_multiplier=self.config.expert_width_multiplier,
            # 中文 L0559 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
            device=self.config.device if self.config.device is not None else "auto",
        # 中文 L0560 闭合上方尚未结束的括号。
        )
        # 中文 L0561 定义把机器人状态投影到语言骨干隐藏维度的线性层。
        self.state_proj = nn.Linear(
            # 中文 L0562 延续上面的语句，self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size。本段处理建立骨干、动作专家和输入输出投影。
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        # 中文 L0563 闭合上方尚未结束的括号。
        )
        # 中文 L0564 将内部动作向量从 max_action_dim 投影到专家的隐藏空间。
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        # 中文 L0565 将专家输出从隐藏空间投影回内部动作向量的宽度。
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        # 中文 L0567 把动作嵌入与时间嵌入拼接后的双倍宽度投影回专家隐藏宽度。
        self.action_time_mlp_in = nn.Linear(
            # 中文 L0568 延续上面的语句，self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size。本段处理建立骨干、动作专家和输入输出投影。
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        # 中文 L0569 闭合上方尚未结束的括号。
        )
        # 中文 L0570 调用 nn.Linear，用它的结果给 “self.action_time_mlp_out” 赋值。
        self.action_time_mlp_out = nn.Linear(
            # 中文 L0571 延续上面的语句，self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size。本段处理建立骨干、动作专家和输入输出投影。
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        # 中文 L0572 闭合上方尚未结束的括号。
        )

        # 中文 L0574 调用 self.set_requires_grad 执行操作。
        self.set_requires_grad()
        # 中文 L0575 读取属性 self.vlm_with_expert.processor.tokenizer.fake_image_token_id给 “self.fake_image_token” 赋值。
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        # 中文 L0576 读取属性 self.vlm_with_expert.processor.tokenizer.global_image_token_id给 “self.global_image_token” 赋值。
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        # 中文 L0577 调用 torch.tensor，用它的结果给 “self.global_image_start_token” 赋值。
        self.global_image_start_token = torch.tensor(
            # 中文 L0578 延续上面的语句，[self.fake_image_token, self.global_image_token], dtype=torch.long。本段处理建立骨干、动作专家和输入输出投影。
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        # 中文 L0579 闭合上方尚未结束的括号。
        )

        # 中文 L0581 读取属性 self.config.add_image_special_tokens给 “self.add_image_special_tokens” 赋值。
        self.add_image_special_tokens = self.config.add_image_special_tokens
        # 中文 L0582 调用 torch.tensor，用它的结果给 “self.image_end_token” 赋值。
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        # 中文 L0583 读取属性 self.config.prefix_length给 “self.prefix_length” 赋值。
        self.prefix_length = self.config.prefix_length
        # 中文 L0584 引用已有的值 rtc_processor给 “self.rtc_processor” 赋值。
        self.rtc_processor = rtc_processor

        # Compile model if requested
        # 中文 L0587 条件 config.compile_model 成立时进入下方分支。
        if config.compile_model:
            # 中文 L0588 调用 torch.compile，用它的结果给 “self.sample_actions” 赋值。
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)  # type: ignore[method-assign]
            # 中文 L0589 调用 torch.compile，用它的结果给 “self.forward” 赋值。
            self.forward = torch.compile(self.forward, mode=config.compile_mode)  # type: ignore[method-assign]

    # 中文 L0591 定义方法 _rtc_enabled。
    def _rtc_enabled(self):
        # 中文 L0592 组合多个布尔判断，按 Python 的短路规则求值作为当前调用的返回值。
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # 中文 L0594 定义方法 set_requires_grad。
    def set_requires_grad(self):
        # 中文 L0595 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for params in self.state_proj.parameters():
            # 中文 L0596 读取属性 self.config.train_state_proj给 “params.requires_grad” 赋值。
            params.requires_grad = self.config.train_state_proj

    # 中文 L0598 定义方法 sample_noise。
    def sample_noise(self, shape, device):
        # 中文 L0599 调用 sample_noise，用它的结果作为当前调用的返回值。
        return sample_noise(shape, device)

    # 中文 L0601 定义方法 sample_time。
    def sample_time(self, bsize, device):
        # 中文 L0602 调用 sample_time_beta，用它的结果作为当前调用的返回值。
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    # 中文 L0604 构造观测前缀，按图像、语言、状态的顺序合并嵌入及其掩码。
    def embed_prefix(
        # 中文 L0605 延续上面的语句，self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None。本段处理把图像、文字与状态组织成观测前缀。
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    # 中文 L0606 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        # 中文 L0610 创建列表，按给定顺序保留元素给 “embs” 赋值。
        embs = []
        # 中文 L0611 创建列表，按给定顺序保留元素给 “pad_masks”（各 token 是否有效的掩码） 赋值。
        pad_masks = []
        # 中文 L0612 创建列表，按给定顺序保留元素给 “att_masks”（注意力块边界标记） 赋值。
        att_masks = []
        # 中文 L0613 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for _img_idx, (
            # 中文 L0614 “img”是当前括号中的一项，逗号将它与下一项分开。
            img,
            # 中文 L0615 “img_mask”是当前括号中的一项，逗号将它与下一项分开。
            img_mask,
        # 中文 L0616 zip 成对遍历图像和掩码。enumerate 同时提供当前相机序号。
        ) in enumerate(zip(images, img_masks, strict=False)):
            # 中文 L0617 条件 self.add_image_special_tokens 成立时进入下方分支。
            if self.add_image_special_tokens:
                # 中文 L0618 调用 expand，用它的结果给 “image_start_token” 赋值。
                image_start_token = (
                    # 中文 L0619 延续上面的语句，self.vlm_with_expert.embed_language_tokens(。本段处理把图像、文字与状态组织成观测前缀。
                    self.vlm_with_expert.embed_language_tokens(
                        # 中文 L0620 延续上面的语句，self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)。本段处理把图像、文字与状态组织成观测前缀。
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    # 中文 L0621 闭合上方尚未结束的括号。
                    )
                    # 中文 L0622 延续上面的语句，.unsqueeze(0)。本段处理把图像、文字与状态组织成观测前缀。
                    .unsqueeze(0)
                    # 中文 L0623 延续上面的语句，.expand(img.shape[0], -1, -1)。本段处理把图像、文字与状态组织成观测前缀。
                    .expand(img.shape[0], -1, -1)
                # 中文 L0624 闭合上方尚未结束的括号。
                )
                # 中文 L0625 调用 torch.ones_like，用它的结果给 “image_start_mask” 赋值。
                image_start_mask = torch.ones_like(
                    # 中文 L0626 延续上面的语句，image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device。本段处理把图像、文字与状态组织成观测前缀。
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                # 中文 L0627 闭合上方尚未结束的括号。
                )
                # 中文 L0628 用增量赋值更新 att_masks，运算逻辑见左侧运算符。
                att_masks += [0] * (image_start_mask.shape[-1])
                # 中文 L0629 调用 embs.append，在列表末尾追加一个元素。
                embs.append(image_start_token)
                # 中文 L0630 调用 pad_masks.append，在列表末尾追加一个元素。
                pad_masks.append(image_start_mask)

            # 中文 L0632 执行视觉编码与连接器，获得 [B,图像 token 数,隐藏维度] 的嵌入。
            img_emb = self.vlm_with_expert.embed_image(img)
            # 中文 L0633 这里把变量赋值给它自身，张量的值和形状都没有发生变化。
            img_emb = img_emb

            # Normalize image embeddings
            # 中文 L0636 用方括号按键或索引读取 img_emb.shape给 “img_emb_dim” 赋值。
            img_emb_dim = img_emb.shape[-1]
            # 中文 L0637 用隐藏宽度的平方根缩放视觉嵌入，标量用同一 dtype 和 device 创建。
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            # 中文 L0639 对 img_emb.shape 做切片，右端索引不包含在结果中给 “(bsize, num_img_embs)” 赋值。
            bsize, num_img_embs = img_emb.shape[:2]
            # 中文 L0640 从每路相机每个样本一个有效位，扩展成该相机每个 token 一个有效位。
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            # 中文 L0642 调用 embs.append，在列表末尾追加一个元素。
            embs.append(img_emb)
            # 中文 L0643 调用 pad_masks.append，在列表末尾追加一个元素。
            pad_masks.append(img_mask)

            # 中文 L0645 把视觉 token 标成同一注意力块的一部分。这个列表记录块边界，另有掩码记录 token 是否有效。
            att_masks += [0] * (num_img_embs)
            # 中文 L0646 条件 self.add_image_special_tokens 成立时进入下方分支。
            if self.add_image_special_tokens:
                # 中文 L0647 调用 expand，用它的结果给 “image_end_token” 赋值。
                image_end_token = (
                    # 中文 L0648 延续上面的语句，self.vlm_with_expert.embed_language_tokens(。本段处理把图像、文字与状态组织成观测前缀。
                    self.vlm_with_expert.embed_language_tokens(
                        # 中文 L0649 延续上面的语句，self.image_end_token.to(device=self.vlm_with_expert.vlm.device)。本段处理把图像、文字与状态组织成观测前缀。
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    # 中文 L0650 闭合上方尚未结束的括号。
                    )
                    # 中文 L0651 延续上面的语句，.unsqueeze(0)。本段处理把图像、文字与状态组织成观测前缀。
                    .unsqueeze(0)
                    # 中文 L0652 延续上面的语句，.expand(img.shape[0], -1, -1)。本段处理把图像、文字与状态组织成观测前缀。
                    .expand(img.shape[0], -1, -1)
                # 中文 L0653 闭合上方尚未结束的括号。
                )
                # 中文 L0654 调用 torch.ones_like，用它的结果给 “image_end_mask” 赋值。
                image_end_mask = torch.ones_like(
                    # 中文 L0655 延续上面的语句，image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device。本段处理把图像、文字与状态组织成观测前缀。
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                # 中文 L0656 闭合上方尚未结束的括号。
                )
                # 中文 L0657 调用 embs.append，在列表末尾追加一个元素。
                embs.append(image_end_token)
                # 中文 L0658 调用 pad_masks.append，在列表末尾追加一个元素。
                pad_masks.append(image_end_mask)
                # 中文 L0659 用增量赋值更新 att_masks，运算逻辑见左侧运算符。
                att_masks += [0] * (image_end_mask.shape[1])
        # 中文 L0660 将整数语言 token ID 查表转成浮点嵌入，新增最后一条隐藏维度轴。
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        # 中文 L0662 用方括号按键或索引读取 lang_emb.shape给 “lang_emb_dim” 赋值。
        lang_emb_dim = lang_emb.shape[-1]
        # 中文 L0663 用语言隐藏向量宽度的平方根缩放嵌入。
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        # 中文 L0665 调用 embs.append，在列表末尾追加一个元素。
        embs.append(lang_emb)
        # 中文 L0666 调用 pad_masks.append，在列表末尾追加一个元素。
        pad_masks.append(lang_masks)

        # 中文 L0668 用方括号按键或索引读取 lang_emb.shape给 “num_lang_embs” 赋值。
        num_lang_embs = lang_emb.shape[1]
        # 中文 L0669 用增量赋值更新 att_masks，运算逻辑见左侧运算符。
        att_masks += [0] * num_lang_embs

        # 中文 L0671 把 [B,状态宽度] 送入线性层，转换为语言模型的隐藏宽度。
        state_emb = self.state_proj(state)
        # 中文 L0672 [:, None, :] 插入长度为 1 的序列轴，使每个状态可作为一个 token 拼接。
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        # 中文 L0673 调用 embs.append，在列表末尾追加一个元素。
        embs.append(state_emb)
        # 中文 L0674 用方括号按键或索引读取 state_emb.shape给 “bsize”（批大小） 赋值。
        bsize = state_emb.shape[0]
        # 中文 L0675 读取属性 state_emb.device给 “device”（运算设备） 赋值。
        device = state_emb.device

        # 中文 L0677 用方括号按键或索引读取 state_emb.shape给 “states_seq_len” 赋值。
        states_seq_len = state_emb.shape[1]
        # 中文 L0678 创建 [B,状态 token 数] 的布尔掩码，状态位置全部有效。
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        # 中文 L0679 调用 pad_masks.append，在列表末尾追加一个元素。
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        # 中文 L0682 在状态 token 处开始新的注意力块。
        att_masks += [1] * (states_seq_len)
        # 中文 L0683 沿序列轴把视觉 token、语言 token 和状态 token 拼成前缀。
        embs = torch.cat(embs, dim=1)
        # 中文 L0684 沿同一条序列轴拼接有效位，以便和前缀嵌入对齐。
        pad_masks = torch.cat(pad_masks, dim=1)
        # 中文 L0685 把 Python 列表变成与有效位掩码在同一设备的布尔 Tensor。
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        # 中文 L0686 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “att_masks”（注意力块边界标记） 赋值。
        att_masks = att_masks[None, :]

        # 中文 L0688 用方括号按键或索引读取 pad_masks.shape给 “seq_len” 赋值。
        seq_len = pad_masks.shape[1]
        # 中文 L0689 条件 seq_len < self.prefix_length 成立时进入下方分支。
        if seq_len < self.prefix_length:
            # 中文 L0690 调用 pad_tensor，用它的结果给 “embs” 赋值。
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            # 中文 L0691 调用 pad_tensor，用它的结果给 “pad_masks”（各 token 是否有效的掩码） 赋值。
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            # 中文 L0692 调用 pad_tensor，用它的结果给 “att_masks”（注意力块边界标记） 赋值。
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        # 中文 L0694 将注意力分块标记广播到整个批次。
        att_masks = att_masks.expand(bsize, -1)

        # 中文 L0696 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return embs, pad_masks, att_masks

    # 中文 L0698 将当前动作估计与采样时间结合成后缀输入。
    def embed_suffix(self, noisy_actions, timestep):
        # 中文 L0699 原文档概述本段功能，把当前动作估计和采样时间组织成动作后缀。
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        # 中文 L0700 创建列表，按给定顺序保留元素给 “embs” 赋值。
        embs = []
        # 中文 L0701 创建列表，按给定顺序保留元素给 “pad_masks”（各 token 是否有效的掩码） 赋值。
        pad_masks = []
        # 中文 L0702 创建列表，按给定顺序保留元素给 “att_masks”（注意力块边界标记） 赋值。
        att_masks = []

        # Fuse timestep + action information using an MLP
        # 中文 L0705 把每个含噪动作向量投影为专家嵌入，保持 [B,T,…] 两个前导轴不变。
        action_emb = self.action_in_proj(noisy_actions)
        # 中文 L0706 读取属性 action_emb.device给 “device”（运算设备） 赋值。
        device = action_emb.device
        # 中文 L0707 用方括号按键或索引读取 action_emb.shape给 “bsize”（批大小） 赋值。
        bsize = action_emb.shape[0]
        # 中文 L0708 读取属性 action_emb.dtype给 “dtype”（元素类型） 赋值。
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 中文 L0710 对当前采样时间创建正弦和余弦时间嵌入。
        time_emb = create_sinusoidal_pos_embedding(
            # 中文 L0711 “timestep”是当前括号中的一项，逗号将它与下一项分开。
            timestep,
            # 中文 L0712 延续上面的语句，self.vlm_with_expert.expert_hidden_size,。本段处理把当前动作估计和采样时间组织成动作后缀。
            self.vlm_with_expert.expert_hidden_size,
            # 中文 L0713 延续上面的语句，self.config.min_period,。本段处理把当前动作估计和采样时间组织成动作后缀。
            self.config.min_period,
            # 中文 L0714 延续上面的语句，self.config.max_period,。本段处理把当前动作估计和采样时间组织成动作后缀。
            self.config.max_period,
            # 中文 L0715 为“device”（运算设备）提供这一项的值，注意它所在的参数表或外层表达式。
            device=device,
        # 中文 L0716 闭合上方尚未结束的括号。
        )
        # 中文 L0717 使时间嵌入与动作嵌入数据类型一致。
        time_emb = time_emb.type(dtype=dtype)

        # 中文 L0719 插入时间位置轴，并把同一采样时间嵌入提供给动作块中的每个位置。
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        # 中文 L0720 沿最后一轴拼接动作和时间嵌入，末维宽度变为专家宽度的两倍。
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        # 中文 L0722 调用 self.action_time_mlp_in，用它的结果给 “action_time_emb” 赋值。
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        # 中文 L0723 使用 SiLU 激活，引入非线性。
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        # 中文 L0724 调用 self.action_time_mlp_out，用它的结果给 “action_time_emb” 赋值。
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        # 中文 L0727 调用 embs.append，在列表末尾追加一个元素。
        embs.append(action_time_emb)

        # 中文 L0729 对 action_time_emb.shape 做切片，右端索引不包含在结果中给 “(bsize, action_time_dim)” 赋值。
        bsize, action_time_dim = action_time_emb.shape[:2]
        # 中文 L0730 动作位置全部有效，构造 [B,T] 的布尔掩码。
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        # 中文 L0731 调用 pad_masks.append，在列表末尾追加一个元素。
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        # 中文 L0734 为动作位置写入注意力分块标记，实际可见性由后续 make_att_2d_masks 计算。
        att_masks += [1] * self.config.chunk_size
        # 中文 L0735 调用 torch.cat，用它的结果给 “embs” 赋值。
        embs = torch.cat(embs, dim=1)
        # 中文 L0736 调用 torch.cat，用它的结果给 “pad_masks”（各 token 是否有效的掩码） 赋值。
        pad_masks = torch.cat(pad_masks, dim=1)
        # 中文 L0737 调用 torch.tensor，用它的结果给 “att_masks”（注意力块边界标记） 赋值。
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        # 中文 L0738 调用 expand，用它的结果给 “att_masks”（注意力块边界标记） 赋值。
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        # 中文 L0739 打包成元组，元素顺序由逗号分隔的位置决定作为当前调用的返回值。
        return embs, pad_masks, att_masks

    # 中文 L0741 这是计算训练损失的 forward 方法。逐步评测使用下方的 sample_actions。
    def forward(
        # 中文 L0742 延续上面的语句，self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None。本段处理构造训练输入并计算流匹配损失。
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    # 中文 L0743 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor:
        # 中文 L0744 原文档概述本段功能，构造训练输入并计算流匹配损失。
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        # 中文 L0745 条件 noise is None 成立时进入下方分支。
        if noise is None:
            # 中文 L0746 调用 self.sample_noise，用它的结果给 “noise”（采样起点噪声） 赋值。
            noise = self.sample_noise(actions.shape, actions.device)

        # 中文 L0748 条件 time is None 成立时进入下方分支。
        if time is None:
            # 中文 L0749 调用 self.sample_time，用它的结果给 “time” 赋值。
            time = self.sample_time(actions.shape[0], actions.device)

        # 中文 L0751 调用 make_flow_matching_inputs，用它的结果给 “(x_t, u_t, _)” 赋值。
        x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)
        # 中文 L0752 调用 self.embed_prefix，用它的结果给 “(prefix_embs, prefix_pad_masks, prefix_att_masks)” 赋值。
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            # 中文 L0753 延续上面的语句，images, img_masks, lang_tokens, lang_masks, state=state。本段处理构造训练输入并计算流匹配损失。
            images, img_masks, lang_tokens, lang_masks, state=state
        # 中文 L0754 闭合上方尚未结束的括号。
        )
        # 中文 L0755 调用 self.embed_suffix，用它的结果给 “(suffix_embs, suffix_pad_masks, suffix_att_masks)” 赋值。
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        # 中文 L0757 调用 torch.cat，用它的结果给 “pad_masks”（各 token 是否有效的掩码） 赋值。
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        # 中文 L0758 调用 torch.cat，用它的结果给 “att_masks”（注意力块边界标记） 赋值。
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # 中文 L0760 调用 make_att_2d_masks，用它的结果给 “att_2d_masks” 赋值。
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # 中文 L0761 计算右侧表达式 torch.cumsum(pad_masks, dim=1) - 1给 “position_ids” 赋值。
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # 中文 L0762 调用 self.vlm_with_expert.forward，用它的结果给 “((_, suffix_out), _)” 赋值。
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            # 中文 L0763 为“attention_mask”提供这一项的值，注意它所在的参数表或外层表达式。
            attention_mask=att_2d_masks,
            # 中文 L0764 为“position_ids”提供这一项的值，注意它所在的参数表或外层表达式。
            position_ids=position_ids,
            # 中文 L0765 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
            past_key_values=None,
            # 中文 L0766 为“inputs_embeds”提供这一项的值，注意它所在的参数表或外层表达式。
            inputs_embeds=[prefix_embs, suffix_embs],
            # 中文 L0767 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
            use_cache=False,
        # 中文 L0768 闭合上方尚未结束的括号。
        )
        # 中文 L0769 按逗号区分的各轴分别切片，保留写成冒号的整轴给 “suffix_out” 赋值。
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        # 中文 L0771 调用 suffix_out.to，用它的结果给 “suffix_out” 赋值。
        suffix_out = suffix_out.to(dtype=torch.float32)
        # 中文 L0772 调用 self.action_out_proj，用它的结果给 “v_t”（当前速度场输出） 赋值。
        v_t = self.action_out_proj(suffix_out)
        # 中文 L0773 调用 F.mse_loss，用它的结果给 “losses” 赋值。
        losses = F.mse_loss(u_t, v_t, reduction="none")
        # 中文 L0774 引用已有的值 losses作为当前调用的返回值。
        return losses

    # 中文 L0776 推理采样入口，一次生成一块动作，在外层动作队列用空后才触发。
    def sample_actions(
        # 中文 L0777 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0778 “images”是当前括号中的一项，逗号将它与下一项分开。
        images,
        # 中文 L0779 “img_masks”（相机输入有效位）是当前括号中的一项，逗号将它与下一项分开。
        img_masks,
        # 中文 L0780 “lang_tokens”（任务文字的 token ID）是当前括号中的一项，逗号将它与下一项分开。
        lang_tokens,
        # 中文 L0781 “lang_masks”（文字 token 的有效位）是当前括号中的一项，逗号将它与下一项分开。
        lang_masks,
        # 中文 L0782 “state”（机器人状态或当前步骤的状态）是当前括号中的一项，逗号将它与下一项分开。
        state,
        # 中文 L0783 为“noise”（采样起点噪声）提供这一项的值，注意它所在的参数表或外层表达式。
        noise=None,
        # 中文 L0784 星号展开可迭代对象或关键字字典。若在函数签名中则收集额外参数。
        **kwargs: Unpack[RTCActionSelectKwargs],
    # 中文 L0785 结束参数表。若箭头后有类型，它标注预期返回值，缩进块是函数体。
    ) -> Tensor:
        # 中文 L0786 原文档概述本段功能，缓存观测条件并从初始噪声迭代生成动作块。
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        # 中文 L0787 取状态批次轴长度。当前单环境 B=1。
        bsize = state.shape[0]
        # 中文 L0788 在与状态相同的设备上生成噪声和时间等张量。当前评测为 CUDA。
        device = state.device

        # 中文 L0790 条件 noise is None 成立时进入下方分支。
        if noise is None:
            # 中文 L0791 形状三个轴依次是环境数、动作块时间长度、内部单步动作宽度，本次为 [1,50,32]。
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            # 中文 L0792 按该形状在目标设备生成 float32 标准正态噪声。
            noise = self.sample_noise(actions_shape, device)
        # 中文 L0793 进入对应的另一条分支，注意用缩进找到配对的判断或异常块。
        else:
            # 中文 L0794 调用者显式提供初始噪声时也将其移到同设备并转成 float32。
            noise = noise.to(device=device, dtype=torch.float32)

        # 中文 L0796 对当前观测做一次前缀嵌入，输出嵌入、有效位和注意力块标记。
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            # 中文 L0797 延续上面的语句，images, img_masks, lang_tokens, lang_masks, state=state。本段处理缓存观测条件并从初始噪声迭代生成动作块。
            images, img_masks, lang_tokens, lang_masks, state=state
        # 中文 L0798 闭合上方尚未结束的括号。
        )
        # 中文 L0799 将一维标记转换成 [B,L前缀,L前缀] 的布尔注意力可见矩阵。
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        # 中文 L0800 对有效位沿序列轴累计计数，得到从零开始的有效 token 位置编号。
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        # 中文 L0802 对观测前缀做一次预计算，取各层的键和值作为后续动作采样的缓存。
        _, past_key_values = self.vlm_with_expert.forward(
            # 中文 L0803 为“attention_mask”提供这一项的值，注意它所在的参数表或外层表达式。
            attention_mask=prefix_att_2d_masks,
            # 中文 L0804 为“position_ids”提供这一项的值，注意它所在的参数表或外层表达式。
            position_ids=prefix_position_ids,
            # 中文 L0805 传入 None 表示传入的这次没有已有缓存，由模型在开启 use_cache 时初始化。
            past_key_values=None,
            # 中文 L0806 两个输入槽位依次是前缀、后缀。这里预处理前缀，后缀为空。
            inputs_embeds=[prefix_embs, None],
            # 中文 L0807 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
            use_cache=self.config.use_cache,
        # 中文 L0808 闭合上方尚未结束的括号。
        )
        # 中文 L0809 从配置取更新轮数，本次为 10。
        num_steps = self.config.num_steps

        # 中文 L0811 将速度预测函数、初始噪声和步数交给通用 Euler 采样循环。
        return euler_integrate(
            # 中文 L0812 lambda 定义只接收当前动作估计与当前采样时间的小函数，对每步调用 denoise_step。
            lambda input_x_t, current_timestep: self.denoise_step(
                # 中文 L0813 为“x_t”（采样时间 t 的当前动作块估计）提供这一项的值，注意它所在的参数表或外层表达式。
                x_t=input_x_t,
                # 中文 L0814 为“prefix_pad_masks”提供这一项的值，注意它所在的参数表或外层表达式。
                prefix_pad_masks=prefix_pad_masks,
                # 中文 L0815 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
                past_key_values=past_key_values,
                # 中文 L0816 为“timestep”提供这一项的值，注意它所在的参数表或外层表达式。
                timestep=current_timestep,
            # 中文 L0817 闭合上方尚未结束的括号。末尾逗号可用于分隔外层的元素。
            ),
            # 中文 L0818 “noise”（采样起点噪声）是当前括号中的一项，逗号将它与下一项分开。
            noise,
            # 中文 L0819 “num_steps”（每块动作生成的积分步数）是当前括号中的一项，逗号将它与下一项分开。
            num_steps,
            # 中文 L0820 为“rtc_processor”提供这一项的值，注意它所在的参数表或外层表达式。
            rtc_processor=self.rtc_processor,
            # 中文 L0821 为“rtc_enabled”提供这一项的值，注意它所在的参数表或外层表达式。
            rtc_enabled=self._rtc_enabled(),
            # 中文 L0822 为“inference_delay”提供这一项的值，注意它所在的参数表或外层表达式。
            inference_delay=kwargs.get("inference_delay"),
            # 中文 L0823 为“prev_chunk_left_over”提供这一项的值，注意它所在的参数表或外层表达式。
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            # 中文 L0824 为“execution_horizon”提供这一项的值，注意它所在的参数表或外层表达式。
            execution_horizon=kwargs.get("execution_horizon"),
        # 中文 L0825 闭合上方尚未结束的括号。
        )

    # 中文 L0827 计算一个采样步骤的速度场，输入当前带噪动作块与观测前缀缓存。
    def denoise_step(
        # 中文 L0828 self 接收当前实例，调用 obj.method() 时由 Python 自动传入。
        self,
        # 中文 L0829 “prefix_pad_masks”是当前括号中的一项，逗号将它与下一项分开。
        prefix_pad_masks,
        # 中文 L0830 “past_key_values”（各注意力层缓存的 K/V）是当前括号中的一项，逗号将它与下一项分开。
        past_key_values,
        # 中文 L0831 “x_t”（采样时间 t 的当前动作块估计）是当前括号中的一项，逗号将它与下一项分开。
        x_t,
        # 中文 L0832 “timestep”是当前括号中的一项，逗号将它与下一项分开。
        timestep,
    # 中文 L0833 闭合上方尚未结束的括号。
    ):
        # 中文 L0834 原文档概述本段功能，根据条件和当前动作估计预测一次速度。
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # 中文 L0835 每个积分步骤都把当前动作估计与时间编码为动作后缀。
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        # 中文 L0837 用方括号按键或索引读取 suffix_pad_masks.shape给 “suffix_len” 赋值。
        suffix_len = suffix_pad_masks.shape[1]
        # 中文 L0838 用方括号按键或索引读取 prefix_pad_masks.shape给 “batch_size”（批大小） 赋值。
        batch_size = prefix_pad_masks.shape[0]
        # 中文 L0839 用方括号按键或索引读取 prefix_pad_masks.shape给 “prefix_len” 赋值。
        prefix_len = prefix_pad_masks.shape[1]
        # 中文 L0840 将观测 token 的有效位广播给所有动作查询，形状为 [B,L动作,L前缀]。
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # 中文 L0842 调用 make_att_2d_masks，用它的结果给 “suffix_att_2d_masks” 赋值。
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # 中文 L0844 沿键位置轴拼接前缀与动作的注意力掩码，后缀查询可按掩码查看两部分。
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        # 中文 L0845 统计每个批次有效观测 token 的数目，作为后缀位置偏移。
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        # 中文 L0846 后缀位置在有效前缀位置后开始计数。
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # 中文 L0848 用注意力模型处理当前动作后缀，并使用之前构造的观测前缀 K/V。
        outputs_embeds, _ = self.vlm_with_expert.forward(
            # 中文 L0849 为“attention_mask”提供这一项的值，注意它所在的参数表或外层表达式。
            attention_mask=full_att_2d_masks,
            # 中文 L0850 为“position_ids”提供这一项的值，注意它所在的参数表或外层表达式。
            position_ids=position_ids,
            # 中文 L0851 为“past_key_values”（各注意力层缓存的 K/V）提供这一项的值，注意它所在的参数表或外层表达式。
            past_key_values=past_key_values,
            # 中文 L0852 这次运行后缀路径，前缀输入位是 None。前缀条件来自缓存。
            inputs_embeds=[None, suffix_embs],
            # 中文 L0853 为“use_cache”提供这一项的值，注意它所在的参数表或外层表达式。
            use_cache=self.config.use_cache,
        # 中文 L0854 闭合上方尚未结束的括号。
        )
        # 中文 L0855 条件 past_key_values is not None 成立时进入下方分支。
        if past_key_values is not None:
            # Self-attention layers append suffix K/V in place; restore the prefix for the next step.
            # 中文 L0857 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
            for layer in past_key_values.layers:
                # 中文 L0858 将曾被临时后缀扩展的层缓存裁回前缀长度，避免上一轮后缀混入下一轮。
                layer.crop(prefix_len - layer.get_seq_length())
        # 中文 L0859 取返回列表的第 2 项，即动作专家处理的后缀。
        suffix_out = outputs_embeds[1]
        # 中文 L0860 截取最后 chunk_size 个位置，确保对齐动作块时间轴。
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # 中文 L0861 将专家输出转为 float32，接下来转回动作空间。
        suffix_out = suffix_out.to(dtype=torch.float32)
        # 中文 L0862 对每个动作位置使用输出投影，速度预测形状为 [B,T,max_action_dim]。
        v_t = self.action_out_proj(suffix_out)
        # 中文 L0863 把速度场交回 Euler 循环，供它更新当前动作估计。
        return v_t
