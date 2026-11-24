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
# 中文 L0015 从模块 dataclasses 导入 dataclass、field。
from dataclasses import dataclass, field

# 中文 L0017 从模块 lerobot.configs 导入 FeatureType、NormalizationMode、PolicyFeature、PreTrainedConfig。
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
# 中文 L0018 从模块 lerobot.optim 导入 AdamWConfig、CosineDecayWithWarmupSchedulerConfig。
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
# 中文 L0019 从模块 lerobot.utils.constants 导入 OBS_IMAGES。
from lerobot.utils.constants import OBS_IMAGES

# 中文 L0021 从模块 ..rtc.configuration_rtc 导入 RTCConfig。
from ..rtc.configuration_rtc import RTCConfig


# 中文 L0024 把下一行处理后的配置类注册到名字 smolvla。这使加载器能从 config.json 的 type 选到该类。
@PreTrainedConfig.register_subclass("smolvla")
# 中文 L0025 把类转为数据类，由字段定义生成 __init__ 等方法。
@dataclass
# 中文 L0026 类名后括号表示继承。SmolVLAConfig 在父类通用项上增加本模型专属字段。
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    # 中文 L0028 默认使用一个观测时间步。
    n_obs_steps: int = 1
    # 中文 L0029 默认每次模型预测 50 个连续动作位置。这个整数对应动作块的时间轴长度。
    chunk_size: int = 50
    # 中文 L0030 默认让 select_action 每次入队最多 50 个动作。它在 50 次控制调用中逐个出队。
    n_action_steps: int = 50

    # 中文 L0032 声明“normalization_mapping”的类型为 dict[str, NormalizationMode]，调用 field，用它的结果作为初始值。
    normalization_mapping: dict[str, NormalizationMode] = field(
        # 中文 L0033 为“default_factory”提供这一项的值，注意它所在的参数表或外层表达式。
        default_factory=lambda: {
            # 中文 L0034 视觉输入在这个归一化处理器上保持原尺度。模型的 prepare_images 另有像素预处理。
            "VISUAL": NormalizationMode.IDENTITY,
            # 中文 L0035 机器人状态按 checkpoint 保存的均值和标准差归一化。
            "STATE": NormalizationMode.MEAN_STD,
            # 中文 L0036 动作特征使用均值和标准差定义尺度，出模型后执行反变换。
            "ACTION": NormalizationMode.MEAN_STD,
        # 中文 L0037 闭合上方尚未结束的括号。
        }
    # 中文 L0038 闭合上方尚未结束的括号。
    )

    # Shorter state and action vectors will be padded
    # 中文 L0041 模型内部把较短的状态向量补齐到 32 维，这次仿真的原始状态是 8 维。
    max_state_dim: int = 32
    # 中文 L0042 模型内部动作向量宽度是 32，这次 checkpoint 对外声明的动作宽度是 7。
    max_action_dim: int = 32

    # Image preprocessing
    # 中文 L0045 传入图像由模型调整并补齐到 512×512。这和仿真相机输出分辨率属于不同阶段。
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    # 中文 L0049 声明“empty_cameras”的类型为 int，使用字面值 0作为初始值。
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    # 中文 L0053 声明“adapt_to_pi_aloha”的类型为 bool，使用布尔值 False作为初始值。
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    # 中文 L0057 声明“use_delta_joint_actions_aloha”的类型为 bool，使用布尔值 False作为初始值。
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    # 中文 L0060 配置分词后的文字最大长度。用缓存前处理文件才能确认这次的实际分词参数。
    tokenizer_max_length: int = 48

    # Decoding
    # 中文 L0063 默认产生一个动作块时计算 10 个 Euler 更新步。它和机器人控制步是两个计数。
    num_steps: int = 10

    # Attention utils
    # 中文 L0066 允许推理在生成动作块时缓存观测前缀的注意力 K/V。
    use_cache: bool = True

    # Finetuning settings
    # 中文 L0069 这是源码里的默认训练设置。原 checkpoint 对这个字段保存的是 False。
    freeze_vision_encoder: bool = True
    # 中文 L0070 这是源码里的默认训练设置。原 checkpoint 对这个字段保存的是 False。
    train_expert_only: bool = True
    # 中文 L0071 声明“train_state_proj”的类型为 bool，使用布尔值 True作为初始值。
    train_state_proj: bool = True

    # Training presets
    # 中文 L0074 声明“optimizer_lr”的类型为 float，使用字面值 0.0001作为初始值。
    optimizer_lr: float = 1e-4
    # 中文 L0075 声明“optimizer_betas”的类型为 tuple[float, float]，打包成元组，元素顺序由逗号分隔的位置决定作为初始值。
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    # 中文 L0076 声明“optimizer_eps”的类型为 float，使用字面值 1e-08作为初始值。
    optimizer_eps: float = 1e-8
    # 中文 L0077 声明“optimizer_weight_decay”的类型为 float，使用字面值 1e-10作为初始值。
    optimizer_weight_decay: float = 1e-10
    # 中文 L0078 声明“optimizer_grad_clip_norm”的类型为 float，使用字面值 10作为初始值。
    optimizer_grad_clip_norm: float = 10
    # 中文 L0079 声明“optimizer_fused”的类型为 bool | None，设为空值 None作为初始值。
    optimizer_fused: bool | None = None

    # 中文 L0081 声明“scheduler_warmup_steps”的类型为 int，使用字面值 1000作为初始值。
    scheduler_warmup_steps: int = 1_000
    # 中文 L0082 声明“scheduler_decay_steps”的类型为 int，使用字面值 30000作为初始值。
    scheduler_decay_steps: int = 30_000
    # 中文 L0083 声明“scheduler_decay_lr”的类型为 float，使用字面值 2.5e-06作为初始值。
    scheduler_decay_lr: float = 2.5e-6

    # 中文 L0085 指定视觉语言骨干的模型标识。
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    # 中文 L0086 控制模型构造时是否加载骨干预训练权重。本文所读 checkpoint 将它设为 True。
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    # 中文 L0088 控制是否在视觉 token 前后再插入特定词表 token，原 checkpoint 关闭这一设置。
    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    # 中文 L0090 配置含 cross 时，填充前缀缓存后，部分层使用交叉注意力。
    attention_mode: str = "cross_attn"

    # 中文 L0092 允许配置最小前缀长度。embed_prefix 只在实际序列比此值短时补齐。
    prefix_length: int = -1

    # 中文 L0094 文字分词的填充选项。原 checkpoint 保存的流水线选择填到最大长度。
    pad_language_to: str = "longest"  # "max_length"

    # 中文 L0096 正数可以指定专家层数，零或负数使专家跟随骨干的实际层数。
    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    # 中文 L0097 视觉语言骨干的文本模型只保留前 16 层。这个字段控制文本层，视觉编码器另有自己的结构。
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    # 中文 L0098 以间隔 2 选取后缀阶段的自注意力层。其余相应层使用交叉注意力。
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    # 中文 L0099 用骨干隐藏宽度乘 0.75 得到专家的隐藏宽度，再取整数。
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    # 中文 L0101 采样时间的正弦余弦嵌入使用的最短周期。
    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    # 中文 L0102 对应的最长周期。两者共同决定时间特征的频率范围。
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    # 中文 L0105 RTC 是可选的实时动作块机制。这个 checkpoint 的值为 None，逐步评测走普通队列。
    rtc_config: RTCConfig | None = None

    # 中文 L0107 该配置是否启用 PyTorch 编译。原 eval 配置为 False。
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    # 中文 L0108 声明“compile_mode”的类型为 str，使用字面值 'max-autotune'作为初始值。
    compile_mode: str = "max-autotune"  # Torch compile mode

    # 中文 L0110 数据类构造会在字段赋值后执行这个校验方法。
    def __post_init__(self):
        # 中文 L0111 super() 沿继承关系查找父类的 __post_init__，先完成通用配置检查。
        super().__post_init__()

        # 中文 L0113 字符串字面量，供上方的调用、容器或错误消息使用。
        """Input validation (not exhaustive)."""
        # 中文 L0114 校验每块要执行的步数不超过模型产生的动作位置数。
        if self.n_action_steps > self.chunk_size:
            # 中文 L0115 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError(
                # 中文 L0116 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                # 中文 L0117 f 字符串片段，花括号中的表达式会被求值并插入文本。
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            # 中文 L0118 闭合上方尚未结束的括号。
            )
        # 中文 L0119 条件 self.use_delta_joint_actions_aloha 成立时进入下方分支。
        if self.use_delta_joint_actions_aloha:
            # 中文 L0120 抛出 NotImplementedError，中断当前路径并交给上层的异常处理。
            raise NotImplementedError(
                # 中文 L0121 字符串字面量，供上方的调用、容器或错误消息使用。
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            # 中文 L0122 闭合上方尚未结束的括号。
            )

    # 中文 L0124 在构造策略时检查并补齐配置的输入特征。
    def validate_features(self) -> None:
        # 中文 L0125 从 checkpoint 或环境等来源解析输入特征后才能执行该校验。
        if self.input_features is None:
            # 中文 L0126 抛出 ValueError，中断当前路径并交给上层的异常处理。
            raise ValueError("`input_features` must be resolved before `validate_features()` is called.")
        # 中文 L0127 逐项迭代右侧的可迭代对象，并把每项绑定给 in 左侧的名字。
        for i in range(self.empty_cameras):
            # 中文 L0128 计算此行的右侧表达式给 “key” 赋值。
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            # 中文 L0129 调用 PolicyFeature，用它的结果给 “empty_camera” 赋值。
            empty_camera = PolicyFeature(
                # 中文 L0130 为“type”提供这一项的值，注意它所在的参数表或外层表达式。
                type=FeatureType.VISUAL,
                # 中文 L0131 为“shape”（各轴的长度）提供这一项的值，注意它所在的参数表或外层表达式。
                shape=(3, 480, 640),
            # 中文 L0132 闭合上方尚未结束的括号。
            )
            # 中文 L0133 引用已有的值 empty_camera给 “self.input_features[key]” 赋值。
            self.input_features[key] = empty_camera

    # 中文 L0135 定义方法 get_optimizer_preset。
    def get_optimizer_preset(self) -> AdamWConfig:
        # 中文 L0136 调用 AdamWConfig，用它的结果作为当前调用的返回值。
        return AdamWConfig(
            # 中文 L0137 为“lr”提供这一项的值，注意它所在的参数表或外层表达式。
            lr=self.optimizer_lr,
            # 中文 L0138 为“betas”提供这一项的值，注意它所在的参数表或外层表达式。
            betas=self.optimizer_betas,
            # 中文 L0139 为“eps”提供这一项的值，注意它所在的参数表或外层表达式。
            eps=self.optimizer_eps,
            # 中文 L0140 为“weight_decay”提供这一项的值，注意它所在的参数表或外层表达式。
            weight_decay=self.optimizer_weight_decay,
            # 中文 L0141 为“grad_clip_norm”提供这一项的值，注意它所在的参数表或外层表达式。
            grad_clip_norm=self.optimizer_grad_clip_norm,
            # 中文 L0142 为“fused”提供这一项的值，注意它所在的参数表或外层表达式。
            fused=self.optimizer_fused,
        # 中文 L0143 闭合上方尚未结束的括号。
        )

    # 中文 L0145 定义方法 get_scheduler_preset。
    def get_scheduler_preset(self):
        # 中文 L0146 调用 CosineDecayWithWarmupSchedulerConfig，用它的结果作为当前调用的返回值。
        return CosineDecayWithWarmupSchedulerConfig(
            # 中文 L0147 为“peak_lr”提供这一项的值，注意它所在的参数表或外层表达式。
            peak_lr=self.optimizer_lr,
            # 中文 L0148 为“decay_lr”提供这一项的值，注意它所在的参数表或外层表达式。
            decay_lr=self.scheduler_decay_lr,
            # 中文 L0149 为“num_warmup_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            num_warmup_steps=self.scheduler_warmup_steps,
            # 中文 L0150 为“num_decay_steps”提供这一项的值，注意它所在的参数表或外层表达式。
            num_decay_steps=self.scheduler_decay_steps,
        # 中文 L0151 闭合上方尚未结束的括号。
        )

    # 中文 L0153 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0154 定义方法 observation_delta_indices。
    def observation_delta_indices(self) -> list:
        # 中文 L0155 创建列表，按给定顺序保留元素作为当前调用的返回值。
        return [0]

    # 中文 L0157 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0158 定义方法 action_delta_indices。
    def action_delta_indices(self) -> list:
        # 中文 L0159 调用 list，用它的结果作为当前调用的返回值。
        return list(range(self.chunk_size))

    # 中文 L0161 属性装饰器。调用方用 obj.name 的形式访问这个方法的结果。
    @property
    # 中文 L0162 定义方法 reward_delta_indices。
    def reward_delta_indices(self) -> None:
        # 中文 L0163 设为空值 None作为当前调用的返回值。
        return None
