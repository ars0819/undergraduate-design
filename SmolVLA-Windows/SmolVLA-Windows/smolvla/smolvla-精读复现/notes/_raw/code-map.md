# SmolVLA 论文 ↔ 官方实现代码逐条对照（paper → code map）

> **用途**：把 SmolVLA 论文（arXiv:2506.01844）的每条方法学陈述落到 LeRobot 官方实现的具体函数/行号，并标出「论文没写但代码这么做」「论文与代码不一致」「复现时一定会踩」的点，供论文精读文档直接引用。
>
> **代码版本**：`/home/lxy/lerobot-smolvla-repro/lerobot`，LeRobot `0.7.1`，`git rev-parse HEAD = 83883ae60`
> （`fix(env): defer signal shutdown until the owner loop (#5295)`）。除特别注明外，下文所有路径都相对该仓库根。
>
> **checkpoint**：`lerobot/smolvla_libero` @ revision `a24f1f380461b665158d52acd140152c79e8b36d`
> （本地缓存 `/home/lxy/lerobot-smolvla-repro/hf-cache/hub/models--lerobot--smolvla_libero/snapshots/a24f1f38.../`），
> 基座 `lerobot/smolvla_base`；VLM 底座 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`。
>
> **公式编号说明**：论文 v1 的 §3.1 公式**没有编号**（PDF 抽取确认无线号）。为引用方便，本文用
> 「§3.1-式(a)」标记论文原文中公式出现的顺序，并在每处附上原文关键记号，不做精确编号承诺。

---

## 0. 一页速查表：论文陈述 → 代码位置

| 论文陈述（章节） | 关键量 | 代码位置 |
|---|---|---|
| 论文 §3.1-式(a)：flow matching 损失 `L(θ)=E‖v_θ(A^τ,o_t) − u(A^τ|A_t)‖²` | 逐元素 MSE | `src/lerobot/policies/smolvla/modeling_smolvla.py:772-773` |
| 论文 §3.1：`A^τ = τA+(1−τ)ε`，`u = ε−A` | 插值/回归目标 | `modeling_smolvla.py:751` + `src/lerobot/policies/common/flow_matching.py:246-253`（`FlowConvention.NOISE_AT_ONE`） |
| 论文 §3.1：τ 取自 Beta 分布（follow π0） | τ ~ Beta(1.5,1.0)·0.999+0.001 | `modeling_smolvla.py:601-602` → `common/flow_matching.py:167-185` |
| 论文 §3.1：expert 隐维 `0.75×d` | 720 = 960×0.75 | `src/lerobot/policies/smolvla/smolvlm_with_expert.py:108-118` |
| 论文 §3.1：CA / SA 交替，"interleaved" | `self_attn_every_n_layers=2` | `smolvlm_with_expert.py:448-476`（路由）；实现 `:220-292`（SA）、`:294-398`（CA） |
| 论文 §3.1：SA 层对 action 用 causal mask | att_masks/cumsum 构造 | `modeling_smolvla.py:733-734` + `common/vla_utils.py:60-89` |
| 论文 §3.1：只用 LLM 前 N=L/2 层（16/32 层） | `num_vlm_layers=16` | `smolvlm_with_expert.py:102-105` |
| 论文 §3.1：每帧只用 64 个视觉 token（pixel shuffle，不用 tiling） | connector pixel-shuffle 4 | `smolvlm_with_expert.py:202-215`；`SmolVLM(connector)`，实测 64 token/图 |
| 论文 §3.1：state 投影成 1 个 token 与图像/语言拼接 | `state_proj: 32→960` | `modeling_smolvla.py:561-563, 671-682` |
| 论文 §3.1：512×512 resize | `resize_imgs_with_padding=(512,512)` | `configuration_smolvla.py:45`；`modeling_smolvla.py:403-413` |
| 论文 §3.3 / Alg.1：异步推理、action queue | `select_action` 队列 | `modeling_smolvla.py:255-288` |
| 论文 §4.3：推理 10 步 flow matching | `num_steps=10` | `configuration_smolvla.py:63`；`modeling_smolvla.py:809-825` |
| 论文 §4.3：chunk n=50 | `chunk_size=50`, `n_action_steps=50` | `configuration_smolvla.py:29-30` |
| 论文 §4.3：只训练 action expert，VLM 冻结（~100M trainable） | 99,880,992 可训练 | `smolvlm_with_expert.py:152-160`（`freeze_vision_encoder=True`,`train_expert_only=True`，即 base 配置） |
| 论文 §4.3：总参数 450M | 实测 450,046,176 | 见 §5；由 checkpoint 张量逐个 numel 求和得到 |
| 论文 §4.3：bf16 + torch.compile | 模型构造即 bf16；compile 默认关闭 | `smolvlm_with_expert.py:92-96`（VLM 以 bfloat16 加载）；`configuration_smolvla.py:107-108`（`compile_model=False`） |

---

## 1. Flow matching 训练目标

### 1.1 论文原文（§3.1 "Flow matching action expert"）

> `L^τ(θ) = E_{p(A_t|o_t), q(A_t^τ|A_t)} [ ‖v_θ(A_t^τ, o_t) − u(A_t^τ | A_t)‖² ]`，
> `A_t^τ = τ A_t + (1−τ) ε`，`ε ~ N(0,I)`，`u(A_t^τ|A_t) = ε − A_t`；
> "In keeping with Black et al. (2024), we sample τ from a Beta distribution"。

### 1.2 代码里的完整训练链路

调用链：`lerobot_train.py` 训练循环 → `SmolVLAPolicy.forward(batch)` → `VLAFlowMatching.forward(...)`。

`src/lerobot/policies/smolvla/modeling_smolvla.py:741-774`（核心 20 行）：

```python
# modeling_smolvla.py:745-773
if noise is None:
    noise = self.sample_noise(actions.shape, actions.device)      # N(0,1), fp32, (B,50,32)
if time is None:
    time = self.sample_time(actions.shape[0], actions.device)     # Beta(1.5,1)*0.999+0.001
x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)
...
(_, suffix_out), _ = self.vlm_with_expert.forward(...)            # 见 §3
suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
v_t = self.action_out_proj(suffix_out)                            # 720 -> 32
losses = F.mse_loss(u_t, v_t, reduction="none")                   # (B, 50, 32)
```

**逐项对应**：

1. **插值与回归目标** `src/lerobot/policies/common/flow_matching.py:246-251`（`FlowConvention.NOISE_AT_ONE`）：

```python
# common/flow_matching.py:246-251
if noise_at_one:
    x_t = expanded_time * noise + (1 - expanded_time) * actions   # x_t = t*ε + (1-t)*A
    velocity_target = noise - actions                             # u = ε - A
```

即代码把论文里的 `τ` 当成 **"噪声坐标" t**：`t=1` 是纯噪声，`t=0` 是干净动作。
`u_t = ε − A` 与论文一致；但论文同时写的 `A^τ = τA+(1−τ)ε` 与 `u=ε−A` **方向不自洽**
（对 `τA+(1−τ)ε` 求导得 `A−ε`，与 `u=ε−A` 相反）。**以代码为准**：`x_t=tε+(1−t)A`、`u=ε−A`、
推理从 `t=1 → 0` 积分（`dt<0`）。复现时若照论文公式字面实现、又照论文用 `ε−A` 作目标，会得到符号反向的速度场。

2. **时间 t 的采样分布**（论文只说 "Beta distribution"）——`modeling_smolvla.py:601-602`：

```python
def sample_time(self, bsize, device):
    return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
```

`common/flow_matching.py:167-185` 实现 `t = Beta(alpha,beta).sample() * scale + offset`，Beta 浓度固定在 CPU 上采样。
实测（200k 采样）：`mean=0.6017, std=0.2612, min=0.0014, max=1.0000, q10=0.2175, q50=0.6320, q90=0.9322`
→ **偏向高噪声端 t≈1**（不是均匀采样，也不是偏向干净端）。这是 π0 系（openpi）的配方，不是 SmolVLA 原创。

3. **噪声**：`common/flow_matching.py:137-164`，`torch.normal(0,1)`，**dtype 固定 float32**，形状 `(B, chunk_size, max_action_dim) = (B,50,32)`
（注意是 `max_action_dim=32` 而不是真实 7 维，见 §4.3）。

4. **loss 的写法与 padding/mask**：模型层返回**未归约**的逐元素 MSE `(B,50,32)`（`modeling_smolvla.py:773`），
真正"只算真实维度 + 只算有效时间步"是在 policy 层做的，`modeling_smolvla.py:318-354`：

```python
# modeling_smolvla.py:318-335
actions = self.prepare_action(batch)                   # 7 -> 32 (zeros pad)
actions_is_pad = batch.get("action_is_pad")
losses = self.model.forward(...)                       # (B,50,32) 逐元素
losses = losses[:, :, : action_feature.shape[0]]       # 先裁到真实动作维 7
...
if actions_is_pad is not None:
    in_episode_bound = ~actions_is_pad                 # (B,50)
    losses = losses * in_episode_bound.unsqueeze(-1)   # 无效时间步乘 0
losses = losses[:, :, : self.config.max_action_dim]    # 此时是空操作（7 < 32）
```

```python
# modeling_smolvla.py:346-354（默认 reduction="mean" 的归一化）
if actions_is_pad is None:
    loss = losses.mean()
else:
    num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
    loss = losses.sum() / num_valid
```

要点：
- **模型层 loss 覆盖 32 维**（含 25 个纯 padding 维）。padding 维的 `x_t = t·noise`（因 clean 端是 0），
  目标 `u = noise`；这些 loss 在 policy 层被裁掉，但**噪声仍然进入了 `action_in_proj` 的前向**（见 §4.3）。
- policy 层先裁维度、再按 `action_is_pad` 置零、再用 `有效时间步数 × 真实维数(7)` 作分母 —— 是"有效元素均值"，
  不是先 mean 再 mask（后者会让分母偏大）。`reduction="none"` 分支（`smolvla` 支持 RA-BC 加权）在 `:337-345`。
- `loss_dict` 里的 `losses_after_forward / losses_after_in_ep_bound / losses_after_rm_padding` 是调试量（`:326,331,335`），会 `.item()` 强制同步。

5. **`action_is_pad` 从哪来**：不在 processor 里，而在数据集侧按 episode 边界生成：
   `src/lerobot/datasets/utils.py:146-160` 的 `delta_window()`（落在 `[ep_start, ep_end)` 之外的位置 mask=True），
   在 `src/lerobot/datasets/dataset_reader.py:300-310` 生成 `padding[f"{key}_is_pad"]`，即 batch 里的 `action_is_pad`。
   `action_delta_indices = list(range(chunk_size))`（`configuration_smolvla.py:157-159`）决定这个 mask 的长度是 50。
   **训练 batch 里没有这个 key 时**（例如直接从 env 采样训练/自造数据），`batch.get("action_is_pad")` 返回 None，
   loss 退化为对全部 50×7 取均值——这是复现时数据管线必须补齐的字段。

---

## 2. 推理采样（sample_actions / Euler / action queue）

### 2.1 `sample_actions` 全流程 —— `modeling_smolvla.py:776-825`

```python
# modeling_smolvla.py:790-825（省略 embed 细节）
if noise is None:
    actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)   # (B,50,32)
    noise = self.sample_noise(actions_shape, device)        # N(0,1), fp32
else:
    noise = noise.to(device=device, dtype=torch.float32)
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)          # 图像+语言+state
prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
_, past_key_values = self.vlm_with_expert.forward(       # 前缀预填充，只算一次
    attention_mask=prefix_att_2d_masks, position_ids=prefix_position_ids,
    past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=self.config.use_cache)
num_steps = self.config.num_steps                         # 10
return euler_integrate(lambda input_x_t, current_timestep: self.denoise_step(...),
                       noise, num_steps, rtc_processor=..., ...)
```

### 2.2 Euler 积分循环 —— `common/flow_matching.py:256-421`

- 步长：`uniform_dt = -1.0/num_steps`（`NOISE_AT_ONE`，`:338`），`time = 1.0 + step*dt`（`:347-348`）。
  `num_steps=10` → 评估时刻 **t = 1.0, 0.9, 0.8, …, 0.1**（共 10 次前向），然后 `x_t += dt·v_t`（`:410-413`）。
  注意 **第一步在 t=1.0（纯噪声点）评估**，最后一步用 t=0.1 的速度更新后结束，不会在 t=0 再评估。
- 初始噪声：标准正态 `(B,50,32)`，fp32；`sample_actions(noise=...)` 可外部注入（RTC / 可复现实验用）。
- 最终输出：`(B, 50, 32)` fp32，随后 policy 层裁到真实 7 维（`modeling_smolvla.py:226`）。
- `rtc_processor`（实时 chunking）默认未启用：checkpoint 里 `rtc_config: null`；启用时由 `euler_integrate` 的
  `guidance.denoise_step(...)` 包住每步速度计算（`:394-408`）。

### 2.3 单步去噪 `denoise_step` —— `modeling_smolvla.py:827-863`

```python
# modeling_smolvla.py:835-862（关键 3 段）
suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)  # 动作+时间嵌入
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1   # 动作位置从 prefix_len 续号
outputs_embeds, _ = self.vlm_with_expert.forward(..., past_key_values=past_key_values,
                                                 inputs_embeds=[None, suffix_embs],
                                                 use_cache=self.config.use_cache)
if past_key_values is not None:
    for layer in past_key_values.layers:                # 关键：把 SA 层追加的 suffix KV 裁掉
        layer.crop(prefix_len - layer.get_seq_length())  # 恢复纯前缀 cache，供下一步复用
```

- 每次 Euler step 只前向 **50 个动作 token**（不是 177+50），前缀 K/V 复用（`use_cache=True`）。
- 自注意力层（偶数层）在去噪时会往 `DynamicCache` **就地追加** suffix 的 K/V，所以每步结束要 `crop` 回前缀长度
  （`:855-858`，注释原文："restore the prefix for the next step"）。**这是 0.7.x 的重要性能/正确性实现细节**。
- 交叉注意力层（奇数层）不去写 cache，只从 `past_key_values.layers[layer_idx]` **读取前缀 K/V**（`smolvlm_with_expert.py:342-348`）。

### 2.4 `use_cache` 指什么

配置项 `configuration_smolvla.py:66` `use_cache: bool = True`，含义 = **推理时是否复用前缀 KV cache**：
- `sample_actions` 的前缀预填充传 `use_cache`（`:807`），`denoise_step` 也传（`:853`）；
- 训练 `forward` 里**硬编码 `use_cache=False`**（`:767`），training 不走 cache 路径。
- ⚠️ **实测：`attention_mode="cross_attn"` 时 `use_cache=False` 推理会崩**（见 §7 坑 1）。所以这个开关在当前实现里
  实际不可关闭，不是"性能选项"。

### 2.5 `chunk_size` 与 `n_action_steps` 的关系 + `select_action` 队列

- 配置：`chunk_size=50`（网络一次输出 50 步）、`n_action_steps=50`（实际执行多少步后重新规划），
  `__post_init__` 校验 `n_action_steps <= chunk_size`（`configuration_smolvla.py:114-118`）。
- 队列初始化：`reset()` 建 `deque(maxlen=n_action_steps)`（`modeling_smolvla.py:172-176`）。
- `select_action`（`modeling_smolvla.py:255-288`）：

```python
# modeling_smolvla.py:278-285
if self._check_get_actions_condition():                    # len(queue)==0 才推理
    actions = self._get_action_chunk(batch, noise=noise)   # (B,50,7) 新 chunk
    # 队列形状是 (n_action_steps, B, *)：transpose(0,1) 取每步动作，[:n_action_steps] 截断
    self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
return self._queues[ACTION].popleft()
```

  即：**队列空 → 生成整 chunk → 只入队前 `n_action_steps` 步 → 每调用一次弹一个**。
  `n_action_steps == chunk_size == 50` 时等价于"每 50 步重新观测+规划一次"（open-loop 50 步）。
  若 `n_action_steps < chunk_size`，chunk 的剩余动作**直接丢弃**（不会重排/插值），重新规划的周期 = `n_action_steps`。
- `predict_action_chunk`（`:239-253`）绕过队列直接返回整 chunk（RTC / 离线评测用），`supports_rtc()` 为 True（`:150-151`）。
- 论文 §3.3 的**异步推理栈**不在 policy 类里：本 fork 把它拆到 `src/lerobot/inference/`
  （`--inference.type=sync|rtc|remote`，见 `src/lerobot/inference/factory.py:17-20`；`rtc` = 后台线程异步推理 +
  chunk 聚合/队列阈值 `queue_threshold`，对应论文 Algorithm 1 的 g 阈值异步模式；`sync` = 执行完整个 chunk 再规划）。
  `SmolVLAPolicy.select_action` 是**同步兜底路径**：队列空才推理，正是论文 Algorithm 1 里 g=0 的 "sequential limit"。
- ⚠️ 论文 §4.3 写 "In simulation, we perform inference by sampling new observations and predicting a new action after each
  executed action"，与 checkpoint 的 `n_action_steps=50` 字面冲突。见 §6.3 的复现待验证项。

---

## 3. 架构：VLM 与 action expert 如何交替

### 3.1 构造 —— `smolvlm_with_expert.py:74-147` + `modeling_smolvla.py:545-589`

1. **VLM 加载**：`load_vlm_weights=True` 时 `AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="bfloat16")`
   （`:90-97`），所以 VLM 权重是 **bf16**；`False` 时 `SmolVLMForConditionalGeneration(config)` 随机初始化（默认 fp32）。
2. **截断 LLM 层**（论文 "only the first 16 layers"）：

```python
# smolvlm_with_expert.py:102-105
if num_vlm_layers > 0:
    self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)   # 16
```

   SmolVLM2-500M 原本 32 层（text hidden 960），SmolVLA 只保留**前 16 层**（`:104`）。vision encoder(12L, 768d) 与
   connector 完整保留。**注意：丢层 ≠ 丢参数**，被丢的 16 层不进入 checkpoint（所以 §5 里 450M 已按 16 层计）。
3. **action expert 构造**（`:107-118`）：`deepcopy(text_config)` → 缩宽 → `AutoModel.from_config(...)`：

```python
# smolvlm_with_expert.py:108-118
lm_expert_config = copy.deepcopy(config.text_config)          # SmolLM2 text config (d=960, heads=15, kv=5, head_dim=64)
lm_expert_config.hidden_size = int(960 * 0.75)                # = 720
lm_expert_config.intermediate_size = get_intermediate_size(int(960 * 0.75))  # = 2048
lm_expert_config.num_hidden_layers = self.num_vlm_layers      # = 16（num_expert_layers<=0 时）
if num_expert_layers > 0:
    ...
    lm_expert_config.num_hidden_layers = num_expert_layers
self.lm_expert = AutoModel.from_config(lm_expert_config)
```

   **`num_expert_layers=0` 的含义**：`if num_expert_layers > 0` 不成立 → 不覆盖，expert 层数 = `num_vlm_layers` = 16
   （`configuration_smolvla.py:96` 注释："Less or equal to 0 is the default where the action expert has the same number of
   layers of VLM"）。**不是 0 层**！这是最容易误读的配置项之一。
4. **层配对**：`get_model_layers`（`smolvlm_with_expert.py:400-412`）

```python
multiple_of = self.num_vlm_layers // self.num_expert_layers   # 16 // 16 = 1
for i in range(self.num_vlm_layers):
    if multiple_of > 0 and i > 0 and i % multiple_of != 0:
        expert_layer = None                                   # 本例不会发生
    else:
        expert_layer = models[1].layers[i // multiple_of]
```
   即 **VLM 第 i 层配 expert 第 i 层，一对一**；若 `num_expert_layers < num_vlm_layers` 则 expert 会"稀疏挂载"
   （部分 VLM 层没有对应 expert）。

### 3.2 "交替"到底怎么交替 —— `self_attn_every_n_layers=2`

`smolvlm_with_expert.py:448-476` 的层路由：

```python
# smolvlm_with_expert.py:448-476（节选）
for layer_idx in range(num_layers):                            # num_layers = 16
    if (fill_kv_cache
        or "cross" not in self.attention_mode
        or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)):
        att_outputs, past_key_values = self.forward_attn_layer(...)        # joint self-attention
    else:
        att_outputs, past_key_values = self.forward_cross_attn_layer(...)  # expert cross-attention
```

`self_attn_every_n_layers=2`（`configuration_smolvla.py:98`）→ **偶数层 = 联合自注意力，奇数层 = 交叉注意力**
（实测层路由：0 SA, 1 CA, 2 SA, …, 15 CA；共 8 SA + 8 CA）。若该值 ≤ 0，则所有层都走 CA 路径
（`or "cross" not in self.attention_mode` / 取模条件都不成立）；`attention_mode="self_attn"` 时则所有层走联合 SA。
⚠️ 不要把它和论文 Table 8 的 "Skip % 2" 消融混淆：后者是"跳过每隔一层的 VLM 层"（不同机制），
`self_attn_every_n_layers` 调的是 CA/SA 交替模式（对应论文 Table 6 的 CA/SA/CA+SA 消融）。

两种层的精确语义（重要，论文描述较简）：

| 层型 | 前缀（VLM token） | 动作 expert | 注意力范围 |
|---|---|---|---|
| **SA 层**（偶数）`forward_attn_layer:220-292` | 更新自己 | 更新自己 | **拼接成一条序列 [prefix; suffix] 做一次联合 attention**：Q/K/V 由两个模型各自的投影产出，K/V 在第 1 维拼接（`:255-257`），**共用一个 softmax**。动作 token 可看图像/语言/state + 之前的动作 |
| **CA 层**（奇数）`forward_cross_attn_layer:294-398` | 仅做前缀自注意力（`:313-338`） | 交叉注意力到 VLM 的 K/V（`:350-395`） | 两套独立 attention：expert Q 只 attend 前缀 K/V；**动作之间不可见**（所以 CA 层不需要 causal mask） |

SA 层里 expert 的 K/V 投影是 `720→320`；CA 层里 expert 的 `k_proj/v_proj` 被**重新定义**为 `320→320`，因为它的输入是
VLM 的 K/V（`5 heads × 64 = 320`），见 `smolvlm_with_expert.py:122-136`：

```python
# smolvlm_with_expert.py:122-136（节选）
if "cross" in attention_mode:
    for layer_idx in range(len(self.lm_expert.layers)):
        if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
            continue                                    # SA 层不改，K/V 来自动作自身 (720→320)
        self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
            config.text_config.num_key_value_heads * config.text_config.head_dim,   # 320 (VLM kv 维)
            lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,       # 320 (expert kv 维)
            bias=lm_expert_config.attention_bias)
```

**CA 层的 K/V 是"VLM 层已经过 RoPE 的 K/V"再经 expert k_proj/v_proj 投影一次**（不是重复加 RoPE，见 §3.5）。

### 3.3 `expert_width_multiplier=0.75` 的精确效果

| 项 | VLM | expert | 说明 |
|---|---|---|---|
| hidden（残差流宽度） | 960 | **720** | `int(960*0.75)` |
| intermediate_size | 2560 | **2048** | `get_intermediate_size(720)`：`int(2*720/3)=480`，`×4=1920`，向上取 256 的整数倍 → 2048 |
| num_attention_heads | 15 | **15** | 不变（deepcopy text_config） |
| num_key_value_heads | 5 | **5** | 不变 |
| head_dim | 64 | **64** | 不变 |
| q_proj 输出 / o_proj 输入 | 960 (=15×64) | **960** | **不随 hidden 缩小**：q_proj 是 720→960，o_proj 是 960→720 |
| k_proj/v_proj（SA 层） | 960→320 | **720→320** | |
| k_proj/v_proj（CA 层） | 960→320 | **320→320** | 输入是 VLM 的 K/V |

⚠️ 关键细节：**0.75 只缩残差流宽度与 MLP，不缩 head 数/head_dim**。于是 expert 的 attention 内维是
`15×64=960`（q/o）与 `5×64=320`（kv），而残差流宽度是 720：`q_proj` 是 720→960、`o_proj` 是 960→720。
也就是说 attention 内维（960/320）**不等于** hidden_size（720）——这是 pi0 系 VLA 的常规做法，不是实现 bug，
但自研实现时很容易顺手把 q_proj 写成 720→720 而形状对不上 checkpoint。

### 3.4 attention mask 的构造（论文 §3.1 "causal mask for SA layers" 的代码实现）

两段式（openpi / big_vision 风格），`common/vla_utils.py:60-89`：

```python
# common/vla_utils.py:86-89
cumsum = torch.cumsum(att_masks, dim=1)
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]   # 行=query, 列=key
pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
return att_2d_masks & pad_2d_masks
```

- `pad_masks`：图像（padding mask / 全 1）、语言（tokenizer attention mask）、state（恒 1）、动作（**恒 1**，动作 padding 只用 loss 排除）。
- `att_masks`（区块标记，`:685-686` 与 `:733-734`）：图像+语言 = 0；state = 1；50 个动作 token = 1 each。
  → `cumsum` 后：图像/语言 = 0，state = 1，动作 = 2,3,…,51。
- 结果（用 3 image/lang + 1 state + 5 action 实测，`1`=可见）：

```
tokens = [L,L,L,S,A,A,A,A,A]        # L=图像/语言, S=state, A=动作
[[1 1 1 0 0 0 0 0 0]                # 图像/语言 token 只看图像+语言
 [1 1 1 0 0 0 0 0 0]
 [1 1 1 0 0 0 0 0 0]
 [1 1 1 1 0 0 0 0 0]                # state 看前缀+自己
 [1 1 1 1 1 0 0 0 0]                # 动作 k 看全部前缀 + 动作 0..k（causal）
 [1 1 1 1 1 1 0 0 0]
 [1 1 1 1 1 1 1 0 0]
 [1 1 1 1 1 1 1 1 0]
 [1 1 1 1 1 1 1 1 1]]
```

**论文没写、但代码里存在的语义**：
1. 图像/语言 token **看不到 state token**（state 用 `att_masks=1` 单独起块，`:682` 注释 "Set attention masks so that
   image and language inputs do not attend to state or actions"）。
2. 动作 token 在 SA 层内是 **causal**（每个动作只看自己之前的动作），与论文 "preventing future action dependencies" 一致。
3. 但**动作 token 能看全部图像/语言/state**（它们是 cumsum=0/1 的"过去"块）。
4. 在 CA 层，expert 的 mask 是 `attention_mask[:, -50:, :prefix_len]`（`smolvlm_with_expert.py:379-381`），
   只保留"动作 query × 前缀 key"的 pad 掩码 → 无动作间 causal（因为动作间根本不互相 attend）。
5. 训练（`use_cache=False`）时 SA 层一次算完整 177+50 的联合 attention；inference 时前缀只算一次（177），每步只算 50。

**实测序列长度**（2 相机、tokenizer max_length=48）：prefix = 2×64(图) + 48(语言) + 1(state) = **177**，
suffix = 50，联合序列 227。3 相机则为 241。这个长度**直接由相机数与 `tokenizer_max_length` 决定**。

### 3.5 RoPE 与位置编码的两个实现细节

- `create_sinusoidal_pos_embedding`（`common/vla_utils.py:41-57`）用于 **flow-matching 时间步** 的正弦编码，
  `min_period=4e-3, max_period=4.0`（`configuration_smolvla.py:101-102`），维度 = `expert_hidden_size=720`；
  用 `float64` 计算后 cast 到模型 dtype（`:51`）。在 `embed_suffix` 里与动作嵌入 concat 后过 `action_time_mlp_in/out`
  （`modeling_smolvla.py:710-724`）。
- **attention 里的 RoPE 是代码手写**：`apply_rope(x, positions, max_wavelength=10_000)`（`smolvlm_with_expert.py:41-64`）。
  ⚠️ SmolVLM2 的 `text_config.rope_theta = 100000`，而这里**硬编码 10_000**（openpi 默认值），没有读 config。
  因为所有 released checkpoint 都是按这个实现训练的，复现必须保持 10_000，但要知道它与 SmolVLM2 预训练时的 RoPE 尺度不同。
  `apply_rope` 内部先 upcast 到 float32 再 cast 回（`:48`），切分方式是 Llama 式 "rotate half"（`rope_interleaved=False`）。
- 位置 id：前缀 `cumsum(pad_masks)-1`（`modeling_smolvla.py:761`），去噪时 suffix `prefix_len + i`（`:845-846`）；
  但**在 CA 层里 expert Q 的位置被平移成从 0 开始**（`smolvlm_with_expert.py:376-378`）：

```python
expert_position_id = expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
```

  于是同一个 expert 在 SA 层用绝对位置（177…226）、在 CA 层用相对位置（0…49）。这是继承自 openpi 的实现，
  论文完全没提；自研实现若统一用一种位置会与 released checkpoint 不一致。

### 3.6 `prefix_length` 是什么、为什么 checkpoint 里是 0

`configuration_smolvla.py:92`：`prefix_length: int = -1`；`modeling_smolvla.py:688-693`：

```python
seq_len = pad_masks.shape[1]
if seq_len < self.prefix_length:                  # -1 或 0 时永不成立
    embs = pad_tensor(embs, self.prefix_length, pad_value=0)
    pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
    att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
```

作用：把可变长前缀**补零到固定长度**，方便 torch.compile / 固定 shape 的 KV cache（原版 openpi 用于静态形状）。
`lerobot/smolvla_base` 与 `smolvla_libero` 的 config 都是 `prefix_length: 0` → **不 pad**，前缀长度随相机数变化。
复现时不要在代码里当成"177"写死；改相机数/语言长度会改变序列长度（`use_cache` 前缀只算一次，影响不大）。

### 3.7 attention 实现只有 eager（没有 sdpa/flash）

`smolvlm_with_expert.py:521-569`：`get_attention_interface()` 永远返回 `self.eager_attention_forward`，
手写 GQA 展开 + **float32 upcast** 的 matmul-softmax（`:548-561`，`big_neg = torch.finfo(float32).min`）。
论文 §4.3 提到的 `torch.compile()` 在本 fork 里是 `config.compile_model`（`configuration_smolvla.py:107-108`），
**默认 False，checkpoint 配置里也没有打开**；即使打开也只 compile `sample_actions/forward` 两个入口（`modeling_smolvla.py:586-589`）。
复现注意：① 不存在 `attn_implementation="sdpa"` 的开关；② 训练 batch=32 时单个 SA 层的 fp32 attention 矩阵是
`32×15×227×227×4B ≈ 99MB`，eager 路径比 flash 吃显存。

---

## 4. 输入输出张量化（processor + policy 内部）

### 4.1 预处理流水线（顺序是 Hub 序列化契约，不能随便换）

`src/lerobot/policies/smolvla/processor_smolvla.py:63-81`：

```python
input_steps = [
    steps.rename_observations,   # observation.images.image -> camera1 等（libreo 用 rename_map）
    steps.add_batch_dim,
    NewLineTaskProcessorStep(),  # task 串末尾补 "\n"
    TokenizerProcessorStep(tokenizer_name=config.vlm_model_name,
        padding=config.pad_language_to, padding_side="right", max_length=config.tokenizer_max_length),
    steps.to_device,
    steps.normalize,             # MEAN_STD: state / action
]
output_steps = [steps.unnormalize, steps.to_cpu]
```

checkpoint 里 `policy_preprocessor.json` 的步骤序列与上完全一致（`rename → batch → smolvla_new_line → tokenizer →
device → normalizer`），`tokenizer` 配置为 `padding="max_length", max_length=48, padding_side="right", truncation=true`。
- `NewLineTaskProcessorStep`：`src/lerobot/processor/newline_task_processor.py:34-54`，任务描述末尾补 `\n`（SmolVLM tokenizer 风格）。
- `TokenizerProcessorStep`：`src/lerobot/processor/tokenizer_processor.py:255-272`（调用 tokenizer），
  输出写进 `observation.language.tokens` / `observation.language.attention_mask`（`:206-207`），mask 转 bool。

### 4.2 state：8 → 32（零填充，归一化之后）

`modeling_smolvla.py:462-466`：

```python
state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]  # n_obs_steps=1 取最后一帧
state = pad_vector(state, self.config.max_state_dim)                                   # 8 -> 32，F.pad 补 0
```

`common/vla_utils.py:116-138` 的 `pad_vector` 只在最后一维 < new_dim 时补零（不复制、不回绕）。
→ 模型看到的是 **归一化后的 8 维 + 24 个 0**；`state_proj` 是 `32→960` 带 bias（`modeling_smolvla.py:561-563`），
所以 padding 维贡献的是 bias（不是 0）。state 作为**前缀最后一个 token**（`state_proj(state)`，`:671-673`）。

### 4.3 action：7 → 32（零填充），训练 loss 只在 7 维、推理输出裁到 7

- 训练：`prepare_action`（`modeling_smolvla.py:468-471`）`pad_vector(batch[ACTION], 32)`；`x_t/u_t` 都是 32 维。
- loss：模型层 32 维（§1），policy 层裁到 `action_feature.shape[0] = 7`（`:325`）。
- 推理：`_get_action_chunk` 末尾 `actions[:, :, : action_feature.shape[0]]`（`:226`）→ 输出 7 维，再交给 postprocessor 反归一化。
- ⚠️ **padding 维并非"无害"**：训练时 `x_t[:, :, 7:] = t·noise`（clean=0），这些随机噪声经过 `action_in_proj`（32→720）
  进入 expert 输入，只是不被监督；推理时同样对 32 维采样噪声。也就是 7 维动作的预测会"看到"25 维噪声输入。
  这是 openpi 系共享的 padding 约定，但论文完全没写。

### 4.4 语言：tokenizer_max_length=48、右 padding、`pad_language_to`

- `configuration_smolvla.py:60` `tokenizer_max_length=48`；`:94` `pad_language_to: str = "longest"`，
  **但 base / libero checkpoint 都是 `"max_length"`**（见 §6 配置表）→ 每条指令都 pad/truncate 到 **48 token**。
- tokenizer 来自 `config.vlm_model_name`（SmolVLM2 的 tokenizer），padding 张量在 tokenizer 里做（`tokenizer_processor.py:265-272`）。
- 前缀布局：`[图像 tokens] + [48 语言 tokens] + [1 state token]`，语言 padding 位置 `pad_mask=False`（被 attention mask 排除）。
- checkpoint 的 `policy_preprocessor.json` 里 `padding="max_length"`、`truncation=true`、`padding_side="right"`。

### 4.5 图像：resize 512×512 + 左/上 padding → [-1,1]，每张 64 token

`modeling_smolvla.py:385-433`：

```python
# modeling_smolvla.py:400-422（节选）
img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]   # n_obs_steps=1 取最后一帧
if img.dtype == torch.uint8:
    img = img.to(torch.float32) / 255.0
if self.config.resize_imgs_with_padding is not None:
    img = resize_with_pad(img, self.config.resize_imgs_with_padding[1],   # 注意：config 存 (width,height)
                          self.config.resize_imgs_with_padding[0], pad_value=0)
img = img * 2.0 - 1.0                                                     # [0,1] -> [-1,1]，SigLIP 输入
mask = batch[f"{key}_padding_mask"].bool() if f"{key}_padding_mask" in batch else torch.ones(bsize, dtype=torch.bool)
```

- **不用 HF 的 image processor**：直接 `vision_model(pixel_values=...)`，`patch_attention_mask=None`
  （`smolvlm_with_expert.py:202-215`）→ 黑边也被当成有效像素。
- `resize_with_pad`（`common/vla_utils.py:218-242`）：保持长宽比 resize，然后 `F.pad(..., (pad_w,0,pad_h,0))`
  → **左/上 padding**（openpi 的 `resize_with_pad_torch` 是居中 padding，SmolVLA 用左/上，见 vla_utils 注释）。
  padding 值 0（[0,1] 域）→ 映射后 = -1（黑）。
- 512/16 = 32×32 patch = 1024，pixel-shuffle 4 → `(32/4)² = 64` token/图；**实测 `embed_image` 输出 `(B,64,960)`**。
- 缺失相机（batch 里没有但 config 里有）：按 `empty_cameras` 补**全 -1 图 + mask=False**（`:424-432`）；
  libero/base 配置 `empty_cameras=0` → 直接不补。
- VLM 侧 embedding 缩放（论文没写）：图像 `img_emb *= sqrt(960)`（`:636-637`）、语言 `lang_emb *= sqrt(960)`（`:661-663`，
  与 openpi 的 `pi0` 一致）。自研实现若直接用 HF SmolVLM2 的 embedding 而不做这个缩放，会与 checkpoint 差的不是一点。

### 4.6 MEAN_STD 归一化在 processor 里怎么实现

`src/lerobot/processor/normalize_processor.py:352-365`：

```python
if norm_mode == NormalizationMode.MEAN_STD:
    mean, std = stats["mean"], stats["std"]
    denom = std + self.eps                       # eps 来自 policy_preprocessor.json: 1e-08
    if inverse:                                  # 反归一化（postprocessor）
        return tensor * std + mean               # 注意：反归一化用 std，不补 eps
    return (tensor - mean) / denom               # 前向
```

- 归一化对象：`state`(8 维) 与 `action`(7 维)；`VISUAL` 是 `IDENTITY`（`configuration_smolvla.py:32-38`），
  图像不做 dataset 统计归一化（只做 [-1,1]）。
- **归一化在 pad 之前**（processor 里是 8/7 维，模型里才 pad 到 32）。
- 正/逆变换的 eps 不对称（前向 `/(std+1e-8)`，逆向 `*std`），量级 1e-8 可忽略，但如果你手写对齐要注意。
- checkpoint 的实际统计量（`policy_preprocessor_step_5_normalizer_processor.safetensors`）：
  `action.mean/std` 形状 (7,)，`observation.state.mean/std` 形状 (8,)；例如 `state.mean=[-0.0465,0.0344,0.7646,2.9722,
  -0.2205,-0.1256,0.0269,-0.0272]`、`action.std=[0.3355,0.3784,0.4447,0.0392,0.0634,0.0780,0.9988]`。
  文件里还存了 min/max/q01/q99 等，但 MEAN_STD 只用 mean/std。

---

## 5. 参数量：按代码结构实算 + 与论文 450M 对账

### 5.1 统计方法（两种互相验证）

**方法 A：直接数 checkpoint 张量**（最可靠，不经任何框架构建）——`safetensors.safe_open` 读 `model.safetensors` 的每个
tensor shape，`numel` 求和并按模块前缀分组。checkpoint 的权重文件 `model.safetensors` 是 865MB（474 个 BF16 + 26 个 F32）。

**方法 B：按 config 构建模型再数**（用于交叉验证与"分组归因"）：

```python
# 关键：把 vlm_model_name 指到本地 snapshot 目录、load_vlm_weights=False，即只建结构不加载 1.9GB 权重
cfg = SmolVLAConfig(vlm_model_name=<local SmolVLM2 snapshot>, load_vlm_weights=False,
                    num_vlm_layers=16, num_expert_layers=0, self_attn_every_n_layers=2,
                    expert_width_multiplier=0.75, attention_mode="cross_attn",
                    prefix_length=0, device="cpu")
model = VLAFlowMatching(cfg)
sum(p.numel() for p in model.parameters())     # 450,046,176
```

两种方法**逐字节一致**：`450,046,176`。第三种旁证：第三方对该架构的模型卡也写着
"Total parameters: 450,046,176 (450M) / Learnable parameters: 99,880,992 (100M)"（见文末 [refs]）。

### 5.2 参数明细（方法 A/B 结果一致，单位：个）

| # | 组件 | 参数量 | 计算式 / 说明 |
|---|---|---:|---|
| A | VLM: SigLIP vision encoder（12 层，d=768，patch 16，输入 512） | **86,433,024** | 每层 7,087,872 ×12 + 非层 1,378,560（patch_embed 590,592 + pos_emb 786,432 + post_ln 1,536） |
| B | VLM: connector（pixel-shuffle 4：768→12288，linear→960，无 bias） | **11,796,480** | 960×12288 |
| C | VLM: text embed_tokens | **47,308,800** | 49280×960 |
| D | VLM: text transformer **16 层**（d=960, ffn=2560, 15 heads/5 kv, head_dim 64） | **157,317,120** | 每层 9,832,320 = q/k/v/o (921,600+307,200+307,200+921,600) + MLP 3×2,457,600 + 2×960 |
| E | VLM: text final RMSNorm | **960** | |
| F | VLM: **lm_head**（49280×960，**冻结且前向从不调用**） | **47,308,800** | 与 embed_tokens 不 tie（`tie_word_embeddings=false`，实测 data_ptr 不同） |
| G | action expert **16 层**（d=720, ffn=2048, 15 heads/5 kv） | **98,245,120** | SA 层(偶数) 6,268,320×8 + CA 层(奇数) 6,012,320×8 + final norm 720 |
| H | state_proj 32→960 (+bias) | **31,680** | |
| I | action_in_proj 32→720 (+bias) | **23,760** | |
| J | action_out_proj 720→32 (+bias) | **23,072** | |
| K | action_time_mlp_in 1440→720 (+bias) | **1,037,520** | |
| L | action_time_mlp_out 720→720 (+bias) | **519,120** | |
| | **合计** | **450,046,176 = 450.05M** | ✓ 与论文 "450 million parameters" 对账完全一致 |

**VLM 部分合计**：86,433,024 + 11,796,480 + 47,308,800 + 157,317,120 + 960 + 47,308,800 = **350,165,184（350.2M）**；
expert + 5 个投影 = **99,880,992（99.9M）**。

**expert 每层明细**（用于理解 0.75 的实际形状）：

| | SA 层（偶数 0,2,…,14，共 8 层） | CA 层（奇数 1,3,…,15，共 8 层） |
|---|---:|---:|
| q_proj 720→960 | 691,200 | 691,200 |
| k_proj | 720→320：230,400 | **320→320：102,400** |
| v_proj | 720→320：230,400 | **320→320：102,400** |
| o_proj 960→720 | 691,200 | 691,200 |
| MLP gate/up/down（720↔2048） | 3×1,474,560 = 4,423,680 | 4,423,680 |
| 2×RMSNorm(720) | 1,440 | 1,440 |
| 小计 | **6,268,320** | **6,012,320** |

### 5.3 与论文 450M 对账：差异来源

- **对得上的原因**：450.05M 是 `nn.Module` 全部 `numel` 之和，**包含**：
  - vision encoder 86.4M（论文说 SmolVLA 用 SmolVLM2-500M 的视觉编码器，应该含）；
  - **未使用的 lm_head 47.3M**（SmolVLM2 的 lm_head 与 embed 不 tie，代码保留但冻结、前向不调用）；
  - 冻结的 VLM 其它层（base 里 VLM 全冻结）。
- 因此常见的错误估算会偏差：
  - 若按"500M VLM × (16/32) + expert"粗算 ≈ 250M + 100M，会得到 ~350M —— 漏掉了 vision encoder（86M）、
    connector（12M）、embed（47M）、lm_head（47M）这些**不随 LLM 层数缩减**的部分；
  - 若把 lm_head 去掉算"有效参数"：**402,737,376（402.7M）**；
  - 论文 §4.3 "with approximately 100 million dedicated to the action expert" 对应的是
    **expert 98.25M + 投影 1.64M = 99.88M**（不是 100M 里再分状态投影等）。
- 对照：底座 `SmolVLM2-500M-Video-Instruct` 本身 = **507,482,304（507.5M）**（32 层 text + 全量 vision/connector/lm_head，实测），
  SmolVLA 只保留其 **69%** 的参数（350.2M/507.5M），再加 expert 得到 450M。

### 5.4 可训练参数（两种训练制度，实算）

| 制度 | 配置 | 冻结了什么 | 可训练参数 |
|---|---|---|---|
| **论文 / base 配置** | `freeze_vision_encoder=True`, `train_expert_only=True` | 整个 `vlm` + vision encoder（`smolvlm_with_expert.py:152-160`），只训 expert + 5 投影 + state_proj | **99,880,992（99.88M）** |
| **libero checkpoint 记录的配置** | `freeze_vision_encoder=false`, `train_expert_only=false` | 只冻 `lm_head`、`text_model.norm.weight`、VLM **最后一层（索引 num_vlm_layers-1 = 15）**（`smolvlm_with_expert.py:161-187`） | **392,904,096（392.9M）** |

第二种制度冻最后一层的理由（代码注释）：训练前向只用 expert 输出 `suffix_out`，VLM 最后一层的 q/o/MLP 输出被丢弃，
在 DDP 下会"unused parameters"报错（`:161-187` 与 `:162-168` 注释）。**注意这个 `Trainable` 数是 checkpoint 记录的配置，
即发布版 libero 微调实际解冻了 VLM/vision**，与论文"train only the action expert"不同（见 §6.2）。

---

## 6. base 与 libero 微调的差异

### 6.1 配置差异（`lerobot/smolvla_base` vs `lerobot/smolvla_libero`，均来自 Hub 实际 config.json）

| 字段 | smolvla_base | smolvla_libero | 含义 |
|---|---|---|---|
| `input_features` | state **6** 维 + 相机 **3** 个（camera1/2/3, 3×256×256） | state **8** 维 + 相机 **2** 个 | state/action 维度与相机数不同 |
| `output_features.action` | 6 维 | 7 维 | |
| `freeze_vision_encoder` | **true** | **false** | libero 解冻了 SigLIP |
| `train_expert_only` | **true** | **false** | libero 解冻了整个 VLM（除 lm_head/norm/最后一层） |
| `scheduler_decay_steps` | 30000 | 25000 | libero 缩短了 cosine |
| `empty_cameras` | 0 | 0 | |
| 其余架构超参 | 全部相同：`chunk_size=50, n_action_steps=50, max_state_dim=32, max_action_dim=32, resize=(512,512), tokenizer_max_length=48, num_steps=10, use_cache=true, load_vlm_weights=true, add_image_special_tokens=false, attention_mode="cross_attn", prefix_length=0, pad_language_to="max_length", num_expert_layers=0, num_vlm_layers=16, self_attn_every_n_layers=2, expert_width_multiplier=0.75, min_period=4e-3, max_period=4.0` | 同左 | **架构完全一致** |

### 6.2 从 base 加载时到底加载了哪些参数

- 训练入口：`lerobot-train --policy.path=lerobot/smolvla_base ...` → `TrainPipelineConfig._resolve_pretrained_from_cli`
  （`src/lerobot/configs/train.py:216-241`）**先按 base 的 config.json 建配置**，再叠加 CLI overrides，并把
  `pretrained_path` 记成 base（`configs/policies.py:243-246` 的 backfill）。
- `make_policy` 在 `cfg.pretrained_path` 存在时走 `policy_cls.from_pretrained(**kwargs)`（`policies/factory.py:396-411`），
  `from_pretrained` 默认 **`strict=False`**（`policies/pretrained.py:197-209`），加载后只 log missing/unexpected keys
  （`pretrained.py:319-351`）。
- base 与 libero 的**参数形状完全相同**（层数/宽度/max_state_dim=32/max_action_dim=32 都一样；6→8、6→7 维只影响
  输入向量长度，而 `state_proj/action_in_proj/action_out_proj` 的权重形状是 `(960,32)/(720,32)/(32,720)`，与真实维数无关）
  → 两者 `state_dict` 的 key 集合与每个张量形状一一对应，**预期 500 个张量全部匹配、无 missing/unexpected**
  （`from_pretrained(strict=False)` 只会在出现不匹配时 log，不会报错；建议复现时打开日志确认一遍）。
  也就是说 libero = base 的权重 + 25k 步 libero 微调。
- ⚠️ `load_vlm_weights=true` 在 `from_pretrained` 建模型时仍会先下载/加载一遍 SmolVLM2-500M（`smolvlm_with_expert.py:90-97`），
  然后被 base 权重覆盖。离线复现时要么准备 SmolVLM2 缓存，要么临时把该字段置 false（但会改变随机初始化路径，
  只要紧接着加载 base 全量权重，结果等价）。CI 脚本里显式传了 `--policy.load_vlm_weights=true`（`scripts/ci/run_sim_benchmark.py:178`）。

### 6.3 发布版 libero 训练超参 vs 论文 §4.3

`train_config.json`（本地缓存快照）实际记录：

| 超参 | 论文 §4.3（simulation fine-tune） | 发布版 smolvla_libero（train_config） | 备注 |
|---|---|---|---|
| steps | **100,000** | **25,000** | 差 4× |
| batch_size | **64** | **32** | 差 2× |
| 是否冻结 VLM | 论文 §4.3：pretrain "train only the action expert, keeping the VLM frozen" | `train_expert_only=false, freeze_vision_encoder=false`（**全解冻**） | 见 §5.4 |
| warmup | 论文 pretrain 用 100 步 | `scheduler_warmup_steps=1000` | 代码默认 1000 |
| peak LR | 1e-4 | 1e-4 | ✓ |
| min LR | 2.5e-6 | 2.5e-6 | ✓ |
| optimizer | AdamW β=(0.9,0.95) | 同（`optimizer_betas=[0.9,0.95], eps=1e-8, wd=1e-10, clip=10`） | ✓（论文只写了 β） |
| scheduler decay steps | 未写 | 25,000 | 与 steps 对齐 |
| `num_steps`（推理 Euler） | 10 | 10 | ✓ |
| chunk n | 50 | 50 | ✓ |
| 图像尺寸 | 512×512 | 512×512 | ✓ |
| 精度 | bf16 + torch.compile() | `use_amp=false`；VLM/expert bf16、投影 fp32；`compile_model` 未开 | 见 §7 坑 3/4 |
| 数据集 | 论文 §4.5 用多任务 LIBERO 训练 | `lerobot/libero`（1693 eps / 273,465 frames / 40 tasks, 2 cams, state 8/action 7） | |
| `rename_map` | — | `observation.images.image→camera1`, `image2→camera2` | 相机顺序即前缀 token 顺序（影响"哪张图在前"） |
| `seed` | — | 1000 | |
| `num_workers` | — | 4 | |

官方 LeRobot 自己的 "smolvla_libero 复现配方"在 CI 里（`scripts/ci/run_sim_benchmark.py:175-181`），除 batch/steps 外与
发布版记录的 flag 一致（`--policy.path=lerobot/smolvla_base --policy.load_vlm_weights=true
--policy.scheduler_decay_steps=25000 --policy.freeze_vision_encoder=false --policy.train_expert_only=false`），
但 CI 的 smoke 只跑 `--steps=1 --batch_size=1`。另外 `AGENT_GUIDE.md:328-336` 明确建议"SmolVLA 默认冻结 vision encoder；
解冻通常显著涨点"，与发布版 checkpoint 的选择一致。

**复现结论（重要）**：若目标是"复现论文的 SmolVLA 微调配方"，应冻结 VLM（trainable≈100M）并训练 100k 步/bs64；
若目标是"复现发布的 `lerobot/smolvla_libero` 权重"，则必须按 checkpoint 记录解冻 VLM（trainable≈393M）且只训 25k 步/bs32。
两者不可混用，否则 loss 曲线/成功率的对照会失真。

---

## 7. 实现层面的坑（复现时最容易踩的点，按危险程度排序）

### 坑 1：`use_cache=False` 时推理直接崩（cross_attn 模式实际不可关 cache）

`sample_actions` 前缀预填充 `inputs_embeds=[prefix_embs, None]`（`modeling_smolvla.py:806`）。若 `use_cache=False`：
`forward` 里 `fill_kv_cache = use_cache and past_key_values is None` 为 False（`smolvlm_with_expert.py:441`），
偶数层走 `forward_attn_layer` 正常，**奇数层走 `forward_cross_attn_layer` 且 `past_key_values=None`**，
该函数会进入"prefix attention"分支（`smolvlm_with_expert.py:313`），随后在 expert 分支对 `inputs_embeds[1]=None`
调用 `layer.input_layernorm(None)` → 崩。**实测报错**：

```
sample_actions (use_cache=False) FAILED: AttributeError 'NoneType' object has no attribute 'dtype'
```

→ 复现时不要试图关 `use_cache` 省显存；训练路径不受影响（训练 forward 硬编码 `use_cache=False` 但输入含 suffix）。

### 坑 2：`num_expert_layers=0` 不是 0 层，`max_state_dim`/`max_action_dim` 也不是真实维数

- `num_expert_layers=0`（checkpoint 值）→ 走 `if num_expert_layers > 0` 的 else，expert 层数 = `num_vlm_layers` = **16**
  （`smolvlm_with_expert.py:112-118`）。`configuration_smolvla.py:96` 的默认值是 `-1`，语义相同（"same number of layers"）。
- `max_state_dim=max_action_dim=32` 是**统一的 padding 上限**，LIBERO 真实是 state 8 / action 7。若照抄 config 但忘了
  `prepare_state/prepare_action` 的 pad，张量形状虽然能跑通（`state_proj` 期望 32），loss 语义会不同。

### 坑 3：`use_amp=false` 不等于"训练用 fp32"，bf16 来自模型构造

- `policy.use_amp` 在本 fork 里**只影响 eval/inference 的 autocast**：
  `scripts/lerobot_eval.py:889`、`inference/sync.py:107,166` 用它包 `torch.autocast`。
- **训练精度由 `--accelerator.mixed_precision` 控制**（默认 `"no"`，`src/lerobot/configs/accelerator.py:348`；
  训练循环 `lerobot_train.py:244` 用 `accelerator.autocast()`）。libero 的 train_config 没有 accelerator 段 → 默认 "no"。
- 模型里的 bf16 来自 `SmolVLMWithExpertModel` 加载 VLM 时的 `torch_dtype="bfloat16"`（`smolvlm_with_expert.py:92-96`）
  以及 expert 从 `text_config.torch_dtype=bfloat16` 构建。**所以「use_amp=false」并不意味着权重是 fp32**。
- 实测 checkpoint dtype：**474 张 BF16 + 26 张 F32**。26 张 F32 是：5 个投影层（state/action in/out/time_mlp 共 10 张）
  + **8 个 CA 层的 k_proj/v_proj（16 张，320×320）**——这些是构造后新 `nn.Linear` 默认 fp32。
  → 训练/推理是**混合 dtype**：VLM 与 expert 主体 bf16，投影与 CA k/v fp32；代码在各层用
  `hidden_states.to(dtype=q_proj.weight.dtype)`（`smolvlm_with_expert.py:244`）显式对齐，不要"顺手"把全模型 `.to(bf16)`，
  否则加载权重/数值路径会变。

### 坑 4：attention 只有手写 eager（无 sdpa/flash），且 mask 是 fp32 全矩阵

`get_attention_interface` 恒返回 `eager_attention_forward`（`smolvlm_with_expert.py:521-523`）；softmax 前 upcast fp32
（`:548-561`）。batch=32、序列 227 时每层 SA 的 attention 矩阵约 `32×15×227×227×4B ≈ 99MB`，比 flash/sdpa 费显存。
同时 `torch.compile` 默认关闭（`compile_model=False`，libero config 无该字段），论文说的 "torch.compile()" 优化在发布配方里**没有生效**。

### 坑 5：RoPE 的 theta 是 10_000，而 SmolVLM2 config 是 100_000

`apply_rope(x, positions, max_wavelength=10_000)`（`smolvlm_with_expert.py:41`），调用处（`:269-270, :332-333, :383`）
从不传参。SmolVLM2 的 `text_config.rope_theta=100000`。凡是 released checkpoint 都是按 10_000 训练的，**复现必须保持 10_000**；
对"为什么和 HF 原生 SmolVLM2 的输出对不上"这个问题，这是原因之一（该实现整体不是 HF SmolVLM2 的前向，而是 openpi 风格的重写）。

### 坑 6：动作 padding 只 mask loss，不 mask attention；模型层 32 维 loss 会先算再裁

- suffix 的 `pad_masks` 恒为 1（`modeling_smolvla.py:730`），所以 25 个 padding 动作维在 attention 里**完全可见**。
- 模型层 MSE 在 32 维上算（`modeling_smolvla.py:773`），policy 层才裁到 7 维（`:325`）——若你直接调
  `VLAFlowMatching.forward()` 做可视化/迁移学习，注意返回的 loss 不是论文意义下的 action loss。
- 同理 `action_is_pad`（episode 边界）只用来置零 loss，不影响 attention。

### 坑 7：论文 §3.1 公式的符号约定与代码相反

见 §1.2 第 1 条。**"x_t = t·ε + (1−t)·A、目标 ε−A、推理 10 步从 t=1 积到 t=0"** 三者是绑定的；只改其一必崩。

### 坑 8：论文说仿真里"每个动作后重新观测"，checkpoint 却是 `n_action_steps=50`

论文 §4.3 原文："For real-world evaluation, we perform synchronous inference: the model samples new observations only after
executing the full chunk of actions. **In simulation, we perform inference by sampling new observations and predicting a new
action after each executed action.**" —— 但 `smolvla_libero` 的 config 是 `chunk_size=50, n_action_steps=50`，
`lerobot-eval` 默认沿用 → 实际是"执行完 50 步再重新观测"。仓库文档里 `n_action_steps=10` 的示例只针对 pi0.5
（`docs/source/libero.mdx:209`，"matching the original OpenPI implementation"），不是 SmolVLA。
**建议复现实验把 `--policy.n_action_steps ∈ {1, 10, 50}` 作为消融项**（`select_action` 对任意 n≤chunk 都成立，见 §2.5）。

### 坑 9：`prefix_length=0` 意味着"不固定前缀长度"，改相机数/语言长度会改序列长度

见 §3.6。若为部署做 ONNX/TensorRT/torch.compile 固定 shape，需要自己补 prefix padding（原版 openpi 用 `prefix_length` 干这个）。
LIBERO（2 相机）前缀 177，base（3 相机）前缀 241。

### 坑 10：图像预处理不用 HF processor，黑边参与 attention；颜色域是 [-1,1]

- 直接喂 `vision_model(pixel_values=...)`，`patch_attention_mask=None`（`smolvlm_with_expert.py:202-212`）→ padding 黑边不算 mask。
- 输入必须是 `[0,1]` 再映射到 `[-1,1]`；若你的数据管线给的是 ImageNet 归一化（mean/std）图，结果会明显变差。
- `resize_imgs_with_padding` 在 config 里存的是 **(width, height)**，而 `prepare_images` 里传参时做了 `[1], [0]` 交换
  （`modeling_smolvla.py:404-410` 注释也点明）；因为 libero/base 都是 (512,512) 所以看不出问题，改成非方形时要小心。

### 坑 11：state/action 的归一化与 padding 顺序、以及"空相机"的 -1 填充

- 先 `MEAN_STD` 归一化（8/7 维），后补零到 32 维。若顺序颠倒（先 pad 再归一化），零 padding 会被减均值/除 std，语义全错。
- 缺失相机时补的是**全 -1 图（即 [0,1] 域的 0，黑）**且 `mask=False`（`modeling_smolvla.py:424-432`）；
  `empty_cameras` 控制最多补几个。libero/base 都是 0。

### 坑 12：`self_attn_every_n_layers` 的名字具有误导性

它不是"每 2 层跳过 1 层"，也不是 VLM 内部的 SA/CA；它只决定**该层用联合 self-attention（SA 路径）还是 expert 交叉注意力（CA 路径）**
（`smolvlm_with_expert.py:448-476`）。设为 `-1/0` 时所有层都走 CA（等价于纯 π0 式 cross-attn 变体）。
论文表 6 的 CA/SA 消融（CA 79.0 / SA 74.5 / CA+SA 85.5）对应的就是这个开关。

### 坑 13：训练时冻结策略决定 DDP 是否报 "unused parameters"

`train_expert_only=false` 时刻意冻结了 `lm_head`、`text_model.norm.weight`、VLM 最后一层（`smolvlm_with_expert.py:161-187`），
因为它们的输出不进入 loss。若你的并行方案（DDP/FSDP）报 unused params，先检查这里；反过来，若你自己写训练循环却只冻
"最后一层"但不冻 `lm_head/norm`，DDP 可能直接报错。

### 坑 14：`policy_preprocessor.json` 的 `rename_map` 与相机顺序

`observation.images.image → camera1`、`image2 → camera2`（`train_config.json` 的 `rename_map`）。
相机进入前缀的顺序影响 attention（先 camera1 再 camera2），如果复现时把两个相机顺序反了，性能会掉但不报错。
另外 `SmolVLA` 对相机数是**软约束**（多余的 config 相机会被 `missing_img_keys`+`empty_cameras` 逻辑丢弃/补空图），
所以"能跑通"不代表"输入正确"。

---

## 8. 复现对照清单（按代码顺序做 sanity check）

1. **数据**：batch 里必须有 `observation.images.camera1/2`（HWC uint8 或 [0,1] float）、`observation.state`(8)、
   `action`(7)、`action_is_pad`(B,50)、`task`(str)。缺失 `action_is_pad` 时 loss 语义变化（§1.2 第 5 点）。
2. **预处理**：`rename → batch → newline → tokenize(max_length=48) → device → MEAN_STD`；`action` 反归一化在 postprocessor。
3. **张量形状**：prefix 177（2 相机）、suffix 50、联合 227；`x_t (B,50,32)`；图像 512×512 [-1,1]。
4. **loss 自检**：模型层返回 `(B,50,32)` 逐元素 MSE；policy 层输出标量 = `Σ(masked loss)/ (有效步数×7)`。
5. **时间采样自检**：`t.mean()≈0.60`（Beta(1.5,1)*0.999+0.001），不是均匀分布。
6. **推理自检**：Euler 评估时刻 t=1.0,…,0.1（10 次）；首步是纯噪声；`use_cache` 必须 True。
7. **参数自检**：`sum(p.numel()) == 450,046,176`；冻结制度见 §5.4（论文 99,880,992 / 发布 ckpt 392,904,096）。
8. **对齐论文数字时**：先确认冻结策略（§6.2）、步数/batch（§6.3）、`n_action_steps`（§7 坑 8）三件事，否则成功率不可比。

---

## 附录 A：关键行号索引（相对 `/home/lxy/lerobot-smolvla-repro/lerobot`）

| 主题 | 文件:行 |
|---|---|
| 训练 forward / loss | `src/lerobot/policies/smolvla/modeling_smolvla.py:741-774`，`:293-354` |
| 时间/噪声采样 | `modeling_smolvla.py:598-602`；`src/lerobot/policies/common/flow_matching.py:105-185` |
| flow matching 插值/目标/欧拉 | `common/flow_matching.py:188-253`，`:256-421` |
| embed_prefix（图像/语言/state/att_masks/prefix_length） | `modeling_smolvla.py:604-696` |
| embed_suffix（动作+时间 MLP） | `modeling_smolvla.py:698-739` |
| sample_actions / denoise_step / KV cache 裁剪 | `modeling_smolvla.py:776-863` |
| select_action / 队列 | `modeling_smolvla.py:172-176, 255-288` |
| prepare_images / state / action | `modeling_smolvla.py:385-433, 462-471` |
| VLM 加载/截断/expert 构造/宽度 | `smolvlm_with_expert.py:88-147` |
| 冻结策略 | `smolvlm_with_expert.py:152-200` |
| SA 层实现 | `smolvlm_with_expert.py:220-292` |
| CA 层实现（K/V 注入、位置平移） | `smolvlm_with_expert.py:294-398` |
| 层配对 | `smolvlm_with_expert.py:400-412` |
| 层路由（SA/CA 交替）+ 残差 | `smolvlm_with_expert.py:414-519` |
| 手写 attention | `smolvlm_with_expert.py:521-569` |
| RoPE（max_wavelength=10_000） | `smolvlm_with_expert.py:41-64` |
| attention mask 构造 | `common/vla_utils.py:60-89` |
| pad_vector / resize_with_pad / 正弦时间编码 | `common/vla_utils.py:116-138, 218-242, 41-57` |
| 配置默认值 | `src/lerobot/policies/smolvla/configuration_smolvla.py:28-108` |
| processor 流水线 | `src/lerobot/policies/smolvla/processor_smolvla.py:63-81` |
| tokenizer / newline / MEAN_STD | `src/lerobot/processor/tokenizer_processor.py:255-272`；`newline_task_processor.py:34-54`；`normalize_processor.py:352-365` |
| `action_is_pad` 生成 | `src/lerobot/datasets/utils.py:146-160`；`src/lerobot/datasets/dataset_reader.py:300-310` |
| 训练循环 AMP / pretrained 加载 | `src/lerobot/scripts/lerobot_train.py:244`；`policies/factory.py:396-411`；`policies/pretrained.py:197-209, 319-351` |
| accelerator 精度默认值 | `src/lerobot/configs/accelerator.py:348` |
| 官方 libero 复现命令 | `scripts/ci/run_sim_benchmark.py:175-181`；`AGENT_GUIDE.md:328-336` |

## 附录 B：本文用到的实测脚本（可复跑，均只读、不改仓库）

**(1) checkpoint 参数普查（不经框架）**

```bash
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python - <<'PY'
from safetensors import safe_open
import collections
CKPT="/home/lxy/lerobot-smolvla-repro/hf-cache/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d/model.safetensors"
f=safe_open(CKPT, framework="pt"); total=0
for k in f.keys():
    n=1
    for s in f.get_slice(k).get_shape(): n*=s
    total+=n
print(len(list(f.keys())), "tensors,", f"{total:,}", "params")
PY
```

**(2) 按 config 建结构数参数（离线）**

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/lxy/miniconda3/envs/lerobot-smolvla-repro/bin/python - <<'PY'
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
LOCAL="/home/lxy/lerobot-smolvla-repro/hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/4cde9d208e19519c36917aa170fffc7e381aa0ad"
cfg=SmolVLAConfig(vlm_model_name=LOCAL, load_vlm_weights=False, num_vlm_layers=16, num_expert_layers=0,
                  self_attn_every_n_layers=2, expert_width_multiplier=0.75, attention_mode="cross_attn",
                  prefix_length=0, freeze_vision_encoder=False, train_expert_only=False, device="cpu")
m=VLAFlowMatching(cfg)
print(sum(p.numel() for p in m.parameters()))                                  # 450046176
print(sum(p.numel() for p in m.parameters() if p.requires_grad))               # 392904096
PY
```

**(3) 其它已验证事实**：`load_vlm_weights=True` 构建后 dtype 计数 = 474 bf16 + 26 fp32（与 checkpoint 完全一致）；
`embed_image` 输出 `(B,64,960)`；prefix `(B,177,960)`；forward 返回 `(B,50,32)` loss；`sample_actions` 返回 `(B,50,32)`；
`sample_time` 200k 采样 mean=0.6017；`use_cache=False` 推理抛 `AttributeError`；Euler 时刻 = `[1.0,0.9,…,0.1]`。

## [refs]

- 论文：<https://arxiv.org/abs/2506.01844>（SmolVLA: A vision-language-action model for affordable and efficient robotics,  
  Shukor et al., 2025；§3.1 Flow matching / Interleaved CA-SA，§4.3 Implementation details，§4.7 消融）
- 基座配置（实测拉取）：<https://huggingface.co/lerobot/smolvla_base/raw/main/config.json>
- 发布版 libero 配置/训练配置（本地缓存）：`.../models--lerobot--smolvla_libero/snapshots/a24f1f38.../{config.json,train_config.json,policy_preprocessor.json}`
- 第三方模型卡对参数量数字的独立交叉验证："Learnable parameters: 99,880,992 (100M) / Total parameters: 450,046,176 (450M)"：
  <https://zenodo.org/records/23685969/preview/MODEL_CARD.md?include_deleted=0>；以及综述笔记
  <https://guidongnan.github.io/notes/papers/2506.01844>（"450M 总参数、action expert 约 100M、训练时 VLM 冻结"）
