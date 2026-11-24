---
title: "独立审查报告：T1′/T3′ 近邻工作全文核查"
reviewer_role: "科研分析（不执行代码、不覆盖实验数据）"
review_date: "2026-11-01"
reviews: ["research/topics/candidate_topics.md", "research/topics/experiment_plan.md"]
trigger: "V11（VLAQuantBench 是否扫过执行长度）+ V12（AAC/KWS 是否建模延迟约束）"
evidence_level: "全文级（arXiv HTML 正文抓取），非摘要级"
status: "待导师/作者确认后合并入 topics 文件"
---

# 独立审查报告：T1′ / T3′ 近邻工作全文核查

> [!info] 本报告的性质
> 这是**审查**，不是修改。按协作约定，`research/topics/` 下的文件在确认前不动。
> 证据等级：本次为**全文级**（抓取了 arXiv HTML 正文并检索原文），高于前几轮的摘要级。所有引用都标了小节号，可复核。

---

## 0. 审查方法与可复现性

| 项 | 内容 |
|---|---|
| 抓取对象 | `arxiv.org/html/2609.25376v1`（VLAQuantBench）、`2604.04161v1`（AAC）、`2609.00908v3`（Knowing When to Stop） |
| 抓取方式 | PowerShell `Invoke-WebRequest` + 去标签，落盘 `work/vlaquantbench.txt`（1097 行）、`work/aac.txt`（984 行）、`work/kws.txt`（657 行） |
| 检索词 | `execution length`, `open-loop`, `replan`, `chunk`, `latency`, `real-time`, `constraint`, `deadline`, `compute`, `fixed-horizon` |
| 已知局限 | ①公式被替换为 `[MATH]`，**部分模型名无法辨识**（原文用 `$\pi$` 记号）；②只做了关键词定位，未逐句通读 28 页；③表格内容被折叠为 `[TABLE]`，数值需回原文表格核对 |

---

## 1. 裁定摘要（先给结论）

| 问题 | 裁定 | 对方案的影响 |
|---|---|---|
| **V11**：VLAQuantBench 是否扫过执行长度 `E`？ | **已裁定：没有。** 唯一一次 `E` 扰动（50 vs 10）作用在 **π 系列**，只报定性「成功率下降」；OFT 侧固定 8 步。无 `E × 比特宽` 交叉 | T3′ **存活**，但必须改写边界与措辞（§3.1） |
| **V12**：AAC / KWS 是否建模延迟约束？ | **AAC 完全没有**（0 处延迟词）；**KWS 有 §IV-C 效率分析**，但只比较方法自身开销（单 A100，多采样 +29.5 ms/cycle），**没有**建立可行性边界 | T1′ **收窄但仍存活**，须降级为「测量与判定」（§3.2） |
| **V13**：量化是不是 12GB 的加速杠杆？ | **出现反证**：VLAQuantBench §5.6 实测「所有被测量化路径都比 BF16 **更慢**」 | **必须修正 T3′ 的价值主张**（§4） |

---

## 2. 全文级发现（逐篇）

### 2.1 VLAQuantBench（2609.25376）

#### 事实 F1：它确实把「预测长度」与「执行步数」区分开了（🟦 §6）

原文（§6 Evaluation-Stack Checks）：

> "**Executed action chunks.** Executing 50 actions before re-observing instead of the intended 10 changed the tested [MATH]-family control loop and reduced success. **Predicted chunk length and executed action count must be distinguished.**"

**这直接命中了上一轮审阅指出的 P/E 混淆问题**——而且该文是把它当作「评测栈正确性检查」提出的，不是当作研究变量。

#### 事实 F2：它的分析与回放使用**固定** `E`（🟦 附录 G）

> "All 20 reference rollouts succeed and supply **219 replanning calls, with ten executed actions between replans**. Each call predicts a non…"

即：开环回放固定 **10 个执行步**。`E` 是常数，**没有与比特宽交叉**。

#### 事实 F3：规模确认（🟦 §3 / 附录 B）

> "a controlled evaluation with **409 runs and 94,574 simulation episodes**"
> "The corpus contains 409 evaluation runs and 94,574 simulation episodes"

上一轮审阅转述的数字**完全属实**（此前我只能核实标题与页数）。

#### 事实 F4：**量化比 BF16 更慢**（🟦 §5.6，这是本次最重要的发现）

> 小节标题：**"Memory reduction does not imply lower latency"**
> "Real-kernel measurements on OpenVLA-OFT (**LLM only, RTX 4090, batch size one, 20 LIBERO episodes**). … **All tested quantized paths reduce peak allocated memory and take longer than BF16** under both control-call and full-inference timing."
> "Three independent repeat sessions preserve the finding that all quantized paths are slower than BF16. … largest within-method ranges are 0.38 ms for control calls and 3.00 ms for full inference"

**推理链（🟨）**：这推翻了「量化 → 更快 → 更适合实时」的默认叙事。在消费级单卡、batch=1 的小 batch 场景下，反量化开销与 kernel 不支持会吃掉收益，**只有显存收益是可靠的**。

#### 事实 F5：它明确区分了「控制调用」与「完整推理」（🟦 §5.6 / 附录 F）

> "Control calls include **cached actions**; full inference generates a new action chunk."

这正是 `E` 造成的两种延迟口径差异：执行缓存动作的步 vs 真正前向。**说明该文已经具备做这个区分的意识**，只是没把它做成研究问题。

#### 事实 F6：硬件与 sm_120 线索（🟦 §4 / 附录 H）

> "The OpenVLA-OFT and X-VLA LIBERO grids run on an **RTX 4090** and the two [MATH] grids run on an **RTX 5080 Laptop GPU**."

🟨 **推断**：RTX 5080 Laptop 属于 Blackwell（sm_120）世代，若该文在其上跑通了 π 系列推理，说明 Blackwell 上的 VLA 推理**链路存在**。⚠️ 但**不能据此断定我们的 5070 可直接复用**——驱动、CUDA、PyTorch 版本均未知，仍须 Gate 0 实测（🟥）。

#### 事实 F7：统计口径（🟦 §4）

> "Reported 95% **Wilson intervals** summarize binomial episode-sampling uncertainty, and they **do not include** uncertainty from training, checkpoint choice, or **task clustering**."
> "Each setting uses **200 episodes**, except X-VLA with 190."

🟨 这对我们有两层含义：①200 episode/配置是该领域的现实规模；②他们**自己承认**没处理 task clustering —— 这正是我们 v2 统计设计里「按 task 区组 + 配对」能补的地方（也是我们可以做得更严谨的点）。

#### 事实 F12：`50 vs 10` 扰动作用在 **π 系列**，且明确点名 OFT 用 8 步（🟦 §6，第二次精查）

> "Executing 50 actions before re-observing instead of the intended 10 changed the tested **π-family** control loop and reduced success. Predicted chunk length and executed action count must be distinguished. Other clients retain their own execution conventions; **OpenVLA-OFT's simulation adapter, for example, uses an eight-action chunk**."

✅ 这一句**独立印证**了我们从 `openvla-oft` 源码读到的 `NUM_ACTIONS_CHUNK = 8`（LIBERO）。
同时它也确认：`50 vs 10` 只做了**一次定性扰动**，没有扫描、没有与精度交叉、没有报数值。

#### 事实 F13：真实机器人的控制频率是 **15 Hz**，且强调 `P` 与 `E` 必须随部署客户端记录（🟦 附录 H.1）

> "The policy control rate is **15 Hz**. Predicted action chunks and the number of actions executed before re-observation are **distinct settings and must be recorded with the deployed client**."

🟨 这是本次最有操作价值的一条：它给了 `Δt ≈ 66.7 ms` 的**参照量级**（其 Franka Research 3 设置，**不是** OpenFleX）。
配合 OFT 的 `P = 8`：`P·Δt ≈ 533 ms` —— 即只要单次推理 `L < 533 ms`，理论上就能不间断闭环。
配合 π0.5 的 `P = 50`（见 F14）：`P·Δt ≈ 3.3 s`。
⚠️ 这组数字**只能当参照**，OpenFleX 的 `Δt` 仍未知（🟥 A5′）。

#### 事实 F14：不同模型的预测长度差异极大（🟦 表 34 说明）

> "**chunk MAE uses the full normalized 50-step output**"

🟨 即 π0.5 的 `P = 50`。与 OFT 的 `P = 8` 相差 6 倍——**「执行长度」这个变量在不同基座上的含义完全不同**，跨模型比较 `E` 的绝对值没有意义，只能比较 `E/P` 的比值。这一点必须写进实验设计。

### 2.2 AAC（2604.04161，CVPR 2026）

#### 事实 F8：完全没有延迟/实时约束建模（🟦 关键词检索）

- `latency` → **0 处**
- `real-time` → **0 处**
- `deadline` → **0 处**
- 唯一相关：`delay` 1 处，指「默认 20 个采样会引入约 **20 ms** 推理延迟」

#### 事实 F9：机制是「动作熵 + 最小动作幅值约束」（🟦 §3–§4）

> "we impose **a minimum action magnitude constraint** on the optimal action chunk size, thereby defining a dynamic threshold"

基座：**GR00T N1.5**（§3.1）。即：**AAC 没在 OpenVLA-OFT / LIBERO 上做**。

🟨 对我们的含义：`E` 的选择信号已经有两条独立路线（熵、cross-attention），**但都是在「精度-效率」空间里选，没有一条把「能不能按时算完」作为约束**。

### 2.3 Knowing When to Stop（2609.00908）

#### 事实 F10：有延迟分析，但是「方法自身开销」比较（🟦 §IV-C Efficiency Analysis）

> "We analyze the computational latency for a single action chunk inference cycle on a **single NVIDIA A100 GPU** … the multi-sampling (MS) method increases the latency of each inference cycle by **29.5 ms**. Although MS employs parallel execution… such overhead remains considerable for **high-frequency closed-loop control**."

🟨 定性：它把延迟当作**选择方法的成本**来比，落点是「我们的信号比多采样便宜」。它**没有**：
- 建立 `L ≤ E·Δt` 这类可行性判据；
- 讨论可行域为空的情形；
- 在消费级 12GB 卡上做。

#### 事实 F11：它也扫了固定执行长度（🟦 §IV-A）

> "X-VLA generates a candidate sequence of [MATH] actions at each inference step. We evaluate **fixed execution horizons of 15, 20, 25 and 30**, while omitting shorter horizons due to their low success rates."

🟨 含义：**「固定 horizon 的扫描」在这两篇里都作为 baseline 出现过**。所以「扫 E」本身**绝对不是贡献**，只能作为复现基线与对照。

---

## 3. 对 T1′ / T3′ 的裁定与改写建议

### 3.1 T3′（量化 × 开环执行长度）：**存活，但边界必须改写**

| 项 | 裁定 |
|---|---|
| VLAQuantBench 是否做过「量化 × E」交叉？ | **否**（F2：固定 10 步；F1：`50 vs 10` 只是一次协议扰动，未与比特宽交叉） |
| 是否已有间接证据？ | **是**（F1：执行 50 步代替 10 步**降低了成功率**——方向与 H3′ 一致，但混淆了控制环改变） |
| 核心主张是否仍成立？ | **成立但收窄**：不再是「量化损失与开环长度无关/有关」的发现，而是「**在必须量化才能装进 12GB 的硬件上，量化惩罚随执行长度如何变化**」 |

**必须做的三处改写**：

1. **把 F1 写进 Related Work 并明确区分**：该文已指出「预测长度 ≠ 执行步数」，且观察到长执行降低成功率；我们的增量是**把 `E` 作为因子与比特宽交叉**，而不是重复这个观察。
2. **删掉「量化带来加速」的任何暗示**：F4 已证明消费级单卡上量化更慢。T3′ 的正当性改为「**12GB 装不下 bf16，量化是准入条件而非加速手段**」。
3. **引用其统计口径**：F7 显示他们用 200 episode/配置且未处理 task clustering；我们的配对设计可以作为**方法学上的改进点**（但要先确认这不是他们附录里已做的）。

> [!success] 重合风险已排除（A1 闭环）
> 我抓取了该文全部 **39 条图表标题** 与 **8 处 `executed` 原文上下文**：
> - 没有任何表格以「执行长度」为因子；
> - 表 34 出现的 "executed-action MAE" 是**度量对象**（被执行动作的误差）区别于 "chunk MAE"（整块 50 步的误差），**不是 `E` 扫描**；
> - 唯一的 `E` 扰动是 π 系列的 `50 vs 10`，定性报告，未与精度交叉。
>
> **结论：T3′ 的「量化 × 开环执行长度」交叉，在本次检索边界内确实未被做过。**
> 剩余不确定性很低，但仍有：若其**公开代码**里含 `E` 的默认参数配置差异，需在复现时核对（→ A3）。

### 3.2 T1′（可行性边界）：**收窄但仍存活**

| 项 | 裁定 |
|---|---|
| 「扫固定 horizon」是否新颖？ | **否**（F11：KWS 扫 15/20/25/30 作为 baseline） |
| 「延迟分析」是否新颖？ | **否**（F10：KWS 有 §IV-C；F5：VLAQuantBench 区分两种延迟口径） |
| 「可行性判据 + 可行域为空」是否新颖？ | **🟥 未定位到**（三篇均未建立 `L ≤ P·Δt` 判据，也未讨论空可行域） |
| 新颖性等级 | **低**：属于「测量 + 判定」，不是新方法 |

**必须做的四处改写**：

1. **降级措辞**：从「我们发现最优 E」改为「**我们在 12GB 单卡上核对：精度最优的 E 是否实时可行**」。
2. **明确引用 F10/F11**：说明固定 horizon 扫描与延迟分析都已有人做，我们做的是**在目标硬件上、以 `Δt` 为约束的可行性判定**。
3. **增加「空可行域」的检验**：这是三篇都没碰的点，也是最可能出结果的点。
4. **接受 T1′ 可能只是审计性工作**：若结论是「一切可行、实时性不是瓶颈」，那就是负面结果，**不能包装成发现**。

---

## 4. 本次核查引出的方案级修正（比选题更重要）

| # | 修正 | 依据 | 影响范围 |
|---|---|---|---|
| M1 | **撤销「量化 = 加速」的隐含前提** | F4（量化实测更慢） | T3′ 的价值主张、`advisor_discussion.md` 的 Q14 |
| M2 | **`E` 的合法性来自代码，且官方警告偏离会掉点** | 上一轮已核实 `num_open_loop_steps` + WARNING | 已在 v2 写入，本次确认无误 |
| M3 | **统计口径需处理 task clustering** | F7（对方自述未处理） | 我们的配对/区组设计反而成为相对优势 |
| M4 | **sm_120 有旁证但不可外推** | F6（5080 Laptop 上跑过 π） | Gate 0 仍是 P0，但风险概率可下调 |
| M5 | **T3′ 的正当性改为「准入」而非「加速」** | F4 + 12GB 容量的算术推算 | 12GB 上 bf16 的 7B 模型放不下（≈14GB 权重），量化是**必要**而非**可选** |

🟨 M5 是本轮最有价值的重构：**在 12GB 上，量化不是「为了更快」，而是「为了能跑」。** 这把 T3′ 从一个可疑的加速故事，变成一个 unavoidable 的部署约束问题——而「量化后该执行多长」正是这个约束下的真问题。

---

## 5. 仍需人工确认的事项（我无法自动化完成）

| # | 事项 | 为什么必须人工 | 优先级 |
|---|---|---|---|
| ~~A1~~ | ~~通读 VLAQuantBench 附录 G / K~~ | ✅ **已由我完成**：抓取全部 39 条图表标题 + 8 处 `executed` 原文上下文，**未发现 `E` 扫描或 `E × 比特宽` 交叉**（见 F12/F14、表 34 仅区分「executed-action MAE」与「chunk MAE」两种度量） | 已闭环 |
| ~~A2~~ | ~~确认 `50 vs 10` 作用在哪个模型~~ | ✅ **已确认**：π 系列（F12 原文） | 已闭环 |
| A3 | 查 VLAQuantBench 是否公开代码/数据 | 409 runs 若可复现，我们的 baseline 成本大降 | P1 |
| A4 | 核对 F4 的绝对延迟数值（ms） | 我只拿到「更慢」的定性 + 0.38/3.00 ms 的组内极差，**没有中位数绝对值**；表 10 有具体数值但被折叠 | P1 |
| A5 | 确认 5080 Laptop 的具体 CUDA/PyTorch 版本 | 决定我们能否复用其环境配方 | P2 |
| **A5′** | **OpenFleX 的控制频率 `Δt`** | 这是 T1′ 可行性判据的唯一输入；目前只有第三方的 15 Hz 参照 | **P0** |

---

## 6. 给 Codex 的可执行任务（若采纳）

> 按分工，以下属于**执行侧**，我不代做。建议按序下发：

| # | 任务 | 产出 | 验收标准 |
|---|---|---|---|
| C1 | 在 `vlaquantbench.txt` 中定位附录 G/K 的 `E` 相关段落并报告 | 摘录 + 段落定位 | 能回答 A1 |
| C2 | 下载并解包 VLAQuantBench 的公开实现（若有） | 目录结构 + README 摘要 | 能回答 A3 |
| C3 | 复现其 RTX 4090 延迟测量脚本的最小版本 | 一份 `latency_probe.py` | 在我们 5070 上跑通 |
| C4 | Gate 0：打印 sm_120 + CUDA + PyTorch 版本并跑通最小前向 | `env_report.md` | B1–B7 全绿 |

---

## 7. 结论

**三篇近邻工作的全文核查，把 T1′ 和 T3′ 都从「有创新点」压到了「有明确边界的小增量」，但两者的存活状态不同：**

- **T1′（可行性边界）——存活但弱**：固定 horizon 扫描与延迟分析都已有人做（KWS 的 15/20/25/30 + §IV-C；VLAQuantBench 区分 control-call vs full-inference 两种延迟口径）。剩下的是「在 12GB 单卡、给定 `Δt` 下判定可行域是否为空」。**这是审计，不是发现。** 但它**恰好是 OpenFleX 远程推理架构最需要的那张表**（见 F13：`P` 与 `E` 必须随部署客户端记录）。
- **T3′（量化 × 开环长度）——存活且边界清晰**：`E` 未被作为因子与比特宽交叉（A1 已闭环）。VLAQuantBench 只在 §6 指出 P/E 必须区分、并定性观察到长执行降低成功率。**我们的增量必须限定为「交互效应」，并把 F1/F12 明确引用为先行观察。**
- **方案级（比选题更重要）**：量化在消费级单卡上**更慢而非更快**（F4），这要求彻底改写「量化 = 加速」的叙事，改为「**量化 = 12GB 准入条件**」。在 12GB 上 bf16 的 7B 模型放不下（权重 ≈14GB），所以量化不是优化项，是必要条件。

### 三条可直接用于开题的参照数字（🟨，非 OpenFleX 实测）

| 量 | 值 | 来源 | 用法 |
|---|---|---|---|
| 参照控制频率 | **15 Hz**（Δt ≈ 66.7 ms） | VLAQuantBench 附录 H.1（Franka Research 3） | 估算 `P·Δt` 的量级 |
| OpenVLA-OFT 的 `P` | **8**（LIBERO 适配器） | F12 原文 + `NUM_ACTIONS_CHUNK` 源码 | `P·Δt ≈ 533 ms` |
| π0.5 的 `P` | **50** | 表 34 说明 | `P·Δt ≈ 3.3 s` |

🟨 由此推得：**只要单次推理 `L < P·Δt` 就能不间断闭环**。若我们的 5070 上 `L` 落在 100–300 ms（待实测），OFT（`P·Δt ≈ 533 ms`）**大概率可行**，π0.5（`P·Δt ≈ 3.3 s`）**更加可行**——这意味着 **T1′ 的「可行域为空」分支可能不成立**，需提前准备叙事转换（转 T5 度量协议）。

**下一步建议**：先做 A5′（向导师/厂商确认 OpenFleX 的 `Δt`）与 A3（查 VLAQuantBench 是否开源）。这两项决定 T1′ 的论点是「发现空可行域」还是「给出可行域边界表」。**这个判断涉及研究叙事，建议连同本报告一起给王强老师看。**
