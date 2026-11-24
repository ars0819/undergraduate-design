# SmolVLA 实验数据数字化（arXiv:2506.01844v1）

> 论文：*SmolVLA: A vision-language-action model for affordable and efficient robotics*（Shukor, Aubakirova, Capuano, …, Cadene；Hugging Face / Sorbonne / valeo.ai / ENS Paris-Saclay；arXiv:2506.01844v1，2025-06-02，24 页）
>
> 本文件用途：SmolVLA 精读时的实验数据底稿。**只做转录 + 核对 + 批判，不引入外部数据**。
>
> 唯一写入文件：本文件。未修改工作区其它任何文件。

---

## 0. 来源、范围与转写约定

### 0.1 事实来源（按权威性排序）

| 来源 | 路径（工作区） | 用途 |
|---|---|---|
| arXiv HTML 全文（权威） | `paper/2506.01844.html` | 所有 caption、正文、表格结构的最终依据 |
| caption 清单 | `paper/tables-and-figures.md` | 交叉核对表号 ↔ caption（caption 在 LaTeXML 中会重复出现，已按 figure 结构配对） |
| 16 个非公式表格网格 | `paper/table-grids-raw.md` | 表格单元格原始文本；**按 id 配对 caption，不按相邻顺序** |
| HTML→Markdown 全文 | `paper/paper-wrapped.md`、`paper/2506.01844.md` | 正文/附录文字检索（两份内容一致，仅排版差异） |
| PDF（工作区） | `paper/2506.01844.pdf` | 开始时该文件为**截断件**（7,516,160 B，`pdfinfo` 报 `Couldn't find trailer dictionary / Couldn't read xref table`，ghostscript 仅恢复 1 页空白）；写本文件期间它已被替换为**完整 24 页 v1 版本**（8,921,356 B）。另存在 `paper/2506.01844-layout.txt`（同版 `pdftotext -layout` 文本）。本文件的页码即按这个完整 24 页版本核对 |
| PDF（备用，仅用于页码核对） | 从 arXiv 同一版本 v1 另行下载的 24 页 PDF（临时目录，未写入工作区） | 在完整件就位前用于**定位页码**与复核排版；与工作区完整件一致；内容与 HTML 一致（个别交叉引用渲染不同，见 C8） |

> **页码说明**：本文件中的 `p.N` 均指 arXiv:2506.01844**v1 完整 24 页 PDF** 的物理页码（与工作区当前 `paper/2506.01844.pdf` 一致）。若 PDF 与 HTML 渲染冲突，以 HTML 为准（已发现 1 处交叉引用差异，见 C8）。

### 0.2 表格清单与编号映射（16 个网格 = 13 张编号表 + Figure 5 的 3 张子表）

| 表号 | 网格 id（`table-grids-raw.md`） | PDF 页 | HTML anchor |
|---|---|---|---|
| Table 1 | `S3.T1.1.1` | p.5 | `S3.T1` |
| Table 2 | `S4.T2.3.1` | p.11 | `S4.T2` |
| Table 3 | `S4.T4.fig1.1.1` | p.11 | `S4.T4.fig1` |
| Table 4 | `S4.T4.fig2.1.1` | p.11 | `S4.T4.fig2` |
| Table 5 | `S4.T5.fig1.3.1` | p.12 | `S4.T5.fig1` |
| Figure 5(a) 性能 | `S4.F5.sf1.3.1` | p.12 | `S4.F5.sf1` |
| Figure 5(b) 完成时间 | `S4.F5.sf2.3.1` | p.12 | `S4.F5.sf2` |
| Figure 5(c) 固定时间完成量 | `S4.F5.sf3.3.1` | p.12 | `S4.F5.sf3` |
| Table 6 | `S4.T6.1.1` | p.13 | `S4.T6` |
| Table 7 | `S4.T7.1.1` | p.13 | `S4.T7` |
| Table 8 | `S4.T8.1.1` | p.13 | `S4.T8` |
| Table 9 | `S4.T9.1.1` | p.13 | `S4.T9` |
| Table 10 | `S4.T10.1.1` | p.14 | `S4.T10` |
| Table 11 | `S4.T11.1.1` | p.14 | `S4.T11` |
| Table 12 | `S4.T12.1.1` | p.14 | `S4.T12` |
| Table 13 | `S4.T13.1.1` | p.14 | `S4.T13` |

**注意**：Table 3 与 Table 4 是同一 LaTeX 浮动体（HTML 外层 `S4.T4`）里的两个并排 panel，各有独立 caption 与编号；Table 5 是独立浮动体。图 5 是"表格型图"，三张子表分别对应 caption (a)(b)(c)。全文没有附录表格，也没有公式表。

### 0.3 转写约定

1. 数字**按原文精度照抄**：如 `82.75`、`86.43`、`56.95`、`13.75`、`2.42`、`1.3`，不四舍五入、不补零。
2. 每格数字均来自 `table-grids-raw.md` 并用 HTML 原始 `<table>` 解析（含 `rowspan/colspan`）复核，二者一致；Table 2、Table 6–13 另与完整 PDF p.11/p.13/p.14 的 `pdftotext -layout` 排版逐格复核，一致。
3. 标「**存疑**」处均附原文片段与出处；「**本笔记推算**」表示我基于原文数字做的算术推导，**不是**论文原文。
4. 每表都给出：caption 原文 → 行列含义 → 完整数字表 → 支持的 claim/作者结论 → 数据来源与可比性 → 算术复核/备注。
5. 表内列名 `S / O / G / 10` 为 LIBERO 的四个 suite：Spatial / Object / Goal / **Long**（第 4 列在 Table 6–13 的原文表头直接印作 `10`，在 Table 2 印作 `Long`；这是原文写法，**不是我的笔误**，含义见 C25）。

### 0.4 论文没有提供的东西（先给结论，细节见 §9）

- 没有附录超参数表；训练细节全部挤在 §4.3 一段散文里（p.10）。
- 没有训练曲线、没有视觉 token 数/数据规模 ablation 表、没有真实机器人更多任务结果表、没有 per-task 明细表。
- 没有 π0.5、Dreamer 等 baseline（Table 2 只有 DP / Octo / OpenVLA / π0 / TinyVLA）；任务描述里提到的这些模型在本版论文中**不存在**。
- 没有 var/seed/置信区间；除 Figure 5(b)(c) 的 Std 外，所有成功率都是**单次运行的单个数**。
- 没有实测推理延迟（ms）、控制频率（Hz）、吞吐（FPS）数字；异步推理只有端到端任务耗时。

---

## 1. 逐表数字化

### 1.1 Table 1 — Community datasets statistics（p.5）

**caption 原文**（HTML `S3.T1`，p.5）：

> Table 1: Community datasets statistics. At $\sim$ 10M episodes, our pretraining set stands at least one order of magnitude smaller than other state-of-the-art. For completeness, we report the list of community-datasets used in Section A.1.

**列/行含义**：三列统计量，仅一行，描述用于预训练的社区数据集池：`# datasets` 数据集个数、`# episodes` episode 数、`# frames` 帧数。

**完整数字表**（p.5）：

| # datasets | # episodes | # frames |
|---|---|---|
| 481 | 22.9K | 10.6M |

**支持哪个 claim / 作者结论**：支撑"数据高效"核心叙事——用比 SOTA 小一个数量级的数据做 VLM→VLA 预训练（摘要 p.1；§1 p.2；§3.2 p.5）。§3.2 原文（p.5）："we selected a subset of 481 community datasets obtained from Hugging Face, filtered according to embodiment type, episode count, overall data quality, and frame coverage (Table 1)."

**数据来源**：作者自己从 Hugging Face/LeRobot 生态筛选的社区数据集；完整名单在 Appendix A.1（p.20–24）。不是引用别人的统计。

**算术复核/备注**：
- 本笔记推算：22.9K episodes / 481 datasets ≈ 47.6 episodes/dataset；10.6M frames / 22.9K episodes ≈ 463 frames/episode（30 fps 下 ≈ 15.4 s/episode）。
- **与 caption 冲突**：caption 写 "At $\sim$ 10M **episodes**"，但表格是 22.9K episodes / 10.6M **frames** → 数量级差约 400×，显然是把 frames 误写成 episodes（C1）。正文与摘要都是 22.9K/≈23K episodes：摘要 p.2 "fewer than 30k episodes"；§5.1 p.15 "approximately 23k trajectories"。
- 筛选标准只有四个名词（embodiment type / episode count / data quality / frame coverage），**无阈值**（C28）。

---

### 1.2 Table 2 — Simulation benchmarks (LIBERO and Meta-World)（p.11）

**caption 原文**（HTML `S4.T2`，p.11）：

> Table 2: Simulation benchmarks (LIBERO and Meta-World). Success rates (%) for various policies. LIBERO tasks correspond to different settings (e.g., spatial, object, goal, long-horizon); Meta-World tasks reflect different difficulty levels. VLA Pt refers pretraining on robotics data. SmolVLA is only initialized from the VLM. Baselines scores from Kim et al. (2024); Wen et al. (2024).

**列/行含义**：
- 列：`Benchmark`（LIBERO / Meta-World 两个块）；`Policy (# Params)` 策略与参数量；`VLA Pt.` 是否在机器人数据上预训练（Yes/No）；`Success Rate (%) — Simulation` 下分列。LIBERO 块分 `Spatial / Object / Goal / Long / Avg.`；Meta-World 块分 `Easy / Medium / Hard / Very Hard / Avg.`。
- 行：每块内各策略一行；SmolVLA 有三档参数 0.24B / 0.45B / 2.25B。

**完整数字表**（p.11，按原文精度）：

**LIBERO**（成功率 %）

| Policy (# Params) | VLA Pt. | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|---|
| Diffusion Policy (Khazatsky et al., 2024) | No | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo (0.09B) (Team et al., 2024) | Yes | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA (7B) (Kim et al., 2024) | Yes | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 (Paligemma-3B) | No | 87 | 63 | 89 | 48 | 71.8 |
| π0 (3.3B) | Yes | 90 | 86 | 95 | 73 | 86.0 |
| SmolVLA (0.24B) | No | 87 | 93 | 88 | 63 | 82.75 |
| SmolVLA (0.45B) | No | 90 | 96 | 92 | 71 | 87.3 |
| SmolVLA (2.25B) | No | 93 | 94 | 91 | 77 | 88.75 |

**Meta-World**（成功率 %）

| Policy (# Params) | VLA Pt. | Easy | Medium | Hard | Very Hard | Avg. |
|---|---|---|---|---|---|---|
| Diffusion Policy (Chi et al., 2023) | No | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA (Zhou et al., 2024) | No | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π0 (3.5B-Paligemma) | No | 80.4 | 40.9 | 36.7 | 44.0 | 50.5 |
| π0 (3.5B) | Yes | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA (0.24B) | No | 86.43 | 46.36 | 35 | 60 | 56.95 |
| SmolVLA (0.45B) | No | 82.5 | 41.8 | 45.0 | 60.0 | 57.3 |
| SmolVLA (2.25B) | No | 87.14 | 51.82 | 70 | 64 | 68.24 |

**支持哪个 claim / 作者结论**：
- 支撑"小模型可比/超过大 VLA"：SmolVLA 最小档（0.24B）LIBERO Avg 82.75 > Octo(75.1)、OpenVLA(76.5)、π0-Paligemma(71.8)，仅略低于 π0-robotics-pretrained(86.0)；0.45B 87.3 > π0 3.3B 86.0；2.25B 88.75 最高。
- Meta-World：0.24B 56.95 / 0.45B 57.3 / 2.25B 68.24，均高于 π0(3.5B) 47.9 与 π0-Paligemma 50.5（p.11 正文 p.10–11）。
- 作者结论（§4.5 p.10）："Despite not being pretrained on robotics data, SmolVLA consistently outperforms the VLM-initialized π0 and performs competitively with the robotics-pretrained version."；同页 "compared to π0, SmolVLA is around 40% faster to train and consumes 6× less memory."
- 注意：作者明说 **SmolVLA 这三行都没有机器人数据预训练**（caption 末句 "SmolVLA is only initialized from the VLM."，且 `VLA Pt. = No`）——见 C4：这与全文"用 481 个社区数据集预训练"的主卖点并不在同一条实验线上。

**数据来源（关键！）**：
- SmolVLA 各行：作者自己跑（但论文未说明评测 rollout 数在 Table 2 内，只有 §4.1 p.8 的 "10 trials per task" 通用协议）。
- Baseline 行：**引用别的论文的数字，不是作者复现**。caption 原文 "Baselines scores from Kim et al. (2024); Wen et al. (2024)."（Kim et al. 2024 = OpenVLA；Wen et al. 2024 = TinyVLA）。但 DP/Octo/π0 的数字并没有在 caption 里给出处；且同一张表里 DP 在 LIBERO 块引 (Khazatsky et al., 2024)、在 Meta-World 块引 (Chi et al., 2023)——见 C2/C3。

**算术复核/备注**：
- 全部 Avg 已重算并与原文一致（四舍五入 1 位）：LIBERO DP=72.35→72.4、Octo=75.075→75.1、OpenVLA=76.5、π0-Pal=71.75→71.8、π0=86.0、0.24B=82.75、0.45B=87.25→87.3、2.25B=88.75；Meta DP=10.45→10.5、TinyVLA=31.575→31.6、π0-Pal=50.5、π0=47.925→47.9、0.24B=56.9475→56.95、0.45B=57.325→57.3、2.25B=68.24。
- 精度不统一：SmolVLA 的 Avg 用两位小数（82.75/86.43/56.95/87.14/51.82/68.24），其他行多为 1 位。
- 参数量标签矛盾：π0 在 LIBERO 块是 "Paligemma-3B / 3.3B"，在 Meta-World 块是 "3.5B-Paligemma / 3.5B"，Table 3 又是 "π0 (3.5B)"，§4.4 正文说 "3.3 billion parameters"（p.10）→ C5。
- `# Params` 列并不完整：Diffusion Policy、TinyVLA、π0-Paligemma 三行都没给具体参数量（C6）。
- VLA Pt 语义：Octo/OpenVLA/π0(3.3B) 为 Yes；DP/TinyVLA/π0-Pal 为 No；SmolVLA 全 No。

---

### 1.3 Table 3 — Real-world benchmarks (SO100)（p.11）

**caption 原文**（HTML `S4.T4.fig1`，p.11）：

> Table 3: Real-world benchmarks (SO100). Success rate (%) across three tasks using policies trained in multi-task and single-task settings.

**列/行含义**：
- 列：`Policy`；`Pick-Place / Stacking / Sorting / Avg.`（真实世界成功率 %，SO100 三任务）。
- 行：按训练范式分组——`Single-task Training`：ACT；`Multi-task Training`：π0 (3.5B)、SmolVLA (0.45B)。

**完整数字表**（p.11）：

| Policy | Pick-Place | Stacking | Sorting | Avg. |
|---|---|---|---|---|
| **Single-task Training** | | | | |
| ACT | 70 | 50 | 25 | 48.3 |
| **Multi-task Training** | | | | |
| π0 (3.5B) | 100 | 40 | 45 | 61.7 |
| SmolVLA (0.45B) | 75 | 90 | 70 | 78.3 |

**支持哪个 claim / 作者结论**：支撑"真实机器人上小模型超过 7× 大的 π0、且超过 per-task 训练的 ACT"。§4.5 p.11："SmolVLA outperforms both ACT …, which is trained individually on each task, and π0, a significantly larger model in terms of parameter count (∼7×)."（注：这句所在段落本身有表号/描述混乱，见 C8/C9。）
- 分任务看作者结论并不总成立：Pick-Place 上 π0 = 100 > SmolVLA 75；SmolVLA 赢在 Stacking（90 vs 40）和 Sorting（70 vs 45）。

**数据来源**：作者自己采集/训练的 SO100 三任务真实数据（§4.1 p.8：三个 SO100 数据集 + 一个 SO101 数据集，每个 50 条 demonstrations；数据集已开源 lerobot/svla_so100_*）。ACT 由作者从零训练（"ACT is trained from scratch on each dataset"，§4.5 p.10–11）；π0 由作者在目标数据集上 finetune（"π0 is finetuned on the respective target datasets"，p.10）。→ 与 Table 2 不同，这里是**作者自己重训的 baseline**。

**算术复核/备注**：
- Avg 重算：ACT (70+50+25)/3=48.33→48.3 ✓；π0 (100+40+45)/3=61.67→61.7 ✓；SmolVLA (75+90+70)/3=78.33→78.3 ✓。
- **跨范式比较**：ACT 是 single-task、SmolVLA/π0 是 multi-task（行分组已明示），直接把 78.3 与 48.3 对比会被训练范式混淆（C9）。
- 无评测 trial 数、无 seed/方差（C10）；`Success rate` 实际是 §4.1 p.8 定义的分段得分平均（0.5/0.5 等），不是二值成功率（C11）。
- 该表 SmolVLA 行 (75/90/70/78.3) 与 Table 5 最后一行（VLA pt Yes, multi-task）完全相同 → 同一组实验在两表重复使用（C13）。

---

### 1.4 Table 4 — Real-world benchmark (SO101)（p.11）

**caption 原文**（HTML `S4.T4.fig2`，p.11）：

> Table 4: Real-world benchmark (SO101). Success rate (%) for the Pick-Place-Lego task using policies trained in single-task setting.

**列/行含义**：列 `Policy / In Distribution / Out of Distribution`；行为 `Single-task Training` 下的 ACT 与 SmolVLA (0.45B)。任务为 SO101 上的 Pick-Place-Lego（把粉色乐高块放进透明盒），ID = 训练分布内，OOD = 乐高块放在训练中未见过的新位置（§4.5 p.11）。

**完整数字表**（p.11）：

| Policy | In Distribution | Out of Distribution |
|---|---|---|
| **Single-task Training** | | |
| ACT | 70 | 40 |
| SmolVLA (0.45B) | 90 | 50 |

**支持哪个 claim / 作者结论**：支撑"跨 embodiment 泛化 + OOD 优于 ACT"。§4.5 p.11："Similarly, on SO101 (see Table 4), SmolVLA surpasses ACT in both in-distribution and out-of-distribution (OOD) settings. For OOD evaluation, the Lego object is placed in novel positions not previously encountered during training."
- 关键设定（§4.1 p.8）："Critically, SmolVLA is not pretrained on any datasets recorded for the SO101." 即 SO101 对 SmolVLA 是全新 embodiment，但它是用 SO101 数据 single-task 微调过的（caption 明示 single-task setting），所以这是"新机器人上小样本微调 + 物体位置 OOD"，不是零样本迁移。

**数据来源**：作者自采 SO101 Pick-Place-Lego 数据（lerobot/svla_so101_pickplace），两个策略都在该数据上 single-task 训练；作者自己评测。

**算术复核/备注**：无 Avg 列（只有 2 列）。无 trial 数、无 seed/方差（C10）。OOD 的 trial 数/位置数未给（C26）。注意 §4.5 正文对该表的描述错位（"trained on a combination of three datasets…per task and average"，见 C8）。

---

### 1.5 Table 5 — Effect of pretraining and multitask learning（p.12）

**caption 原文**（HTML `S4.T5.fig1`，p.12）：

> Table 5: Effect of pretraining and multitask learning. Success rate (%) across three tasks using SmolVLA trained in multi-task and single-task settings. Results with SO100 robot.

**列/行含义**：列 `Policy / VLA pt.（是否在机器人数据上预训练）/ Pick-Place / Stacking / Sorting / Avg.`；行按 single-task / multi-task 分组，三行都是 SmolVLA (0.45B)。

**完整数字表**（p.12）：

| Policy | VLA pt. | Pick-Place | Stacking | Sorting | Avg. |
|---|---|---|---|---|---|
| **Single-task Training** | | | | | |
| SmolVLA (0.45B) | No | 55 | 45 | 20 | 40 |
| **Multi-task Training** | | | | | |
| SmolVLA (0.45B) | No | 80 | 40 | 35 | 51.7 |
| SmolVLA (0.45B) | Yes | 75 | 90 | 70 | 78.3 |

**支持哪个 claim / 作者结论**：支撑"社区数据预训练有效 + 多任务微调有额外收益"。§4.5 p.11："pretraining on community datasets leads to a substantial performance improvement (from 51.7 to 78.3). Furthermore, multitask finetuning yields further gains"。
- 拆开看：多任务带来的提升是 40.0 → 51.7（+11.7，无预训练）；预训练带来的提升是 51.7 → 78.3（+26.6，多任务）。**但缺 single-task + pretrained 这一格**，两个因素没有完全交叉（C12）。

**数据来源**：作者自己的 SO100 真实实验；"Yes/No" 指是否有 §3.2 的 481 数据集预训练（§4.5 p.10 "For real-world evaluation, SmolVLA is pretrained on community-collected datasets."）。

**算术复核/备注**：
- Avg 重算：(55+45+20)/3=40.0 ✓；(80+40+35)/3=51.67→51.7 ✓；(75+90+70)/3=78.33→78.3 ✓。
- 最后一行与 Table 3 的 SmolVLA 行完全重复（C13）。
- 无 trial 数/方差；单格单次（C10）。

---

### 1.6 Figure 5 的三张子表 — Synchronous vs Asynchronous inference（p.12）

**外层 caption 原文**（HTML `S4.F5`，p.12）：

> Figure 5: Comparison between synchronous (Sync) and asynchronous (Async) inference. We evaluate SmolVLA on three real-world tasks under both inference modes. Asynchronous inference achieves similar success rates (left) but is significantly faster (middle) and complete more tasks (right) in fixed-time settings. Hyperparameters have been optimized for Pick-Place and reused across tasks.

**子表 caption 原文**：(a) Performance (success rates). / (b) Task completion time. / (c) Performance in fixed time.

#### Figure 5(a) — Performance (success rates)（p.12）

列：`Inference`（Sync/Async）；`Pick-Place / Stacking / Sorting / Avg`（真实世界成功率 %）。

| Inference | Pick-Place | Stacking | Sorting | Avg |
|---|---|---|---|---|
| Sync | 75 | 90 | 70 | 78.3 |
| Async | 80 | 90 | 50 | 73.3 |

#### Figure 5(b) — Task completion time（p.12）

列：`Inference`；`Total / Avg / Std`（时间，秒；单位 s）。

| Inference | Total | Avg | Std |
|---|---|---|---|
| Sync | 137.5 | 13.75 | 2.42 |
| Async | 97.0 | 9.70 | 2.95 |

#### Figure 5(c) — Performance in fixed time（p.12）

列：`Inference`；`Total / Avg / Std`（成功 pick-and-place 的方块个数）。

| Inference | Total | Avg | Std |
|---|---|---|---|
| Sync | 9 | 1.8 | 0.45 |
| Async | 19 | 3.8 | 1.3 |

**支持哪个 claim / 作者结论**：支撑"异步推理在不牺牲（太多）成功率的前提下显著提速、固定时间内完成更多"。
- §4.6 p.12 原文："On average, it completes the task in 9.7 seconds, compared to 13.75 seconds in the synchronous setting (∼30% faster). Furthermore, under the fixed-time evaluation, the asynchronous mode allows the robot to complete 19 successful pick-and-place cycles, compared to only 9 in the synchronous case."
- 本笔记推算：9.70/13.75 → 快 29.5%（原文 ∼30% ✓）；19/9 ≈ 2.11×。

**数据来源**：作者自己的真实机器人实验（Pick-Place 计时 + fixed-time 两套评测；§4.6 p.12："we measure the time taken to complete the task across 10 trials and 5 different cube positions"；"we fix a time limit (e.g., 60 seconds)"）。**Hyperparameters optimized for Pick-Place and reused across tasks**（caption 明示）。

**算术复核/备注**：
- (a) Avg：(75+90+70)/3=78.33→78.3 ✓；(80+90+50)/3=73.33→73.3 ✓。
- (b) Total/Avg：137.5/10=13.75；97.0/10=9.70 → "Total" 的分母是 **10 次**（本笔记推算）；std 的 n 未说明。
- (c) Total/Avg：9/5=1.8；19/5=3.8 → "Total" 分母是 **5**（本笔记推算）；两子表分母不同（10 vs 5），原文没有解释各自 trial 数（C16）。
- Sync 行 (75/90/70/78.3) 与 Table 3/Table 5 的 SmolVLA 行完全相同 → 同一组 sync 结果被重复使用。
- "similar success rates" 的说法偏乐观：Avg 73.3 vs 78.3（−5.0），Sorting 50 vs 70（−20）；只有 Pick-Place 提升（+5）（C14）。
- "e.g., 60 seconds"：固定时间上限不是确定值（C17）。

---

### 1.7 Table 6 — Cross vs self-attention（p.13）

**caption 原文**（HTML `S4.T6`，p.13）：

> Table 6: Cross vs self-attention. Interleaved cross (CA) and self-attention (SA) yield the best results.

**列/行含义**：列 `Attention mechanism`（CA = 纯 cross-attention；SA = 纯 causal self-attention；CA+SA = 作者的交错方案）；`Success Rate (%) — LIBERO` 分 `S / O / G / 10 / Avg`（S = **Spatial**、O = Object、G = Goal、10 = **Long**，见 §0.3 约定与 C25）。行 = 三种注意力机制。

**完整数字表**（p.13）：

| Attention mechanism | S | O | G | 10 | Avg |
|---|---|---|---|---|---|
| CA | 87 | 92 | 83 | 54 | 79.0 |
| SA | 80 | 94 | 84 | 40 | 74.5 |
| CA+SA (ours) | 86 | 99 | 90 | 67 | 85.5 |

**支持哪个 claim / 作者结论**：支撑"交错 CA+SA 最好"。§4.7 p.12–13："cross-attention outperforms self-attention significantly. Interleaving both yields the best results, highlighting their complementary strengths."（CA+SA 85.5 vs CA 79.0 vs SA 74.5，+6.5/+11.0。）
**数据来源**：作者自己的 LIBERO ablation；"All ablations are conducted on the LIBERO benchmark. Unless otherwise noted, models are trained from scratch without any pretraining on robotics data. The VLM backbone is frozen, and only the action expert is trained from scratch."（§4.7 p.12）。
**算术复核**：Avg 全对（316/4=79.0；298/4=74.5；342/4=85.5）。无 seed/方差（C18）。CA 行 79.0 与 Table 11 "Prefix CA" 80.3 不一致 → C22。

---

### 1.8 Table 7 — Bidirectional vs causal attention（p.13）

**caption 原文**（HTML `S4.T7`，p.13）：

> Table 7: Bidirectional vs causal attention. Preventing future action leakage improves performance.

**列/行含义**：列 `Attention mask`（Bidir = 动作 token 间双向自注意力；Causal = 因果掩码）；LIBERO 分 `S / O / G / 10 / Avg`。

**完整数字表**（p.13）：

| Attention mask | S | O | G | 10 | Avg |
|---|---|---|---|---|---|
| Bidir | 79 | 86 | 82 | 23 | 67.5 |
| Causal | 80 | 94 | 84 | 40 | 74.5 |

**支持 claim / 结论**：撑"因果掩码防止未来动作泄漏、显著更好"（+7.0 Avg，尤其 Long 23→40）。§4.7 p.13："Table 7 shows causal self-attention performs best, while bidirectional interaction harms performance. Surprisingly, the no-interaction (CA-only) setup performs competitively"（CA-only 即 Table 6 的 CA=79.0）。
**数据来源**：同上，作者 LIBERO ablation。
**算术复核**：67.5=(270/4) ✓；74.5=(298/4) ✓。Causal 行与 Table 6 的 SA 行完全相同（同一配置，正常）。无方差。

---

### 1.9 Table 8 — Skipping VLM layers（p.13）

**caption 原文**（HTML `S4.T8`，p.13）：

> Table 8: Skipping VLM layers. Skipping layers from a large VLM (here 500M parameters) yields better results than downsizing the VLM. Skipping every second layer is a competitive baseline.

**列/行含义**：列 `N`（保留前 N 个 LLM 层）；LIBERO 分 `S / O / G / 10 / Avg`。另有 `Skip %2`（每隔一层取一层，深度减半、容量保留）与 `VLM-256M`（把 backbone 换成 256M VLM）两个对照行。

**完整数字表**（p.13）：

| N | S | O | G | 10 | Avg |
|---|---|---|---|---|---|
| 8 | 77 | 88 | 86 | 49 | 75.0 |
| 16 | 88 | 91 | 91 | 44 | 78.5 |
| 24 | 86 | 97 | 86 | 49 | 79.5 |
| 32 | 89 | 94 | 85 | 53 | 80.3 |
| Skip %2 | 84 | 90 | 83 | 45 | 75.5 |
| VLM-256M | 86 | 83 | 75 | 59 | 75.8 |

**支持 claim / 结论**：
- 主图/正文默认用 N=16（§3.1 p.4 "N=L/2"；§4.3 p.10 "first 16 layers"）→ 表里 16 层 78.5，接近全 32 层 80.3，作者称"good trade-off between performance and compute"（§4.7 p.13）。
- caption：从 500M VLM 跳层优于缩小 VLM（16 层 78.5 > VLM-256M 75.8 ✓）。
- **但正文与数字矛盾**：§4.7 p.13 "Table 8 indicates skipping every second layer performs better than training a smaller VLM, but worse than using the first N<L layers directly." 而 Skip %2 = 75.5 < VLM-256M = 75.8 → 第一半句被自己的表推翻（C19）。
- 表内 N=8 < 16 < 24 < 32 基本单调上升，作者仍选 16 层（效率动机）；但论文没有给速度/内存的量化收益。

**数据来源**：作者 LIBERO ablation（VLM frozen, expert from scratch）。caption "here 500M parameters" 暗示 ablation 的 backbone 是 500M VLM——但主模型 0.45B=450M total（§4.3 p.10），两者关系原文未解释（见 §4 与 C22/C23）。
**算术复核**：Avg 全对（75.0/78.5/79.5/80.25→80.3/75.5/75.75→75.8）。无方差。

---

### 1.10 Table 9 — Expert capacity（p.13）

**caption 原文**（HTML `S4.T9`，p.13）：

> Table 9: Expert capacity. Adjusting the expert's hidden size affects performance. Larger capacities yield better success rates.

**列/行含义**：列 `Expert width (w.r.t. VLM)`，即 action expert 隐藏维 = 倍数 × VLM 维度 d（×1.00 / ×0.75 / ×0.50 / ×0.25）；LIBERO 分 `S / O / G / 10 / Avg`。

**完整数字表**（p.13）：

| Expert width (w.r.t. VLM) | S | O | G | 10 | Avg |
|---|---|---|---|---|---|
| ×1.00 | 87 | 96 | 90 | 56 | 82.3 |
| ×0.75 | 82 | 89 | 84 | 55 | 77.5 |
| ×0.50 | 89 | 94 | 85 | 53 | 80.3 |
| ×0.25 | 76 | 97 | 83 | 39 | 73.8 |

**支持 claim / 结论**：作者结论是"0.75×d 是性能/效率的好平衡"（§3.1 p.4 说主模型用 0.75×d；§4.7 p.13 再次确认）。但数字并不支持 caption "Larger capacities yield better success rates"：×1.00（82.3）> ×0.50（80.3）> ×0.75（77.5）> ×0.25（73.8），**非单调**，且主模型选的 ×0.75 恰是中等偏下的点（C20）。论文未给任何内存/延迟数字来支撑"效率平衡"。
**数据来源**：作者 LIBERO ablation。
**算术复核**：×1.00：(87+96+90+56)/4=82.25→82.3（原文 82.3，half-up；若按 round-half-even 会是 82.2，属可接受的舍入差异）；其余 77.5/80.25→80.3/73.75→73.8 均对。无方差。

---

### 1.11 Table 10 — Training objective（p.14）

**caption 原文**（HTML `S4.T10`，p.14）：

> Table 10: Training objective. We compare our training objective based on Flow matching to regression with L1 loss.

**列/行含义**：列 `Training objective`（Flow matching vs Regression L1）；LIBERO 分 `S / O / G / 10 / Avg`。

**完整数字表**（p.14）：

| Training objective | S | O | G | 10 | Avg |
|---|---|---|---|---|---|
| Flow matching | 89 | 94 | 85 | 53 | 80.25 |
| Regression | 92 | 85 | 86 | 38 | 75.25 |

**支持 claim / 结论**：撑"flow matching 优于 L1 回归"（+5.0 Avg；Long 38→53）。§4.7 p.13："flow matching significantly outperforms regression, suggesting flow matching provides better inductive bias for modeling complex, multimodal action distributions."
**数据来源**：作者 LIBERO ablation。
**算术复核**：80.25=(321/4) ✓、75.25=(301/4) ✓；注意本表 Avg 保留 2 位小数，而其它 ablation 表保留 1 位（80.25 与 80.3 是同一个数）。Flow matching 行 = Table 11 Prefix CA = Table 12 chunk 50 = Table 13 exec 1 = 80.3/80.25，即同一"默认配置"的重复出现。无方差（C18/C22）。

---

### 1.12 Table 11 — States as prefix vs. suffix（p.14）

**caption 原文**（HTML `S4.T11`，p.14）：

> Table 11: States as prefix vs. suffix. Feeding states to the VLM (prefix) leads to better performance than feeding them to the expert (suffix).

**列/行含义**：列 `States`（Prefix = 状态投影为 token 进 VLM；Suffix = 状态直接进 action expert）× `Attention`（CA / SA）；LIBERO 分 `S / O / G / 10 / Avg`。

**完整数字表**（p.14）：

| States | Attention | S | O | G | 10 | Avg |
|---|---|---|---|---|---|---|
| Prefix | CA | 89 | 94 | 85 | 53 | 80.3 |
| Suffix | CA | 86 | 82 | 78 | 47 | 73.3 |
| Prefix | SA | 62 | 74 | 57 | 20 | 53.3 |
| Suffix | SA | 80 | 92 | 80 | 47 | 74.8 |

**支持 claim / 结论**：caption 说 "prefix 更好"；CA 变体成立（80.3 vs 73.3，+7.0），**但 SA 变体完全反了**：Prefix SA 53.3 vs Suffix SA 74.8（−21.5）。正文 §4.7 p.13 更明确地写 "including state information in the VLM leads to significantly better performance **for both the CA and SA variants**" → 与表中 SA 行直接矛盾（C21）。
**数据来源**：作者 LIBERO ablation。
**算术复核**：80.25→80.3、73.25→73.3、53.25→53.3、74.75→74.8，全对。Suffix SA 74.8 ≈ Table 7 Causal 74.5（见 C22 的跨表不一致讨论）。无方差。

---

### 1.13 Table 12 — Action chunk size（p.14）

**caption 原文**（HTML `S4.T12`，p.14）：

> Table 12: Action chunk size. A chunk size between 10 and 50 strikes a good balance between action prediction frequency and performance.

**列/行含义**：列 `Chunk Size` n（每次预测的动作块长度 1/10/30/50/100）；LIBERO 分 `S / O / G / 10 / Avg`。

**完整数字表**（p.14）：

| Chunk Size | S | O | G | 10 | Avg |
|---|---|---|---|---|---|
| 1 | 45 | 77 | 54 | 24 | 50.0 |
| 10 | 90 | 94 | 94 | 58 | 84.0 |
| 30 | 85 | 94 | 87 | 48 | 78.5 |
| 50 | 89 | 94 | 85 | 53 | 80.3 |
| 100 | 83 | 88 | 85 | 42 | 74.5 |

**支持 claim / 结论**：撑"过小过大都差，10–50 是好区间"（§4.7 p.14）。但表里 **n=10 最好（84.0）**，主模型却用 **n=50**（§4.3 p.10 "output chunks of n=50 actions"），n=50 只有 80.3，比 n=10 低 3.7；作者只用"reactivity/efficiency"定性解释（C24）。
**数据来源**：作者 LIBERO ablation。
**算术复核**：50.0/84.0/78.5/80.25→80.3/74.5 全对。无方差。

---

### 1.14 Table 13 — Action execution steps（p.14）

**caption 原文**（HTML `S4.T13`，p.14）：

> Table 13: Action execution steps. Sampling new observations more frequently (e.g., every 1 or 10 steps) significantly improves performance.

**列/行含义**：列 `Action Steps`（执行多少步后重新观测/重新预测）；LIBERO 分 `S / O / G / 10 / Avg`。

**完整数字表**（p.14）：

| Action Steps | S | O | G | 10 | Avg |
|---|---|---|---|---|---|
| 1 | 89 | 94 | 85 | 53 | 80.3 |
| 10 | 89 | 94 | 91 | 57 | 82.8 |
| 30 | 76 | 91 | 74 | 42 | 70.8 |
| 50 | 54 | 70 | 58 | 25 | 51.8 |

**支持 claim / 结论**：撑"更新观测越频繁越好，但存在速度/精度 trade-off"（§4.7 p.14）。细节：1 步 80.3、10 步 82.8（**10 步反而最高**），之后骤降（30→70.8，50→51.8）。
**数据来源**：作者 LIBERO ablation。
**算术复核**：80.25→80.3、82.75→82.8、70.75→70.8、51.75→51.8 全对。
**备注**：§4.3 p.10 说"模拟中每执行一个动作就重新采样观测并预测"（即 1 步），而本表 1 步（80.3）不是最优、10 步（82.8）更高；同时 Table 2 的 LIBERO 主结果（87.3）远高于本表任何一行 → 主结果与 ablation 的协议/预算并未对齐（C23/C24）。

---

## 2. 训练配置与超参（全部来自 §4.3 p.10，另有标注除外）

### 2.1 总表

| 项目 | 预训练（community data） | 模拟基准微调 | 真实机器人微调 |
|---|---|---|---|
| 训练步数 | 200,000 steps | 100,000 steps | 200,000 steps |
| Batch size | **global** 256 | 64 | 未给出（原文只给步数） |
| 学习率调度 | 100-step warmup 后 cosine，1e-4 → 最低 2.5e-6 | 未单独给出（推测同预训练调度，原文未说） | 未给出 |
| Optimizer | AdamW，β1=0.9，β2=0.95 | 未说明 | 未说明 |
| Weight decay / grad clip / EMA / dropout | **均未披露** | — | — |
| 图像分辨率 | 512×512（与 VLM 输入一致） | 同左（推测） | 同左（推测） |
| VLM backbone | SmolVLM-2（Marafioti et al., 2025） | 同左 | 同左 |
| 可训练部分 | 仅 action expert（VLM 冻结） | 仅 action expert（VLM 冻结） | 仅 action expert（VLM 冻结） |
| LLM 层数 | 只用前 16 层（"first 16 layers of the LLM"；§3.1 p.4 说 N=L/2） | 同左 | 同左 |
| flow matching 推理步数 | — | 固定 10 steps | 固定 10 steps |
| action chunk 长度 | n=50 | n=50（§4.3） | n=50（§4.3） |
| 推理方式 | — | 每执行 1 个动作就重新观测+预测（异步/同步未提；§4.3 p.10） | **同步推理**：执行完整个 chunk 才采新观测（§4.3 p.10） |
| 精度/编译 | bfloat16 + `torch.compile()` | — | — |
| 分布式 | Hugging Face `accelerate`，混合精度；预训练用 **4 GPUs** | — | — |
| 计算量 | 项目总体 **≈30k GPU hours**（含全部实验，非仅预训练） | — | — |

**原文依据**（§4.3 p.10，逐句）：
- "During pretraining, we train for 200,000 steps with a global batch size of 256 on all our community datasets. After a 100-step warmup, we use a cosine learning rate schedule starting at 1e-4 and decaying to a minimum of 2.5e-6. We use the AdamW optimizer with β1 = 0.9, β2 = 0.95."
- "Training is performed after resizing the images to 512×512… We use SmolVLM-2 … The action expert is trained with flow matching to output chunks of n = 50 actions."
- "For real-world evaluation, we perform synchronous inference… In simulation, we perform inference by sampling new observations and predicting a new action after each executed action. During inference, the flow matching is fixed to 10 steps. We train only the action expert module, keeping the VLM frozen."
- "Our main model, contains 450 million parameters, with approximately 100 million dedicated to the action expert. We use only the first 16 layers of the large language model (LLM) within the VLM."
- "For fine-tuning on simulation benchmarks, we train for 100,000 steps with a batch size of 64, while for real-world tasks, we fine-tune for 200,000 steps."
- "we leverage bfloat16 precision and torch.compile()… we maintain a fixed sequence length and batch size, discarding any excess frames… For multi-GPU and multi-node training, we utilize… accelerate… Pretraining was conducted using 4 GPUs… Overall, the project consumed approximately 30k GPU hours."

### 2.2 架构/输入相关数字

| 项目 | 值 | 出处 |
|---|---|---|
| 每帧视觉 token 上限 | **64 / frame**（不用 image tiling，只取 global image + pixel shuffle） | §3.1 p.4 |
| 状态 token 数 | 传感器状态线性投影成 **1 个 token** | §3.1 p.4 |
| LLM 层裁剪 | 丢弃顶部 L−N 层，默认 N=L/2（主模型 16 层） | §3.1 p.4、§4.3 p.10 |
| Action expert 宽度 | **0.75×d**（d = VLM hidden dim） | §3.1 p.4 |
| Action expert 参数量 | ≈100M（主模型 450M 总参数） | §4.3 p.10 |
| 注意力结构 | 交错 CA / causal SA | §3.1 p.4 |
| Flow matching τ 采样 | 从 Beta 分布采样（未给 Beta 参数） | §3.1 p.4 |
| 相机/图像数 | 真实：SO100 用 top+wrist；SO101 用 top+side（Figure 4 caption p.9）；模拟：未说明 | Figure 4 p.9 |
| 真实机器人自由度 | SO100/SO101 均 6-DoF、位置控制；Panda 7-DoF（LIBERO）；Sawyer 4-DoF（Meta-World） | §4.2 p.9 |

### 2.3 数据规模汇总

| 数据集/用途 | 规模 | 出处 |
|---|---|---|
| 预训练社区数据 | **481 datasets / 22.9K episodes / 10.6M frames** | Table 1 p.5 |
| 预训练数据来源 | 从 Hugging Face 筛选的 LeRobot 社区数据集（单机器人类型 SO100，见局限性 p.15；但 A.1 清单有可疑条目，见 C28） | §3.2 p.5、§5.1 p.15 |
| LIBERO 训练数据 | `physical-intelligence/libero`，**1,693 episodes** 覆盖全部任务（40 tasks = 4 suite × 10） | §4.1 p.8 |
| Meta-World 训练数据 | `lerobot/metaworld_mt50`，**2,500 episodes（50 tasks × 50 demos）** | §4.1 p.8 |
| 真实机器人数据（SO100） | 3 个任务（Pick-Place / Stacking / Sorting），每任务 **50 demonstrations**（10 trajectories × 5 starting positions；sorting 为 2 种颜色配置 × 5 ep × 5 位置） | §4.1 p.8 |
| 真实机器人数据（SO101） | 1 个任务（Pick-Place-Lego），**50 demonstrations**（同 10×5 规则；数据集 lerobot/svla_so101_pickplace） | §4.1 p.8–9 |
| 数据总量对比 | 论文自述 ≈23k trajectories vs OpenVLA ≈1M trajectories | §5.1 p.15 |

### 2.4 未披露的训练细节（影响复现）

GPU 型号/显存；预训练 wall-clock 时长；模拟/真实微调各自的计算量；微调阶段的学习率与 optimizer；weight decay / warmup 之外的调度细节；随机种子；数据增强；真实微调是否多数据集混合及采样权重；action normalization 方式；state 维度/关节顺序；语言指令增强；checkpoint 选择标准（按什么指标选）。以上论文正文与附录均**没有**给出。

---

## 3. 评测协议

### 3.1 LIBERO（模拟）

- 基准：LIBERO（Liu et al., 2023a），4 个 suite：Spatial / Object / Goal / Long，**每 suite 10 个任务，共 40 个任务**（§4.1 p.8）。
- 训练数据：physical-intelligence/libero，1,693 episodes（§4.1 p.8）。
- 评测：**每任务 10 trials**（"evaluate with 10 trials per task"），二值成功率（1 = 任务完成，0 = 失败），报告平均成功率（§4.1 p.8）。
- → 每 suite 10 tasks × 10 trials = 100 rollouts；四 suite 共 400 rollouts（本笔记推算）。
- 训练方式：多任务训练（"using a multi-task training setup"，§4.5 p.10）；Table 2 中 SmolVLA 不加载机器人预训练权重（caption p.11）。
- baseline 数字**不是作者复现**：Table 2 caption 说引用 Kim et al. (2024)、Wen et al. (2024)；其余 baseline 来源未注明（C2）。

### 3.2 Meta-World（模拟）

- 基准：Meta-World（Yu et al., 2020），**50 个任务**，难度分 Easy / Medium / Hard / Very Hard（Seo et al., 2023）（§4.1 p.8）。
- 训练数据：lerobot/metaworld_mt50，**2,500 episodes（50 tasks × 50 demos）**；作者还自采了一份新 Meta-World 数据（"we collected a new dataset for MetaWorld comprising of 50 demonstrations for each of the 50 tasks"，§4.1 p.8）。
- 评测：**每任务 10 trials**，与 LIBERO 相同协议；只有完整完成才算 1（§4.1 p.8）。
- → 50 tasks × 10 trials = 500 rollouts（本笔记推算）。
- baseline 数字同样为引用（Table 2 caption p.11）。

### 3.3 真实机器人（SO100 / SO101）

| 项目 | 内容 | 出处 |
|---|---|---|
| 机器人 | SO100（3 任务）、SO101（1 任务）；均 6-DoF 低惯量舵机、位置控制 | §4.2 p.9 |
| 相机 | SO100：top + wrist；SO101：top + side | Figure 4 caption p.9 |
| 任务 | Pick-Place（方块→小盒）、Stacking（红方块叠蓝方块）、Sorting（红/蓝方块分盒，长程）、SO101 Pick-Place-Lego（粉乐高块→透明盒） | §4.1 p.8–9 |
| 示范数据 | 每数据集 50 demos（10 trajectories × 5 个起始位置；sorting 10/位置） | §4.1 p.8 |
| 评估指标 | 分段得分平均：Pick-Place 抓 0.5 + 放 0.5；Stacking 抓 0.5 + 叠 0.5；Sorting 0.25×4（抓任一 + 两次颜色-盒匹配）；以 "success rate (%)" 报告 | §4.1 p.8 |
| 每任务评测 trial 数 | **未报告**（Table 3/4/5 均无 N、无方差） | §4.1 p.8、Tables 3–5 |
| OOD 设置 | SO101：Lego 放到训练未见位置 | §4.5 p.11 |
| 状态/动作 | 位置控制；action chunk n=50；同步推理（除 Figure 5 的 async 对比） | §4.3 p.10 |
| 待定项 | 真实推理跑在什么硬件上、控制频率、每次评估耗时 | 未报告 |

### 3.4 异步推理协议（Figure 5）

- 两套实验（§4.6 p.12）：
  1. **任务完成时间**：Pick-Place，"across 10 trials and 5 different cube positions"，从机器人开始运动计时（Table 5b 的 Total/Avg 表明 Total 是 10 次之和 → 每 trial 定义仍不够明确）；
  2. **固定时间完成量**：设定时间上限（原文 "e.g., 60 seconds"），统计成功 pick-and-place 的方块数（Table 5c 的 Total/Avg=5 → 5 次 run）。
- 同步 vs 异步：sync = 执行完整个 chunk 再预测（标准评测设置）；async = 预测与执行并行，队列剩余比例低于阈值 g 时触发新观测（Algorithm 1）。
- "Hyperparameters have been optimized for Pick-Place and reused across tasks"（Figure 5 caption p.12）→ 超参只针对 Pick-Place 调过，其他任务（含 Sorting）直接复用。

### 3.5 试次/方差汇总（批判用）

| 实验 | 每任务 trial | seed/方差 | 出处 |
|---|---|---|---|
| Table 2 LIBERO | 10 trials/task（作者自评） | 无；baseline 来自他文，协议未必一致 | §4.1 p.8、Table 2 p.11 |
| Table 2 Meta-World | 10 trials/task | 同上 | §4.1 p.8 |
| Tables 3/4/5 真实机器人 | **未报告** | 无 | Tables 3–5 p.11–12 |
| Figure 5(a) | 未报告（与 Table 3 同一 sync 数据） | 无 | p.12 |
| Figure 5(b) | Total/Avg 表明 10 次 | 有 Std（2.42 / 2.95），n 未说明 | p.12 |
| Figure 5(c) | Total/Avg 表明 5 次 | 有 Std（0.45 / 1.3），n 未说明 | p.12 |
| Tables 6–13 消融 | 未在表内报告（推测同 10 trials/task） | 无 | p.13–14 |

---

## 4. 参数规模 0.24B / 0.45B / 2.25B 分别对应什么配置？

### 4.1 原文明确说的

- 主模型（论文主推）**0.45B**：§4.3 p.10 "Our main model, contains 450 million parameters, with approximately 100 million dedicated to the action expert. We use only the first 16 layers of the large language model (LLM) within the VLM."
- Table 2 caption p.11："SmolVLA is only initialized from the VLM."（三档都没有机器人数据预训练）
- Table 8 caption p.13："Skipping layers from a large VLM (here **500M parameters**) yields better results than downsizing the VLM."，并有一行 **VLM-256M**（把 backbone 换成 256M 的 VLM）。
- §3.1 p.4：VLM backbone = SmolVLM-2；每帧 64 visual token；N=L/2；expert 宽 0.75d。

### 4.2 论文**没有**明说的（存疑）

论文从头到尾没有一段文字定义 0.24B / 0.45B / 2.25B 三档的具体组成（backbone 型号、层数、expert 宽度、是否同比例缩放）。只能从线索推断：

| 档位 | 论文依据 | 合理推断（**存疑，属本笔记推断**） |
|---|---|---|
| 0.24B | Table 2 p.11 行标签；Table 8 p.13 的 `VLM-256M` 对照行 | 基于 SmolVLM-2 **256M** 级 VLM + action expert 的小模型；总参数约 0.24B |
| 0.45B | §4.3 p.10 "450 million parameters…~100M action expert…first 16 layers" | 主模型：500M 级 VLM（SmolVLM-2 "large VLM" per Table 8 caption）裁到 16/32 层（约半个 LLM）+ ~100M expert ≈ 450M total |
| 2.25B | Table 2 p.11 行标签 | 基于 SmolVLM-2 **2.2B** 级 VLM 的同架构放大版（具体层数/expert 宽度未报告） |

- **不能确认**：0.24B 与 2.25B 是否也用 N=L/2、expert 0.75d、frozen VLM；论文只对主模型给了这些设定。
- **需要回查**：正式发表的 v2/v3 或 HF 模型卡（本笔记只用 v1 素材，不引入外部信息）。
- 另注意：Table 2 三档的参数量标签 vs §4.3 的 450M 主模型 vs Table 8 caption 的 "500M VLM" —— 三者量级能勉强自洽（500M VLM 被裁半 + 100M expert ≈ 450M），**但论文没有给出这段算术**（C22、C23）。

---

## 5. 异步推理的延迟 / 控制频率 / 时间数字

### 5.1 机制（§3.3 p.5–8，Algorithm 1）

- 设定：chunk size n、阈值 g∈[0,1]、joint-space 相似度过滤阈值 ε；`RobotClient` 向 `PolicyServer` 发观测 o_t，异步拿回动作块 A_t；队列消耗到 `|A_t|/n < g` 时触发新观测/新 chunk；新 chunk 与旧 chunk 在重叠时间步上聚合 `A_{t+1}←f(A_t, Ã_{t+1})`。
- 三种 g（Figure 3 讨论）：g=0（sequential limit，队列会空）、**g=0.7（作者实践值）**、g=1（compute-intensive limit，每个控制 tick 一次 forward）。
- 解析式（§3.3 p.7）：
  - `E[ℓ] = E[t_{C→S}] + E[ℓ_S] + E[t_{S→C}] ≈ E[ℓ_S]`（通信时延忽略）；
  - "With a real-world frame-rate of **30 frames per second**, **Δt = 33 ms**."；
  - 不空队列条件：`g ≥ (E[ℓ_S]/Δt)/n`；
  - 触发间隔 ≈ `(1−g)·n·Δt`（无相似度过滤时），收到新 chunk 间隔 ≈ `(1−g)·n·Δt + E[ℓ_S]`。
- 本笔记推算（非原文）：n=50、g=0.7、Δt=33ms → 名义上每消耗 0.3×50=15 个动作（≈0.5 s）触发一次新 chunk 预测。

### 5.2 论文给出的全部"速度"数字

| 数字 | 值 | 出处 |
|---|---|---|
| 环境帧率/控制周期 | **30 fps → Δt = 33 ms** | §3.3 p.7 |
| 异步阈值 g（实践最优） | **0.7**（Figure 3 讨论 g=0 / 0.7 / 1） | §3.3 p.7 |
| chunk size n | 50 | §4.3 p.10、Table 12 p.14 |
| 端到端任务完成时间（Pick-Place） | Sync **13.75 s** vs Async **9.70 s**（Total 137.5 vs 97.0 s；Std 2.42 vs 2.95） | Figure 5(b) p.12 |
| 提速比例 | **≈30%**（本笔记：29.5%） | §4.6 p.12 |
| 固定时间内完成 pick-and-place 数 | Sync **9**（Avg 1.8, Std 0.45）vs Async **19**（Avg 3.8, Std 1.3） | Figure 5(c) p.12 |
| 成功率对比 | Sync 78.3 / Async 73.3（Avg） | Figure 5(a) p.12 |
| 训练对比 π0 | 训练快 **~40%**、显存 **6× 更少** | §4.5 p.10 |

### 5.3 缺失（重要）

- **没有实测单次 chunk 推理延迟 E[ℓ_S] 的数值**（公式里留作符号；Figure 3 只有队列长度曲线，本笔记未从图中读数）。
- **没有控制频率（Hz）的实测值**：30 fps / 33 ms 是"real-world frame-rate"的设定值，不是测得的策略下发频率。
- **没有吞吐/FLOPs/显存绝对值**、没有 RobotClient/PolicyServer 硬件与网络时延数据。
- 摘要/intro 的 "allowing higher control rates"（p.1）与 §3.3 的 "higher control rates" 没有对应的量化实验（C18）。

---

## 6. 附录摘录（Appendix A.1，p.20–24）

### 6.1 附录内容清单

v1 的 Appendix 只有 **A.1 Community datasets**，包含两块：
1. **Task annotation** 的 VLM prompt 模板（p.20）；
2. **List of datasets**：481 个 Hugging Face dataset handle 的清单（p.20–24）。

**没有任何附录表格、没有额外实验、没有训练曲线、没有超参表**（与 §0.2 的 16 个网格清单一致：全部 16 个非公式表格都在正文，编号 Table 1–13 + Figure 5 三个子表）。

### 6.2 Task annotation prompt 原文（p.20，逐字）

> Here is a current task description: {current_task}. Generate a very short, clear, and complete one-sentence describing the action performed by the robot arm (max 30 characters). Do not include unnecessary words. Be concise. Here is some examples: Pick up the cube and place it in the box, open the drawer and so on. Start directly with an action verb like "Pick", "Place", "Open", etc. Similar to the provided examples, what is the main action done by the robot arm?

- 用途（§3.2 p.5）：社区数据集的原始语言标注噪声很大（出现 `task desc`、`Hold`、`Up` 之类占位/过短指令，或完全没有指令），作者用 **Qwen2.5-VL-3B-Instruct** 对每个数据集采样代表性帧 + 原始指令，重写为一句简短动作描述。
- 备注：prompt 里写 "(max 30 characters)"，但示例 "Pick up the cube and place it in the box" 就不止 30 字符；这是 prompt 自身的不一致（不算论文数据矛盾，但复现时需注意）。
- 相机命名标准化（§3.2 p.5）：人工把每个相机映射为 top/wrist/side，重命名为 `OBS_IMAGE_1/2/3`；多余视角训练时丢弃。

### 6.3 数据集清单（p.20–24）

- 形式：以 "owner/repo" 开头的 Hugging Face 数据集名，逗号/空格混排，多栏排版。
- 可核验性问题：
  - 排版中大量条目之间**缺逗号/粘连**（例如 p.21 附近出现 "andy309/so100_0314_fold_cloths jchun/so100_pickplace_small_20250323_120056"），因此无法可靠地逐个计数来核对 "481"（C28）。
  - 清单中包含 **`Ashton3/lerobot-aloha`**（p.21）等非 SO100 命名的条目，与 §5.1 p.15 "Our pretraining currently uses datasets collected from a single robot type (SO100)" 可能存在出入（也可能只是数据集命名/格式）→ 存疑。
  - 清单包含大量双机械臂/多相机/工业/domotic 任务（`ZCM5115/so100_2Arm3cameras_movebox`、`andreasBihlmaier/dual_arm_transfer_2025_02_16`、`kantine/industrial_robothon_*` 等），说明"任务多样性"远超 4 个真实评测任务，但论文没有给出任务类别/时长的统计。
- 论文只给了三列总量（Table 1），**没有给出 episode 长度分布、相机数、动作维度、机器人型号的分布**。

### 6.4 附录里没有的"额外实验"（对照任务清单）

- 不同数据集规模的训练曲线：**没有**。
- 视觉 token 数 ablation：**没有专门表格**（只有 §3.1 p.4 说每帧 64 token，无对比实验）。
- 真实机器人更多结果：**没有**（只有 Tables 3/4/5 与 Figure 5，共 4 个任务）。
- prompt 模板：**有**（§6.2）。
- 附录超参表：**没有**（训练细节只在 §4.3 p.10）。

---

## 7. 批判性清单（数字矛盾 / 缺方差 / baseline 不可比 / 消融不完整）

> 格式：**C编号. 现象** — 原文出处（页/表/§）→ 影响判断。

**C1. Table 1 caption 的 "~10M episodes" 是单位笔误。**
Table 1 caption（p.5）："At ~10M episodes, our pretraining set stands at least one order of magnitude smaller than other state-of-the-art."；但同表是 **22.9K episodes / 10.6M frames**。摘要（p.2）"fewer than 30k episodes"、§5.1（p.15）"approximately 23k trajectories" 都支持 22.9K 才是 episode 数。→ 10M 应指 frames；引用时必须写 22.9K episodes，不能写 10M episodes。

**C2. Table 2 的 baseline 数字是引用而非复现，且只说了一半出处。**
caption（p.11）："Baselines scores from Kim et al. (2024); Wen et al. (2024)." → 只交代 OpenVLA 与 TinyVLA 的出处；Diffusion Policy / Octo / π0 的数字来源未注明。作者自己的 LIBERO/Meta-World 评测协议是"每任务 10 trials"（§4.1 p.8），而各 baseline 原论文的 trial 数/评测版本未必相同 → Table 2 的跨策略比较不是同协议复现，弱于 Table 3 的"作者重训"。

**C3. Table 2 内引用错误（自相矛盾）。**
- LIBERO 块 "Diffusion Policy (**Khazatsky et al., 2024**)"：Khazatsky et al. 2024 是 **DROID 数据集**论文（参考文献条目），Diffusion Policy 是 Chi et al. 2023；同一张表的 Meta-World 块又写 "Diffusion Policy (**Chi et al., 2023**)" → 同一模型两个出处。
- Meta-World 块 "TinyVLA (**Zhou et al., 2024**)"：§2（p.3）与参考文献都把 TinyVLA 记为 **Wen et al., 2024**；Zhou et al. 2024 是 **TinyLLaVA**（VLM，非 VLA）。
→ 表 2 的 baseline 引用不可靠，回查原始数字时要按模型名而非按引文。

**C4. Table 2 的 SmolVLA 结果没有用"社区数据预训练"——与全文主卖点不在同一条线上。**
caption（p.11）："SmolVLA is only initialized from the VLM."，且三行 `VLA Pt. = No`；§4.7（p.12）也说所有 ablation "trained from scratch without any pretraining on robotics data"。而摘要（p.1）、§1（p.2）、§3.2（p.5）的主贡献是"用 481 个社区数据集预训练、<30k episodes"。→ Table 2 证明的是"小架构+模拟数据微调"的能力，**没有**证明社区预训练的价值；预训练的收益只在真实机器人 Table 5（51.7→78.3，p.12）出现。写作/引用时不能把 Table 2 当作"数据高效预训练"的证据。

**C5. π0 的参数量标签三处不一致。**
§4.4（p.10）："π0 … total model size of **3.3** billion parameters"；Table 2 LIBERO（p.11）：π0 (Paligemma-3B) / π0 (**3.3B**)；Table 2 Meta-World（p.11）：π0 (**3.5B**-Paligemma) / π0 (**3.5B**)；Table 3（p.11）：π0 (**3.5B**)。→ 同一 baseline 有两个参数量，且 "Paligemma-3B" 又是第三种写法；"~7×"（§4.5 p.11）按 3.3B/0.45B=7.3 或 3.5/0.45=7.8 都能凑上，但引用时需注明版本。

**C6. `# Params` 列不完整。**
Table 2（p.11）中 Diffusion Policy、TinyVLA、π0 (Paligemma-3B) 三行都没有给参数量；"Paligemma-3B" 是模型家族名而非行内声明。→ 无法从表内验证"参数效率"对比。

**C7. Table 2 的 Avg 精度与舍入不统一。**
SmolVLA 行为两位小数（82.75、87.3、88.75；Meta 56.95、57.3、68.24），其他行 1 位；Table 10 的 Avg 又保留 2 位（80.25/75.25）而其它同值表写 80.3。数值本身算术正确（见各表复核），但排版不一致，引用时注意别把 82.75 当成更高精度的测量。

**C8. §4.5 真实机器人段落的表号与描述互相错位（PDF 与 HTML 渲染还不同）。**
原文（完整 PDF p.11）："In **Table 3**, we evaluate SmolVLA on **four real-world tasks**. For the **SO101** benchmark, the model is trained on a combination of **three datasets**, and success rates are reported per task as well as on average."；arXiv **HTML 同一句渲染为 "In Table 4"**。
- Table 3 = SO100 三任务（Pick-Place/Stacking/Sorting），不是"四个任务"；
- "SO101 benchmark 训练于三个数据集组合、报告 per-task 和 average"描述的其实是 Table 3（SO100 三数据集）；
- Table 4 = SO101 单任务（Pick-Place-Lego），只有 ID/OOD 两列，无"三个数据集"、无 per-task 平均。
→ 该段落无论按 PDF（Table 3）还是 HTML（Table 4）都指错/描述错位；引用这段时要自己拆开对应 Table 3 与 Table 4。

**C9. Table 3 把 single-task 的 ACT 与 multi-task 的 SmolVLA/π0 放在一起比较。**
caption（p.11）与行分组明示：ACT 在 "Single-task Training" 组，π0/SmolVLA 在 "Multi-task Training" 组；§4.5 p.11 仍说 "SmolVLA outperforms both ACT, which is trained individually on each task, and π0…"。→ 78.3 vs 48.3 的差距同时混入了"算法/模型大小/训练范式/每个任务的数据量"多个变量，"超过 ACT"的结论不是受控比较（ACT 也没有 multi-task 版本作为对照）。

**C10. 真实机器人（Tables 3/4/5、Figure 5a）完全没有评测 trial 数与方差。**
只有训练示范数（每任务 50 demos，§4.1 p.8），没有说每任务评测几次、几个起始位置、是否重复测量；所有成功率都是单个数。→ 78.3 vs 61.7、90 vs 50 这类 5–20 分的差距无法判断显著性。

**C11. 真实机器人的 "success rate" 实际是分段得分平均，不是二值成功率。**
§4.1 p.8："For real-world evaluations, we adopt a more fine-grained scoring approach by decomposing each task into subtasks… a score of 0.5 for successfully picking the cube and an additional 0.5 for correctly placing it…"；Sorting 为 0.25×4。→ Table 3/4/5 的 "%" 是部分完成得分；baseline（ACT/π0）是否用完全相同的评分脚本没有说明，跨策略比较有度量口径风险。

**C12. Table 5 缺 "single-task + pretrained" 这一格，两个因素没有完全交叉。**
表（p.12）只有：single/no-pt（40.0）、multi/no-pt（51.7）、multi/pt（78.3）；缺 single/pt。→ 预训练收益（51.7→78.3）与多任务收益（40→51.7）各自都不是在对方固定水平下测的，无法估计交互作用；且单任务组只有 3 个任务各自训练，和"多任务"组的数据量/步数是否对齐也未说。

**C13. Table 5 最后一行与 Table 3 的 SmolVLA 行是同一组数字（重复使用同一实验）。**
Table 3（p.11）SmolVLA 75/90/70/78.3 = Table 5（p.12）multi-task/pt 行 75/90/70/78.3；Figure 5(a) Sync 行也是 75/90/70/78.3。→ 阅读时别把三处当成三次独立验证；论文没有说明它们是否为同一次评测。

**C14. Figure 5 的 "similar success rates" 偏乐观。**
caption（p.12）"Asynchronous inference achieves similar success rates"；实际 Figure 5(a)：Avg **73.3 vs 78.3（−5.0）**，Sorting **50 vs 70（−20）**，只有 Pick-Place 80 vs 75（+5）。→ 异步推理在长程/更敏感任务上明显掉点，"similar" 只在 Pick-Place/Stacking 成立（Stacking 持平 90）。

**C15. Figure 5 的超参只在 Pick-Place 上调过，却用于三任务结论。**
caption（p.12）："Hyperparameters have been optimized for Pick-Place and reused across tasks." → 异步的 g、相似度阈值 ε 等如果对任务敏感，Sorting 掉 20 分可能来自调参偏差而非 async 本身；论文没有做 per-task 调参对照。

**C16. Figure 5(b)/(c) 的 "Total/Avg/Std" 分母不一致。**
(b) Total/Avg：137.5/10=13.75、97.0/10=9.70 → 10 次；(c) Total/Avg：9/5=1.8、19/5=3.8 → 5 次。正文（§4.6 p.12）只说第一套实验 "across 10 trials and 5 different cube positions"，第二套只说 "from different positions"。→ 两个子表对应的 run 数不同（10 vs 5）且 Std 的 n 未注明，比较 "19 vs 9" 时要小心统计口径。

**C17. 固定时间评测的时间上限不是确定值。**
§4.6 p.12："we fix a time limit (**e.g., 60 seconds**)" → "e.g." 说明 60 s 只是举例，论文没有给出实际使用的时限；Figure 5(c) 的 "19 vs 9" 无法严格复现。

**C18. 全文没有任何实测推理延迟/控制频率/吞吐数字。**
仅有：30 fps→Δt=33 ms 的设定（§3.3 p.7）、g=0.7（p.7）、端到端任务时间（Figure 5b p.12）；摘要（p.1）与 §3.3 的 "higher control rates" 没有量化支撑。→ "异步推理降低延迟/提高控制频率" 目前只有间接证据（任务更快、固定时间内更多），不是直接测量；E[ℓ_S]（单 chunk 推理时延）在公式里始终是符号。

**C19. Table 8 正文与数字矛盾（skip-half vs 缩小 VLM）。**
§4.7 p.13："Table 8 indicates skipping every second layer performs better than training a smaller VLM"；表：Skip %2 Avg **75.5** vs VLM-256M Avg **75.8** → 更小 VLM 反而略好（+0.3），正文说反了（当然 0.3 分在无方差情况下也说明不了谁更好）。

**C20. Table 9 的 caption/正文与数字不符，且主模型选的 ×0.75 不是最优点。**
caption（p.13）："Larger capacities yield better success rates"；数字：×1.00 **82.3** > ×0.50 **80.3** > ×0.75 **77.5** > ×0.25 73.8（非单调）；§4.7 p.13 称 ×0.75 "achieves a good balance between performance and efficiency"，但论文**没有给出任何效率数字**（显存/延迟/FLOPs），且 ×0.75 比 ×0.50 低 2.8 分、比 ×1.00 低 4.8 分。→ "0.75×d" 的选择缺乏数据支撑。

**C21. Table 11 正文与 SA 行数字直接矛盾。**
§4.7 p.13："including state information in the VLM leads to significantly better performance **for both the CA and SA variants**"；表：Prefix SA **53.3** vs Suffix SA **74.8**（prefix 反而差 21.5）。CA 行才支持正文（80.3 vs 73.3）。→ Prefix SA 的崩点（S 62、Long 20）很可能实现/配置有问题，但论文未解释。

**C22. 同一名义配置在不同表给出不同数字，且没有 seed/方差来解释。**
- Table 6 "CA" Avg **79.0**（p.13） vs Table 11 "Prefix CA" Avg **80.3**（p.14）；
- Table 7 "Causal"（=Table 6 SA）Avg **74.5** vs Table 11 "Suffix SA" Avg **74.8**；
- Table 2 SmolVLA(0.45B) LIBERO Avg **87.3**（p.11） vs 消融默认配置 80.3/80.25（Tables 10/11/12/13，p.14）。
→ 若这些是同配置，则单次运行的波动至少 ~1–3 分（更不用说 87.3 vs 80.3 的 7 分差）；而论文用 0.3–2.8 分的差距下结论（C19/C20）。另外 Table 11 的 "CA/SA" 是否等于 Table 6 的纯 CA/纯 SA 也没有说明。

**C23. 消融实验的 base（80.3）无法对应主实验（87.3），消融协议不完整。**
§4.7 p.12 只说 "All ablations are conducted on LIBERO…trained from scratch without any pretraining…VLM frozen, only action expert trained from scratch"；没有说消融用多少数据、多少训练步、什么 batch、评测多少 trials（推测沿用 §4.1/§4.3，但未明说）。而 Table 2 的 SmolVLA(0.45B) LIBERO Avg 87.3，比本应同架构的消融基准高约 7 分。→ 读者无法把 ablation 的结论（如"16 层足够"）映射回主模型的实际配置。

**C24. 主模型默认设置与自身消融结论相反/不一致。**
- 主模型 chunk **n=50**（§4.3 p.10），但 Table 12 显示 n=10 Avg **84.0** > n=50 **80.3**（p.14）；caption 只说 "between 10 and 50 good balance"，没有给效率测量来解释为何选 50。
- 模拟推理"每执行 1 步就重新观测"（§4.3 p.10），对应 Table 13 的 1 步 = **80.3**；而 10 步 = **82.8** 更高（p.14）。
→ 主结果的配置不是消融里最好的，且"效率 vs 精度"的取舍只有定性描述。

**C25. Ablation 表的第 4 列写 "10" 而 Table 2 写 "Long"，含义未统一。**
Tables 6–13（p.13–14）列头为 `S / O / G / 10 / Avg`；Table 2（p.11）为 `Spatial / Object / Goal / Long`。合理理解是 LIBERO 的 Long suite（10 个任务），但正文没有交代；引用这些列时不能把 "10" 误读成"10 个任务的平均"或数值 10。此外，消融是否也跑满 4×10 tasks × 10 trials 未说明。

**C26. Table 4 的 OOD 协议过于简略。**
§4.5 p.11："For OOD evaluation, the Lego object is placed in novel positions not previously encountered during training."；没有说明 OOD 位置数、每位置 trial 数、位置偏移范围，也没有说明 ID/OOD 各占总评测多少。→ "90/50 vs 70/40（Table 4）" 的 OOD 泛化结论可复现性低。

**C27. 训练/硬件细节不足以复现。**
只有 §4.3 p.10 一段：4 GPUs（型号未说）、~30k GPU hours（项目级，非分阶段）、bfloat16+torch.compile、accelerate；没有微调阶段 lr/optimizer/weight decay/seed/数据采样权重/动作归一化/评测 checkpoint 选择标准（详见 §2.4）。论文 §5 声称 "provide detailed instructions to facilitate full reproducibility"（p.14），但 v1 正文可复现信息明显不足。

**C28. 社区数据集的筛选与清单不可复现/不可核对。**
§3.2 p.5："filtered according to embodiment type, episode count, overall data quality, and frame coverage"（无阈值）；A.1 清单（p.20–24）逗号/空格混排、条目粘连，无法可靠计数核对 481；清单中 `Ashton3/lerobot-aloha`（p.21）等命名与 §5.1 p.15 "datasets collected from a single robot type (SO100)" 可能冲突。→ "481 datasets / 22.9K episodes / 10.6M frames" 三个数只能整体采信，无法从附录重建。

**C29. "10× larger"（摘要 p.1）与 "~7×"（§4.5 p.11）量级口径不一。**
摘要："performance comparable to VLAs that are 10× larger"；§4.5 对 π0 说 "~7×"；若对 OpenVLA(7B) 则是 ~15.6×（本笔记推算 7B/0.45B）。→ 宣传语与正文的对比对象不同，引用时以表格里的具体参数量为准。

**C30. "训练快 40% / 显存少 6×" 缺少测量协议与绝对值。**
§4.5 p.10："compared to π0, SmolVLA is around 40% faster to train and consumes 6× less memory."；没有给硬件、batch/token 数、步数、绝对显存/耗时，也没有说明是否同 batch 同硬件。→ 只能当作定性线索。

**C31. 真实机器人数据量与"多任务"的说法需要留意。**
§4.1 p.8："we collected three datasets using the SO-100 robot arm … each corresponding to a different manipulation task. Each dataset … 50 demonstrations"；§4.5 HTML/PDF 段落又说 "trained on a combination of three datasets"（见 C8）。→ SO100"多任务"= 3 任务 × 50 demos = 150 demos 的混合，数据规模很小；"multi-task" 的泛化结论不宜外推。

---

## 8. 存疑点汇总表（不确定的数字/说法）

| # | 存疑内容 | 原文片段 / 依据 | 出处 | 判断/建议 |
|---|---|---|---|---|
| 1 | Table 1 的 "~10M episodes" | "At $\sim$ 10M episodes, our pretraining set…"；表内 22.9K episodes / 10.6M frames | Table 1 caption p.5；摘要 p.2；§5.1 p.15 | **笔误**，应为 ~10M frames；episodes = 22.9K（≈23K） |
| 2 | 0.24B / 2.25B 的配置组成 | 正文只给主模型 450M = VLM(16/32 层)+~100M expert；Table 8 出现 "500M VLM" 与 "VLM-256M" | §4.3 p.10、§3.1 p.4、Table 8 caption p.13 | **论文未定义**；推断 0.24B≈256M VLM 版、2.25B≈2.2B VLM 版（存疑） |
| 3 | π0 参数量 3.3B 还是 3.5B | "3.3 billion parameters" vs 表中 3.5B | §4.4 p.10；Table 2 p.11；Table 3 p.11 | 同一模型两种标法，**以表格为引用口径并注明** |
| 4 | E[ℓ_S]（单 chunk 推理延迟）具体值 | 只出现符号，未给数值 | §3.3 p.7 | **论文未报告**；"异步降低延迟"只有端到端时间证据 |
| 5 | 控制频率 Hz | 只有 "real-world frame-rate of 30 frames per second, Δt=33 ms" | §3.3 p.7 | 30 Hz 是设定值；无实测下发频率 |
| 6 | Figure 5(b) 的 10 次 vs Figure 5(c) 的 5 次 | Total/Avg 反推分母不同；正文只说 "10 trials and 5 positions" | Figure 5 p.12、§4.6 p.12 | 统计口径**存疑**，Std 的 n 未给 |
| 7 | 固定时间上限 | "we fix a time limit (**e.g., 60 seconds**)" | §4.6 p.12 | 60 s 只是举例，实际值**未报告** |
| 8 | Table 8 §4.7 关于 Skip%2 的结论 | 正文 "performs better than training a smaller VLM" vs 75.5 < 75.8 | §4.7 p.13、Table 8 p.13 | 正文与数字**矛盾** |
| 9 | Table 9 的 "Larger capacities yield better" | ×0.75=77.5 < ×0.50=80.3 | Table 9 p.13 | caption/正文与数字**不符**，且无效率测量 |
| 10 | Table 11 的 "state in VLM 对 SA 也更好" | Prefix SA 53.3 vs Suffix SA 74.8 | §4.7 p.13、Table 11 p.14 | 正文与数字**矛盾**（SA 反向） |
| 11 | §4.5 段落表号（Table 3 还是 Table 4） | PDF p.11 印 "In Table 3"；HTML 同句印 "In Table 4"；两者都与描述不符 | §4.5 p.11（PDF）/ HTML | 原文/版本渲染差异 + 描述错位，见 C8 |
| 12 | 消融 base 与主结果差 7 分 | Table 2 SmolVLA(0.45B) 87.3 vs 消融默认 80.3 | Table 2 p.11 vs Tables 10–13 p.14 | 协议不明，**不能直接互推** |
| 13 | 481 个数据集清单可否核对 | A.1 清单缺逗号粘连、多栏排版 | A.1 p.20–24 | 计数不可靠；筛选阈值未给 |
| 14 | "single robot type (SO100)" | 清单出现 `Ashton3/lerobot-aloha` 等 | §5.1 p.15；A.1 p.21 | **存疑**（也可能只是命名） |
| 15 | 真实机器人每任务评测 trial 数 | Tables 3/4/5 无 N | p.11–12 | **未报告** |
| 16 | TinyVLA 引文作者 | 表内 (Zhou et al., 2024)，但 §2/参考文献为 Wen et al. 2024（Zhou et al. 2024 = TinyLLaVA） | Table 2 p.11；§2 p.3；参考文献 | **引文错误** |
| 17 | Diffusion Policy 引文 | 表内 LIBERO 块 (Khazatsky et al., 2024)=DROID 论文；Meta 块 (Chi et al., 2023) | Table 2 p.11；参考文献 | **引文错误/不一致** |
| 18 | 异步 vs 同步"成功率相似" | 73.3 vs 78.3（Sorting 50 vs 70） | Figure 5 caption p.12 | 说法偏乐观，见 C14 |
| 19 | 30k GPU hours 是否只算预训练 | "the project consumed approximately 30k GPU hours" | §4.3 p.10 | 项目级总消耗，非预训练单项 |
| 20 | LIBERO "10 trials/task" 是否也用于 baseline 引文 | §4.1 说自家协议；Table 2 caption 仅说 "Baselines scores from…" | §4.1 p.8；Table 2 caption p.11 | baseline 协议**不可比**风险 |

---

## 9. 覆盖报告与缺口

### 9.1 已覆盖

- **全部 13 张编号表**（Table 1–13）：caption 原文、列/行含义、逐格数字、支持的 claim/作者结论、数据来源与可比性、算术复核。
- **Figure 5 的三张子表**：(a) 性能、(b) 任务完成时间、(c) 固定时间内完成量；含 caption 原文与 Sync/Async 全部数字。
- **训练超参**：pretrain/sim-finetune/real-finetune 三套（步数、batch、warmup、cosine lr、AdamW、图像尺寸、chunk、flow 步数、frozen VLM、bfloat16、torch.compile、accelerate、4 GPUs、30k GPU hours）。
- **数据量**：Table 1 总量、LIBERO 1,693 episodes、Meta-World 2,500 episodes、真实数据 4×50 demos、与 OpenVLA 1M trajectories 的对比。
- **评测协议**：LIBERO/Meta-World 每任务 10 trials；真实机器人部分（缺 trial 数）；异步推理两套实验。
- **参数规模三档** 0.24B/0.45B/2.25B 的原文依据与推断边界。
- **异步推理的延迟/控制频率/时间数字**：30 fps/Δt=33 ms、g=0.7、n=50、13.75→9.70 s、137.5→97.0 s、9→19 cubes；并明确指出**没有**实测 ms/Hz/吞吐。
- **附录摘录**：A.1 的 task annotation prompt 原文 + 数据集清单说明 + 附录无额外实验的结论。
- **批判性清单 31 条**（C1–C31）与 **20 条存疑点**（§8）。

### 9.2 任务清单里提到但论文中**不存在**的内容（避免误引）

- **pi0.5 / Dreamer**：v1 没有任何表格或正文与它们比较（Table 2 只有 DP / Octo / OpenVLA / π0 / TinyVLA）。
- **附录超参数表 / 附录实验表**：不存在；全部 16 个网格都在正文。
- **不同数据集规模的训练曲线、视觉 token 数 ablation**：正文没有对应表格/图（每帧 64 token 只在 §3.1 p.4 陈述，无对比实验）。
- **真实机器人更多任务/更多 trial 的补充结果**：不存在（4 个任务，Tables 3/4/5 + Figure 5）。
- **per-task 明细表**：不存在；只有 suite 级/任务级成功率，没有逐 episode 结果。

### 9.3 建议回查位置（按主题）

| 主题 | 位置 |
|---|---|
| 表格数字/caption | HTML `S3.T1`、`S4.T2`、`S4.T4.fig1/fig2`、`S4.T5.fig1`、`S4.F5.sf1–sf3`、`S4.T6–T13`；或 `table-grids-raw.md` + `tables-and-figures.md` |
| 训练细节 | §4.3 p.10 |
| 数据统计 | Table 1 p.5、§3.2 p.5、§4.1 p.8、§5.1 p.15 |
| 评测协议 | §4.1 p.8–9、§4.2 p.9 |
| 异步推理机制 | §3.3 p.5–8、Algorithm 1 p.6、Figure 2/3 p.6–7 |
| 异步推理结果 | §4.6 p.12、Figure 5 p.12 |
| 消融 | §4.7 p.12–14、Tables 6–13 p.13–14 |
| 限制与失败案例 | §5.1 p.15 |
| 任务标注 prompt / 数据集清单 | Appendix A.1 p.20–24 |

---

*（本文件为数字化底稿；所有「存疑」「本笔记推算」均已显式标注，未引入论文以外的数据。）*
