# 精读优先级（Reading Priority）

> 原则：**先读"会推翻我方案"的论文，再读"支撑我方案"的论文。**
> 时间预算：12 篇合计约 14 小时（中文笔记 + 每篇 10 行要点）。分两批：**第一批 4 篇（3 天）**必须先读，因为它决定题目还成不成立。

---

## 第 0 批：必读，决定题目生死（3 天，4 篇）

### 1. RTC — Real-Time Execution of Action Chunking Flow Policies ★★★★★
`arXiv:2506.07339v2`（NeurIPS 2025 自述）

**先读**
- 摘要第 2 句起（问题定义："does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries"）
- 方法节：**freezing**（哪些动作被冻结）与 **inpainting**（其余如何补）的具体判据
- 实验节：Kinetix 12 任务 + 6 个真机双臂任务的设置；**延迟是如何注入的**（这是关键）

**为什么必须读**：这是我方案里 A1 的**最大威胁**。它解决的问题和 A1 几乎同义。
**读后要能回答**：① RTC 的"延迟"是注入到哪一层？② 它是否假设异步架构？③ 如果我的评测入口是同步阻塞，RTC 的结论能不能直接搬？

---

### 2. FutureRTC — Anticipatory-Conditioned Action Chunking ★★★★★
`arXiv:2607.24008v1`

**先读**
- 摘要（它如何描述"prediction-execution misalignment"）
- 三个模块：state correction / observation prediction（motion-aware feature transport）/ policy consistency loss
- 实验：延**迟鲁棒性**的量化方式

**为什么必须读**：RTC 之后还有人做，说明这个方向**没有收敛**，但也说明门槛在快速抬高。
**读后要能回答**：RTC + FutureRTC 之后，"延迟鲁棒"还剩什么可做？我的同步入口 + 机械耦合是否落在它们的假设之外？

---

### 3. KWS — Knowing When to Stop ★★★★★
`arXiv:2609.00908v3`

**先读**
- cross-attention 熵随预测时域上升并趋于平台的**那张图**（这是全文的支点）
- 训练无关截断机制的判据（"sustained high-entropy plateau" 如何定义、阈值怎么定）
- 固定 E 的网格实验（LIBERO `E ∈ {2,4,6,8,10}`、RoboTwin 的 `E` 网格）

**为什么必须读**：**自适应执行长度已经被它做掉**。同时它是唯一给出固定 E 网格的论文，可作为我 F1/F2 的实验设计参照。
**读后要能回答**：① 它的固定 E 网格里，**执行的是前缀还是后缀**（有没有声明）？② 它的 E 默认值与 `P` 是多少？③ 它的"自适应"和我的"固定 E 扫描"是不是同一件事？

> **这一篇决定 G1 的新颖性。** 如果它声明了实现语义，我的 G1 直接失效，题目必须立刻转向 F2。

---

### 4. OpenVLA-OFT — Fine-Tuning VLAs: Optimizing Speed and Success ★★★★★
`arXiv:2502.19645v2`（RSS 2025 自述）

**先读**
- §Fine-tuning recipe 四个组件：parallel decoding / action chunking / continuous action representation / L1 regression
- LIBERO 结果表（76.5 → 97.1）与吞吐 26× 的**测量口径**（相对什么基线、什么硬件）
- 附录里的 `num_open_loop_steps` 说明（如果存在）

**为什么必须读**：这是我唯一的代码基座；也是 v3 的 `P=8`、后缀行为的来源。
**读后要能回答**：① 论文自己怎么定义 open-loop steps？② 有没有说明"从第几步开始执行"？③ 吞吐 26× 是端到端还是仅动作生成？

---

## 第 1 批：构建论证（4 天，4 篇）

### 5. AAC — Adaptive Action Chunking at Inference-time ★★★★★
`arXiv:2604.04161v2`（CVPR 2026 自述）
**先读**：动作熵作为信号的定义；"最小动作幅值"约束为什么必要；真机实验的硬件声明。
**读后要能回答**：它与 KWS 的信号有何本质差异？两条路线并存说明什么（是不是说明"选 E"这件事本身没有唯一解）？

### 6. SmolVLA — A VLA Model for Affordable and Efficient Robotics ★★★★★
`arXiv:2506.01844v1`
**先读**：**异步推理栈**那一节（"decoupling perception and action prediction from action execution"）；单卡训练的配置；消费级 GPU/CPU 部署的具体声明。
**为什么读**：这是"异步执行"在**系统侧**的已发表设计，也是我能否在 12–24 GB 卡上跑通的关键。
**读后要能回答**：它的异步栈与 RTC 的算法层异步是什么关系？我若用同步入口，是否等于放弃了它的优势？

### 7. From Inference Efficiency to Embodied Efficiency ★★★★★
`arXiv:2603.19131v1`
**先读**：三类受控研究（模型压缩 / token 稀疏化 / 动作序列压缩）的设计；三条反直觉发现；具身效率指标的定义（完成时间、轨迹平滑度、累计关节转角、运动能量）。
**为什么读**：它给我"怎么论证指标"的模板，也给我"别只报 FLOPs"的弹药。
**读后要能回答**：我该报哪几个具身效率指标？jerk 怎么算？

### 8. VLAQuantBench ★★★★★
`arXiv:2609.25376v1`
**先读**：明确指出 `P` 与 `E` 必须区分的那段（它的措辞）；Table 10（显存 vs 延迟）；"保护一个 28,672 参数输出投影"的实验。
**为什么读**：它是我引用最多的"同领域已认识到 P≠E"的证据，也是"省显存不等于省时间"的唯一来源。
**读后要能回答**：它做过 50 vs 10 的扰动，我得说清我和它的差别在哪。

---

## 第 2 批：机械侧（3 天，3 篇）

### 9. Fin-Ray 优化设计（FREM） ★★★★★
`arXiv:2606.03798v1`
**先读**：4 条准则（tip displacement / total deflection / stress distribution / contact force）；参数化方式；FREM 与 ANSYS 的误差对比（3% vs 2%）；最终构型参数（30 mm / 10 mm / 7 筋 −15° / 1 mm）。
**为什么读**：**这是我机械章节的写作文本模板**。本科可辩护的"设计过程"长什么样，看它就够了。
**要学什么**：怎么把一个夹爪设计写成"有准则、有模型、有误差、有最优解"的工程论证，而不是"我画了个夹爪"。

### 10. TacFR-Gripper ★★★★
`arXiv:2311.10611v1`
**先读**：kinematic analysis / workspace analysis / FEA 三节；多种构型下的抓取成功率实验设计。
**要学什么**：机械论文里"运动学 + 工作空间 + 有限元 + 抓取成功率"这条链怎么串；实验对象（everyday objects）怎么选。

### 11. LIBERO ★★★★
`arXiv:2306.03310v2`
**先读**：四个 task suite 的构成（130 任务）、评测协议（每任务多少 episode、初始状态如何固定）。
**为什么读**：我的所有仿真实验都在它上面；`control_freq` 与 `Δt_sim` 从它的代码/文档里来（v3 的 U5）。
**要学什么**：把"评测协议"写进论文的方法节。

---

## 第 3 批：按需查阅（不精读）

| 文献 | 什么时候查 |
|---|---|
| `2503.02310` PD-VLA | 若要解释"为什么并行解码能降 L 而不改 E" |
| `2501.09747` FAST | 若要解释动作 tokenization 与执行长度正交 |
| `2510.24795` Efficient VLA Survey | 写绪论时找分类框架（注意：它的三支柱里**没有执行策略**） |
| `2602.03782` QVLA / `2506.07530` BitVLA | 若后续要做量化分支 |
| `2410.24164` π0 / `2504.16054` π0.5 | 若要换基座 |
| `2510.10273` TIAGo-Isaac / `2307.01350` SATYRR / `2410.06911` | 若方案③（差速底盘）被重新提起 |

---

## 每日节奏建议（与 11-13 定题截止对齐）

| 日期 | 做什么 |
|---|---|
| 11-03 / 11-04 | 读完 RTC + OFT（第 0 批前两篇），**先回答"我的 A1 还剩什么"** |
| 11-05 / 11-06 | 读完 KWS + FutureRTC，**决定 G1 是否还有意义** |
| 11-07 | 写 1 页"与 RTC/KWS 的差异说明"，发给导师 |
| 11-08—11-10 | 第 1 批 4 篇 + 第 2 批 3 篇（略读为主） |
| 11-11 / 11-12 | 把 `initial_research_report.md` 的第 4/7/9 节改写成本人语气，准备与导师面谈 |
| 11-13 | 完成互选 |

---

## 每篇读完必须产出的 10 行笔记模板

```
标题 / arXiv / 版本日期：
它解决什么问题：
它的核心机制（一句话）：
它假设了什么（硬件/架构/数据）：
它报了什么数字（口径 + 硬件）：
它没报什么（在检索边界内）：
它的代码/权重是否公开：
它与我的关系（威胁 / 支撑 / 工具）：
我要引它写在哪一句：
读完后我改了什么判断：
```
