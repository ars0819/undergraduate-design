# 文献矩阵（Literature Matrix）

> 检索日期：2026-11-03｜检索通道：arXiv Atom API（唯一可用，见 `search_log.md`）
> 核验状态定义：**M** = 元数据（标题/作者/日期/摘要）经 Atom 条目核验；**A** = 仅摘要核对；**F** = 全文；**X** = 未获取
> 本矩阵所有条目均为 **M/A 级**，**无一条达到 F 级全文核验**（OFT 除外，其数值沿用 v3 的正文核对记录）。

---

## 主题 1：动作分块、执行策略与推理延迟（核心）

| # | 标题 | 主要作者 | 首发 / 更新 | arXiv | venue（论文自述，未第三方核实） | 核验 | 与毕设的关联 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| 1 | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT) | T. Z. Zhao, V. Kumar, S. Levine, C. Finn | 2023-04-23 | 2304.13705v1 | — | M | 动作分块起点；10 min 演示达 80–90% | ★★★ |
| 2 | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, S. Song | 2023-03-07 / 2024-03-14(v5) | 2303.04137v5 | 期刊版（原 RSS 2023） | M | 滚动时域与分块功能重叠；+46.9% | ★★ |
| 3 | **Real-Time Execution of Action Chunking Flow Policies (RTC)** | K. Black, M. Y. Galliker, S. Levine | 2025-06-09 / 2025-12-05(v2) | 2506.07339v2 | NeurIPS 2025 | M | **直接覆盖"延迟→块边界停顿"**；冻结+修补，免重训 | ★★★★★ |
| 4 | **FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking** | H. Jiang, Y. Zou, B. Liang, B. Liu, F. Meng, S. Liu | 2026-07-27 | 2607.24008v1 | — | M | 异步执行的预测-执行错位；状态校正+观测预测+一致性损失 | ★★★★★ |
| 5 | **Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs (KWS)** | R. Xu, X. Shan, S. Dai, Y. Wang, J. Yu | 2026-09-01 / 2026-09-16(v3) | 2609.00908v3 | — | M | **自适应执行长度的最强先行工作**；训练无关截断 | ★★★★★ |
| 11 | Adaptive Action Chunking at Inference-time for VLAs (AAC) | Y. Liang, X. Wang, K. Wang, S. Wang, X. Peng, H. Chen, D. K. H. Chua, P. Vadakkepat | 2026-04-05 / 2026-04-10(v2) | 2604.04161v2 | CVPR 2026 | M | 动作熵作自适应信号；与 KWS 构成两条独立路线 | ★★★★★ |
| 7 | SmolVLA: A VLA Model for Affordable and Efficient Robotics | M. Shukor, D. Aubakirova, F. Capuano, et al.（HF） | 2025-06-02 | 2506.01844v1 | — | M | **自带异步推理栈**；单卡训练、消费级/CPU 部署 | ★★★★★ |
| 18 | PD-VLA: Accelerating VLA Integrated with Action Chunking via Parallel Decoding | W. Song, J. Chen, P. Ding, et al. | 2025-03-04 / 2026-02-25(v2) | 2503.02310v2 | IROS 2025 | M | 并行定点迭代；7-DoF 执行频率 2.52× | ★★★ |
| 16 | FAST: Efficient Action Tokenization for VLAs | K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, S. Levine | 2025-01-16 | 2501.09747v1 | — | M | DCT+BPE tokenization，**与执行长度正交** | ★★ |
| 10 | **Fine-Tuning VLAs: Optimizing Speed and Success (OpenVLA-OFT)** | M. J. Kim, C. Finn, P. Liang | 2025-02-27 / 2025-04-28(v2) | 2502.19645v2 | RSS 2025 | **F**（v3 正文核对） | 本项目的代码基座；`P=8`；LIBERO 76.5→97.1，吞吐 26× | ★★★★★ |
| 9 | OpenVLA: An Open-Source VLA Model | M. J. Kim, K. Pertsch, S. Karamcheti, et al. | 2024-06-13 / 2024-09-05(v3) | 2406.09246v3 | — | M | 7B / 970k；+16.5% vs RT-2-X；LoRA 微调 + 量化部署 | ★★★ |
| 12 | π0: A Vision-Language-Action Flow Model for General Robot Control | K. Black, N. Brown, D. Driess, et al. | 2024-10-31 / 2026-01-08(v4) | 2410.24164v4 | RSS 2025 | M | 流匹配动作专家；含移动操作本体数据 | ★★★ |
| 13 | π0.5: a VLA Model with Open-World Generalization | Physical Intelligence, K. Black, N. Brown, et al. | 2025-04-22 | 2504.16054v1 | — | M | KWS/AAC 的核心评测基座 | ★★★ |
| 14 | RT-1: Robotics Transformer for Real-World Control at Scale | A. Brohan, N. Brown, J. Carbajal, et al. | 2022-12-13 / 2023-08-11(v2) | 2212.06817v2 | — | M | 规模化的历史起点 | ★ |
| 15 | RT-2: VLA Models Transfer Web Knowledge to Robotic Control | A. Brohan, N. Brown, J. Carbajal, et al. | 2023-07-28 | 2307.15818v1 | — | M | 动作即文本 token；6k 评测 trial | ★ |
| 17 | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, P. Stone | 2023-06-05 / 2023-10-14(v2) | 2306.03310v2 | — | M | 4 套件 130 任务的评测基准 | ★★★★ |

---

## 主题 2：效率度量与部署（方法学支撑）

| # | 标题 | 主要作者 | 首发 / 更新 | arXiv | venue | 核验 | 与毕设的关联 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| 6 | **From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for VLAs** | Z. Li, H. Yang, Z. Chen, Y. Chen, Y. Lin, C. Li | 2026-03-19 | 2603.19131v1 | — | M | **质疑 FLOPs/吞吐类效率度量**；压缩可能增加端到端成本 | ★★★★★ |
| 8 | **VLAQuantBench: Closed-Loop Evaluation of PTQ for VLAs** | J. Xu, Q. Jin, M. Chen, S. Wang, Y. Sui, Y. Shi | 2026-09-21 | 2609.25376v1 | — | M | 409 runs / 94,574 episodes；**明确指出 P 与 E 必须区分** | ★★★★★ |
| 20 | A Survey on Efficient Vision-Language-Action Models | Z. Yu, B. Wang, P. Zeng, H. Zhang, J. Zhang, Z. Wang, L. Gao, J. Song, N. Sebe, H. T. Shen | 2025-10-27 / 2026-09-28(v3) | 2510.24795v3 | TPAMI | M | 三支柱分类里**没有"执行策略"**，本身是 gap 信号 | ★★★ |
| 19 | VLA-Cache: Efficient VLA Manipulation via Adaptive Token Caching | S. Xu, Y. Wang, C. Xia, D. Zhu, T. Huang, C. Xu | 2025-02-04 / 2025-10-21(v2) | 2502.02175v2 | NeurIPS 2025 | M | 训练无关加速；1.7× CUDA 延迟、+15% 控制频率 | ★★ |
| 21 | BitVLA: 1-bit VLA Models for Robotics Manipulation | H. Wang, C. Xiong, R. Wang, X. Chen | 2025-06-09 / 2026-03-01(v2) | 2506.07530v2 | 自述 Work in progress | M | 三值权重；显存 11.0×、端到端延迟 4.4× | ★★ |
| 22 | QVLA: Not All Channels Are Equal in VLA Model's Quantization | Y. Xu, Y. Yang, Z. Fan, Y. Liu, Y. Li, B. Li, Z. Zhang | 2026-02-03 | 2602.03782v1 | ICLR 2026 | M | 通道级比特分配；29.2% 显存 / 98.9% 性能 / 1.49× | ★★ |

---

## 主题 3：末端执行器与抓取机构（机械设计侧，v3 缺失部分）

| # | 标题 | 主要作者 | 首发 | arXiv | venue | 核验 | 与毕设的关联 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| 23 | **Optimal Design and Analytical Modeling of a Soft Fin-Ray Effect Gripper Finger Using the Finite Rigid Elements Method** | S. Adeli, H. Sayyaadi | 2026-06-02 | 2606.03798v1 | — | M | **可直接复刻的设计流程**：4 准则→参数→FREM+ANSYS（误差 3%/2%）→最优构型 | ★★★★★ |
| 24 | **TacFR-Gripper: A Reconfigurable Fin Ray-Based Compliant Robotic Gripper with Tactile Skin** | Q. Cong, W. Fan, D. Zhang | 2023-11-17 | 2311.10611v1 | — | M | **运动学 + 工作空间 + FEA 的完整论证链**；多种构型的抓取成功率 | ★★★★ |
| 25 | AllTact Fin Ray: A Compliant Robot Gripper with Omni-Directional Tactile Sensing | S. Liang, Y. Guan, J. Xu, H. Qian, X. Zhang, D. Wu, W. Ding, R. Chen | 2025-04-25 | 2504.18064v1 | — | M | 一体成形硅胶指 + 视觉触觉；构型参考 | ★★★ |
| 26 | GelSight Fin Ray: Incorporating Tactile Sensing into a Soft Compliant Robotic Gripper | S. Q. Liu, E. H. Adelson | 2022-04-14 | 2204.07146v1 | IEEE RoboSoft 2022 | M | Fin Ray 系列起点；柔顺与感知的接口问题 | ★★ |
| 27 | GelSight Baby Fin Ray: A Compact, Compliant, Flexible Finger | S. Q. Liu, Y. Ma, E. H. Adelson | 2023-03-27 | 2303.14883v1 | IEEE RoboSoft 2023 | M | 小型化与柔顺性改进 | ★★ |

---

## 主题 4：移动操作（方案③评估用）

| # | 标题 | 主要作者 | 首发 / 更新 | arXiv | venue | 核验 | 与方案③的关联 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| 28 | Integration of the TIAGo Robot into Isaac Sim with Mecanum Drive Modeling and Learned S-Curve Velocity Profiles | V. Schoenbach, M. Wiedemann, R. Memmesheimer, M. Mosbach, S. Behnke | 2025-10-11 / 2025-10-14(v2) | 2510.10273v2 | IEEE CASE 2025（DOI 10.1109/CASE58245.2025.11163930） | M | 轮式底盘动力学的**仿真建模**，不是抓取表现研究 | ★★ |
| 29 | Dynamic Mobile Manipulation via Whole-Body Bilateral Teleoperation of a Wheeled Humanoid | A. Purushottam, Y. Jung, C. Xu, J. Ramos | 2023-07-03 | 2307.01350v1 | — | M | 形态最接近 OpenFleX；全身遥操作；抓 0.4 m/s 移动目标 | ★★★ |
| 30 | Combining Planning and Diffusion for Mobility with Unknown Dynamics | Y. Ravan, Z. Yang, T. Chen, T. Lozano-Pérez, L. P. Kaelbling | 2024-10-09 | 2410.06911v1 | 自述投 ICRA 2025 | M | 移动操作的长程规划；非抓取类操作 | ★ |

---

## 汇总裁量

| 主题 | 条目数 | 其中核心（★★★★以上） |
|---|---|---|
| 1 动作分块与执行策略 | 16 | 7 |
| 2 效率度量与部署 | 6 | 3 |
| 3 末端执行器（机械） | 5 | 2 |
| 4 移动操作 | 3 | 2 |
| **合计** | **30** | **12**（去重后核心 9 篇） |

> **覆盖度自评（诚实）**：目标 20–30 篇，实际 30 篇，**数量达标**；但
> (a) **机械侧仅 5 篇且全部为柔顺/Fin Ray 类**，缺平行二指刚体夹爪的力学分析文献——因为 arXiv 对机械工程期刊覆盖很差；
> (b) **无一篇来自 IEEE/ASME 期刊或中文核心期刊**（通道不可达）；
> (c) 全部为 M/A 级核验，**没有一篇在本轮读到全文**。
> 因此矩阵可作为"起步索引"，不可作为"领域全景"。

---

## 关键交叉证据（跨文献，用于报告中的判定）

| 判定 | 支撑条目 | 性质 |
|---|---|---|
| "调执行长度"已被做掉 | [11] AAC、[5] KWS | 两篇均自适应决定 E |
| "延迟导致块边界问题"已被做掉 | [3] RTC、[4] FutureRTC、[7] SmolVLA 异步栈 | 三篇独立解法 |
| "P 与 E 必须区分"已有他人提出 | [8] VLAQuantBench | 但**未讨论保留前 E 还是后 E** |
| 机械–策略耦合未见研究 | 主题 1 全部 16 篇均无机械自变量；主题 3 全部 5 篇均无策略执行长度因子 | **交叉为空集** |
| 移动底盘对抓取表现：无直接对照 | [28][29][30] | 三篇均非"底盘运动 → 抓取成功率" |
