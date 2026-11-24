# 检索日志（Search Log）

> 检索执行日期：**2026-11-03**（Asia/Shanghai）
> 检索执行者：DSH 文献调研会话
> 目的：记录实际发生的检索动作，使本次调研的边界可被复查。**未发生的检索不写入。**

---

## 1. 通道可用性实测（先测通道，再检索）

| 通道 | 目标 | 结果 | 时间 |
|---|---|---|---|
| `curl.exe` | `https://api.crossref.org/works/10.1038/nature12373` | **HTTP 000**（连接失败） | 12:23 UTC |
| `curl.exe` | `https://export.arxiv.org/api/query?id_list=2502.19645` | **HTTP 000** | 12:23 UTC |
| `curl.exe` | `https://api.openalex.org/works?filter=doi:10.1038/nature12373` | **HTTP 000** | 12:23 UTC |
| PowerShell `Invoke-WebRequest` | 上述三个 + `10.1109/ICRA.2019.8793971` | 全部失败：`基础连接已经关闭: 接收时发生错误。` | 12:23 UTC |
| `web_fetch` | `https://export.arxiv.org/api/query?...` | ✅ **HTTP 200**，返回 Atom XML | 12:23 UTC |
| `web_fetch` | `https://arxiv.org/abs/2502.19645` | ✅ HTTP 200，返回完整 HTML | 12:23 UTC |
| `web_fetch` | `http://export.arxiv.org/api/query?...`（http 而非 https） | ❌ `cross-origin redirect ... not followed`；改用 `https://export.arxiv.org` 直接域名后成功 | 12:23 UTC |
| `web_fetch` | `https://huggingface.co/docs/lerobot/so101` | ❌ `URL hostname "huggingface.co" resolves to a non-public IP address` | 12:23 UTC |
| `web_fetch` | `https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/README.md` | ❌ 同上 | 12:23 UTC |

**结论**：进程级出网（curl / Invoke-WebRequest）被环境阻断；仅 harness 的 `web_fetch` 与 `web_search` 可出网。
**因此 `paper-lookup` skill 规定的执行方式（curl 直连 Crossref/arXiv/OpenAlex/S2）在本环境不可用**，已按 skill 的「Error recovery → Report the failure」要求记录，并改用 `web_fetch` 直连 arXiv 官方 API 作为等效主通道。

---

## 2. 实际执行的检索式

### 2.1 arXiv 官方 API（主通道，`export.arxiv.org/api/query`）

| # | 类型 | 参数 | 返回 | 归入 |
|---|---|---|---|---|
| S1 | `id_list` | `2506.01844,2506.07339,2609.00908,2609.25376,2603.19131,2502.02175`（`max_results=6`） | 6/6 | 主题 1、2 |
| S2 | `id_list` | `2304.13705,2406.09246,2410.24164,2504.16054,2604.04161,2306.03310` | 6/6 | 主题 1 |
| S3 | `id_list` | `2607.24008,2501.09747,2503.02310,2506.07530,2602.03782,2510.24795` | 6/6 | 主题 1、2 |
| S4 | `id_list` | `2303.04137,2212.06817,2307.15818` | 3/3 | 主题 1 |
| S5 | `search_query` | `all:"Fin Ray" AND all:gripper`，`max_results=5`，`sortBy=relevance` | `totalResults=21`，取前 5 | 主题 3 |
| S6 | `search_query` | `abs:"mobile manipulation" AND abs:wheeled AND abs:grasping`，`max_results=4`，`sortBy=submittedDate desc` | `totalResults=5`，取前 4 | 主题 4 |
| S7 | `abs` 页 | `https://arxiv.org/abs/2502.19645` | 200 | 交叉核对 OFT（作者/日期/venue/数字） |

**S5/S6 的检索边界说明**：arXiv API 的 `all:` 检索覆盖标题/摘要/全文元数据；`totalResults=21` 与 `5` 为 API 自报总数，本次分别只取了前 5 与 4 条，**未做穷尽式分页**（依 skill 的 bounded-call 原则）。因此"机械侧只找到 5 篇"是**采样结果，不是领域总量**。

### 2.2 `web_search`（辅助发现，4 批 12 条 query）

| # | queries | 有效产出 |
|---|---|---|
| W1 | `adaptive underactuated two-finger gripper design grasping force analysis paper`；`3D printed robot end effector strength stiffness PLA PETG gripping force experiment`；`hand-eye calibration camera mounting robot arm grasping success rate study` | 提出若干出版社页面（ScienceDirect / IEEE / MM Science Journal）**均不可访问**；未得到可核验条目 |
| W2 | `Fin Ray effect gripper design compliant finger geometry grasping paper`；`two-finger parallel jaw gripper design optimization grasp force friction coefficient`；`SO-101 SO-100 LeRobot arm payload specs Feetech STS3215` | 定位到 Fin Ray 方向，但**返回的大多是索引页/PDF 片段**；SO-101 规格只找到第三方站（roboticscenter.ai），**非官方，按任务要求不采用** |
| W3 | `mobile manipulator wheeled base arm grasping success rate study 2025`；`FutureRTC action chunking VLA arXiv`；`SmolVLA asynchronous inference stack chunked action generation control rate` | **确认 FutureRTC 存在**（2607.24008）→ 直接转 arXiv API 核验 |
| W4 | 见 W1/W2/W3 合并 | — |

**W 系列的可靠性限制**：返回的是搜索引擎摘要与索引入口，**不能作为元数据来源**。因此 W 系列只用于「发现候选 → 拿到 arXiv ID → 回 arXiv API 核验」。凡未经 S1–S7 核验的条目不进入 `literature_matrix.md`。

---

## 3. 不可访问的来源（逐项列出）

| 来源 | 用途 | 状态 | 影响 |
|---|---|---|---|
| Crossref API | DOI 注册性检查、期刊卷期 | ❌ 连接失败 | **未做 DOI 注册性核验**；不改写任何 DOI |
| OpenAlex | 引用数、机构、主题 | ❌ | 无法给引用量 |
| Semantic Scholar | 引用图、TLDR | ❌ | 未做引用追踪（citation chaining） |
| IEEE Xplore | 机械工程期刊（TMECH/RA-L/T-ASE） | ❌ | **机械侧覆盖严重不足** |
| ScienceDirect / Elsevier | 机械设计类论文 | ❌（搜索可见标题，不可访问） | 同上 |
| ASME Digital Collection | 机构学/设计理论 | ❌ | 同上 |
| HuggingFace（LeRobot 文档、模型卡） | SO-101 官方规格、SmolVLA 权重 | ❌ 非公网 IP | **SO-101 规格缺失**；模型可用性未知 |
| GitHub（raw） | SO-ARM100 仓库、官方代码 | ❌ | 未核验代码仓库内容 |
| 中文数据库（CNKI/万方） | 国内机械设计工作 | ❌ | 无中文文献覆盖 |
| arXiv 全文 HTML | 读正文表格 | ⚠️ 未使用（避免超长上下文） | 全部条目停在摘要级；除 OFT 沿用 v3 正文核对 |

---

## 4. 去重与筛选规则

1. **以 arXiv ID 去重**：`id_list` 天然去重；`search_query` 结果与 `id_list` 交叉时按 ID 合并（本次 S5/S6 与 S1–S4 **无重叠**）。
2. **纳入标准（inclusion）**：
   - I1 主题相关：动作分块/执行策略/推理延迟/效率度量/末端执行器/移动操作；
   - I2 可核验：能从 arXiv Atom 条目读到标题、作者、日期、摘要；
   - I3 时效：以 2024-01 之后为主，经典工作（RT-1/RT-2/ACT/DP/LIBERO/Fin Ray 系列）不受此限。
3. **排除标准（exclusion）**：
   - E1 无 arXiv 条目且无法核验（大量 IEEE/Elsevier 候选）；
   - E2 仅关键词命中但主题不符（如 W2 返回的轮胎滑移率文献、医学信息学文献）；
   - E3 第三方聚合站数据（roboticscenter.ai 的 SO-101 规格）。
4. **去重结果**：30 条，无重复。其中核心（★★★★及以上）12 条，去重后核心 9 篇。

---

## 5. 检索边界的明确声明

以下表述在本报告及后续文档中**必须**保持同样口径：

- ✅ 允许：「**在本次 arXiv 检索边界内**未见报告」
- ❌ 禁止：「没有人做过」「首次发现」「填补空白」
- ❌ 禁止：对任何期刊/会议等级、影响因子、分区作断言（Crossref 与出版社均不可达）
- ❌ 禁止：把 arXiv `comment` 字段里的 venue 自述当作已核实事实（本报告统一标注为"论文自述，未第三方核实"）
- ❌ 禁止：把 `totalResults` 当作领域总量（仅前 N 条被读取）

---

## 6. 未完成、建议在下一轮补做

| # | 项 | 为什么重要 | 建议通道 |
|---|---|---|---|
| 1 | 机械工程期刊的平行二指夹爪力学分析（TMECH / Mechanism and Machine Theory / ASME JMD） | 现 5 篇全为柔顺/Fin Ray，缺刚体夹爪接触力学 | 学校图书馆 / IEEE Xplore（换网络环境） |
| 2 | SO-101 官方规格（负载、指长、关节力矩、电机型号） | B1 的载荷与强度校核无法开始 | 本机直连 HuggingFace LeRobot 文档 |
| 3 | KWS / AAC / RTC / FutureRTC 的**正文**（尤其是否声明实现语义、`E` 默认值） | 直接决定 G1 的新颖性判定 | 下载 PDF 精读（见 `reading_priority.md`） |
| 4 | LIBERO 的 `control_freq` 与 `Δt_sim` | 预算判据的仿真侧输入（v3 U5） | 读 LIBERO 仓库与论文正文 |
| 5 | 引用图（谁引用了 RTC/KWS） | 判断该方向是否已收敛 | Semantic Scholar（换网络） |
