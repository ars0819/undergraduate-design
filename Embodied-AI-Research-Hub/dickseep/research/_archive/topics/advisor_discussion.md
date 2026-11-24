---
title: "导师讨论材料：VLA 高效推理选题"
status: "for-discussion"
revision: 2
revision_date: "2026-11-01"
changes_v2: "按审阅意见下调 T1/T3 新颖性；新增 Q13/Q14；新增 V11–V13；补记近邻文献遗漏"
evidence_cutoff: "2026-11-01"
depends_on: ["research/topics/literature_matrix.md", "research/topics/candidate_topics.md", "research/topics/experiment_plan.md"]
---

# 导师讨论材料

> 本文只做三件事：**给出推荐顺序与理由**、**列出必须先确认的事**、**列出需要导师拍板的问题**。
> 所有判断都标注证据等级（🟦 已核实 / 🟨 推断 / 🟥 待核实），不把推测写成结论。

---

## 1. 推荐顺序与理由

| 顺序 | 选题 | 推荐理由 | 主要顾虑 |
|---|---|---|---|
| **第一** | **T1′** 执行长度 × 重规划频率 × **实时可行性边界** | 复现代码公开（🟦）；只需推理；`E` 只有 8 个合法取值，可全枚举；结论是「能不能闭环」的是非判定，即使阴性也可写 | **新颖性已下调**：AAC（CVPR 2026）与 Knowing When to Stop 已占据「自适应选执行长度」，T1′ 只能做「延迟约束下的可行性判定 + 复现」 |
| **第二（与 T1′ 合并）** | **T3′** 量化误差 × 开环执行长度 | 与 T1′ 共用测量台；是少数量化文献未交叉的维度；机制清楚、可证伪 | **必须先过精读闸门**：VLAQuantBench 有 35 张表，若已扫过 `E`，T3′ 就只剩复现 |
| 第三 | **T5** 效率度量协议 | 工作量小，能撑起方法论章节 | 本质依附 T1′，需与 T1′ 合并计算 |
| 第四 | **T2** 多方法横向帕累托 | 对实验室长期价值最大 | 体量 12–16 人周，本科单干风险高 → 博士第一年 |
| 第五 | **T4** 双系统 + 延迟注入 | 与 OpenFleX 的 ROS2 架构契合 | 依赖「有机载算力吗」未确认（🟥） |

**v2 一句话建议**：**T1′ 是保交付的判定性工作（不再声称发现最优 K），T3′ 是唯一的增量创新但设了精读闸门，T5 并入 T1′**。

> [!warning] 必须向导师说清楚的新颖性变化
> v1 曾把「逐模块量化敏感性 + 混合精度保护」当作 T3 的创新点。**该主张已被 VLAQuantBench 覆盖**（🟦 其摘要明写：对 OpenVLA-OFT 只保护一个 28,672 参数输出投影即可恢复近基线，层范围 126→167 使 π0.5 从 7.0% 升到 70.5%）。
> 同理，v1 的「找最优分块长度」已被 AAC（CVPR 2026）与 Knowing When to Stop 占据。
> **这两个方向只能做复现与交叉验证，不能写成主要创新。** 完整的更正清单见 `literature_matrix.md` §11。

### 1.1 为什么不是「先做大模型加速」

任务给的硬约束是 12GB。做纸面推算（🟨）：

- 7B 模型 bf16 权重 ≈ **14 GB**，**仅加载就超 12GB**；
- 全参微调需「权重 + 梯度 + Adam 状态」≈ 84 GB 量级，差一个数量级；
- LoRA 也要先放底座，底座仍放不下；只有先量化底座的 QLoRA 路线可能可行，但那会把「量化误差」变成混淆变量。

**结论（🟨）**：在现有硬件上，**「训练大 VLA」不在可行域内**，因此任何以「训练」为前提的选题（包括 BitVLA 式的量化感知训练）都不适合作为本科毕设主线。

---

## 2. 选题对比（决策用）

| 维度 | T1′ | T3′ | T2 | T4 | T5 |
|---|---|---|---|---|---|
| 是否需要训练 | 否 | 否（仅校准） | 否 | 否（需小策略） | 否 |
| 12GB 可行性 | 高 | 中 | 高 | 中 | 高 |
| 需要真机 | 否 | 否 | 否 | 否（但真机更有说服力） | 否 |
| 体量（人周） | 9 | 10.5 | 12–16 | 10–14 | 3–4（并入 T1′） |
| 新颖性（v2 判定） | **低**（判定性工作，非新算法） | **中**（须过精读闸门） | 中（工具型） | 中（🟥 待检索） | 中（方法学） |
| 失败后仍有产出 | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| 与博士衔接 | 强 | 很强 | 强 | 中 | 强 |
| 依赖未确认事实 | sm_120 兼容性 + `Δt` | 同左 + 量化 kernel | 同左 | **OpenFleX 算力** | 同左 |

---

## 3. 必须向导师确认的问题

> 按「不确认就会返工」的严重程度排序。前 4 条是**阻塞级**。

### 阻塞级（不解决无法开工）

**Q1｜环境改动授权。** Gate 0（`experiment_plan.md` §0.1）必然要安装 PyTorch / transformers / 仿真器。当前机器上已有 PyTorch/CUDA 环境（背景信息），但版本未知。**是否允许我在隔离环境（conda/venv）里安装，而不动现有全局环境？如果必须动全局环境，需要谁批准？**
> 我目前的处理：**没有安装任何东西**，也未改动现有 ML 环境、Zotero、Obsidian，符合任务要求。

**Q2｜权重与数据下载授权。** 需要下载开源 checkpoint（ACT 级约百 MB 量级；OpenVLA 级为 GB 量级）与 LIBERO 仿真资产。**下载额度与存放位置谁定？是否有内网镜像/合规要求？**

**Q3｜OpenFleX 是否有机载 GPU？** 这决定真实部署是「机载推理」还是「工作站远程推理 + 网络下发」。🟥 我目前在公开渠道**没有找到官方算力说明**。如果是远程推理，**网络延迟会主导端到端延迟**，模型加速的收益会被显著稀释——这会直接改变 T4 的价值判断。

**Q4｜OpenFleX 的交付时间与仿真资产。** 🟦 官方仓库 `OpenFleX-Wheeled-Humanoid/OpenFLEX` 存在（2026-10-22 最后更新），内容是**文档 + 组件 yaml + 安装脚本**，我**没有在仓库里看到 URDF 或 MuJoCo 资产**。🟥 有第三方视频称「MuJoCo 仿真开源、URDF 一键转 MuJoCo」，但我无法核实。**实验室是否已拿到 URDF？如果有，T1/T4 可以顺带做「仿真到本体的运动学一致性」检查，价值会明显提升。**

### 重要级（影响选题方向）

**Q5｜毕设是否要求真机实验？** 若要求，而 OpenFleX 无法按期到位，则需要提前申请替代平台（如实验室现有机械臂）或明确「仿真为主、真机为补充」的验收口径。

**Q6｜「效率」在导师语境下的定义是什么？** 是 ①单帧延迟 ②控制频率 ③任务吞吐（单位时间完成任务数）④能耗 ⑤显存？这决定主指标。🟦 文献综述 [@efficiencymetrics2026] 明确批评「用 FLOPs/吞吐衡量效率」，我倾向选 **②+⑤+动作抖动** 三件套，请导师确认。

**Q7｜是否要对齐实验室内部已有的 VLA baseline？** 如果有内部代码/权重，T1 的 baseline 应换成它，而不是 OpenVLA-OFT——这会显著影响可比性与工作量。

**Q8｜目标发表节点？** 若希望毕设产出可投稿结果（如 ICRA/IROS 2027 或 CoRL），时间线要倒推，T3 的核心实验必须在第 17 周前完成。

### 战略级（博士衔接）

**Q9｜博士方向是否锁定「高效推理」？** 还是「高效推理 + 数据/架构」的更大版图？这决定 T2/T5 是「毕设副产品」还是「博士基础设施」。

**Q10｜是否允许使用外部算力（云 GPU / 实验室服务器）？** 若允许，可解锁「训练小模型」与「更大规模横评」；若不允许，则全部设计必须锁死在 12GB。

**Q11｜OpenFleX 的 ROS2 版本与实时性要求？** 🟦 仓库自述为 `ros2_control` 架构。控制周期目标是多少（100 Hz？500 Hz？）——这决定「模型需要多快」，是所有加速工作的目标函数。

**Q12｜是否接受「负面结果」作为毕设贡献？** 我的方案里 T1′ 有明确的证伪分支（若 `argmax SR(E) = argmax U(E)` 且空闲比例 < 2%，则「实时性是瓶颈」不成立）。如果学院/导师要求「必须有正向前沿」，需要在开题时就把叙事改成「可行性判定 + 可复现基线」而非「新方法」。

### v2 新增（针对审阅意见）

**Q13｜实验室认定哪个 VLA baseline「真的能跑」？** 我需要一个**已验证**的 checkpoint：权重可下载、12GB 能加载、`P` 与 `E` 可配置、LIBERO 能评。**如果实验室已有内部 baseline，请优先给我它**——这比我从 OpenVLA-OFT 起重新趟一遍可靠得多，也能避免「用 ACT 冒充 VLA」这类降级（v1 犯过这个错）。

**Q14｜是否有可用的真实量化工具链？** 我需要区分「模拟量化」（只加数值噪声，不省显存）与「真实量化」（int4/int8 存储 + 量化 kernel）。**动作头通常没有现成 kernel**，若实验室有 TensorRT-LLM / torchao / GPTQ 的落地经验，T3′ 的「显存/延迟」主张才站得住；否则只能做数值分析，不能报加速。

**Q15｜OpenFleX 的控制周期目标是多少？** 这个 `Δt` 是 T1′ 可行性判据的输入（判据为 `L ≤ P·Δt`）。**没有它，我无法判断「模型够不够快」**，只能报一堆与部署无关的毫秒数。

---

## 4. 待核实事项清单（我可以自己推进的部分）

| # | 事项 | 为什么重要 | 验证方式 | 优先级 | 状态 |
|---|---|---|---|---|---|
| V1 | RTX 5070（sm_120）与目标 PyTorch/CUDA 的兼容性 | 全线阻塞 | 本机打印 capability 与版本，跑最小前向 | P0 | 🟥 未做 |
| V2 | OpenVLA-OFT 在 12GB 下的实测显存 | 决定 baseline 是否可用 | Gate 0 实测 | P0 | 🟥 未做 |
| V3 | LIBERO 在 12GB 上的评测速度（每 episode 秒数） | 决定能跑多少 seed | 跑 1 个 suite 计时 | P0 | 🟥 未做 |
| V4 | 「分块长度 K 是否存在最优」是否已有人系统研究 | 决定 T1 的新颖性 | 检索「action chunk size ablation」类工作；查 ACT/OFT/RTC 正文 | P1 | 🟥 待检索 |
| V5 | 「动作头对量化更敏感」是否已有直接证据 | 决定 T3 的新颖性 | 检索 PTQ-for-VLA 的逐模块分析 | P1 | 🟥 待检索 |
| V6 | 双系统在延迟注入下的交叉点是否已被研究 | 决定 T4 的新颖性 | 检索 dual-system + latency/jitter | P2 | 🟥 待检索 |
| V7 | OpenFleX 官方规格书（算力/相机/负载/价格） | 决定真机路线 | 找厂商官网/手册，而非第三方目录 | P1 | 🟥 未找到 |
| V8 | OpenFleX 是否提供 URDF/MuJoCo 资产 | 决定仿真保真度 | 问导师/厂商；查仓库 PDF 目录 | P1 | 🟥 未找到 |
| V9 | 各 checkpoint 的许可证（能否用于毕设/论文） | 合规 | 查 HF model card | P2 | 🟥 未做 |
| V10 | 我核实的 46 篇中，哪些正文有摘要未报告的延迟/显存数字 | 补全矩阵 | 逐篇读正文表格 | P1 | 🟥 未做（本次只读摘要） |
| V11 | **VLAQuantBench 的 35 张表是否扫过执行长度 `E`** | **决定 T3′ 是创新还是复现** | 精读该文全部表格与实验设置 | **P0** | 🟥 未做（v2 新增，阻塞 T3′） |
| V12 | AAC / KWS 的自适应选择是否已考虑**延迟或算力约束** | 决定 T1′ 的可行性判定是否也已被占 | 精读两篇的方法与实验 | **P0** | 🟥 未做（v2 新增） |
| V13 | BitVLA / QVLA / QuantVLA 各自在什么 `E` 下报「不掉点」 | 是 H3′ 的关键反证材料 | 查各自的评测配置 | P1 | 🟥 未做（v2 新增） |

> [!note] V4/V5/V6 是「新颖性检索」，必须在开题报告里写清检索边界。按 `hypothesis-generation` 的规范：只能说「**在本次检索边界内未定位到**」，不能说「**没人做过**」。

---

## 5. Scientific Agent Skills 安装记录（附带交付说明）

任务要求「检查当前可用的 Scientific Agent Skills」，并在此后提出「把 skills 装到全局、让 Codex 也能调用」。以下是完整记录。

### 5.1 安装前的状态（🟦 已核实）

- 本会话初始**没有任何** Scientific Agent Skills：调用 `paper-lookup` 返回 `skill unknown or no longer available`。
- `C:\Users\Administrator\.dsh\skills` **不存在**；`C:\Users\Administrator\.codex\skills` 存在但只有 `.system`。
- `codex` / `npx` / `gh` 三个 CLI **都不在 PATH**，所以官方安装器（`npx skills add`、`gh skill install`、`codex plugins install`）都用不了。

### 5.2 安装方式与位置（🟦 已核实）

采用「**一份规范 checkout + 两个全局根目录做 junction 链接**」的方式：

| 项 | 路径 | 说明 |
|---|---|---|
| 规范 checkout | `C:\Users\Administrator\.scientific-agent-skills` | 完整仓库（含 `.git`），177 个 skill，约 20.9 MB |
| DSH 全局根 | `C:\Users\Administrator\.dsh\skills\<skill>` | 177 个 junction → 上面 `skills/<skill>` |
| Codex 全局根 | `C:\Users\Administrator\.codex\skills\<skill>` | 177 个 junction；原有 `.system` 未触碰 |

**为什么用 junction 而不是复制**：只有一份实体文件，`git pull` 后两个宿主同时更新；避免 177×2 份副本漂移。**代价**：如果删掉规范 checkout，链接会断。

### 5.3 验证结果（🟦 已核实）

| 验证项 | 结果 |
|---|---|
| 目录数 | DSH 177 / Codex 177（Codex 另有 `.system`，共 178 项） |
| `SKILL.md` 存在性 | 177/177 通过，0 缺失 |
| frontmatter `name` 是否等于目录名 | 177/177 匹配，0 不一致 |
| junction 可穿透读取 | 通过（`paper-lookup/SKILL.md` 经链接读到 26452 字节） |
| DSH 是否真的加载 | **通过**：安装后本会话的技能目录**热更新**，177 个 skill（含 `paper-lookup`、`literature-review`、`scientific-brainstorming`、`hypothesis-generation`、`experimental-design`）当场可调用 |
| Codex 是否认这个路径 | **通过（间接）**：从 `codex.exe` 二进制中检索到字符串 `Installs into $CODEX_HOME/skills/<skill-name> (defaults to ~/.codex/skills)`，与安装位置一致 |

### 5.4 使用与维护

```powershell
# 更新（两个宿主同时生效，无需重装）
git -C C:\Users\Administrator\.scientific-agent-skills pull

# 完全卸载：只删链接，保留 checkout
Get-ChildItem C:\Users\Administrator\.dsh\skills, C:\Users\Administrator\.codex\skills -Directory |
  Where-Object { $_.Name -ne ".system" } | ForEach-Object { (Get-Item $_.FullName).Delete() }
```

**注意事项**：

1. **Codex 需要重启**才会扫描到新的 skill（DSH 是热更新，Codex 不是）。
2. **上下文成本**：177 个 skill 的描述会进入每次会话的技能目录。若觉得太吵，可以只保留研究相关的子集（例如 `paper-lookup`、`literature-review`、`scientific-brainstorming`、`hypothesis-generation`、`experimental-design`、`citation-management`、`scientific-writing`、`statistical-analysis`、`scientific-visualization`、`markdown-mermaid-writing`），其余删链接即可，需要时再建。
3. **许可证**：仓库 README 标注 MIT，`LICENSE.md` 存在（我未逐条阅读许可条款）。
4. **引用**：该仓库对应论文 arXiv:2609.00065（README 所述，我未独立核实该文）。

### 5.5 本次 Skill 使用与产出的对应关系

| 交付文件 | 主要依据的 Skill | 具体体现 |
|---|---|---|
| `literature_matrix.md` | `paper-lookup` + `literature-review` | provenance 逐条记录、检索边界声明、按主题而非按论文组织 |
| `candidate_topics.md` | `hypothesis-generation` + `scientific-brainstorming` | 观察/问题/假设/竞争解释/证伪条件六段式；创新性主张标注「待检索/待实验」 |
| `experiment_plan.md` | `experimental-design` | 实验单元=任务、区组=suite、伪重复警告、判定阈值前置 |
| `advisor_discussion.md` | —（综合） | 阻塞级问题优先、风险登记 |

---

## 6. 风险登记表

| # | 风险 | 概率 | 影响 | 触发信号 | 缓解措施 |
|---|---|---|---|---|---|
| R1 | sm_120 与官方依赖不兼容 | 中 | 阻塞 | Gate 0 报 unsupported GPU / illegal instruction | 升级 PyTorch；降级到 CPU 侧小实验；申请云 GPU（Q10） |
| R2 | 12GB 放不下目标 checkpoint | 中高 | 降级 | OOM | 预置 ACT 级 fallback；用 4-bit 加载 |
| R3 | 仿真评测太慢，seed 不够 | 中 | 结论不稳 | 单 episode > 60 s | 减 episode、增 seed；只做子集 |
| R4 | 文献新颖性主张被推翻 | **高（已发生）** | 返工 | V11/V12 检索命中 | 立即改写叙事为「可行性判定 + 测量协议」；已建立 §11 更正记录机制 |
| R5 | OpenFleX 迟迟不交付 | 高 | 真机缺失 | 无交付日期 | 全仿真设计（本方案已满足） |
| R6 | 量化把成功率打到 0 | 中 | 实验失败 | 4-bit SR = 0 | 从 8-bit 逐级降；先做敏感度不追求极致压缩 |
| R7 | 时间不够（22 周） | 中 | 延期 | 第 10 周仍未出 T1 曲线 | 砍 T3 的逐层扫描，只做三模块 |
| R8 | 把「相对倍数」当成「绝对性能」 | 中 | 结论错误 | 论文里出现只有 ×倍 没有 ms/GB | 强制 §0.2 口径，交付物清单卡这一条 |

---

## 7. 与博士阶段的衔接（三步走）

| 阶段 | 时间 | 内容 | 与本毕设的关系 |
|---|---|---|---|
| 第一步 | 毕设（本科最后一年） | T1 + T3：建立测量台，做出「显存-频率-成功率」与「模块敏感度」两张图 | 产出可复用的 `measure.py` 与第一作者级结果 |
| 第二步 | 博士第 1 年 | T2 + T5：把测量台扩成「多方法统一横评 + 效率度量协议」，接入 OpenFleX 真机 | 毕设的测量代码直接升级为实验室基础设施 |
| 第三步 | 博士第 2–3 年 | 在统一评测下提出**新的加速机制**（例如延迟感知的分块调度、动作头专用量化），并解决 F3 指出的「指标失真」问题 | 前两步提供了「别人没有的诚实基线」 |

🟨 这条路径的核心判断：**这个领域目前缺的不是又一个 ×2 加速技巧，而是可信的横向测量**。先做测量、再做方法，是风险最低且最容易形成持续产出的顺序。

---

## 8. 本次交付的自我限制（必须让导师知道）

1. **文献只读到摘要级**，46 篇的正文表格未读；所有 `未报告` 可能只是摘要没写。
2. **检索非穷尽，且 v1 确实漏掉了三篇关键近邻**（AAC / Knowing When to Stop / VLAQuantBench）。这不是能力问题而是策略问题：我按「方法关键词」找论文，而不是按「某个主张是否已有人做过」检索。**v2 已把「主张级检索」设为开题前的硬闸门**，见 `candidate_topics.md` §3。
3. **代码仓库只做了存在性探测**，未验证是否为官方实现、是否含全部实验。
4. **没有做任何实验**，没有下载任何模型，没有改动任何环境（符合任务要求）。
5. **12GB 可行性全部是纸面推算**（🟨），唯一可信的方式是 Gate 0 实测。
6. **OpenFleX 规格来自第三方目录 + 仓库文档**，不是厂商规格书；「未找到」不等于「不存在」。
7. **统计设计已按功效分析重算**，但「每 episode 墙钟时间」仍是待实测参数（🟥），所以 §6 的工时预算是区间而非承诺。
8. **v1 的五个实质错误已全部更正并留档**（`literature_matrix.md` §11），其中 4 个属于「把假设当成了已验证事实」，1 个属于「用错了统计量」。
