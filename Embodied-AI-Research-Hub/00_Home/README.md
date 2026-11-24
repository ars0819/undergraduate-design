---
title: 知识库总览
type: index
updated: 2026-11-06
---

# Embodied-AI-Research-Hub

> 一个面向五年以上使用的个人科研知识库。目标是让下一次会话在五分钟内知道：**我在研究什么、已经有什么证据、现在做到哪里、下一步做什么、哪些话不能说。**

建立日期：2026-11-06（Asia/Shanghai）。本库位于 `D:\Desktop\Embodied-AI-Research-Hub`，与既有科研文件夹并列，不放在任何短期毕设仓库、Python 虚拟环境或模型缓存目录内。

## 1. 先读什么

新会话的固定顺序：

1. 根目录 [`AGENTS.md`](../AGENTS.md)：工作规则和禁区。
2. 本文件：资料版图、目录含义、当前边界。
3. [`Current_Status.md`](Current_Status.md)：**唯一权威学习状态**。
4. [`Next_Actions.md`](Next_Actions.md)：当前可执行动作。
5. 当前项目入口：[`04_Projects/Undergraduate_Thesis/README.md`](../04_Projects/Undergraduate_Thesis/README.md)。

需要“每天/每周具体怎么操作”时，看 [`使用手册.md`](使用手册.md)。

不要先读旧 Excel 推断“已完成”，也不要把“有笔记”当作“已掌握”。

## 2. 研究定位与当前阶段

- 长期定位：**Efficient AI / ML Systems for Embodied Intelligence**。
- 当前关注：具身模型（尤其 VLA）的轻量化、高效推理、性能测量与实际部署；未来衔接 OpenFleX 轮式双臂平台。
- 本科阶段：允许并优先做**已有工作的复现、评测和工程验证**，不把 Agent、Robot-Use Agent、世界模型或“机器人大脑”作为主研究目标。
- 博士阶段：在上述方向上继续深入，但不把能力锁死在某一代 VLA 架构上。
- 当前真实阶段：本科毕设准备与基础补强。**尚无任何 VLA 模型在本机或 A100 上运行的真实日志。**

导师方向摘要见 [`Advisor_Feedback.md`](../04_Projects/Undergraduate_Thesis/Advisor_Feedback.md)。注意：那里记录的是用户转述的方向，不是逐字会议纪要；正式材料以导师/学院确认件为准。

## 3. 当前最重要的边界

### 已经能确认的

- 三份源文件真实存在并已读取：V3.0 路线 DOCX、原始学习笔记 DOCX、七工作表 XLSX。
- 原始笔记 473 段、1 个表、5 张内嵌图；5 张图均已按字节保存并可被 Markdown 引用。
- 原始 Excel 七个工作表均存在，均可见；没有公式；原工作簿未修改。
- 本地检测到 NVIDIA GeForce RTX 5070，12227 MiB 显存，驱动 617.14，compute capability 12.0；系统 Python 3.12.10；**torch 未安装，nvcc 未找到**。
- 旧笔记中有明确日期 2026-09-26 与 2026-10-06 的数学学习记录；Python/ML 部分没有独立日期。
- 旧进度表在 2026-10-13 记录“第1周”，但那是旧计划快照，不是今天的实际进度。

### 不能声称的

- 不能声称 Python 已“学完”或可独立写完整程序；只能确认有课程摘录和课程步骤记录。
- 不能声称 18.01 已掌握；只能确认接触过若干主题，且 Newton 部分没有内容。
- 不能声称吴恩达机器学习完成到某一 P 数；源文件没有记录。
- 不能声称 SmolVLA、OpenVLA-OFT、LIBERO 或 OpenFleX 已复现、已部署、已测速。
- 不能把旧计划中的“预计完成时间”当实际完成日期。
- 不能把本库的整理文字当作用户当时的原始理解。

完整状态见 [`Current_Status.md`](Current_Status.md)。

## 4. 目录地图

| 目录 | 作用 | 现在有什么 |
|---|---|---|
| `00_Home/` | 入口、操作手册、权威状态、下一步、决策记录 | README、使用手册、Dashboard、Current_Status、Next_Actions、Decisions |
| `01_Roadmap/` | 长期路线、能力地图、基础计划、VLA 推理轨、旧版对照 | 新版双轨路线与旧 V3 对照 |
| `02_Learning/` | 按主题整理的学习笔记 | Python、MIT 18.01、吴恩达 ML、NumPy/Matplotlib 四篇整理笔记 |
| `03_Research/` | 研究概念、论文索引、方向备忘、实验室观察 | 推理链路、指标、压缩方法、部署清单、论文索引入口 |
| `04_Projects/` | 本科毕设与复现项目入口 | 毕设六件套、SmolVLA 复现入口、实验索引 |
| `05_Reviews/` | 日/周/月复盘 | 复盘规范；不预生成空白日志 |
| `06_Resources/` | 课程、论文、工具环境、链接、既有资源索引 | 课程表、论文候选、环境事实、外部路径 |
| `07_Templates/` | 七类可复制模板 | 日志、周复盘、课程、概念、论文、实验、问题排查 |
| `90_Archive/` | 原始资料、旧路线、迁移报告 | 三份原件、哈希、图片、Word/Excel 机械提取 |
| `99_Assets/` | 图片与附件 | 5 张学习图片及索引 |
| `tools/` | 维护脚本 | `verify_hub.py`：相对链接、图片、哈希检查 |

### 对建议目录的必要调整

1. **根目录多了一个 `README.md`**：只作为最短入口，详细说明仍在 `00_Home/README.md`，避免新会话在多层目录里找不到起点。
2. **`02_Learning/` 不是每个子目录都放笔记**：只写有真实内容的部分；`Deep_Learning/`、`PyTorch/`、`Robotics/`、`ML_Systems/`、`CUDA_Triton/` 先保留目录，不制造空笔记。
3. **`03_Research/Literature/` 只放索引入口**：论文条目统一在 `06_Resources/Papers_Index.md`，避免同一篇论文有两套状态。
4. **新增 `06_Resources/Existing_Resources_Index.md`**：记录既有 Zotero、Obsidian Vault、文献库、毕设工作区的真实路径与“只读”边界。
5. **新增 `tools/verify_hub.py`**：五年维护需要可执行的链接/哈希检查，而不是靠记忆。
6. **没有创建 `.obsidian/` 配置**：本库可直接作为 Vault 打开，但不擅自安装插件或迁移现有 Vault。
7. **新增 `00_Home/使用手册.md`**：把“怎么记录、怎么自测、怎么更新状态、怎么维护”集中到一页，避免每次重新解释。

## 5. 来源与迁移

| 源 | 归档 | 哈希 | 处理方式 |
|---|---|---|---|
| `D:\Desktop\机器人学习与具身智能科研路线_V3.0_简化执行版.docx` | `90_Archive/Original_Documents/` | `8c1e4c09…54ee3` | 原样复制；完整机械提取 |
| `D:\Desktop\926150.docx` | `90_Archive/Original_Documents/` | `05c1e731…6d1e0` | 原样复制；逐段提取 + 主题整理 |
| `D:\Desktop\机器人学习与具身智能科研路线_表格版.xlsx` | `90_Archive/Original_Documents/` | `00c5460c…82dec` | 原样复制；七表逐单元格提取 |

### 派生文件来源映射

| 新文件/目录 | 主要来源/依据 | 性质 |
|---|---|---|
| `02_Learning/` 四篇笔记 | 926150 的 P0000–P0049、P0050–P0414、P0415–P0442、P0443–P0472 | 原文整理 + 标注的 AI 解释 |
| `01_Roadmap/Original_V3_Comparison.md` | V3 全文、V3 表 13、Excel“32周主线” | 历史对照，不改旧版 |
| `01_Roadmap/Current_Roadmap.md`、`Foundation_Plan.md`、`VLA_Inference_Track.md`、`Long_Term_Research.md` | V3 的基础/课程部分 + 2026-11-06 用户方向 + 既有毕设资料的静态核查记录 | 新路线综合，不是实验结果 |
| `06_Resources/Courses.md` | V3 第二部分、Excel“课程资源/数学与MLS/框架与CS课程” | 资源索引；P 数截止 2026-10-13 |
| `06_Resources/Papers_Index.md` | 既有毕设资料的候选引用；本库未逐篇复核 | 候选清单，不是已读清单 |
| `06_Resources/Tools_and_Environment.md` | 2026-11-06 本机命令输出 | 实测事实快照 |
| `06_Resources/Existing_Resources_Index.md` | 外部目录/README 只读检查 | 外部路径索引 |
| `04_Projects/Undergraduate_Thesis/*` | 用户 2026-11-06 方向转述 + 既有毕设文档只读参考 | 工作版计划，未批准、未运行 |
| `03_Research/Concepts`、`Efficient_Inference`、`Model_Compression`、`Deployment` | 通用方法/AI 整理 + 既有项目文档的部分概念 | 教学/方法备忘，需实践验证 |
| `07_Templates/`、`AGENTS.md` | 本库的维护规范 | 新建治理内容，不含源数据 |
| `90_Archive/` 机械提取 | 三份源文件 | 逐段/逐表忠实转换 |

> 这张表不是“每句话的引用”，而是让后续会话知道**哪些是原文、哪些是综合、哪些是实测、哪些是计划**。

- 任务说明里的 `926150(1).docx` 在磁盘上不存在；实际文件是 `926150.docx`。没有伪造第二份副本。详见 [`SOURCE_MANIFEST.md`](../90_Archive/Original_Documents/SOURCE_MANIFEST.md)。
- 机器提取文件：
  - [`926150_原始笔记逐段提取.md`](../90_Archive/Original_Documents/926150_原始笔记逐段提取.md)
  - [`V3.0_完整机械提取.md`](../90_Archive/Old_Roadmaps/V3.0_完整机械提取.md)
  - [`Excel_七工作表结构化提取.md`](../90_Archive/Old_Roadmaps/Excel_七工作表结构化提取.md)
- 图片索引：[`IMAGE_INDEX.md`](../99_Assets/Images/IMAGE_INDEX.md)。
- 迁移验收：[`迁移与完整性报告.md`](../90_Archive/Migration_Reports/迁移与完整性报告.md)。

## 6. 进度系统：只允许一个权威位置

**唯一权威：** [`Current_Status.md`](Current_Status.md)。

- `Dashboard.md` 是它的展示视图，不独立维护完成状态；两者冲突时以 `Current_Status.md` 为准。
- 旧 Excel 是历史参考，**不再实时维护**，也不与 Markdown 并行更新进度。
- 只有通过独立验收，才允许把状态改为“已通过独立验收”；有笔记、看过视频、目录里存在文件都不够。
- 不确定的项目统一写“待自测”，不写“已掌握”。

状态词只有五个：

| 状态 | 含义 |
|---|---|
| 未开始 | 没有可靠学习证据，或只有计划 |
| 学习中 | 正在投入，且有可核对的笔记/代码/课程记录，但尚未独立验收 |
| 已接触但待复习 | 看过或记过，但缺少独立完成证据；需要自测/复习 |
| 已通过独立验收 | 在无参考答案条件下完成规定任务，并有测试/运行记录 |
| 暂缓 | 当前方向不需要，或有明确阻塞；不是“已掌握” |

## 7. 日常维护

### 用户说什么，就做什么

- “记录今天的学习内容” → 用 [`Daily_Log.md`](../07_Templates/Daily_Log.md) 建当天记录；只写实际发生的事。
- “整理我这周的学习进度” → 汇总已有记录，按 [`Weekly_Review.md`](../07_Templates/Weekly_Review.md) 生成复盘，不补造完成项。
- “我学会了某个知识点” → 先给 3–5 个自测题；通过后才改 `Current_Status.md`。
- “整理这篇论文” → 用 [`Paper_Note.md`](../07_Templates/Paper_Note.md)，先读原文，再写；与当前研究问题无关的论文不自动进必读。
- “更新我的本科实验进展” → 只从真实日志/原始输出更新 [`Experiment_Index.md`](../04_Projects/Undergraduate_Thesis/Experiment_Index.md)。
- “我不知道接下来学什么” → 先看 `Current_Status.md` 的前置关系和 `Next_Actions.md`，只推荐一个最小动作。
- “把学习计划调整一下” → 先出变更提案和前后对比，旧版本进 `90_Archive/Old_Roadmaps/`。

### 每次改完

运行：

```bash
python tools/verify_hub.py
```

它检查：Markdown 相对链接、图片引用、代码围栏与 Python 语法、归档文件 SHA256、关键文件存在性。检查不通过就不算完成。

## 8. VS Code、Obsidian、Zotero

- **VS Code**：直接打开本文件夹；用 Markdown Preview 看相对链接；不要用会重写全库的批量格式化。建议只在需要时打开 `04_Projects/` 或 `02_Learning/`。
- **Obsidian**：可以把本文件夹作为**新 Vault** 打开。当前不创建 `.obsidian/`，不安装插件，不迁移已有 Vault。标准 Markdown、相对链接、Mermaid 和 LaTeX 可直接用。
- **Zotero**：继续作为文献条目、PDF 与引用管理工具。本库只写个人理解、方法比较、阅读笔记、实验关联；**不直接修改 Zotero 数据库**。已有 Vault `D:\Desktop\Road` 装有 `zotlit`，不要在本库擅自复制或改动其配置。
- **DSH/Codex**：工作目录设为本库；先读 `AGENTS.md` 和本文件。不要让它自动“整理全库”、上传、同步或删除。

## 9. 当前限制

- 原始笔记是个人速记/课程摘录，不是正式教材；部分代码行在 WPS/Word 中丢失换行，整理笔记只做**排版修复**并标注，不改原意。
- 公式以课程截图为主，未可靠转写为 LaTeX；笔记中保留截图并标注待核实。
- 没有真实实验日志、GPU 性能数据、模型权重或数据集。
- A100 40GB 的可用性没有验证；本机只有 RTX 5070 的硬件识别结果，未验证完整计算栈。
- 外部实验室主页与部分课程官方页面未在当前网络环境完成逐页核验；链接仅作入口，内容需另行核对。
- 旧 V3.0 的 B 站 BV 号核对截止 2026-10-13；平台目录可能变化。

## 10. 五年使用的原则

1. **先证据，后叙述。**
2. **先一个最小实验，后一个宏大方向。**
3. **先测量，后优化。**
4. **先保留旧版本，后改路线。**
5. **先写清不知道什么，再写知道了什么。**

如果未来某次会话只能记住一句话：**这个库的价值不在于看起来完整，而在于它不会把“计划”伪装成“完成”。**
