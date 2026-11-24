# Scientific Skills调用与证据记录

执行日期：2026-11-06，北京时间。执行者为本聊天AI助手；学生、导师和学院尚未人工批准。此记录把技能说明读取、流程实际应用、脚本执行、网络核验和GPU实验严格区分。

## 1. 实际安装检查

六项指定Skill均存在于本机 `C:/Users/Administrator/.codex/skills/<名称>/SKILL.md`，已实际读取；绝对路径、字节数与SHA256见 `证据/Skills实际安装清单.json`。Skill是本地程序知识/流程，不是一个名为“scientific agent”的远程执行器；不存在“一次调用自动证明全部科学结论”的能力。

| Skill | 本机版本 | 实际执行及结果 | 可追溯产物 |
|---|---|---|---|
| [literature-review](C:/Users/Administrator/.codex/skills/literature-review/SKILL.md) | 1.11 | 定义限定问题/纳排范围，检索arXiv与官方项目/代码；11篇核心书目本地规范化、去重脚本实际退出0 | 03范围与矩阵；`核心书目规范化.json`、`核心书目去重.json`、`literature_dedup.txt` |
| [paper-lookup](C:/Users/Administrator/.codex/skills/paper-lookup/SKILL.md) | 2.4 | 读取arXiv接口陷阱说明；实时Atom请求被本机网络策略阻断；转网页直接核对元数据，并对3份11-05缓存调用`arxiv_atom.py`重新解析成功 | `网络获取记录.json`；`缓存重新解析_metadata*.json`；下面失败记录。缓存解析不声称实时检索成功 |
| [scientific-critical-thinking](C:/Users/Administrator/.codex/skills/scientific-critical-thinking/SKILL.md) | 1.5 | 实际应用证据/代理指标审查：加载≠闭环、存储≠显存、模拟量化≠真实加速、单帧误差≠成功率；识别VLAQuantBench设备/协议限制 | 02测量纪律与竞争解释、03证据/资源矩阵、05毕业指标风险；该技能为推理流程，本次没有虚构不存在的批判评分脚本 |
| [hypothesis-generation](C:/Users/Administrator/.codex/skills/hypothesis-generation/SKILL.md) | 2.4 | 建立H1编译开销收益及R1饱和、R2缓存/预热偏差、R3编译内存成本；明确可反驳预测、回放与闭环区别；`lint_causal_claims.py`退出0 | 02第7节与`hypothesis_lint.txt`。英文词法触发为0，对中文检测能力有限，不作为因果或科学正确性认证 |
| [experimental-design](C:/Users/Administrator/.codex/skills/experimental-design/SKILL.md) | 1.5 | 实际import官方`assign_factorial_runs`，在每个task–init区组随机original/optimized顺序；生成3×10×2=60行计划，环境记录已保存 | `配对试点顺序_未执行.csv`、`实验设计运行环境.json`；未生成或填造任务结果；试点不作正式功效声明 |
| [scientific-writing](C:/Users/Administrator/.codex/skills/scientific-writing/SKILL.md) | 2.3 | 将事实/估算/待核/未测分层，保存来源manifest（human verification仍unverified）；参考语法检查及01–06写作lint实际执行 | `source_manifest.json`、`Scientific脚本执行结果.json`、`writing_*.txt`；07另做最终lint。没有代替人工批准 |

版本以本机SKILL.md为准；若表中某版本与未来更新不一致，应查看本次hash而不追认新版本。

## 2. 原始资料实际读取与保存

本次通过脚本本地读取并登记145项原始文件/源文件（含培养方案PDF与两份DOCX）；列表、字节与hash见 `原始材料读取清单.json`。这包括旧任务书讨论稿、旧毕业要求审查、实验方案、导师交流提纲、24篇共120份精读笔记、横向总结、现有专题报告及精简交流材料。自动批量读取与关键摘录不能冒充每份笔记重新全文精读；重点复核了模型/任务/指标/资源/旧方案冲突段落。24篇笔记本次重读摘录保存在 `24篇精读重读摘录.txt`。

部分用户提及材料实际在相邻目录：`D:/Desktop/library/Literature_Notes/`、`D:/Desktop/dickseep/research/thesis/2026-11-04_导师交流初稿/`、桌面培养方案与任务书。依据旧项目“实际读取文件清单”定位后只读访问，未覆盖这些源文件。

当前工作区 `D:/Desktop/bishe/AGENTS.md` 未检出，父级候选也未发现有效文件；实际存在并阅读了相邻文献库 `D:/Desktop/library/AGENTS.md`，它只作为资料整理背景，不冒称当前工作区指令。用户最新范围限制高于旧材料中的Agent/世界模型兴趣与泛化精读模板。本次不更改文献库、Zotero或旧稿。

培养方案：直接提取原PDF第53–59页，并查看旧的第59页矩阵渲染图，八项H与3.2空白见05。`原文件保持检查.json`最后核对145项hash，变化项为0；新输出均位于用户指定V2目录。

## 3. 检索与网络调用记录

| 目的 | 实际检索/打开内容 | 结果与限制 |
|---|---|---|
| 指定研究主页 | `https://zhouzypaul.github.io/research` | 网页工具成功打开，确认作者为Zhiyuan “Paul” Zhou、Levine研究线索及QGF链接；shell直接访问失败，没有伪造本地HTML快照 |
| Han Lab | `https://hanlab.mit.edu/news`；AWQ/SmoothQuant/SVDQuant官方项目与仓库 | 网页读取成功；只选量化/内核/部署，未把新闻中的规划/视频生成扩成主线 |
| 两模型 | 搜索`SmolVLA official LIBERO checkpoint evaluation 500M`；官方SmolVLA模型卡/config；OFT仓库/SETUP/LIBERO/pyproject/评测脚本 | 确认存在官方LIBERO权重与源码接口；Hub页面成功、API获取完整revision失败；代码快照成功 |
| 定向量化 | `VLAQuantBench paper github`、`AWQ SmoothQuant SVDQuant arxiv official`、`QuantVLA quantization vision language action models` | 从发现结果追到原论文和作者代码；第三方博客仅作发现，不作为核心方法证据 |
| GPU支持 | `PyTorch Blackwell 2.7 CUDA12.8`，官方release | 确认新GPU环境不能照搬OFT torch2.2；不把有架构支持当所有扩展kernel可用 |
| 书目信息 | 逐ID打开11篇arXiv页；SmoothQuant PMLR、AWQ MLSys、LIBERO NeurIPS记录 | 作者/首发与会议分别记录；Zhou NeurIPS2026、QuantVLA CVPR2026仍明确作者标注/独立核查缺口 |

可重复定位依据：03包含全部原始URL；`网络获取记录.json`、`补充获取记录.json`保存shell逐URL状态、文件名、字节与hash；网页工具的访问记录在本聊天可见。并非所有网页均下载到本地；本轮没有伪造“全部原始PDF已下载”的说法。

本次书目去重输入11条、输出11条。这是人工限定的核心清单数，不是搜索引擎总命中数，也不是PRISMA全库筛选数。检索不穷尽，不能用未纳入推断无相关工作。

## 4. 已获取的确定代码版本

| 组件 | 本次GitHub API返回SHA |
|---|---|
| LeRobot | `d490e472a5d1a207cfe0255caaaf74b6d66e4656` |
| OpenVLA-OFT | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| LIBERO原仓库 | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| VLAQuantBench | `4a2cb7c1bf9160ae683fe598ec583d7c9b9c13c9` |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| OFT transformers fork | `bc339d9ad707454c0c115970db43c260067c61ab` |
| OFT dlimp fork | `040105d256bd28866cc6620621a3d5f7b6b91b46` |

VLAQuantBench安装脚本另外固定AWQ `d6e797a42b9ef7778de8ee2352116e0f48a78d61`、SmoothQuant `c61476d728e42ae0d8a35e7e78494edcac3237b5`；这是已读取的作者脚本内容，本次未安装验证这些内核。完整源码响应与SHA表均在证据目录。

## 5. 真实失败、修复与未完成项

1. 系统PATH没有`python`命令；调用Codex的workspace dependencies后改用捆绑Python3.12.14。该捆绑环境未装torch，只说明当前文档执行环境，不能据此断言用户Ubuntu未装torch。
2. `collect_evidence.py`直接请求arXiv/研究主页/Han Lab等域名遭WinError10013网络限制；GitHub源码请求成功；这些公开页面用网页工具补查。未绕过访问限制、未申请安装或改系统网络。
3. Hugging Face API无论带不带blobs参数均返回HTTP502；网页API也不可访问。`git ls-remote`备用尝试报Windows schannel `SEC_E_NO_CREDENTIALS`。因此模型完整SHA、LFS/分片hash和本地可下载性仍待Ubuntu执行，不宣称权重已锁好。
4. `supplement.py`首次在本地Atom解析子进程因默认Windows输出编码与UTF8解码不匹配中断。后用`python -X utf8`单独重跑3份缓存解析，实际成功；不把原网络请求失败改写为成功。
5. `scientific-writing/check_references.py`首次因source manifest缺`authors`数组返回INVALID_INPUT；补为显式空数组并保留完整作者在03后重跑，退出0。此检查只查本地语法/重复，未自动验证作者完整性或网络内容。human verification保持未审核。
6. 01–06写作lint退出0，但均有`CONFIDENTIALITY_GATE_NOT_CHECKED`提示：未使用正式投稿的人工审批manifest。本稿只在本地处理用户资料，外部检索仅公共论文/代码查询；不称已获人工保密/投稿批准。
7. hypothesis词法lint退出0但中文触发计数0，不能作为中文科学推断合格证。critical-thinking以逐条分析应用，不虚构自动评分结果。
8. 未运行安装、CUDA模型forward、真实kernel量化、任务episode、功率测量或真机动作；未取得A100连接/配额，也未验证OpenFleX。两周计划及60行顺序表都明确未执行。

## 6. 可执行规划与文档质量检查

已额外提供模型快照下载校验、实际环境probe、SmolVLA新块计时三个脚本。语法编译及CLI帮助已检查；不安装模型即可做的检查不等于GPU运行测试。计时脚本只测已预处理观测的模型段，完整端到端/模块trace/任务评测仍按02接入官方脚本执行。README明确输入准备和未验证范围。

文档检查只帮助发现缺字段、引用语法与误导措辞，不能证明实验可复现或满足培养指标。最终交付保留未测/待核项，供学生实跑和导师/学院审核。

## 7. 软件引用与AI辅助披露

Kassis, Timothy; Agarwal, Vinayak; He, Yuhuan; Patel, Darshil; Brueckner, Aubrey M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。本次网页核对当前v2（2026-09-02）；引用不加版本后缀。已按Skill要求加入任务书和矩阵的软件说明，不计入技术核心11篇。

AI辅助用途：检索、源码静态阅读、文献与资源条件对比、实验设计和Markdown草稿生成。实验数据、人工审核、学院认可和科研结论均不由AI文本生成替代。
