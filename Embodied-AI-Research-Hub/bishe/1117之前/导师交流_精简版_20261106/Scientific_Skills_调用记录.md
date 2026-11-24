# Scientific Agent Skills 实际调用记录
日期2026-11-06。本轮为材料复用、重点核查和选题讨论，不是全面系统综述或已运行机器人实验。

## 安装与环境检查
三个候选根目录均实际存在。`.scientific-agent-skills/skills`与`.dsh/skills`中可见literature-review/SKILL.md。本轮统一读取并使用`C:/Users/Administrator/.codex/skills/`下的七项版本，避免混用副本。

运行环境由Codex的load_workspace_dependencies取得，使用其Python与Node，未全局安装依赖。pypdf、pypdfium2、NumPy、pandas、ReportLab及Artifact Tool可用。PyMuPDF/fitz不可用，PDF提取改用pypdf，渲染改用pypdfium2。缺少Word/LibreOffice和传统DOC解析工具，指导手册.doc已定位但本轮未读正文，不声称读过。无需文献API凭据，公开网页查询与原文缓存可用。

## 实际贡献与工具执行
| Skill及版本 | 应用的流程及具体产出 | 工具实际状态与边界 |
|---|---|---|
| literature-review 1.11 | 读取SKILL与core_workflow，复用24篇/120笔记与36研究条目，按动作/效率/执行/评测主线筛选六篇 | 完成有边界的叙述综合，见重点论文证据；未声称穷尽检索或本人全篇精读 |
| paper-lookup 2.4 | 读取arxiv参考，用明确ID核查六篇的标题作者日期，区分版本和出版状态 | 实际运行arxiv_atom.py三次重新解析现有Atom XML，输出三份JSON；另在线打开六篇arXiv、作者页及RTC正式论文集，不把缓存日期当在线更新日期 |
| scientific-critical-thinking 1.5 | 审查平滑/响应、吞吐/时延、等待/陈旧、模型/系统效应和已有工作重叠 | 修正旧稿“支撑充分”及机械件必须改变成功率的过强说法；指出仿真必须持续推进物理时间，见指标核查和实验方案 |
| scientific-brainstorming 1.4 | 明确约束与定性标准，只保留A评测/B预算/C协同及反对理由，D长期线索延期 | CLI session_scaffold.py实际失败：Windows无os.fchmod；随后实际调用同脚本build_scaffold与validate_register函数生成登记/检查JSON，未改技能源码。单AI辅助，不伪称多人独立头脑风暴 |
| hypothesis-generation 2.4 | 三条候选假设，列最近邻、现有证据、竞争解释、反例与不确定结果 | 实际运行lint_causal_claims.py输出假设措辞检查JSON；其英语词法规则未命中中文因果词，不能替代中文人工审查，更不是假设验证 |
| experimental-design 1.5 | 区组配对、episode为实验单元、固定控制变量、预设主要对比与失败处理 | 实际导入randomization.py的assign_factorial_runs，生成60行计划顺序并存Python/NumPy/pandas版本；未运行机器人试验，20种子为工程试点设计而非功效结论 |
| scientific-writing 2.3 | 区分事实/个人理解/计划/学院待确认，写两版中文稿与问答 | 实际运行lint_manuscript.py，无错误，警告CONFIDENTIALITY_GATE_NOT_CHECKED（未提供正式manuscript manifest）；本轮本地材料未发外部服务，搜索只用公开论文名/ID。中文仍逐段审阅，未声称投稿或学院审核就绪 |

## 附加制作技能
读取并应用presentations与pdf技能。PPT采用Artifact Tool、原生文本/表格/连接线，可编辑；原始培养方案页与最终幻灯片逐页渲染核查。PDF为同版幻灯片PNG组成的预览文件，文字不可搜索，PPTX文本与结构图保持可编辑。没有在Microsoft PowerPoint里打开，不能把渲染视为PowerPoint原生验证。

## 软件引用与复现材料
Scientific Agent Skills论文已在线核对并加入《重点论文证据与引用》软件方法披露：Kassis et al., Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, 2026, https://arxiv.org/abs/2609.00065 。

实际命令和流程在PPT_生成源码/prepare.py、finish.py、build.mjs，输出解析JSON、随机顺序表、候选登记、检查记录与源文件哈希。未调用research-lookup，本任务不依赖它或付费图像服务。
