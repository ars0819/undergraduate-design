from pathlib import Path
import json, re, hashlib, zipfile, sys, subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
import pypdfium2 as pdfium
O=Path(__file__).resolve().parent.parent; S=O/'PPT_生成源码'; B=O.parent/'.精简版_build'
images=sorted((B/'previews').glob('slide-*.png')); assert len(images)==11
c=canvas.Canvas(str(O/'本科毕设方向交流_精简版.pdf'),pagesize=(1200,675)); c.setTitle('本科毕设方向初步探索与文献调研'); c.setAuthor('刘鑫宇')
for im in images:c.drawImage(ImageReader(str(im)),0,0,1200,675); c.showPage()
c.save()
# Render the actual delivered PDF pages for independent visual inspection.
(B/'pdf-check').mkdir(exist_ok=True)
d=pdfium.PdfDocument(str(O/'本科毕设方向交流_精简版.pdf'))
for i in range(len(d)): d[i].render(scale=4/3).to_pil().save(B/'pdf-check'/f'page-{i+1:02d}.png')
d.close()
# Printable one-page cue card uses an installed Chinese font.
pdfmetrics.registerFont(TTFont('YaHei','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
c=canvas.Canvas(str(O/'提词卡_1页.pdf'),pagesize=(595,842)); c.setTitle('导师交流提词卡'); c.setFont('YaHei',20);c.drawString(40,795,'导师交流提词卡');c.setFont('YaHei',11)
lines=['核心请求：主线、3.3系统设计口径、两三周优先级','',
'第1页 / 20秒：讨论稿，尚未定题',
'第2页 / 40秒：24篇笔记与36研究条目有交叉，AI辅助整理',
'第3页 / 35秒：规划、动作、控制、反馈分工，世界模型可选',
'第4页 / 45秒：动作生成、效率、异步、衔接、状态对齐、延迟评测',
'第5页 / 65秒：更快未必同比改善任务，平滑未必及时响应',
'第6页 / 60秒：首选评测，预算与协同为扩展，无已证创新点',
'第7页 / 45秒：兼容、完整任务、基线、小规模延迟对照',
'第8页 / 70秒：八项H已核原PDF，3.3系统设计待学院确认',
'第9页 / 50秒：主线？工作量？专业交付？两三周先做什么？','',
'追问底线：没有本人模型或闭环实验结果。',
'不保证创新、收益或3.3达标，OpenFleX为未来平台。','',
'若时间紧：重点讲第5、6、8、9页，使用3分钟稿。']
for i,l in enumerate(lines):c.drawString(40,747-i*34,l)
c.showPage();c.save();assert len(PdfReader(O/'提词卡_1页.pdf').pages)==1
doc=pdfium.PdfDocument(str(O/'提词卡_1页.pdf'));doc[0].render(scale=1.4).to_pil().save(B/'提词卡检查.png');doc.close()

def wr(n,t): (O/n).write_text(t.strip()+'\n',encoding='utf-8')
# Correct section locators to the actual cached PDF version.
p=O/'重点论文证据与引用.md';t=p.read_text(encoding='utf-8')
t=t.replace('方法§III与实验§IV；本地PDF页3—7','§2策略、§3关键设计、§5评估；本地PDF页2—9').replace('§III设计选择与§IV实验；本地PDF页3—7','§IV微调设计、§V/VI仿真与真机实验；本地PDF页3—9').replace('§3模型、异步执行和§4实验；本地PDF页3—9','§3.3异步推理、Algorithm 1及§4.6异步实验；本地PDF页5—6、12—13').replace('§3与Algorithm 1、§4；本地PDF页3—8','§3 Real-Time Chunking via Inpainting、§4 Experiments；本地PDF页4—8').replace('§III方法及§IV实验；本地PDF页3—6','§IV VLASH与§V Experiments；本地PDF页3—7')
p.write_text(t,encoding='utf-8')
# Use the scaffold's in-memory workflow after its CLI fails on os.fchmod on Windows.
sys.path.insert(0,'C:/Users/Administrator/.codex/skills/scientific-brainstorming/scripts')
from session_scaffold import build_scaffold
from validate_register import validate_register
r=build_scaffold(session_id='vla-undergraduate-20261106',title='本科方向收敛',question='怎样以最小工程实验连接VLA推理与机器人闭环表现？',participant_ids=['ai-assistant'],session_date='2026-11-06',in_scope=['评测','预算','执行协同'],out_of_scope=['通用世界模型训练','新基础模型'],constraints=['兼容checkpoint和任务未确认'],perspectives=['方法','工程实施'])
r['ideas']=[];r['assumptions']=[]
for i,(statement,assumption) in enumerate([('A 固定策略的延迟与闭环性能评测','存在兼容且能完成任务的权重与环境'),('B 预算与执行长度对照','实现允许独立改变采样步数和执行长度'),('C 一种实际瓶颈的协同改进','基线揭示可干预且可重复的瓶颈')],1):
 r['assumptions'].append({'id':f'A00{i}','statement':assumption,'category':'feasibility','status':'untested','test_or_check':'兼容准入或基线试验，尚未实施','owner_id':'ai-assistant','evidence_refs':[]})
 r['ideas'].append({'id':f'I00{i}','statement':statement,'provenance':{'origin':'ai-assisted','contributor_ids':['ai-assistant'],'recorded_stage':'post-check','source_refs':['SmolVLA','VLALagBench','RTC','VLASH'],'ai_tool':'Codex'},'assumption_ids':[f'A00{i}'],'predicted_observations':['可能得到负结果，候选不是已验证创新'],'uncertainties':['兼容性','与已有工作重叠','学院交付口径'],'evidence_status':'mixed','status':'candidate'})
r['criteria']=[{'criterion':'可执行性、测量有效性、重叠风险、工程交付','method':'定性比较，无自动打分或人类最终决定'}]
r['notices'].append('单一AI辅助整理，未进行多人独立头脑风暴，未冒充导师作出决策。A仅为供讨论首选。')
(S/'候选方向登记.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
(S/'候选方向结构检查.json').write_text(json.dumps(validate_register(r),ensure_ascii=False,indent=2),encoding='utf-8')

wr('Scientific_Skills_调用记录.md','''# Scientific Agent Skills 实际调用记录
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
''')

wr('PPT_生成源码/README.md','''# 源码与复现说明
prepare.py读取本地材料、生成配套稿件并实际调用experimental-design的随机顺序函数。build.mjs使用Codex自带Artifact Tool生成11页原生可编辑PPTX、渲染和finalizePresentation校验。finish.py生成预览PDF、单页提词卡、候选登记与交付记录。

使用Codex load_workspace_dependencies返回的Node/Python；环境路径写在build.mjs及实际调用记录中。构建中间件与回执位于D:/Desktop/bishe/.精简版_build，最终输出位于本目录上级。重新运行prepare.py会重建生成文档，finish.py负责更新具体段落定位。原始任务书、培养方案和文献笔记只读。

PDF为已核查渲染页的阅读预览，无可搜索文字；可编辑版本为PPTX。提词卡PDF使用微软雅黑，独立一页。没有PowerPoint原生打开验证。
''')
with zipfile.ZipFile(O/'本科毕设方向交流_精简版.pptx') as z:
 slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)];assert len(slides)==11
 assert len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==11
 assert sum(z.read(n).count(b'<a:tbl>') for n in slides)==6
inv=json.loads((S/'实际读取文件清单.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in inv)
count=len(re.findall('[\u4e00-\u9fff]',(O/'交流稿_完整版.md').read_text(encoding='utf-8')))
check={'slides':11,'main':9,'appendix':2,'pdf_pages':len(PdfReader(O/'本科毕设方向交流_精简版.pdf').pages),'native_tables':6,'source_files_read':len(inv),'original_sources_unchanged':True,'speech_chinese_characters':count,'powerpoint_opened':False,'pdf_type':'raster_preview','robot_experiments_run':False}
(S/'交付结构检查.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(check,ensure_ascii=False))
