from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
import hashlib,json,re,io
from lxml import etree
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
O=Path('本科毕设_正式任务书填写_20261106'); Q=O/'校验材料'
S=Path('D:/Desktop/附件6：信通院2027届本科毕设常用特殊表格.docx')
sha=lambda b:hashlib.sha256(b).hexdigest()
D=Document(S)
(Q/'artifact.md').write_text(f'''# 原模板编辑约定
原文件：{S}；SHA256：{sha(S.read_bytes())}。
A4纵向，一节；原页边距、标题、页眉页脚、表格网格和单元格合并全部保留。正文继承原宋体与Times New Roman、10.5磅默认字号及原段落格式。首表14行8网格列；后三表8×8、17×12、28×14。
仅首表基本信息空栏、四项任务、主要内容、文献、阶段内容允许修改；适配版替换四行指标选项，对照版原样保留。首表前新增一段讨论稿说明。题目类型、项目来源、签字和日期保留空选项。长单元格复用原段落样式，保持最小行高及自动换行，不重建表格。
只替换包中word/document.xml，其他包部件逐字节保留。首表之外全部原有正文元素逐一比较；新增说明为唯一额外正文元素。原文件缓存7页，真实页数未知。渲染器因缺少soffice.exe失败，Word未检出且无连接会话；分页、外观及Word可打开性待人工验收，不把结构检查当视觉验收。
''',encoding='utf-8')
cn='面向机器人操作的视觉语言动作模型高效推理与部署研究'
en='Efficient Inference and Deployment of Vision-Language-Action Models for Robotic Manipulation'
tasks=[
'任务1：文献研究与技术方案分析。调研视觉语言动作模型结构、动作生成与高效推理技术，重点比较SmolVLA、OpenVLA-OFT及代表性量化方法的模型规模、资源需求、延迟、部署复杂度与任务性能。目标是形成有依据的主线和替代方案；交付文献矩阵、兼容性清单、需求分析及英文技术概要。',
'任务2：机器人模型推理与部署实验系统设计。面向机器人操作的输入输出、硬件资源与实时性约束，设计感知—推理—动作执行链路，明确数据归一化、动作语义、控制周期、执行时序、范围约束及异常处理。目标是形成可验证的部署方案；交付系统架构图、接口与时序图、方案比选及工程验证计划。3.3最终支撑口径待学院确认。',
'任务3：开源VLA推理与机器人任务基线复现。选择一种成熟开源模型，首选SmolVLA及匹配的LIBERO权重与环境，备用OpenVLA-OFT。完成环境安装、版本固定、加载与推理测试、完整仿真回合及性能测量。目标是建立可重复的原始模型基线；交付配置、运行脚本、延迟/显存/吞吐与成功率原始记录、视频索引及问题日志。',
'任务4：已有高效推理方法复现与工程评测。优先复现torch.compile；不兼容时经导师确认改用流匹配迭代次数优化或经条件核查的低比特量化。固定权重、任务和主要条件，比较延迟、显存、动作误差与闭环成功率，分析额外开销、失效条件及资源成本。目标是给出可信的适用性结论，不强求正向加速；交付对照数据、工程验证报告、论文及复现包。3.3最终支撑口径待学院确认。'
]
inds=['2.3','3.3','6.2','7.2','8.2','10.3','11.2','12.2']
links=[['2.3','10.3','12.2'],['3.3','6.2','7.2','11.2'],['2.3','8.2','11.2','12.2'],['3.3','6.2','7.2','8.2','10.3','11.2','12.2']]
contents=[
'内容一：VLA模型和高效推理技术研究现状及需求分析。依据文献[1—10]与官方实现梳理视觉、语言、状态输入及动作生成机制；比较候选模型与编译、减步数、量化的条件和代价，形成文献矩阵、需求约束与选型报告。以多方案论证支撑2.3，以英文概要及技术问答记录拟支撑10.3，以源码学习、问题归纳和学习记录拟支撑12.2。',
'内容二：机器人感知、模型推理和动作执行实验系统总体设计。明确仿真机器人动作空间、坐标系和时间边界，设计相机与状态接口、动作格式、归一化、控制时序、动作校验、超时和异常终止；比较计算平台与部署方式，形成架构图、接口表、时序图及需求—测试追溯表。以设计取舍和验证方案拟支撑3.3（待学院确认）；以风险、许可和责任边界分析支撑6.2；以能耗测量边界、设备复用及产品周期分析支撑7.2；以资源、工时和成本决策支撑11.2。',
'内容三：代表性开源VLA模型部署、原始性能测试和机器人仿真任务复现。固定代码、依赖、权重及环境版本，核对观测键、处理器与动作维度，完成推理到闭环回合的逐层验收；测量新动作块延迟、峰值显存、吞吐和成功数/总回合数，保留运行配置、种子、初态与失败记录。以兼容性诊断和替代方案分析拟支撑2.3，以真实日志、许可及负结果报告支撑8.2，以资源预算和进度决策支撑11.2，以独立排障及复盘支撑12.2。',
'内容四：高效推理方法复现、性能对照、误差分析与工程验证。在同设备、同权重、同精度、同任务初态及主要配置下比较原模型与一种优化；分别记录编译冷启动、预热后延迟、显存、动作误差和闭环成功率，报告样本范围与不确定性，定位收益或负结果。完成维度错误、非有限输入、动作越界、推理超时四类仿真故障测试；形成对照报告、系统改进与适用边界、资源能耗和成本分析、英文摘要及复现说明。以上证据分别拟支撑3.3（待学院确认）、6.2、7.2、8.2、10.3、11.2、12.2；不得把仿真保护称为真机安全认证。'
]
refs=[
'SHUKOR M, AUBAKIROVA D, CAPUANO F, et al. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics[EB/OL]. (2025-06-02)[2026-11-06]. https://arxiv.org/abs/2506.01844.',
'KIM M J, FINN C, LIANG P. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success[C/OL]//Robotics: Science and Systems. 2025[2026-11-06]. https://arxiv.org/abs/2502.19645.',
'BLACK K, BROWN N, DRIESS D, et al. π0: A Vision-Language-Action Flow Model for General Robot Control[C/OL]//Robotics: Science and Systems. 2025[2026-11-06]. https://arxiv.org/abs/2410.24164.',
'PHYSICAL INTELLIGENCE, BLACK K, BROWN N, et al. π0.5: a Vision-Language-Action Model with Open-World Generalization[EB/OL]. (2025-04-22)[2026-11-06]. https://arxiv.org/abs/2504.16054.',
'LIN J, TANG J, TANG H, et al. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration[C/OL]//Proceedings of Machine Learning and Systems. 2024, 6[2026-11-06]. https://arxiv.org/abs/2306.00978.',
'XIAO G, LIN J, SEZNEC M, et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models[C/OL]//Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023, 202: 38087-38099[2026-11-06]. https://arxiv.org/abs/2211.10438.',
'LI M, LIN Y, ZHANG Z, et al. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models[C/OL]//International Conference on Learning Representations. 2025[2026-11-06]. https://arxiv.org/abs/2411.05007.',
'XU J, JIN Q, CHEN M, et al. VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models[EB/OL]. (2026-09-21)[2026-11-06]. https://arxiv.org/abs/2609.25376.',
'ZHANG J, HSIEH Y, WAN Z, et al. QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models[EB/OL]. (2026-02-23)[2026-11-06]. https://arxiv.org/abs/2602.20309.',
'LIU B, ZHU Y, GAO C, et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning[C/OL]//Advances in Neural Information Processing Systems. 2023, 36[2026-11-06]. https://arxiv.org/abs/2306.03310.'
]
stages=[
'开展文献调研，比较模型、硬件与优化方案；完成需求清单和技术路线，提交开题材料初稿。',
'完成系统架构、接口、时序及测试方案设计，形成资源预算和风险预案；按学院安排完成开题检查。',
'安装模型与仿真环境，固定版本及权重；完成观测/动作接口验收、原始推理和完整回合测试，形成初步基线。',
'补齐仿真基线与测量脚本，复现一种已有推理优化，完成首轮同条件对照和误差分析，准备中期材料。',
'完成阶段性能对照及失败分类；提交中期检查材料、设计包、原始数据和后续计划，参加学院中期检查与答辩。',
'根据中期意见完善系统设计和优化实现；补充配对回合与故障注入，验证动作约束、超时及异常处理。',
'完成工程验证、资源能耗与成本分析；整理适用边界、图表和复现包，完成论文初稿。',
'修改论文，核对引用、数据和实验可重复性，按学院要求参加查重检测与论文内审。',
'5月14日前配合完成查重内审；根据意见修订论文，准备中英文摘要及答辩展示，5月17日起按通知参加终审答辩。',
'按学院安排参加答辩，完成问题复核与论文修订，整理代码、日志、设计文档及归档清单。',
'6月11日前完成学院安排的终审答辩及相关工作；6月14—18日提交总结和归档材料。'
]
def replace_p(p,text):
 rp=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
 for ch in list(p._p):
  if ch.tag!=qn('w:pPr'):p._p.remove(ch)
 r=p.add_run(text)
 if rp is not None:r._r.insert(0,rp)
 return p

def setcell(c,text):replace_p(c.paragraphs[0],text)
def fill_cell(c,lines):
 ps=list(c.paragraphs); base=deepcopy(ps[1]._p if len(ps)>1 else ps[0]._p)
 for p in ps[1:]:c._tc.remove(p._p)
 for text in lines:
  pe=deepcopy(base); pe.attrib.clear(); c._tc.append(pe); replace_p(Paragraph(pe,c),text)

outputs=[]
for adapted in [True,False]:
 d=Document(S); t=d.tables[0]
 vals={(0,1):'现代邮政学院（自动化学院）',(0,6):'机械工程（机器人试验班）',(1,1):'刘鑫宇',(1,4):'2023212129',(1,7):'2023211704',(2,1):'王强',(2,4):'信息与通信工程学院',(2,7):'待确认',(3,1):'（中文，拟定）'+cn,(4,1):'（英文，拟定）'+en}
 for (r,c),v in vals.items():setcell(t.cell(r,c),v)
 note='机械工程专业适配版·导师讨论稿。指标选项修改及3.3支撑口径待学院批准；勾选仅为拟对应关系。' if adapted else '保留原模板选项的对照讨论稿。原信通院指标均不勾选，机械工程拟对应关系见独立审核说明。'
 note+='学生信息暂据旧讨论稿，待正式材料核验；题目类型建议研究设计类，未作正式勾选。'
 pe=deepcopy(d.paragraphs[2]._p); pe.attrib.clear(); t._tbl.addprevious(pe); replace_p(Paragraph(pe,d),note)
 ps=t.cell(9,0).paragraphs
 for i,task in enumerate(tasks):
  replace_p(ps[1+i*2],task)
  if adapted: replace_p(ps[2+i*2],'拟支撑指标点：'+'；'.join(('☑' if x in links[i] else '□')+x for x in inds))
 fill_cell(t.cell(10,0),contents+['本课题以已有工作复现、系统设计与工程验证为主，不将原创架构、论文发表或真机实验列为必达目标；OpenFleX仅作为后续部署衔接背景。所有实验及交付均为计划，达成情况以实际成果和学院审核为准。'])
 fill_cell(t.cell(11,0),[f'[{i+1}] {r}' for i,r in enumerate(refs)])
 # Preserve each original stage heading and paragraph properties, fill first following blank.
 ps=t.cell(12,0).paragraphs; si=0
 for i,p in enumerate(ps):
  if re.match('第[一二三四五六七八九十]+阶段',p.text): replace_p(ps[i+1],stages[si]); si+=1
 assert si==11
 name='本科毕业设计任务书_机械工程适配版_导师讨论稿.docx' if adapted else '本科毕业设计任务书_保留原模板_待学院确认版.docx'
 # Save through python-docx, then retain all untouched original package parts verbatim.
 b=io.BytesIO();d.save(b)
 with ZipFile(io.BytesIO(b.getvalue())) as z:xml=z.read('word/document.xml')
 with ZipFile(S) as src,ZipFile(O/name,'w') as out:
  for info in src.infolist():out.writestr(info,xml if info.filename=='word/document.xml' else src.read(info.filename))
 outputs.append(name)
# Full structural checks after saving/reopening both outputs.
checks={}
for name in outputs:
 d=Document(O/name)
 with ZipFile(S) as a,ZipFile(O/name) as b:
  changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
 orig=list(D.element.body);new=list(d.element.body)
 # Drop first table and the single inserted note, then compare every original outside element.
 original_other=[etree.tostring(e) for e in orig if e is not D.tables[0]._tbl]
 new_other=[etree.tostring(e) for e in new if e is not d.tables[0]._tbl and not (e.tag==qn('w:p') and ''.join(e.itertext()).startswith(('机械工程专业适配版','保留原模板选项')))]
 # itertext duplicates text harmlessly for prefix detection.
 assert original_other==new_other,'outside first table changed'
 assert changed==['word/document.xml']
 assert all(etree.tostring(a._tbl)==etree.tostring(b._tbl) for a,b in zip(D.tables[1:],d.tables[1:]))
 t=d.tables[0];reftexts=[p.text for p in t.cell(11,0).paragraphs if re.match(r'^\[\d+\]',p.text)]
 assert len(reftexts)==10
 assert all(re.search(r'20(2[2-6])',r) and 'https://arxiv.org/abs/' in r and re.search(r'\[(C|EB)/OL\]',r) for r in reftexts)
 assert len(re.findall(r'任务[1-4]：',t.cell(9,0).text))==4
 assert all('内容'+x+'：' in t.cell(10,0).text for x in '一二三四')
 assert t.cell(13,2).text==''
 assert all(t.cell(i,1).text==D.tables[0].cell(i,1).text for i in [5,6,7,8])
 dates=lambda t: re.findall(r'第[一二三四五六七八九十]+阶段（[^）]+）',t.cell(12,0).text)
 assert dates(t)==dates(D.tables[0]) and len(dates(t))==11
 checks[name]={'zip_crc':ZipFile(O/name).testzip(),'python_docx_reopened':True,'changed_package_parts':changed,'all_original_body_outside_task_table_unchanged':True,'other_three_tables_xml_identical':True,'references':len(reftexts),'stages_unchanged':11,'tasks':4,'source_unchanged':sha(S.read_bytes()),'visual_review':'NOT_COMPLETED','word_open_test':'NOT_COMPLETED'}
(Q/'自动验收.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
(Q/'内容数据.json').write_text(json.dumps({'cn':cn,'en':en,'tasks':tasks,'contents':contents,'references':refs,'links':links},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
