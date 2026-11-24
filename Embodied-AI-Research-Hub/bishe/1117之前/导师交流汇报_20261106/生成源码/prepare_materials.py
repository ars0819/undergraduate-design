import pathlib,json,hashlib,re
S=pathlib.Path(__file__).parent; O=S.parent; W=O.parent
N=pathlib.Path('D:/Desktop/library/Literature_Notes')
sources=[N/'00_文献总索引.md',N/'00_横向比较与总结.md',
 W/'具身智能系统架构与协同调度_深度研究报告.md',
 pathlib.Path('D:/Desktop/Embodied-Research-Library/08-本科毕设候选/本科毕设候选方向分析.md'),
 pathlib.Path('D:/Desktop/Embodied-Research-Library/09-论文对比表/按技术主题分类的文献矩阵.md')]
inventory=[]
for d in sorted(N.glob('Paper_0*')):
 files=sorted(d.glob('*.md'))
 texts=[f.read_text(encoding='utf-8-sig') for f in files]
 first=next((t for f,t in zip(files,texts) if f.name.startswith('01_')),'')
 title=re.search(r'^title_en:\s*(.*)$',first,re.M)
 inventory.append({'folder':d.name,'note_count':len(files),'title':title.group(1).strip('"') if title else None,
  'files':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]})
assert len(inventory)==24 and all(x['note_count']==5 for x in inventory)
manifest={'date':'2026-11-05','meeting_date':'2026-11-06','sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
 'notes':inventory,'note_count':sum(x['note_count'] for x in inventory),
 'personal_reading_count':'unverified; not equated with AI-assisted notes',
 'identity':'刘鑫宇，来源于旧导师讨论材料明确标记的本人提供姓名，未复用其他个人信息'}
(S/'source_inventory.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
es=json.loads((W/'sources/included_evidence.json').read_text(encoding='utf-8'))
byid={x['arxiv_id']:x for x in es}
ids=['2506.01844','2506.07339','2512.01031','2606.00537','2611.03103','2611.05155','2502.19645','2502.19417','2611.01011','2611.01390','2610.34717','2510.03342','2303.04137','2602.15922']
intro='''# 参考文献与数据出处

汇报日期：2026年11月6日。材料整理截至2026年11月5日。本次以已有原始论文、核读报告与笔记为基础，没有重新逐篇核验24篇笔记中的全部事实。没有修改原始报告、论文、知识库或Zotero。

## 1. 实际找到的本地资料

'''
intro+='\n'.join(f'- `{p}`' for p in sources)
intro+='''

- 姓名依据：`D:/Desktop/dickseep/research/thesis/2026-11-04_导师交流初稿/11_两份表格待补充信息与审批清单.md`。仅使用“学生姓名：刘鑫宇”，该表注明来源为本人提供。姓名仍建议交流前由本人确认。
- 工作区未找到独立AGENTS.md。已读取资料库`D:/Desktop/library/AGENTS.md`，按其证据与原始资料保护规则使用笔记。
- 目录扫描实际找到24个Paper目录、每篇5个Markdown笔记，共120个分篇文件，另有总索引和横向比较。目录完整不代表本人完成阅读。
- 当前工作区系统报告纳入36篇研究条目，15篇重点核读方法和实验。这个统计描述AI辅助核读范围，不描述学生个人精读数。
- 两组文献有交叉，没有计算“60篇”，也没有据不完整去重估计本人总阅读量。

## 2. 主讲结论的来源映射

|幻灯片|资料与结论|证据边界|
|---|---|---|
|1–2|用户背景与交流目标，旧材料的本人姓名|候选方向和博士平台计划尚待导师讨论|
|3|24篇索引、120个分篇文件、36篇系统报告|材料统计不是个人阅读统计|
|4|Hi Robot、Gemini Robotics 1.5、DreamZero，系统报告§3–4与§10|组合架构有实例，图是教学综合，不是某篇论文原图|
|5|24篇横向比较、系统报告§4–7|只展示技术分类，不复制性能排名|
|6|SmolVLA、RTC、VLALagBench|200ms是教学示意，未冒充本机实测|
|7|六篇原论文的身份与机制，系统报告§6–7|最新预印本不升级为正式会议成果|
|8–9|系统报告§11–12、已有候选方向文件|可验证问题与可行性判断，非原创性证明|
|10–11|系统报告§13–15|拟议路线，尚未跑通代码和软硬件|
|12|用户希望导师判断的内容|不代替导师或学院批准|
|A1|文献清单与链接|年份按首次预印本，身份见下列条目|
|A2|24篇目录与两份汇总|重新按主题分组，每篇只计一次|
|A3|系统报告候选分析及相关论文|方法收益、创新性与失效条件均待验|
|A4|系统报告§9与§13|时间戳合同为拟议设计，跨主机需时钟处理|
|A5|用户硬件背景和系统报告资源边界|GPU兼容、配额与平台接口待核实|

## 3. 主要论文与原始链接

'''
for i,aid in enumerate(ids,1):
 e=byid[aid]; au=e['authors']; au='、'.join(au[:4])+('等' if len(au)>4 else '')
 intro+=f"{i}. **{e['title']}**。{au}。首次 {e['published'][:10]}，所核版本 {e['arxiv_id_versioned']}。 [arXiv](https://arxiv.org/abs/{aid})。\n\n"
intro+='''15. **TinyVLA: Toward Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation**。Junjie Wen、Yichen Zhu等。笔记记载IEEE RA-L 2025，DOI：[10.1109/LRA.2025.3544909](https://doi.org/10.1109/LRA.2025.3544909)。本次只用于“轻量化路线”示例，未重新核验该条目的性能数据与代码状态。

附录A3涉及的通用服务系统：

- [Clockwork，OSDI 2020官方论文页](https://www.usenix.org/conference/osdi20/presentation/gujarati)。服务接纳与性能可预测性，不是机器人闭环安全证明。
- [Paella，作者PDF](https://vincen.tl/files/ng23paella.pdf)。SOSP 2023的软件定义GPU调度，不在本PPT引用其性能倍数。
- [Reflex](https://arxiv.org/abs/2607.14695)。2026预印本，系统报告指出其部分实机统计解释待澄清，本次只作最近邻条目，不引用其成功率。

## 4. 官方异步代码与版本边界

系统报告实际核过LeRobot v0.4.4的固定提交：`8fff0fde7c79f23a93d845d1a50e985de01f8b8a`。

- [robot_client.py](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/async_inference/robot_client.py)
- [policy_server.py](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/async_inference/policy_server.py)
- [configs.py](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/async_inference/configs.py)

后续真正复现应锁定匹配的Python、PyTorch、GPU驱动和机器人接口。不能在浮动main上直接复制旧目录命令，也不能把版本核读说成已经运行官方基线。

## 5. 旧资料中的不一致与保留意见

- Cue the Flow在总索引与横向比较中的年份不同，本次只在附录列简称，不据此标注会议或年份。
- TinyVLA的索引表与单篇笔记对代码公开状态存在差异，本次不声称其已开放可运行代码。
- 旧比较中的“第一篇”“必须用生成模型”“加速最有效的方式”等绝对判断，未带入主讲。
- 不把不同论文的延迟、成功率或平台结果直接排名。PPT没有展示未经同条件核查的性能倍数。
- 本人掌握程度、GPU实测、A100配额、机器人到位时间、OpenFleX接口，以及候选方向创新性，都需要本人或导师进一步确认。

## 6. 文件性质与检查范围

PPTX由可编辑文字、原生表格与原生流程形状组成，没有机器人伪照片或实验截图。讲解稿约13分45秒，附录不计入主讲。已将所有17页渲染为1600×900图像并逐页检查，修正过流程方向、表格与换行。此检查是Artifact Tool渲染与结构检查，不是已在本机Microsoft PowerPoint中打开检查。

原生PDF导出因运行库缺少标准字体模块失败，最终预览PDF由同版幻灯片PNG构成，用于阅读，文字不可编辑或搜索。PPTX保持原生可编辑对象。不同机器的字体替换或PowerPoint渲染仍可能产生细小变化，交流前建议本人打开一次。
'''
(O/'参考文献与数据出处.md').write_text(intro,encoding='utf-8')
readme='''# 生成源码使用说明

`build_deck.mjs`以`@oai/artifact-tool`生成17页可编辑PPTX，`content.mjs`保存讲解与参考条目，`prepare_materials.py`记录来源清点和配套说明，`make_preview.py`生成PDF预览。

本机使用Codex自带Node与Python，未安装或修改全局依赖。`node_modules`为指向现有运行库的目录联接，不是复制的依赖。移到其他电脑时需提供同等Artifact Tool环境，源码中的SKILL、PY与运行库路径需对应修改。

构建过程的候选PPTX、17页PNG、渲染数据与校验回执在工作区`.pptx_build_20261106/`，不属于演示内容。重新构建会生成新的验证副本，再复制到交付文件名，不修改原始调研资料。
'''
(S/'README.md').write_text(readme,encoding='utf-8')
print('24 folders, 120 notes; companion sources written')
