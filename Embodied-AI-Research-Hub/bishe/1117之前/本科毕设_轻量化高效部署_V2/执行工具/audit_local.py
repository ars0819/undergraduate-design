from pathlib import Path
import hashlib,json,sys,subprocess,importlib.util
from pypdf import PdfReader
from docx import Document
root=Path(__file__).resolve().parents[1]; out=root/'证据'
paths=[]
for p in [Path('D:/AGENTS.md'),Path('D:/Desktop/AGENTS.md'),Path('D:/Desktop/bishe/AGENTS.md')]:
    if p.exists(): paths.append(p)
paths += list(Path('D:/Desktop/dickseep/research/thesis/2026-11-04_导师交流初稿').glob('*.md'))
paths += list(Path('D:/Desktop/library/Literature_Notes').rglob('*.md'))
paths += [Path('D:/Desktop/bishe/具身智能系统架构与协同调度_深度研究报告.md')]
paths += list(Path('D:/Desktop/bishe/导师交流_精简版_20261106').glob('*.md'))
manifest=[]; excerpts=[]
for p in paths:
    data=p.read_bytes(); text=data.decode('utf-8-sig')
    manifest.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'read':True})
    if p.name=='01_论文精读.md':
        lines=[s for s in text.splitlines() if s.strip() and any(w in s for w in ['核心','开源','复现','显存','量化','推理','局限'])]
        excerpts.append(str(p)+'\n'+'\n'.join(lines[:15]))
(out/'原始材料读取清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
(out/'24篇精读重读摘录.txt').write_text('\n\n'.join(excerpts),encoding='utf8')
pdf=Path('D:/Desktop/2. 现代邮政学院（自动化学院）2023级本科生培养方案.pdf')
r=PdfReader(pdf)
(out/'培养方案原PDF页53至59.txt').write_text('\n\n'.join(f'=== PDF页 {i+1} ===\n'+r.pages[i].extract_text() for i in range(52,59)),encoding='utf8')
manifest.append({'path':str(pdf),'pages':len(r.pages),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()})
for p in [Path('D:/Desktop/09_本科毕业设计任务书_已填写_导师讨论稿.docx'),Path('D:/Desktop/信通院本科毕业设计（论文）任务书-面向多自由度机械臂操作的具身智能算法设计与实验.docx')]:
    d=Document(p); txt='\n'.join(x.text for x in d.paragraphs)+'\n'+'\n'.join(' | '.join(c.text for c in row.cells) for t in d.tables for row in t.rows)
    (out/(p.stem+'.txt')).write_text(txt,encoding='utf8')
    manifest.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'read':True})
(out/'原始材料读取清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
skills=['literature-review','paper-lookup','scientific-critical-thinking','hypothesis-generation','experimental-design','scientific-writing']
inventory=[]
for name in skills:
    p=Path('C:/Users/Administrator/.codex/skills')/name/'SKILL.md'; t=p.read_text(encoding='utf8')
    inventory.append({'name':name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(out/'Skills实际安装清单.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')
sys.path.insert(0,'C:/Users/Administrator/.codex/skills/experimental-design/scripts')
from randomization import assign_factorial_runs
import pandas as pd,numpy as np
rows=[]
for task in [0,1,2]:
    for init in range(10):
        block=assign_factorial_runs(pd.DataFrame([{'task_id':task,'init_state_id':init,'condition':c,'noise_seed':10000+task*100+init,'status':'NOT_RUN'} for c in ['original','optimized']]),seed=20261106+task*100+init)
        rows+=block.to_dict('records')
pd.DataFrame(rows).to_csv(out/'配对试点顺序_未执行.csv',index=False,encoding='utf8')
(out/'实验设计运行环境.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'units':30,'planned_runs':60,'status':'schedule_only_not_results'},indent=2),encoding='utf8')
print('Read files',len(manifest),'PDF pages',len(r.pages),'paired schedule',len(rows))
