from pathlib import Path
from docx import Document
from pypdf import PdfReader
import json
O=Path('本科毕设_正式任务书填写_20261106/校验材料')
paths=[Path('D:/Desktop/附件6：信通院2027届本科毕设常用特殊表格.docx'),Path('D:/Desktop/09_本科毕业设计任务书_已填写_导师讨论稿.docx'),Path('D:/Desktop/信通院本科毕业设计（论文）任务书-面向多自由度机械臂操作的具身智能算法设计与实验.docx'),*Path('.').glob('*.docx')]
for k,p in enumerate(paths):
 d=Document(p); lines=[str(p)]+[f'P{i}: {x.text}' for i,x in enumerate(d.paragraphs)]
 for ti,t in enumerate(d.tables):
  lines.append(f'TABLE {ti} {len(t.rows)}x{len(t.columns)}')
  for ri,r in enumerate(t.rows):
   seen=set()
   for ci,c in enumerate(r.cells):
    if c._tc in seen:continue
    seen.add(c._tc); lines.append(f'R{ri}C{ci}: '+c.text)
 (O/f'doc{k}.txt').write_text('\n'.join(lines),encoding='utf-8')
p=PdfReader('D:/Desktop/2. 现代邮政学院（自动化学院）2023级本科生培养方案.pdf')
(O/'培养方案原文.txt').write_text('\n'.join(f'PAGE {i+1}\n'+p.pages[i].extract_text() for i in range(52,59)),encoding='utf-8')
print('read',len(paths),'documents',len(p.pages),'pdf pages')
