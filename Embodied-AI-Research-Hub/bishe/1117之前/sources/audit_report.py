import pathlib,re,json,sys,hashlib,importlib.util
R=pathlib.Path(__file__).parent; P=R.parent/'具身智能系统架构与协同调度_深度研究报告.md'
t=P.read_text(encoding='utf-8'); body=t.split('<!-- REFERENCES_INSERT -->')[0]
refs={int(x) for x in re.findall(r'^\*\*\[(\d+)\]',t,re.M)}
cites={int(x) for x in re.findall(r'\[(\d+)\]',body)}
heads=re.findall(r'^## (\d+)\.',t,re.M)
assert heads==[str(i) for i in range(1,19)],heads
assert refs==set(range(1,39)),refs
assert not cites-refs
assert t.count('```')%2==0
deep=json.loads((R/'included_evidence.json').read_text(encoding='utf-8'))
assert sum(x['evidence_depth']=='F' for x in deep)==15
assert all(x['published'][:10]<='2026-11-05' and x['updated'][:10]<='2026-11-05' for x in deep)
spec=importlib.util.spec_from_file_location('randomization','C:/Users/Administrator/.codex/skills/experimental-design/scripts/randomization.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
design=[]
for task in ['static_transport','target_shift']:
 for delay in ['native','fixed','tail']:
  for initial_state in range(10):
   for method in ['official_sync','official_async_tuned','adaptive_candidate']:
    design.append(dict(task=task,delay=delay,initial_state=initial_state,method=method))
# Randomize inside paired task/delay/initial-state blocks; this is a proposed pilot, not collected data.
import pandas as pd
frames=[]
for i in range(0,len(design),3):
 f=m.assign_factorial_runs(pd.DataFrame(design[i:i+3]),seed=20261105+i)
 f['block_id']=i//3; frames.append(f)
schedule=pd.concat(frames,ignore_index=True)
schedule.to_csv(R/'pilot_randomized_schedule.csv',index=False)
result=dict(reference_count=38,research_paper_count=36,focused_fulltext_count=15,
 body_chinese_characters=sum('\u4e00'<=c<='\u9fff' for c in body),
 total_characters=len(t),missing_citations=sorted(cites-refs),chapter_count=len(heads),
 mermaid_blocks=len(re.findall(r'```mermaid',t)),
 mermaid_check='manual syntax inspection; parser/render not executed; bundled mermaid unavailable',
 pilot_rows=len(schedule),pilot_status='design only; no experimental results',
 report_sha256=hashlib.sha256(P.read_bytes()).hexdigest())
(R/'final_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
