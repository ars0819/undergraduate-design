from pathlib import Path
import json,sys,os,subprocess,re,hashlib
root=Path(__file__).resolve().parents[1];out=root/'证据'
ids=['2506.01844','2502.19645','2410.24164','2504.16054','2306.00978','2211.10438','2411.05007','2609.25376','2602.20309','2606.11087','2306.03310']
titles=['SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics','Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success','π0: A Vision-Language-Action Flow Model for General Robot Control','π0.5: a Vision-Language-Action Model with Open-World Generalization','AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration','SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models','SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models','VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models','QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models','Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning','LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning']
records=[dict(title=t,year=2000+int(i[:2]),url='https://arxiv.org/abs/'+i,source='arXiv',arxiv_id=i) for t,i in zip(titles,ids)]
(out/'核心书目规范化.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
manifest={'schema_version':'1.0','sources':[dict(evidence_id=f'E{n:03}',title=r['title'],year=r['year'],identifiers={'url':r['url']},source_type='preprint',confidentiality='public',verification={'source_opened':True,'status':'unverified','verified_by':'','verified_on':''},locator='See literature matrix; assistant source access, human review pending') for n,r in enumerate(records,1)]}
(out/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
for source in manifest['sources']:
 source['authors']=[]  # Full bibliographic author strings remain in document 03; not inferred here.
(out/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
commands=[('literature_dedup','literature-review/scripts/search_databases.py',[str(out/'核心书目规范化.json'),'--deduplicate','--format','json','--output',str(out/'核心书目去重.json'),'--summary']),('hypothesis_lint','hypothesis-generation/scripts/lint_causal_claims.py',[str(root/'02_实验基线与复现规划_V1.md')]),('writing_refs','scientific-writing/scripts/check_references.py',[str(out/'source_manifest.json')])]
for doc in sorted(root.glob('0[1-6]_*.md')):
 commands.append(('writing_'+doc.name[:2],'scientific-writing/scripts/lint_manuscript.py',[str(doc)]))
logs=[]
for label,script,args in commands:
 env=dict(os.environ,PYTHONUTF8='1')
 proc=subprocess.run([sys.executable,'-X','utf8',str(Path('C:/Users/Administrator/.codex/skills')/script),*args],capture_output=True,env=env)
 stdout=proc.stdout.decode('utf8',errors='replace');stderr=proc.stderr.decode('utf8',errors='replace')
 (out/(label+'.txt')).write_text(stdout+'\nSTDERR:\n'+stderr,encoding='utf8')
 logs.append(dict(name=label,script=script,arguments=args,exit_code=proc.returncode,output=label+'.txt'))
(out/'Scientific脚本执行结果.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding='utf8')
old=json.loads((out/'原始材料读取清单.json').read_text(encoding='utf8'))
changed=[]
for row in old:
 p=Path(row['path'])
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:changed.append(str(p))
(out/'原文件保持检查.json').write_text(json.dumps(dict(checked=len(old),changed=changed),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(logs,ensure_ascii=False,indent=2));print('Original files changed:',changed)
