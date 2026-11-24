from pathlib import Path
import urllib.request,json,hashlib,subprocess,sys
out=Path(__file__).resolve().parents[1]/'证据'; log=[]
def get(name,url):
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ThesisAudit'}),timeout=15) as r: b=r.read()
  (out/name).write_bytes(b);log.append({'file':name,'url':url,'status':'ok','sha256':hashlib.sha256(b).hexdigest()}); print(name,len(b),flush=True);return b
 except Exception as e: log.append({'file':name,'url':url,'error':str(e)});print(name,str(e),flush=True)
for label,repo in [('smolvla','lerobot/smolvla_libero'),('oft_spatial','moojink/openvla-7b-oft-finetuned-libero-spatial')]:
 b=get(label+'_api.json','https://huggingface.co/api/models/'+repo)
 if b:
  sha=json.loads(b)['sha']
  for f in ['config.json','train_config.json','policy_preprocessor.json','policy_postprocessor.json','README.md']:
   get(label+'_'+f,'https://huggingface.co/'+repo+'/resolve/'+sha+'/'+f)
sha=json.loads((out/'版本锁定.json').read_text())['quantbench']
for f in ['scripts/env/setup_openvla_oft.sh','scripts/profile_kernels.py','docs/protocol.md','pyproject.toml']:
 get('vqb_'+f.replace('/','__'),'https://raw.githubusercontent.com/jiuyixu25/VLAQuantBench/'+sha+'/'+f)
for repo,label in [('moojink/transformers-openvla-oft','oft_transformers'),('moojink/dlimp_openvla','oft_dlimp')]:
 get(label+'_commit.json','https://api.github.com/repos/'+repo+'/commits/HEAD')
(out/'补充获取记录.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf8')
records=[]
for f in ['metadata.xml','metadata_extra.xml','metadata_related.xml']:
 p=Path('D:/Desktop/bishe/sources')/f
 proc=subprocess.run([sys.executable,'C:/Users/Administrator/.codex/skills/paper-lookup/scripts/arxiv_atom.py',str(p)],capture_output=True,text=True,encoding='utf8')
 (out/('缓存重新解析_'+f+'.json')).write_text(proc.stdout,encoding='utf8')
 records.append({'script':'arxiv_atom.py','input':str(p),'exit_code':proc.returncode,'stderr':proc.stderr})
(out/'paper_lookup执行记录.json').write_text(json.dumps(records,indent=2),encoding='utf8')
