import retrieve, pathlib, urllib.request, json, time
retrieve.IDS=['2602.15922','2610.20442','2512.10675','2207.05608','2209.07753']
import urllib.parse,subprocess,sys,os
feed=retrieve.ROOT/'metadata_extra.xml'
retrieve.get('https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'id_list':','.join(retrieve.IDS),'max_results':10}),feed)
env=dict(os.environ,PYTHONUTF8='1')
p=subprocess.run([sys.executable,'C:/Users/Administrator/.codex/skills/paper-lookup/scripts/arxiv_atom.py',str(feed)],capture_output=True,encoding='utf-8',env=env)
(retrieve.ROOT/'metadata_extra.json').write_text(p.stdout,encoding='utf-8')
retrieve.main()
