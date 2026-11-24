import retrieve,urllib.parse,subprocess,sys,os
retrieve.IDS=['2512.01031','2605.08168','2606.31132']
feed=retrieve.ROOT/'metadata_related.xml'
retrieve.get('https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'id_list':','.join(retrieve.IDS),'max_results':5}),feed)
p=subprocess.run([sys.executable,'C:/Users/Administrator/.codex/skills/paper-lookup/scripts/arxiv_atom.py',str(feed)],capture_output=True,encoding='utf-8',env=dict(os.environ,PYTHONUTF8='1'))
(retrieve.ROOT/'metadata_related.json').write_text(p.stdout,encoding='utf-8')
retrieve.main()
