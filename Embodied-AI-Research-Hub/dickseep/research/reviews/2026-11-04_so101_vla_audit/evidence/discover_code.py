from collect import *
sha=json.loads((OUT/'lerobot_commit.json').read_text(encoding='utf8'))['sha']
url=f'https://api.github.com/repos/huggingface/lerobot/git/trees/{sha}?recursive=1'
try:tree=json.load(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EvidenceAudit'}),timeout=30))
except Exception as e:
 print(str(e));tree={'tree':[]}
(OUT/'lerobot_tree.json').write_text(json.dumps(tree),encoding='utf8')
paths=[x['path'] for x in tree['tree'] if any(s in x['path'] for s in ['so101','so100','phone_teleop','isaac_teleop','robot_client.py','policy_server.py']) and x['path'].endswith(('.py','.md'))]
print('\n'.join(paths))
items=[('doc_phone_teleop','https://huggingface.co/docs/lerobot/phone_teleop'),('doc_isaac_teleop','https://huggingface.co/docs/lerobot/isaac_teleop')]
items += [('resolved_'+p.replace('/','_'),f'https://raw.githubusercontent.com/huggingface/lerobot/{sha}/{p}') for p in paths]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,items))
(OUT/'resolved_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
