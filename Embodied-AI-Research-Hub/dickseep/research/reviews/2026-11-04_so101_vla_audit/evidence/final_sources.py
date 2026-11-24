from collect import *
sha=json.loads((OUT/'lerobot_commit.json').read_text(encoding='utf8'))['sha']
tree=json.loads((OUT/'lerobot_tree.json').read_text(encoding='utf8'))
paths=[x['path'] for x in tree['tree'] if x['path'].endswith('.py') and (x['path'].startswith('src/lerobot/robots/so_follower/') or x['path'].endswith('/writer.py') or 'async' in x['path'] and 'test' not in x['path'])]
print('\n'.join(paths))
items=[('final_'+p.replace('/','_'),f'https://raw.githubusercontent.com/huggingface/lerobot/{sha}/{p}') for p in paths]
items += [('full_2610.14603v2','https://arxiv.org/html/2610.14603v2'),('full_2603.19131v1','https://arxiv.org/html/2603.19131v1')]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,items))
(OUT/'final_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
