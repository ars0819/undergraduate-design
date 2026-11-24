from collect import *
items=[('doc_phone_main','https://huggingface.co/docs/lerobot/main/phone'),('doc_teleop_isaac','https://huggingface.co/docs/lerobot/main/teleop_isaac'),('doc_il','https://huggingface.co/docs/lerobot/il_robots'),('doc_async_main','https://huggingface.co/docs/lerobot/main/async'),('doc_dataset','https://huggingface.co/docs/lerobot/lerobot-dataset-v3')]
req=urllib.request.Request('https://api.github.com/repos/huggingface/lerobot/commits/main',headers={'User-Agent':'EvidenceAudit'})
commit=json.load(urllib.request.urlopen(req,timeout=30));sha=commit['sha']
(OUT/'lerobot_commit.json').write_text(json.dumps(commit,ensure_ascii=False,indent=2),encoding='utf8')
paths=['src/lerobot/scripts/lerobot_record.py','src/lerobot/robots/so101_follower/so101_follower.py','src/lerobot/robots/so100_follower/so100_follower.py','src/lerobot/policies/act/modeling_act.py','src/lerobot/policies/smolvla/modeling_smolvla.py','src/lerobot/policies/pi05/modeling_pi05.py','src/lerobot/datasets/lerobot_dataset.py','src/lerobot/async_inference/robot_client.py','src/lerobot/async_inference/policy_server.py','pyproject.toml']
items += [('code_'+p.split('/')[-1].replace('.py',''),f'https://raw.githubusercontent.com/huggingface/lerobot/{sha}/{p}') for p in paths]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(fetch,items))
(OUT/'more_manifest.json').write_text(json.dumps({'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha':sha,'sources':results},ensure_ascii=False,indent=2),encoding='utf8')
print(sha); print(json.dumps([r for r in results if 'error'in r]))
