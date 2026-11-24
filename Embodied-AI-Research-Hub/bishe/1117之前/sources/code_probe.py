import urllib.request,json,pathlib
R=pathlib.Path(__file__).parent
urls={
'lerobot_tree.json':'https://api.github.com/repos/huggingface/lerobot/git/trees/9e5ee9859480e425a97a6836bb3fee479539beb5?recursive=1',
'lerobot_tags.json':'https://api.github.com/repos/huggingface/lerobot/tags?per_page=20',
'lerobot_commit.json':'https://api.github.com/repos/huggingface/lerobot/commits?until=2026-11-05T15:59:59Z&per_page=1',
'lerobot_async.mdx':'https://raw.githubusercontent.com/huggingface/lerobot/main/docs/source/async.mdx',
'lerobot_robot_client.py':'https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/async_inference/robot_client.py',
'lerobot_policy_server.py':'https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/async_inference/policy_server.py',
'lerobot_configs.py':'https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/async_inference/configs.py',
'lerobot_helpers.py':'https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/async_inference/helpers.py',
'openvla_oft_README.md':'https://raw.githubusercontent.com/moojink/openvla-oft/main/README.md',
}
sha=None
for name,url in urls.items():
  try:
    if sha and 'raw.githubusercontent.com/huggingface/lerobot/main/' in url: url=url.replace('/main/','/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/')
    req=urllib.request.Request(url,headers={'User-Agent':'ResearchEvidence/1.0'})
    data=urllib.request.urlopen(req,timeout=30).read(); (R/name).write_bytes(data)
    if name=='lerobot_commit.json': sha=json.loads(data)[0]['sha']
    print(name,len(data),'sha='+str(sha),flush=True)
  except Exception as e: print(name,str(e),flush=True)
