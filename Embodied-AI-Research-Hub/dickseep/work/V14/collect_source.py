import datetime, hashlib, json, pathlib, urllib.request
root = pathlib.Path('work/V14')
root.mkdir(parents=True, exist_ok=True)
repo = 'moojink/openvla-oft'
meta = json.load(urllib.request.urlopen(f'https://api.github.com/repos/{repo}/commits/main'))
sha = meta['sha']
(root / 'commit.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
paths = ['prismatic/vla/constants.py', 'experiments/robot/libero/run_libero_eval.py', 'experiments/robot/robot_utils.py', 'experiments/robot/openvla_utils.py', 'prismatic/extern/hf/modeling_prismatic.py', 'prismatic/models/action_heads.py', 'experiments/robot/libero/libero_utils.py']
manifest = {'repository': repo, 'branch_resolved': 'main', 'commit': sha, 'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': []}
for path in paths:
    url = f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
    content = urllib.request.urlopen(url).read()
    target = root / 'source' / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    manifest['files'].append({'path': path, 'url': url, 'sha256': hashlib.sha256(content).hexdigest()})
(root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
