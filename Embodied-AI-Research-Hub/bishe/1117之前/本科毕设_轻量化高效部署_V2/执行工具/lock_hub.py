"""Download an immutable Hub snapshot and record local hashes. Not run in this audit."""
import argparse, hashlib, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('repo_id'); p.add_argument('output'); p.add_argument('--revision',default='main')
 p.add_argument('--repo-type',choices=['model','dataset'],default='model')
 a=p.parse_args()
 from huggingface_hub import HfApi,snapshot_download
 target=Path(a.output)
 if target.exists() and any(target.iterdir()): raise SystemExit('Output must be new/empty; refusing overwrite')
 info=HfApi().repo_info(a.repo_id,repo_type=a.repo_type,revision=a.revision)
 sha=info.sha
 if not sha or len(sha)!=40: raise SystemExit('No full immutable revision')
 snapshot_download(a.repo_id,repo_type=a.repo_type,revision=sha,local_dir=target)
 files=[]
 for f in sorted(target.rglob('*')):
  if not f.is_file() or '.cache' in f.relative_to(target).parts: continue
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
  files.append(dict(file=f.relative_to(target).as_posix(),bytes=f.stat().st_size,sha256=h.hexdigest()))
 index=next(target.glob('*.safetensors.index.json'),None)
 if index:
  refs=set(json.loads(index.read_text())['weight_map'].values())
  missing=[f for f in refs if not (target/f).is_file()]
  if missing:raise SystemExit(f'Missing shards: {missing}')
 (target/'download_manifest.json').write_text(json.dumps(dict(repo_id=a.repo_id,revision=sha,files=files,status='downloaded_not_forward_verified'),ensure_ascii=False,indent=2),encoding='utf8')
 print(sha, len(files),'files; forward and processor compatibility still require verification')

if __name__=='__main__':main()
