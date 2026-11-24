"""Run in the actual experiment environment; does not install anything."""
import argparse,json,platform,subprocess,sys,importlib.metadata
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
 dest=Path(a.out)
 if dest.exists():raise SystemExit('Refusing overwrite')
 r=dict(os=platform.platform(),python=sys.version,status='probe_only')
 for package in ['torch','torchvision','transformers','lerobot','mujoco','robosuite','hf-libero','torchcodec']:
  try:r[package]=importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:r[package]=None
 try:
  q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv'],capture_output=True,text=True,timeout=15)
  r['nvidia_smi']=q.stdout;r['nvidia_smi_error']=q.stderr;r['nvidia_smi_returncode']=q.returncode
 except Exception as e:r['nvidia_smi_error']=str(e)
 try:
  import torch
  r['cuda_available']=torch.cuda.is_available();r['torch_cuda']=torch.version.cuda
  r['compiled_arches']=torch.cuda.get_arch_list()
  if torch.cuda.is_available():
   r['device']=torch.cuda.get_device_name();r['capability']=torch.cuda.get_device_capability()
   x=torch.ones((64,64),device='cuda');y=x@x;torch.cuda.synchronize()
   r['cuda_matmul_finite']=bool(torch.isfinite(y).all())
 except Exception as e:r['torch_probe_error']=str(e)
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
 print('Probe saved; model forward/render/episode still unverified')
if __name__=='__main__':main()
