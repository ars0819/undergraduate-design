"""Prepared-observation model-only new-chunk benchmark; NOT a closed-loop evaluator.

Input .pt must come from the frozen official preprocessor on real observations,
containing tensor keys incl. tokenized language, state and mapped images.
Does not time preprocessing, D2H or environment. GPU execution unverified.
"""
import argparse,contextlib,csv,hashlib,json,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True);p.add_argument('--batch',required=True)
 p.add_argument('--out',required=True);p.add_argument('--compile',action='store_true')
 p.add_argument('--precision',choices=['fp32','bf16_amp'],default='fp32')
 p.add_argument('--steps',type=int,default=10);p.add_argument('--warmup',type=int,default=20)
 p.add_argument('--iters',type=int,default=200);p.add_argument('--seed',type=int,default=20261106)
 a=p.parse_args();out=Path(a.out)
 if out.exists():raise SystemExit('Use a new output directory for each session')
 if a.iters<1 or a.warmup<1 or a.steps<1:raise SystemExit('Positive counts required')
 out.mkdir(parents=True)
 import torch,numpy as np
 from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
 from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
 if not torch.cuda.is_available():raise SystemExit('CUDA required')
 if a.precision=='bf16_amp' and not torch.cuda.is_bf16_supported():raise SystemExit('BF16 unsupported')
 cfg=SmolVLAConfig.from_pretrained(a.checkpoint)
 cfg.compile_model=a.compile;cfg.num_steps=a.steps;cfg.device='cuda'
 torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 policy=SmolVLAPolicy.from_pretrained(a.checkpoint,config=cfg).to(device='cuda',dtype=torch.float32).eval()
 torch.cuda.synchronize();load_s=time.perf_counter()-t
 load_peak=torch.cuda.max_memory_allocated()
 data=torch.load(a.batch,map_location='cpu',weights_only=True)
 batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in data.items()}
 if not all(torch.is_tensor(v) for v in batch.values()):raise SystemExit('Prepared batch must contain only tensors')
 b=batch['observation.state'].shape[0]
 if b!=1:raise SystemExit('First baseline requires batch=1')
 g=torch.Generator(device='cuda').manual_seed(a.seed)
 noise=torch.randn((b,cfg.chunk_size,cfg.max_action_dim),generator=g,device='cuda')
 def call():
  policy.reset()
  ctx=torch.autocast('cuda',dtype=torch.bfloat16) if a.precision=='bf16_amp' else contextlib.nullcontext()
  with torch.inference_mode(),ctx:return policy.predict_action_chunk(dict(batch),noise=noise.clone())
 torch.cuda.synchronize();t=time.perf_counter();first=call();torch.cuda.synchronize()
 first_s=time.perf_counter()-t;first_peak=torch.cuda.max_memory_allocated()
 if first.shape[-1]!=7 or not torch.isfinite(first).all():raise SystemExit('Invalid action chunk')
 torch.save(first.cpu(),out/'first_normalized_chunk.pt')
 for _ in range(a.warmup):call()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();rows=[]
 for i in range(a.iters):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  torch.cuda.synchronize();t=time.perf_counter();start.record();y=call();end.record();torch.cuda.synchronize()
  wall=(time.perf_counter()-t)*1000;gpu=start.elapsed_time(end)
  if not torch.isfinite(y).all():raise SystemExit('Nonfinite output')
  rows.append(dict(index=i,wall_ms=wall,cuda_ms=gpu,new_chunk=True))
 with (out/'latency.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 lat=np.array([r['wall_ms'] for r in rows]);chunk_s=1000/lat.mean()
 summary=dict(status='model_only_not_task_evaluation',args=vars(a),torch=torch.__version__,cuda=torch.version.cuda,
  gpu=torch.cuda.get_device_name(),parameter_dtypes=sorted({str(p.dtype) for p in policy.parameters()}),batch_sha256=hashlib.sha256(Path(a.batch).read_bytes()).hexdigest(),
  load_s=load_s,first_call_including_compile_s=first_s,load_peak_bytes=load_peak,first_call_peak_bytes=first_peak,
  steady_peak_allocated_bytes=torch.cuda.max_memory_allocated(),steady_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
  p50_ms=float(np.percentile(lat,50)),p95_ms=float(np.percentile(lat,95)),chunks_per_s=chunk_s,
  predicted_actions_per_s=chunk_s*cfg.chunk_size,action_shape=list(first.shape),
  excludes=['preprocessing','postprocessing','D2H','environment','NVML_memory','module_profiler'])
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 print('Saved model-only timing; requires separate processor and episode validation')

if __name__=='__main__':main()
