from pathlib import Path
import urllib.request, json, hashlib, time
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'证据'; OUT.mkdir(exist_ok=True)
log=[]
def fetch(name,url):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'ThesisReproducibilityAudit/1.0'})
        with urllib.request.urlopen(req,timeout=25) as r: data=r.read(); status=r.status
        (OUT/name).write_bytes(data)
        log.append(dict(file=name,url=url,status=status,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        print(name,status,len(data),flush=True)
        return data
    except Exception as e:
        log.append(dict(file=name,url=url,error=str(e))); print(name,str(e),flush=True)
        return None
repos={'lerobot':'huggingface/lerobot','oft':'moojink/openvla-oft','libero':'Lifelong-Robot-Learning/LIBERO','quantbench':'jiuyixu25/VLAQuantBench','openpi':'Physical-Intelligence/openpi'}
pins={}
for k,repo in repos.items():
    d=fetch(k+'_commit.json',f'https://api.github.com/repos/{repo}/commits/HEAD')
    if d:
        sha=json.loads(d)['sha']; pins[k]=sha
        for f in {'lerobot':['pyproject.toml','src/lerobot/policies/smolvla/modeling_smolvla.py','src/lerobot/policies/smolvla/configuration_smolvla.py','src/lerobot/envs/configs.py','src/lerobot/scripts/lerobot_eval.py','docs/source/libero.mdx'], 'oft':['README.md','SETUP.md','LIBERO.md','pyproject.toml','experiments/robot/libero/run_libero_eval.py','experiments/robot/libero/libero_requirements.txt','experiments/robot/openvla_utils.py'], 'libero':['requirements.txt'], 'quantbench':['README.md'], 'openpi':['README.md','examples/libero/README.md']}[k]:
            fetch(k+'_'+f.replace('/','__'),f'https://raw.githubusercontent.com/{repo}/{sha}/{f}')
for k,model in {'smolvla':'lerobot/smolvla_libero','smolvla_base':'lerobot/smolvla_base','oft_spatial':'moojink/openvla-7b-oft-finetuned-libero-spatial'}.items():
    d=fetch(k+'_hub.json',f'https://huggingface.co/api/models/{model}?blobs=true')
    if d:
        sha=json.loads(d)['sha']; pins[k]=sha
        for f in ['config.json','train_config.json','README.md']:
            fetch(k+'_'+f,f'https://huggingface.co/{model}/resolve/{sha}/{f}')
ids='2506.01844,2502.19645,2410.24164,2504.16054,2306.00978,2211.10438,2411.05007,2609.25376,2602.20309,2606.11087,2306.03310,2609.00065'
fetch('arxiv_atom.xml','https://export.arxiv.org/api/query?id_list='+ids+'&max_results=20')
for k,url in {'zhou':'https://zhouzypaul.github.io/research/','hanlab':'https://hanlab.mit.edu/news','quantbench_paper':'https://arxiv.org/html/2609.25376v1','quantvla_paper':'https://arxiv.org/html/2602.20309','smolvla_paper':'https://arxiv.org/html/2506.01844','oft_paper':'https://arxiv.org/html/2502.19645','awq':'https://raw.githubusercontent.com/mit-han-lab/llm-awq/main/README.md','smoothquant':'https://raw.githubusercontent.com/mit-han-lab/smoothquant/main/README.md','svdquant':'https://hanlab.mit.edu/projects/svdquant'}.items():
    fetch(k+'.html' if k not in ['awq','smoothquant'] else k+'.md',url)
(OUT/'版本锁定.json').write_text(json.dumps(pins,indent=2),encoding='utf-8')
(OUT/'网络获取记录.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
