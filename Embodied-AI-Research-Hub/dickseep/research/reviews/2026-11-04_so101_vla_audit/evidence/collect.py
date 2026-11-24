from pathlib import Path
import urllib.request, urllib.error, re, json, hashlib, concurrent.futures, datetime
from html.parser import HTMLParser
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'research/reviews/2026-11-04_so101_vla_feasibility'
class Text(HTMLParser):
 def __init__(self): super().__init__(); self.parts=[]; self.skip=0; self.meta={}
 def handle_starttag(self,t,a):
  a=dict(a)
  if t in ('script','style'): self.skip+=1
  if t in ('p','div','h1','h2','h3','h4','tr','li','section'): self.parts.append('\n')
  if t=='meta' and a.get('name','').startswith('citation_'): self.meta.setdefault(a['name'],[]).append(a.get('content',''))
 def handle_endtag(self,t):
  if t in ('script','style'): self.skip=max(0,self.skip-1)
  if t in ('p','div','h1','h2','h3','tr','li'): self.parts.append('\n')
 def handle_data(self,s):
  if not self.skip:self.parts.append(s)
 def value(self):return '\n'.join(re.sub(r'\s+',' ',s).strip() for s in ''.join(self.parts).splitlines() if s.strip())
def parse_bib(s):
 entries=[]; pos=0
 while True:
  m=re.search(r'@(\w+)\s*\{\s*([^,]+),',s[pos:])
  if not m:break
  start=pos+m.start(); i=pos+m.end(); fields={}
  while True:
   while s[i].isspace() or s[i]==',': i+=1
   if s[i]=='}':i+=1;break
   f=re.match(r'(\w+)\s*=\s*',s[i:]); assert f,(i,s[i:i+80]); key=f[1];i+=f.end()
   if s[i]=='{':
    i+=1;j=i;depth=1
    while depth:
     if s[i]=='\\':i+=2;continue
     if s[i]=='{':depth+=1
     if s[i]=='}':depth-=1
     i+=1
    value=s[j:i-1]
   else:
    f=re.match(r'[A-Za-z0-9]+',s[i:]);assert f;i+=f.end();value=f[0]
   assert key not in fields;fields[key]=value
  entries.append(dict(key=m[2],type=m[1],line=s[:start].count('\n')+1,**fields));pos=i
 return entries
def fetch(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'AcademicEvidenceAudit/1.0'})
  with urllib.request.urlopen(req,timeout=45) as r: data=r.read();status=r.status;final=r.url
  (OUT/(name+'.html')).write_bytes(data)
  p=Text();p.feed(data.decode('utf-8',errors='replace'));(OUT/(name+'.txt')).write_text(p.value(),encoding='utf-8')
  return dict(name=name,url=url,final_url=final,status=status,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),metadata=p.meta)
 except Exception as e:return dict(name=name,url=url,error=str(e))
def local():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for p in SRC.iterdir():
  if p.is_file():manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 (OUT/'original_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 entries=parse_bib((SRC/'references.bib').read_text(encoding='utf8'))
 (OUT/'bib_parsed.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf8')
 for p in Path('D:/Desktop').glob('*.pdf'):
  if p.name.startswith(('2610.34717','2611.00283','2611.03103','2. 现代')):
   try:
    reader=PdfReader(p); pages=[pg.extract_text() or '' for pg in reader.pages]
    name='curriculum' if p.name.startswith('2.') else p.stem
    (OUT/(name+'_local.txt')).write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{v}' for i,v in enumerate(pages)),encoding='utf8')
    manifest.append(dict(path=str(p),bytes=p.stat().st_size,pages=len(pages),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   except Exception as e:manifest.append(dict(path=str(p),error=str(e)))
 (OUT/'local_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 print('local files',len(manifest),'bib entries',len(entries),'unique keys',len({x['key'] for x in entries}),'unique ids',len({x['eprint'] for x in entries}))
def network():
 entries=json.loads((OUT/'bib_parsed.json').read_text(encoding='utf8'))
 ids=[e['eprint'] for e in entries]+['2610.36420','2606.08881','2603.13966','2610.14603','2609.00065']
 items=[('abs_'+i,'https://arxiv.org/abs/'+i) for i in ids]
 full=['2610.36420v1','2606.08881v2','2603.13966v3','2610.14603v1','2506.07339v2','2502.19645v2','2506.01844v1','2604.04161v2','2609.00908v3','2607.24008v1','2609.25376v1']
 items += [('full_'+i,'https://arxiv.org/html/'+i) for i in full]
 docs=['so101','installation','smolvla','pi05','teleoperate','record','phone','async','rtc','cameras','act']
 items += [('doc_'+i,'https://huggingface.co/docs/lerobot/'+i) for i in docs]
 items += [('lerobot_readme','https://raw.githubusercontent.com/huggingface/lerobot/main/README.md'),('lerobot_pyproject','https://raw.githubusercontent.com/huggingface/lerobot/main/pyproject.toml'),('openpi_readme','https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/README.md'),('so101_bom','https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/README.md')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(fetch,items))
 (OUT/'web_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 print('web',len(results),'ok',sum('status' in r for r in results));print(json.dumps([r for r in results if 'error'in r],ensure_ascii=False))
if __name__=='__main__':
 import sys
 if '--network'in sys.argv:network()
 else:local()
