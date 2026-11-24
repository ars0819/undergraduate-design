import pathlib,re,sys,json
R=pathlib.Path(__file__).parent
for aid in sys.argv[1:]:
 p=R/(aid+'.txt')
 if not p.exists(): continue
 t=p.read_text(encoding='utf-8'); lines=t.splitlines()
 print('\n===== '+aid+' =====\n'+t[:1700])
 # Extract section starts and table neighborhoods, preserving page markers.
 keys=r'^(?:[3-6](?:\.[0-9]+)?\s+[A-Z]|[3-6]\s*$|Table [1-6]|4\.\d|5\.\d)|A100|H100|RTX|4090|3090|University|INSAIT|Institute|Trials|rollouts|Limitations|limitations'
 inds=[i for i,l in enumerate(lines) if re.search(keys,l)]
 used=set(); out=[]
 for i in inds:
  for j in range(max(0,i-2),min(len(lines),i+11)):
   if j not in used: out.append(f'{j+1}: {lines[j]}'); used.add(j)
 print('\n'.join(out)[:17000])
