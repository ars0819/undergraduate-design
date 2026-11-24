from collect import *
import sys
sys.stdout.reconfigure(encoding='utf8')
for ident in ['2610.36420','2606.08881','2611.00283','2603.13966','2610.14603','2611.03103','2610.34717','2506.07339','2502.19645','2506.01844','2604.04161','2609.00908','2607.24008','2609.25376']:
 s=(OUT/('abs_'+ident+'.html')).read_text(encoding='utf8')
 links=sorted(set(re.findall(r'href="(https?://[^"<>]+)"',s)))
 print(ident,[l for l in links if any(k in l for k in ['github','huggingface','pi.website']) and 'arxiv' not in l])
