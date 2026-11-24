from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
p=Path(__file__).parent
spec={
'full_2607.24008v1.txt':r'Hardware Setup|Real-world Evaluation|Baselines\.|Limitations|Franka|Aloha|UR5|4090',
'full_2609.00908v3.txt':r'A100|Hardware|ARX|Baseline|Limitations|0.2661',
'full_2604.04161v2.txt':r'A800|Hardware|Real-World|Real-world|Franka|Agile|UR5',
'full_2502.19645v2.txt':r'LIBERO|ALOHA|A100|Hardware|RTX|Summary|Our Recipe',
'full_2506.01844v1.txt':r'SO-100|SO-101|A100|450M|500M|asynchronous|Asynchronous',
'full_2603.13966v3.txt':r'Experimental|Hardware|A100|H100|4090|47|Limitations|Servers',
'full_2610.14603v1.txt':r'Experiments|RoboArena|TRI|Baseline|Assumption|Bernoulli|Limitations',
'full_2609.25376v1.txt':r'4090|Table 10|execution horizon|Limitations|simulation|200 episodes',
}
for name,pat in spec.items():
 lines=(p/name).read_text(encoding='utf8').splitlines(); print('\nFILE',name)
 hits=[(i+1,s) for i,s in enumerate(lines) if re.search(pat,s)]
 for i,s in hits[:14]:print(i,s[:1500])
