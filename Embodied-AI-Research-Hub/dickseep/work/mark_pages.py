# -*- coding: utf-8 -*-
import io, re

src = r'D:\Desktop\dickseep\out\vla_layout.txt'
t = io.open(src, encoding='utf-8').read()
parts = t.split('\f')
out = []
for i, p in enumerate(parts, 1):
    lines = []
    for ln in p.splitlines():
        ln = re.sub(r'[ \t]{2,}', ' | ', ln.rstrip())
        if ln.strip(' |'):
            lines.append(ln)
    out.append(f'\n===== PDF PAGE {i} =====\n' + '\n'.join(lines))
io.open(r'D:\Desktop\dickseep\out\vla_layout_pages.txt', 'w', encoding='utf-8').write(''.join(out))
print('done', len(parts))
