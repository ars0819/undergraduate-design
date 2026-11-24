# -*- coding: utf-8 -*-
import io, re

src = r'D:\Desktop\dickseep\out\vla_text.txt'
t = io.open(src, encoding='utf-8').read()
t = t.split('=' * 90, 1)[1]
parts = t.split('----PAGE----')


def clean(s):
    s = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), s)
    s = s.replace('\x02', 'fi').replace('\x03', 'fl').replace('\x01', '')
    s = s.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2013', '-').replace('\u2014', '--')
    return s


out = []
for i, p in enumerate(parts, 1):
    out.append(f'\n\n===== PDF PAGE {i} =====\n' + clean(p.strip()))
io.open(r'D:\Desktop\dickseep\out\vla_clean.txt', 'w', encoding='utf-8').write(''.join(out))
print('done', len(out))
