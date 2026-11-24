import sys
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph


def dump(path, out_path):
    d = docx.Document(path)
    body = d.element.body
    out = []
    ti = 0
    for child in body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            t = Paragraph(child, d).text.strip()
            if t:
                out.append(t)
        elif tag == 'tbl':
            tb = Table(child, d)
            out.append('[[TABLE %d]]' % ti)
            seen = set()
            for r in tb.rows:
                cells = []
                for c in r.cells:
                    txt = c.text.strip().replace('\n', ' / ')
                    cells.append(txt)
                # collapse horizontally merged duplicates
                merged = []
                for c in cells:
                    if merged and c == merged[-1]:
                        continue
                    merged.append(c)
                line = ' | '.join(merged)
                if line.strip(' |') and line not in seen:
                    seen.add(line)
                    out.append(line)
            out.append('[[/TABLE %d]]' % ti)
            ti += 1
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'{out_path}: paras={len(d.paragraphs)} tables={len(d.tables)} lines={len(out)}')


if __name__ == '__main__':
    dump(sys.argv[1], sys.argv[2])
