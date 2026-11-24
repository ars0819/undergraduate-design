#!/usr/bin/env python3
"""把 arXiv HTML (LaTeXML) 转成可读 Markdown，保留标题/段落/公式 alttext/表格。

用法: python3 html2md.py 2506.01844.html > 2506.01844.md
仅依赖标准库。
"""
import sys
import re
from html.parser import HTMLParser

BLOCK = {
    "p", "div", "section", "figure", "figcaption", "table", "tr", "li",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "math",
}
INLINE_SKIP = {"script", "style", "annotation", "annotation-xml"}


class MD(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.buf = []
        self.skip = 0
        self.in_math = 0
        self.table = None
        self.row = None
        self.cell = None
        self.cell_has_block = False
        self.figure_depth = 0
        self.figcap = None
        self.figcap_buf = []
        self.in_bib = False
        self.href = None
        self.list_depth = 0

    # ---------- helpers ----------
    def flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if not text:
            return
        if self.table is not None:
            if self.cell is not None:
                self.cell.append(text)
            return
        self.out.append(text + "\n\n")

    def close_block(self):
        self.flush()

    # ---------- parser callbacks ----------
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in INLINE_SKIP:
            self.skip += 1
            return
        if tag == "math":
            alt = a.get("alttext")
            if alt:
                self.buf.append(f" ${alt.strip()}$ ")
            self.in_math += 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.flush()
            level = int(tag[1])
            self.buf.append("\n" + "#" * level + " ")
            return
        if tag == "table":
            self.flush()
            self.table = []
            return
        if tag == "tr" and self.table is not None:
            self.row = []
            return
        if tag in ("td", "th") and self.row is not None:
            self.cell = []
            return
        if tag == "figure":
            self.flush()
            self.figure_depth += 1
            return
        if tag == "figcaption":
            self.flush()
            return
        if tag == "li":
            self.flush()
            self.buf.append("- ")
            return
        if tag == "br":
            self.buf.append("  \n")
            return
        if tag in BLOCK:
            # block start: flush pending inline text
            self.flush()
        if tag == "a" and a.get("class", "").startswith("ltx_ref"):
            pass

    def handle_endtag(self, tag):
        if tag in INLINE_SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "math":
            self.in_math = max(0, self.in_math - 1)
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.flush()
            return
        if tag == "table":
            self.flush()
            self._emit_table()
            self.table = None
            return
        if tag == "tr" and self.row is not None:
            self.flush()
            if self.row:
                self.table.append(self.row)
            self.row = None
            return
        if tag in ("td", "th") and self.cell is not None:
            self.flush()
            txt = re.sub(r"\s+", " ", "".join(self.cell)).strip()
            self.row.append(txt)
            self.cell = None
            return
        if tag == "figure":
            self.figure_depth = max(0, self.figure_depth - 1)
            return
        if tag in BLOCK:
            self.flush()

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_math:
            return
        self.buf.append(data)

    # ---------- table emit ----------
    def _emit_table(self):
        rows = [r for r in self.table if any(c.strip() for c in r)]
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        md = []
        for i, r in enumerate(rows):
            md.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
            if i == 0:
                md.append("| " + " | ".join("---" for _ in range(width)) + " |")
        self.out.append("\n".join(md) + "\n\n")


def main():
    src = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    # 丢掉参考文献之前不需要的部分? 保留全部; 但移除 <nav>/<footer>
    src = re.sub(r"<(nav|footer|header)[^>]*>.*?</\1>", "", src, flags=re.S | re.I)
    p = MD()
    p.feed(src)
    p.flush()
    text = "".join(p.out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
