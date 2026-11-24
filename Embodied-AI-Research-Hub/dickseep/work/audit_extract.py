# -*- coding: utf-8 -*-
"""Extract structural evidence from arXiv HTML (LaTeXML) papers.

Modes:
  tables  -> dump every <figure class="ltx_table"> with caption + rows
  index   -> compact index of tables (number, caption, shape, header)
  grep    -> regex hits with section context

Usage:
  python audit_extract.py tables URL OUT.txt
  python audit_extract.py index  URL
  python audit_extract.py grep   URL "pattern" [max_hits]
"""
import re
import sys
import urllib.request
from html.parser import HTMLParser

UA = {"User-Agent": "Mozilla/5.0 (research-audit)"}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode("utf-8", errors="replace")


class TableParser(HTMLParser):
    """Collect figure.ltx_table -> caption + rows of cell texts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []          # list of dicts
        self.cur = None
        self.in_caption = False
        self.in_cell = False
        self.buf = []
        self.row = None
        self.in_table_tag = False
        self.fig_depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "figure" and "ltx_table" in cls:
            self.cur = {"caption": "", "rows": [], "id": a.get("id", "")}
            self.fig_depth = 1
            return
        if self.cur is None:
            return
        if tag == "figcaption":
            self.in_caption = True
            self.buf = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.buf = []
        elif tag == "table":
            self.in_table_tag = True
        elif tag == "figure":
            self.fig_depth += 1

    def handle_endtag(self, tag):
        if self.cur is None:
            return
        if tag == "figcaption" and self.in_caption:
            self.in_caption = False
            self.cur["caption"] = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        elif tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            if self.row is not None:
                self.row.append(re.sub(r"\s+", " ", "".join(self.buf)).strip())
        elif tag == "tr" and self.row is not None:
            if any(c for c in self.row):
                self.cur["rows"].append(self.row)
            self.row = None
        elif tag == "figure":
            self.fig_depth -= 1
            if self.fig_depth <= 0:
                self.tables.append(self.cur)
                self.cur = None

    def handle_data(self, data):
        if self.cur is None:
            return
        if self.in_caption or self.in_cell:
            self.buf.append(data)


def parse_tables(html: str):
    p = TableParser()
    p.feed(html)
    return p.tables


def table_number(caption: str, idx: int) -> str:
    m = re.match(r"Table\s+(\d+)", caption)
    return m.group(1) if m else f"?{idx}"


def main():
    import os
    _out = os.environ.get("AUDIT_OUT")
    if _out:
        sys.stdout = open(_out, "w", encoding="utf-8", newline="\n")
    mode, url = sys.argv[1], sys.argv[2]
    html = fetch(url)

    if mode == "grep":
        pattern = sys.argv[3]
        maxhits = int(sys.argv[4]) if len(sys.argv) > 4 else 12
        # strip tags for readability but keep section markers
        txt = re.sub(r"(?is)<(script|style|math).*?</\1>", " ", html)
        txt = re.sub(r"(?is)<h([123])[^>]*>", r"\n@@H\1 ", txt)
        txt = re.sub(r"<[^>]+>", " ", txt)
        import html as htmlmod
        txt = htmlmod.unescape(txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        section = ""
        hits = 0
        for i, l in enumerate(lines):
            if l.startswith("@@H"):
                section = l[4:][:90]
                continue
            if re.search(pattern, l, re.I):
                print(f"[{section}] {l[:400]}")
                hits += 1
                if hits >= maxhits:
                    break
        print(f"--- {hits} hits for /{pattern}/ ---")
        return

    tables = parse_tables(html)
    if mode == "index":
        print(f"TOTAL TABLES: {len(tables)}")
        for i, t in enumerate(tables):
            num = table_number(t["caption"], i)
            ncol = max((len(r) for r in t["rows"]), default=0)
            head = " | ".join(t["rows"][0][:6])[:150] if t["rows"] else ""
            print(f"T{num:>3} rows={len(t['rows']):>3} cols={ncol:>2} :: {t['caption'][:150]}")
            if head:
                print(f"        hdr: {head}")
        return

    if mode == "tables":
        out = sys.argv[3]
        with open(out, "w", encoding="utf-8") as f:
            for i, t in enumerate(tables):
                num = table_number(t["caption"], i)
                f.write(f"\n=== Table {num} (id={t['id']}) ===\n")
                f.write(f"CAPTION: {t['caption']}\n")
                for r in t["rows"]:
                    f.write("  " + " | ".join(r) + "\n")
        print(f"wrote {len(tables)} tables -> {out}")
        return


if __name__ == "__main__":
    main()
