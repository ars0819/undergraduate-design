#!/usr/bin/env python3
"""从 arXiv LaTeXML HTML 中抽取 <figure class="ltx_table"> 的表号/标题/表格，
输出 Markdown（同时抽取图片 figure 的图号/标题/图片 URL）。

用法: python3 extract_tables.py 2506.01844.html > tables-and-figures.md
仅标准库。
"""
import re
import sys
import html as htmllib

MATH_RE = re.compile(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', re.S)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean(fragment: str) -> str:
    def math_sub(m):
        alt = htmllib.unescape(m.group(1))
        return " $" + alt.strip() + "$ "

    fragment = MATH_RE.sub(math_sub, fragment)
    fragment = TAG_RE.sub(" ", fragment)
    fragment = htmllib.unescape(fragment)
    return WS_RE.sub(" ", fragment).strip()


def find_figures(src: str, cls: str):
    """返回 [(id, inner_html)]，按出现顺序。"""
    out = []
    pos = 0
    pat = re.compile(r'<figure[^>]*class="[^"]*' + re.escape(cls) + r'[^"]*"[^>]*id="([^"]*)"[^>]*>', re.S)
    # 有的 figure 先 id 后 class，退化处理：匹配所有 <figure ...> 再筛选 class
    for m in re.finditer(r"<figure([^>]*)>", src):
        attrs = m.group(1)
        if cls not in attrs:
            continue
        start = m.end()
        depth = 1
        i = start
        while depth > 0:
            nxt = re.compile(r"</?figure[^>]*>").search(src, i)
            if not nxt:
                i = len(src)
                break
            if nxt.group(0).startswith("</"):
                depth -= 1
            else:
                depth += 1
            i = nxt.end()
        inner = src[start:i]
        idm = re.search(r'id="([^"]*)"', attrs)
        out.append((idm.group(1) if idm else "", inner))
    return out


def table_md(inner: str) -> str:
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", inner, re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)
        if not cells:
            continue
        rows.append([clean(c) for c in cells])
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = []
    for i, r in enumerate(rows):
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(lines)


def caption(inner: str) -> str:
    m = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", inner, re.S)
    return clean(m.group(1)) if m else ""


def main():
    src = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
    print("# 表（按 figure 结构抽取，表号以 caption 为准）\n")
    for fid, inner in find_figures(src, "ltx_table"):
        cap = caption(inner)
        print(f"\n## [{fid}] {cap}\n")
        md = table_md(inner)
        print(md if md else "_（未解析到表格网格，可能为图片或复杂布局）_")
    print("\n\n# 图（图片 URL 相对本文件目录）\n")
    for fid, inner in find_figures(src, "ltx_figure"):
        cap = caption(inner)
        imgs = re.findall(r'<img[^>]*src="([^"]*)"', inner)
        print(f"\n## [{fid}] {cap}\n")
        for s in imgs:
            print(f"- `{s}`")


if __name__ == "__main__":
    main()
