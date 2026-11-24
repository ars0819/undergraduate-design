# -*- coding: utf-8 -*-
"""Minimal PDF text extractor: parse page content streams, use ToUnicode CMaps."""
import re
import sys
import zlib


def get_objects(data):
    objs = {}
    for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", data):
        num = int(m.group(1))
        start = m.end()
        end = data.find(b"endobj", start)
        if end == -1:
            continue
        objs[num] = data[start:end]
    return objs


def stream_of(body):
    m = re.search(rb"stream\r?\n?", body)
    if not m:
        return None
    start = m.end()
    end = body.find(b"endstream", start)
    raw = body[start:end]
    if b"/FlateDecode" in body[: m.start()]:
        try:
            raw = zlib.decompress(raw.rstrip(b"\r\n"))
        except zlib.error:
            try:
                raw = zlib.decompressobj().decompress(raw)
            except Exception:
                return None
    return raw


def parse_cmap(cs):
    """Return dict code->unicode for a ToUnicode CMap stream."""
    mapping = {}
    for m in re.finditer(rb"beginbfchar(.*?)endbfchar", cs, re.S):
        for pair in re.finditer(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", m.group(1)):
            src = int(pair.group(1), 16)
            dsthex = pair.group(2).decode()
            chars = "".join(
                chr(int(dsthex[i: i + 4], 16)) for i in range(0, len(dsthex), 4)
            )
            mapping[src] = chars
    for m in re.finditer(rb"beginbfrange(.*?)endbfrange", cs, re.S):
        block = m.group(1)
        for r in re.finditer(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block
        ):
            lo = int(r.group(1), 16)
            hi = int(r.group(2), 16)
            dst = int(r.group(3), 16)
            for i in range(hi - lo + 1):
                mapping[lo + i] = chr(dst + i)
    return mapping


def extract(path):
    data = open(path, "rb").read()
    objs = get_objects(data)

    # build font resource name -> cmap, per object
    cmaps = {}
    for num, body in objs.items():
        if b"/ToUnicode" in body:
            m = re.search(rb"/ToUnicode\s+(\d+)\s+\d+\s+R", body)
            if m:
                tgt = objs.get(int(m.group(1)))
                if tgt:
                    cs = stream_of(tgt)
                    if cs:
                        cmaps[num] = parse_cmap(cs)

    # map font object -> its /BaseFont name
    font_names = {}
    for num, body in objs.items():
        m = re.search(rb"/BaseFont\s*/([#\w+.\-]+)", body)
        if m:
            font_names[num] = m.group(1).decode("latin-1")

    # resource name /F1 -> font object, needs page resources; do globally
    res_map = {}
    for num, body in objs.items():
        for m in re.finditer(rb"/(F\d+)\s+(\d+)\s+\d+\s+R", body):
            res_map[m.group(1).decode()] = int(m.group(2))

    out = []
    for num, body in objs.items():
        head = body[: body.find(b"stream")] if b"stream" in body else body
        if b"/Contents" in head:
            continue
        cs = stream_of(body)
        if not cs or (b"Tj" not in cs and b"TJ" not in cs):
            continue
        out.append(extract_text(cs, res_map, cmaps))
    return "\n----PAGE----\n".join(out)


def extract_text(cs, res_map, cmaps):
    lines = []
    cur_font = None
    buf = []

    def flush():
        nonlocal buf
        if buf:
            lines.append("".join(buf))
            buf = []

    tokens = re.finditer(
        rb"/(F\d+)\s+[\d.]+\s+Tf|\[(.*?)\]\s*TJ|<([0-9A-Fa-f]*)>\s*Tj|(T\*|Td|TD|ET|BT)",
        cs,
        re.S,
    )
    for t in tokens:
        if t.group(1):
            cur_font = res_map.get(t.group(1).decode())
        elif t.group(2) is not None:
            for piece in re.finditer(rb"<([0-9A-Fa-f]*)>|(-?[\d.]+)", t.group(2)):
                if piece.group(1) is not None:
                    buf.append(decode_hex(piece.group(1), cmaps.get(cur_font, {})))
                else:
                    try:
                        if float(piece.group(2)) < -180:
                            buf.append(" ")
                    except ValueError:
                        pass
        elif t.group(3) is not None:
            buf.append(decode_hex(t.group(3), cmaps.get(cur_font, {})))
        else:
            flush()
    flush()
    return "\n".join(x for x in lines if x.strip())


def decode_hex(h, cmap):
    s = h.decode()
    if len(s) % 4 == 0 and len(s) >= 4:
        width = 2 if max(cmap.keys(), default=0) > 255 else 1
        step = 4 if width == 2 else 2
        return "".join(
            cmap.get(int(s[i: i + step], 16), "") for i in range(0, len(s), step)
        )
    return ""


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=" * 90)
        print("FILE:", p)
        try:
            print(extract(p))
        except Exception as e:
            print("ERROR:", type(e).__name__, e)
