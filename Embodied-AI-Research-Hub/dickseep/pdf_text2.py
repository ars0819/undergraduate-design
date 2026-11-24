# -*- coding: utf-8 -*-
"""PDF text extractor with per-page font resource resolution."""
import re
import sys
import zlib


class PDF:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        self.objs = {}
        for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", self.data):
            num = int(m.group(1))
            start = m.end()
            end = self.data.find(b"endobj", start)
            if end != -1:
                self.objs[num] = self.data[start:end]

    def body(self, num):
        return self.objs.get(num, b"")

    def resolve_ref(self, blob, key):
        m = re.search(rb"/" + key + rb"\s+(\d+)\s+\d+\s+R", blob)
        return int(m.group(1)) if m else None

    def stream(self, num):
        body = self.body(num)
        m = re.search(rb"stream\r?\n?", body)
        if not m:
            return None
        start = m.end()
        end = body.find(b"endstream", start)
        raw = body[start:end]
        head = body[: m.start()]
        if b"/FlateDecode" in head:
            for trim in (0, 2):
                chunk = raw[:-trim] if trim else raw
                try:
                    return zlib.decompress(chunk)
                except zlib.error:
                    pass
            try:
                d = zlib.decompressobj()
                return d.decompress(raw)
            except Exception:
                return None
        return raw

    def dict_of(self, num):
        """Return the merged dictionary text of an object (following /Resources refs later)."""
        return self.body(num)

    def font_info(self, num):
        """Return (cmap, is_two_byte, widths) for a font object."""
        b = self.body(num)
        is_two = b"/Type0" in b
        cmap = {}
        tu = self.resolve_ref(b, b"ToUnicode")
        if tu:
            cs = self.stream(tu)
            if cs:
                cmap = parse_cmap(cs)
        if not cmap:
            # simple font with /Differences
            enc = resolve_encoding(b, self)
            if enc:
                cmap = enc
        return cmap, is_two

    def pages(self):
        """Yield (resources_dict_bytes, content_bytes)."""
        result = []
        for num, body in self.objs.items():
            if re.search(rb"/Type\s*/Page\b", body):
                res = self.resolve_ref(body, b"Resources")
                resb = self.body(res) if res else body
                # contents may be array
                contents = b""
                m = re.search(rb"/Contents\s+(\d+)\s+\d+\s+R", body)
                if m:
                    cs = self.stream(int(m.group(1)))
                    if cs:
                        contents += cs
                else:
                    m = re.search(rb"/Contents\s*\[(.*?)\]", body, re.S)
                    if m:
                        for r in re.finditer(rb"(\d+)\s+\d+\s+R", m.group(1)):
                            cs = self.stream(int(r.group(1)))
                            if cs:
                                contents += cs + b"\n"
                result.append((num, resb, contents))
        return result


def resolve_encoding(fontbody, pdf):
    """Build code->unicode from /Encoding /Differences using glyph names."""
    m = re.search(rb"/Differences\s*\[(.*?)\]", fontbody, re.S)
    if not m:
        return {}
    mapping = {}
    code = 0
    for tok in re.finditer(rb"(\d+)|/([\w.\-]+)", m.group(1)):
        if tok.group(1):
            code = int(tok.group(1))
        else:
            name = tok.group(2).decode("latin-1")
            ch = glyph_to_char(name)
            if ch is not None:
                mapping[code] = ch
            code += 1
    return mapping


GLYPHS = {
    "space": " ", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "period": ".", "comma": ",", "hyphen": "-", "colon": ":", "percent": "%",
    "yen": "¥", "dollar": "$", "slash": "/", "parenleft": "(", "parenright": ")",
    "asterisk": "*", "plus": "+", "equal": "=",
}


def glyph_to_char(name):
    if name in GLYPHS:
        return GLYPHS[name]
    m = re.fullmatch(r"uni([0-9A-Fa-f]{4})", name)
    if m:
        return chr(int(m.group(1), 16))
    return None


def parse_cmap(cs):
    mapping = {}
    for m in re.finditer(rb"beginbfchar(.*?)endbfchar", cs, re.S):
        for pair in re.finditer(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", m.group(1)):
            src = int(pair.group(1), 16)
            dsthex = pair.group(2).decode()
            mapping[src] = "".join(
                chr(int(dsthex[i: i + 4], 16)) for i in range(0, len(dsthex) - 3, 4)
            )
    for m in re.finditer(rb"beginbfrange(.*?)endbfrange", cs, re.S):
        block = m.group(1)
        for r in re.finditer(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block
        ):
            lo, hi, dst = (int(r.group(i), 16) for i in (1, 2, 3))
            for i in range(hi - lo + 1):
                mapping[lo + i] = chr(dst + i)
    return mapping


def extract(path):
    pdf = PDF(path)
    out = []
    for pno, res, contents in pdf.pages():
        if not contents:
            continue
        # name -> font object from this page's resources
        names = {}
        m = re.search(rb"/Font\s*<<(.*?)>>", res, re.S)
        if not m:
            fo = pdf.resolve_ref(res, b"Font")
            if fo:
                m = re.search(rb"(\d+)\s+\d+\s+R", pdf.body(fo))
        if m:
            for fm in re.finditer(rb"/([\w.\-]+)\s+(\d+)\s+\d+\s+R", m.group(1)):
                names[fm.group(1).decode()] = int(fm.group(2))
        fonts = {k: pdf.font_info(v) for k, v in names.items()}
        out.append(render(contents, fonts))
    return "\n----PAGE----\n".join(out)


def render(cs, fonts):
    lines, buf = [], []
    cur = ({}, False)

    def flush():
        if buf:
            s = "".join(buf)
            if s.strip():
                lines.append(s)
            buf.clear()

    tok = re.finditer(
        rb"/([\w.\-]+)\s+[\d.]+\s+Tf"
        rb"|\[((?:[^\[\]\\]|\\.)*)\]\s*TJ"
        rb"|<([0-9A-Fa-f]*)>\s*Tj"
        rb"|\(((?:[^()\\]|\\.)*)\)\s*Tj"
        rb"|(T\*|Td|TD|ET|BT|Tm)",
        cs,
        re.S,
    )
    for t in tok:
        if t.group(1):
            cur = fonts.get(t.group(1).decode(), ({}, False))
            continue
        if t.group(2) is not None:
            for piece in re.finditer(rb"<([0-9A-Fa-f]*)>|\(((?:[^()\\]|\\.)*)\)|(-?[\d.]+)", t.group(2)):
                if piece.group(1) is not None:
                    buf.append(dec(piece.group(1), cur))
                elif piece.group(2) is not None:
                    buf.append(dec_lit(piece.group(2), cur))
                else:
                    try:
                        if abs(float(piece.group(3))) >= 60:
                            buf.append(" ")
                    except ValueError:
                        pass
            continue
        if t.group(3) is not None:
            buf.append(dec(t.group(3), cur))
            continue
        if t.group(4) is not None:
            buf.append(dec_lit(t.group(4), cur))
            continue
        flush()
    flush()
    return "\n".join(lines)


def dec(h, cur):
    cmap, two = cur
    s = h.decode()
    if not s:
        return ""
    step = 4 if two else 2
    if two and len(s) % 4:
        step = 2
    return "".join(
        cmap.get(int(s[i: i + step], 16), "") for i in range(0, len(s) - step + 1, step)
    )


def dec_lit(b, cur):
    cmap, two = cur
    try:
        s = b.decode("latin-1")
    except Exception:
        return ""
    return "".join(cmap.get(ord(c), c) for c in s)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=" * 90)
        print("FILE:", p.split("\\")[-1])
        try:
            print(extract(p))
        except Exception as e:
            import traceback

            traceback.print_exc()
