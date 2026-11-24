"""Minimal PDF text extractor for WPS-generated Chinese PDFs (layout-aware).

Handles uncompressed object tables, FlateDecode content streams, and
Type0/Identity-H CID fonts with ToUnicode CMaps. Reconstructs lines by
tracking the text matrix (Tm/Td/TD/T*), grouping glyph runs by baseline
and sorting within a line by x position.
"""
import re
import sys
import zlib

TOKEN_RE = re.compile(
    rb"\((?:[^()\\]|\\.)*\)"
    rb"|<[0-9A-Fa-f\s]*>"
    rb"|/[^\s/\[\]<>(){}%]*"
    rb"|[-+]?[0-9]*\.?[0-9]+"
    rb"|[\[\]]"
    rb"|[A-Za-z'\"*]+"
)


def load_objects(data):
    objs = {}
    for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj", data):
        num = int(m.group(1))
        start = m.end()
        end = data.find(b"endobj", start)
        if end < 0:
            continue
        objs[num] = data[start:end]
    return objs


def obj_stream(body):
    pos = body.find(b"stream")
    if pos < 0:
        return None
    head = body[:pos]
    start = pos + 6
    if body[start:start + 2] == b"\r\n":
        start += 2
    elif body[start:start + 1] in (b"\n", b"\r"):
        start += 1
    end = body.rfind(b"endstream")
    if end < 0:
        end = len(body)
    raw = body[start:end]
    if b"/FlateDecode" in head:
        try:
            return zlib.decompress(raw)
        except Exception:
            try:
                return zlib.decompressobj().decompress(raw)
            except Exception:
                return None
    return raw


def balanced_dict(body, pos):
    depth = 0
    i = pos
    while i < len(body) - 1:
        pair = body[i:i + 2]
        if pair == b"<<":
            depth += 1
            i += 2
            continue
        if pair == b">>":
            depth -= 1
            i += 2
            if depth == 0:
                return body[pos:i], i
            continue
        i += 1
    return body[pos:], len(body)


def resolve_dict(objs, body, key):
    m = re.search(rb"/" + key + rb"\s*(\d+)\s+0\s+R", body)
    if m:
        return objs.get(int(m.group(1)), b"")
    m = re.search(rb"/" + key + rb"\s*<<", body)
    if m:
        d, _ = balanced_dict(body, m.end() - 2)
        return d
    return None


def get_ref_num(body, key):
    m = re.search(rb"/" + key + rb"\s*(\d+)\s+0\s+R", body)
    return int(m.group(1)) if m else None


def parse_cmap(cmap_bytes):
    mapping = {}

    def u16(hexs):
        hexs = re.sub(rb"\s+", b"", hexs)
        if len(hexs) % 2:
            hexs += b"0"
        try:
            return bytes.fromhex(hexs.decode()).decode("utf-16-be", "ignore")
        except Exception:
            return ""

    for block in re.findall(rb"beginbfchar(.*?)endbfchar", cmap_bytes, re.S):
        for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", block):
            mapping[int(src, 16)] = u16(dst)
    for block in re.findall(rb"beginbfrange(.*?)endbfrange", cmap_bytes, re.S):
        for lo, hi, dst in re.findall(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block
        ):
            lo_i, hi_i = int(lo, 16), int(hi, 16)
            base = bytes.fromhex(dst.decode() if len(dst) % 2 == 0 else dst.decode() + "0")
            if len(base) >= 2:
                b0 = int.from_bytes(base[-2:], "big")
                for off in range(hi_i - lo_i + 1):
                    mapping[lo_i + off] = chr(b0 + off)
        for lo, hi, arr in re.findall(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]", block, re.S
        ):
            lo_i = int(lo, 16)
            for off, it in enumerate(re.findall(rb"<([0-9A-Fa-f]*)>", arr)):
                mapping[lo_i + off] = u16(it)
    return mapping


def build_fonts(objs):
    fonts = {}
    for num, body in objs.items():
        if not re.search(rb"/Type\s*/Font", body):
            continue
        two_byte = bool(re.search(rb"/Identity-H", body)) or bool(
            re.search(rb"/Type\s*/Type0", body)
        )
        cmap = {}
        tu = get_ref_num(body, b"ToUnicode")
        if tu is not None and tu in objs:
            stream = obj_stream(objs[tu])
            if stream:
                cmap = parse_cmap(stream)
        fonts[num] = (cmap, two_byte)
    return fonts


def unescape_pdf_string(s):
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i:i + 1]
        if c == b"\\" and i + 1 < len(s):
            n = s[i + 1:i + 2]
            mapping = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
                       b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}
            if n in mapping:
                out += mapping[n]
                i += 2
                continue
            if n.isdigit():
                m = re.match(rb"[0-7]{1,3}", s[i + 1:i + 4])
                if m:
                    out.append(int(m.group(0), 8) & 0xFF)
                    i += 1 + len(m.group(0))
                    continue
            out += n
            i += 2
            continue
        out += c
        i += 1
    return bytes(out)


def decode_string(raw, cmap, two_byte):
    if two_byte:
        chars = []
        for i in range(0, len(raw) - 1, 2):
            chars.append(cmap.get(int.from_bytes(raw[i:i + 2], "big"), ""))
        return "".join(chars)
    return "".join(cmap.get(b, chr(b) if 32 <= b < 127 else "") for b in raw)


def mat_mul(m, n):
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (
        a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2, e1 * b2 + f1 * d2 + f2,
    )


def extract_items(content, page_fonts, fonts):
    items = []
    stack = []
    array_mode = None
    tm = (1, 0, 0, 1, 0, 0)
    tlm = (1, 0, 0, 1, 0, 0)
    leading = 0.0
    font_size = 1.0
    current_font = None

    def show(item):
        if not (isinstance(item, bytes) and (item.startswith(b"(") or item.startswith(b"<"))):
            return
        if item.startswith(b"("):
            raw = unescape_pdf_string(item[1:-1])
        else:
            hexs = re.sub(rb"\s+", b"", item[1:-1])
            if len(hexs) % 2:
                hexs += b"0"
            try:
                raw = bytes.fromhex(hexs.decode())
            except Exception:
                return
        if current_font is None or current_font not in fonts:
            txt = raw.decode("latin-1", "ignore")
        else:
            cmap, two_byte = fonts[current_font]
            txt = decode_string(raw, cmap, two_byte)
        if txt:
            items.append((round(tm[5], 1), round(tm[4], 1), font_size, txt))

    for m in TOKEN_RE.finditer(content):
        tok = m.group(0)
        if tok == b"[":
            array_mode = []
            continue
        if tok == b"]":
            stack.append(array_mode if array_mode is not None else [])
            array_mode = None
            continue
        if array_mode is not None:
            array_mode.append(tok)
            continue
        if tok.startswith(b"(") or tok.startswith(b"<") or tok.startswith(b"/"):
            stack.append(tok)
            continue
        if tok[:1].isdigit() or tok[:1] in b"+-.":
            try:
                stack.append(float(tok))
            except ValueError:
                stack.append(0.0)
            continue
        op = tok.decode("latin-1")
        nums = [x for x in stack if isinstance(x, float)]
        try:
            if op == "Tf" and len(stack) >= 2:
                name = stack[-2]
                if isinstance(name, bytes) and name.startswith(b"/"):
                    current_font = page_fonts.get(name[1:].decode("latin-1"))
                font_size = stack[-1] if isinstance(stack[-1], float) else 1.0
            elif op == "TL" and nums:
                leading = nums[-1]
            elif op == "Tm" and len(nums) >= 6:
                tm = tuple(nums[-6:])
                tlm = tm
            elif op == "Td" and len(nums) >= 2:
                tx, ty = nums[-2], nums[-1]
                tlm = mat_mul((1, 0, 0, 1, tx, ty), tlm)
                tm = tlm
            elif op == "TD" and len(nums) >= 2:
                tx, ty = nums[-2], nums[-1]
                leading = -ty
                tlm = mat_mul((1, 0, 0, 1, tx, ty), tlm)
                tm = tlm
            elif op == "T*":
                tlm = mat_mul((1, 0, 0, 1, 0, -leading), tlm)
                tm = tlm
            elif op == "BT":
                tm = (1, 0, 0, 1, 0, 0)
                tlm = tm
            elif op == "Tj":
                if stack and isinstance(stack[-1], bytes):
                    show(stack[-1])
            elif op == "'":
                tlm = mat_mul((1, 0, 0, 1, 0, -leading), tlm)
                tm = tlm
                if stack:
                    show(stack[-1])
            elif op == '"':
                tlm = mat_mul((1, 0, 0, 1, 0, -leading), tlm)
                tm = tlm
                if stack:
                    show(stack[-1])
            elif op == "TJ":
                for item in reversed(stack):
                    if isinstance(item, list):
                        for sub in item:
                            show(sub)
                        break
        except Exception:
            pass
        stack = []
    return items


def assemble(items):
    if not items:
        return ""
    items = sorted(items, key=lambda t: (-t[0], t[1]))
    lines = []
    cur_y = items[0][0]
    cur = []
    for y, x, size, txt in items:
        if abs(y - cur_y) > max(3.0, size * 0.5):
            lines.append(cur)
            cur = []
            cur_y = y
        cur.append((x, txt, size))
    if cur:
        lines.append(cur)

    out = []
    for line in lines:
        line.sort(key=lambda t: t[0])
        buf = []
        prev_x = None
        prev_size = 10.0
        for x, txt, size in line:
            if prev_x is not None:
                gap = x - prev_x
                approx_w = prev_size * 1.0 * max(1, len(txt))
                if gap > max(6.0, prev_size * 0.6):
                    spaces = min(8, max(1, int(gap / max(prev_size * 0.5, 1))))
                    buf.append(" " * spaces)
            buf.append(txt)
            prev_x = x + len(txt) * size * 0.55
            prev_size = size
        out.append("".join(buf).rstrip())
    return "\n".join(out)


def page_list(objs):
    catalog = None
    for num, body in objs.items():
        if re.search(rb"/Type\s*/Catalog", body):
            catalog = body
            break
    order = []

    def walk(num, depth=0):
        body = objs.get(num, b"")
        if depth > 20:
            return
        if re.search(rb"/Type\s*/Pages", body):
            kids = re.search(rb"/Kids\s*\[(.*?)\]", body, re.S)
            if kids:
                for ref in re.finditer(rb"(\d+)\s+0\s+R", kids.group(1)):
                    walk(int(ref.group(1)), depth + 1)
        elif re.search(rb"/Type\s*/Page\b", body):
            order.append(num)

    if catalog:
        root = get_ref_num(catalog, b"Pages")
        if root:
            walk(root)
    if not order:
        order = sorted(n for n, b in objs.items() if re.search(rb"/Type\s*/Page\b", b))
    return order


def main(path, out_path):
    data = open(path, "rb").read()
    objs = load_objects(data)
    fonts = build_fonts(objs)
    pages = page_list(objs)
    chunks = []
    for idx, pnum in enumerate(pages, 1):
        body = objs[pnum]
        res = resolve_dict(objs, body, b"Resources")
        page_fonts = {}
        if res:
            fdict = resolve_dict(objs, res, b"Font")
            if fdict:
                for name, ref in re.findall(rb"/([^\s/\[\]<>()]+)\s+(\d+)\s+0\s+R", fdict):
                    page_fonts[name.decode("latin-1")] = int(ref)
        if not page_fonts:
            for b2 in objs.values():
                for name, ref in re.findall(rb"/(F\d+)\s+(\d+)\s+0\s+R", b2):
                    page_fonts.setdefault(name.decode("latin-1"), int(ref))
        contents = []
        m = re.search(rb"/Contents\s*(\d+)\s+0\s+R", body)
        if m:
            contents = [int(m.group(1))]
        else:
            m = re.search(rb"/Contents\s*\[(.*?)\]", body, re.S)
            if m:
                contents = [int(x) for x in re.findall(rb"(\d+)\s+0\s+R", m.group(1))]
        content = b""
        for c in contents:
            s = obj_stream(objs.get(c, b""))
            if s:
                content += s + b"\n"
        text = assemble(extract_items(content, page_fonts, fonts))
        chunks.append(f"\n===== PAGE {idx} =====\n{text}")
        if idx % 25 == 0:
            print(f"  ...{idx}/{len(pages)}", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))
    print(f"WROTE {out_path}  pages={len(pages)} fonts={len(fonts)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
