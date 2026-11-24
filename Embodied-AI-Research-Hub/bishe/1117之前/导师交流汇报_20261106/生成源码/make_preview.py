"""Create a read-only preview PDF from reviewed slide renders; PPTX stays editable."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
import pypdfium2 as pdfium
import json,hashlib,zipfile,xml.etree.ElementTree as ET
S=Path(__file__).parent; O=S.parent; W=O.parent; B=W/'.pptx_build_20261106'
images=sorted((B/'previews').glob('slide-*.png'))
assert len(images)==17
pdf=O/'本科毕设方向与文献调研汇报.pdf'
c=canvas.Canvas(str(pdf),pagesize=(1200,675))
c.setTitle('本科毕设方向与文献调研汇报')
c.setAuthor('刘鑫宇')
c.setSubject('导师交流预览，2026-11-06；由同版幻灯片渲染页构成')
for p in images:
 c.drawImage(ImageReader(str(p)),0,0,1200,675)
 c.showPage()
c.save()
assert len(PdfReader(pdf).pages)==17
rendered=B/'pdf-check';rendered.mkdir(exist_ok=True)
pdfdoc=pdfium.PdfDocument(str(pdf))
for i in range(len(pdfdoc)):
 page=pdfdoc[i];bitmap=page.render(scale=4/3);bitmap.to_pil().save(rendered/f'page-{i+1:02d}.png')
 page.close()
pdfdoc.close()
deck=O/'本科毕设方向与文献调研汇报.pptx'
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(deck) as z:
 pages=[n for n in z.namelist() if __import__('re').fullmatch(r'ppt/slides/slide\d+\.xml',n)]
 assert len(pages)==17
 native=[]; text=[]; diagrams=[]
 for name in sorted(pages,key=lambda n:int(__import__('re').search(r'slide(\d+)',n).group(1))):
  root=ET.fromstring(z.read(name));native.append(len(root.findall('.//a:tbl',ns)));diagrams.append(len(root.findall('.//p:cxnSp',ns)));text.append(''.join(x.text or '' for x in root.findall('.//a:t',ns)))
 assert sum(native)==6
 assert '刘鑫宇' in text[0] and '2026 年 11 月 6 日' in text[0]
 assert '60 篇' in text[2] and '不合计' in text[2]
 assert not any('感谢各位评委' in x or '恳请批准开题' in x for x in text)
 assert len([n for n in z.namelist() if __import__('re').fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==17
manifest=json.loads((S/'source_inventory.json').read_text(encoding='utf-8'))
for row in manifest['sources']:
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],'Source was modified'
for folder in manifest['notes']:
 for row in folder['files']:
  assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],'Note was modified'
receipts=sorted((B/'receipts').glob('*.json'),key=lambda p:p.stat().st_mtime)
validation=json.loads(receipts[-1].read_text(encoding='utf-8'))
check={'slides':17,'main_slides':12,'appendix_slides':5,'native_tables':sum(native),'connectors_per_slide':diagrams,
 'pdf_pages':17,'pdf_type':'raster preview from final rendered slides','personal_reading_count':'not claimed',
 'source_files_unchanged':True,'pptx_sha256':hashlib.sha256(deck.read_bytes()).hexdigest(),
 'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
 'layout_warnings':validation.get('presentationLayout',{}).get('warnings',[]),
 'powerpoint_native_open_test':False,'visual_inspection':'17 slides individually inspected; changed slides reinspected'}
(B/'final_delivery_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(check,ensure_ascii=False,indent=2))
