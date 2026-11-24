from collect import *
m=json.loads((OUT/'web_manifest.json').read_text(encoding='utf8'))
for x in m:
 if x['name'].startswith('abs_'):
  print(x['name'],json.dumps({k:v for k,v in x['metadata'].items() if k not in ('citation_abstract','citation_pdf_url')},ensure_ascii=False))
print('LINKS')
print('\n'.join(sorted(set(re.findall(r'href="([^"]*(?:phone|isaac|async|teleoperat)[^"]*)"',(OUT/'doc_so101.html').read_text(encoding='utf8'))))))
