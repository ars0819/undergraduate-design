import hashlib, json, pathlib
root=pathlib.Path('.')
manifest=json.loads((root/'work/V14/source_manifest.json').read_text(encoding='utf-8-sig'))
for item in manifest['files']:
    p=root/'work/V14/source'/item['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'], str(p)
before=json.loads((root/'work/V14/research_hashes_before.json').read_text(encoding='utf-8-sig'))
for item in before:
    p=pathlib.Path(item['Path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest().upper()==item['Hash'], str(p)
report=root/'research/audit/V14_execution_audit.md'
s=report.read_text(encoding='utf-8-sig')
assert s.count('```')%2==0
assert manifest['commit'] in s
for heading in ['动作输出','索引追踪','同步与异步','最小修复','T1′','未验证']:
    assert heading in s, heading
validation=json.loads((root/'work/V14/logic_validation.json').read_text())
assert len(validation['cases'])==8 and all(x['assertions']=='PASS' for x in validation['cases'])
summary={'source_snapshot_hashes':'PASS (7 files)', 'original_research_files_unchanged':len(before), 'logic_tests':'PASS', 'report_bytes':report.stat().st_size, 'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
(root/'work/V14/final_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
