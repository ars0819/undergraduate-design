from collect import *
import sys, unicodedata
sys.stdout.reconfigure(encoding='utf8')
entries=json.loads((OUT/'bib_parsed.json').read_text(encoding='utf8'))
web={x['name'][4:]:x for x in json.loads((OUT/'web_manifest.json').read_text(encoding='utf8')) if x['name'].startswith('abs_')}
aliases=['ACT','Diffusion Policy','RTC','FutureRTC','KWS','Embodied Efficiency','SmolVLA','VLAQuantBench','OpenVLA','OpenVLA-OFT','AAC','π₀','π₀.₅','RT-1','RT-2','FAST','LIBERO','PD-VLA','VLA-Cache','A Survey on Efficient','BitVLA','QVLA','Finite Rigid','TacFR','AllTact','GelSight','Baby Fin Ray','TIAGo','SATYRR','Combining Planning','Cowling','HWBench','LagBench']
body='\n'.join(p.read_text(encoding='utf8').split('## 12. 参考文献')[0] for p in SRC.glob('*.md') if not p.name.startswith('文献'))
out=['# 33 条 BibTeX 逐条核验附表','', '核验日期：2026-11-04；原文件不变。下列元数据来自已保存的 arXiv 官方页面；没有将 venue comment 当作出版社确认。语法由本目录 collect.py 的受限 BibTeX 解析器完整消费当前 33 个 @misc 条目（支持嵌套花括号与裸 month 宏）；不是仅数括号，也不是通用 BibTeX 引擎认证。33 key、33 eprint、标准化标题均唯一。', '', '| key / 原行 | 官方题名 | 官方作者（完整） | 首发 / 当前版本日期 | 原条目核查 | 正文使用线索 |', '|---|---|---|---|---|---|']
def norm(s):return re.sub(r'[^a-z0-9]','',s.lower())
for e,a in zip(entries,aliases):
 w=web[e['eprint']];m=w['metadata'];t=m['citation_title'][0];authors='; '.join(m.get('citation_author',[]));s=(OUT/('abs_'+e['eprint']+'.txt')).read_text(encoding='utf8')
 versions=re.findall(r'arXiv:'+re.escape(e['eprint'])+r'(v\d+)',s);v=versions[0] if versions else '见快照'
 issue='题名、ID、年份及作者一致'
 if 'others' in e['author']:issue='题名、ID、年份一致；作者用 others 合法截短，建议完整列出'
 if e['key']=='vlalagbench2026':issue='作者错误：Ruan, Chaoyu → Ruan, Chaoyi；Liu, Shanghai → Liu, Shanghao'
 if e['key']=='embedeff2026':issue='题名/ID一致；HTML meta 将 Yingyan 与 Lin 拆开，页面可见作者为 Yingyan (Celine) Lin；勿机械替换成两人'
 if e['key']=='plandiff2024':issue='题名/ID一致；网页重音编码损坏，保留原 BibTeX Lozano-Pérez, Tomás 的转义'
 note='正文出现 '+a if a.lower() in body.lower() else '未检出正文明确题名/简称；仅继承条目或检索日志'
 out.append(f"| `{e['key']}` / L{e['line']} | [{t}](https://arxiv.org/abs/{e['eprint']}) | {authors} | {m['citation_date'][0]} / {v} {m['citation_online_date'][0]} | {issue} | {note} |")
out+=['','## 引用完整性解释','','正文采用作者—年份与简称，未使用可解析的 BibTeX cite key；因此不能声称已通过自动引用链接检查。上述“出现”只说明能定位文本，不证明所有相关论断都有准确引文，π₀/π₀.₅、RTC/FutureRTC、OpenVLA/OFT 有词串包含关系，需按上下文区分。未引用的背景条目不等于假文献，应移入扩展阅读或解释保留理由。SO-101 Bench、Yu/Qiu、vla-eval、AnyRank 在原库缺失，建议以本轮元数据新增；本轮没有改原库。','','原文件页首仍写 2026-11-03 和 [1]—[30] 对应旧 literature_matrix，实际已有 33 条，应改版本说明。33 条均为 @misc/arXiv 记录，未发现凭空编造的 journal/booktitle 字段；note 中会议归属尚须主办方记录二次验证，不能因为 arXiv 有 comment 就改成已核实发表。论文 2606.08881 的正文仍带会议模板占位符，不能将它当正式会议。','','当前版本日期不等于首发日期。VLA-HWBench 首发 2026-10-26，v2 为 10-27；原条目 note 所写 v2 日期并非错误，但主文不能将它称首发。原文作者错误是明确纠正项，其他字符规范不夸大为身份错误。']
(OUT/'bib_audit_33.md').write_text('\n'.join(out),encoding='utf8')
print('Written bibliography table; entries',len(entries));print('unmentioned:',[e['key'] for e,a in zip(entries,aliases) if a.lower() not in body.lower()])
