import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from 'file:///C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const S=path.dirname(fileURLToPath(import.meta.url)), O=path.dirname(S), W=path.dirname(O), B=path.join(W,'.精简版_build');
const SK='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.12130/skills/presentations';
const PY='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const N=JSON.parse(await fs.readFile(path.join(S,'slides_notes.json'),'utf8'));
const FONT='Microsoft YaHei', C={navy:'#17375E',ink:'#20334B',gray:'#586879',line:'#D4DEE8',teal:'#16787B',amber:'#986020',pale:'#EEF4F8',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1600,height:900}}), geometry=[];
function tx(s,t,x,y,w,h,size=33,color=C.ink,bold=false){let q=s.shapes.add({geometry:'textbox',name:t.slice(0,32),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=t;q.text.style={typeface:FONT,fontSize:size,bold,color,alignment:'left',verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0};geometry.push({slide:p.slides.items.length,text:t,x,y,w,h,size});return q;}
function line(s,x,y,w){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:2},fill:C.line,line:{fill:C.line,width:0}});}
function sl(title,sub,src='',label='') {let s=p.slides.add();s.background.fill=C.white;let i=p.slides.items.length;tx(s,title,80,58,1450,75,46,C.navy,true);if(sub)tx(s,sub,83,148,1430,60,27,C.gray);line(s,80,822,1440);tx(s,src||'导师交流讨论稿　题目、实验与专业交付要求尚待确认',82,843,1340,34,18,C.gray);tx(s,label||String(i).padStart(2,'0'),1450,843,70,34,20,C.gray);s.speakerNotes.text=N[i-1]+'\n\n'+src;return s;}
function tab(s,vals,widths,{y=220,h=500,font=28,header=28,pad=7}={}) {let t=s.tables.add({rows:vals.length,columns:vals[0].length,left:80,top:y,width:1440,height:h,columnWidths:widths,values:vals});t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<vals.length;r++){t.rows[r].height=h/vals.length;for(let c=0;c<vals[0].length;c++){let z=t.getCell(r,c);z.fill=r===0?C.navy:C.white;z.text.style={typeface:FONT,fontSize:r===0?header:font,color:r===0?C.white:C.ink,bold:r===0};}}t.cells.block({row:0,column:0,rowCount:vals.length,columnCount:vals[0].length}).assign({margins:{left:12,right:12,top:pad,bottom:pad},anchor:'center'});return t;}
function node(s,t,x,y,w,h=85){let z=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:C.pale,line:{fill:C.line,width:1.5}});z.text=t;z.text.style={typeface:FONT,fontSize:30,color:C.navy,alignment:'center',verticalAlignment:'middle',wrap:'square',autoFit:'none',insets:8};return z;}
function ar(s,a,b,f='right',t='left',d=false){s.shapes.connect(a,b,{kind:f==='right'&&t==='left'?'straight':'elbow',fromSide:f,toSide:t,line:{fill:C.teal,width:2.5,style:d?'dashed':'solid'},tail:{type:'triangle',width:'med',length:'med'}});}
{
 let s=p.slides.add();s.background.fill=C.white;
 tx(s,'本科毕设方向初步探索\n与文献调研',100,160,1400,200,67,C.navy,true);
 tx(s,'从 VLA 高效推理到机器人闭环执行',105,415,1380,68,38,C.gray);
 tx(s,'北京邮电大学\n机械工程（机器人试验班）',105,550,1300,105,33);
 tx(s,'刘鑫宇　　2026年11月6日',105,705,1300,50,30,C.gray);
 tx(s,'导师交流 · 讨论稿 · 尚未确定最终选题',105,788,1380,48,29,C.amber);
 s.speakerNotes.text=N[0]+'\n姓名来自已填写任务书09及原清单11，仅引用姓名，不披露学号或联系方式。\n总体主题：VLA高效推理、机器人闭环执行与未来具身系统研究。';
}
{
 let s=sl('近期工作与调研进展','从广泛整理，转向重点精读与代码复现','本地：文献总索引、横向比较与总结、具身智能系统架构与协同调度研究报告');
 tx(s,'已形成的资料',85,235,700,60,36,C.navy,true);
 tx(s,'24 篇结构化文献笔记',85,325,760,65,41,C.teal,true);
 tx(s,'36 篇专题研究条目',85,430,760,65,41,C.teal,true);
 tx(s,'两套集合可能重叠，不简单相加',85,552,760,95,31,C.amber);
 tx(s,'当前学习主线',925,235,590,60,36,C.navy,true);
 tx(s,'VLA 基础与动作生成\n模型高效推理\n动作分块与异步执行\nAgent、世界模型与系统架构',925,325,590,260,32);
 line(s,85,687,1430);
 tx(s,'整理以 AI 辅助为主，论文数不等于本人完整精读数\n尚未完成代码复现与闭环实验',85,717,1430,90,29,C.gray);
}
{
 let s=sl('对具身智能技术体系的初步认识','规划、动作生成与控制执行各有职责','概念综合示意，非某一论文或已实现系统；参考本地报告及SmolVLA、RTC');
 let labels=['任务指令','Agent / 任务规划','VLA / 动作生成','机器人控制执行','环境反馈'];
 let a=labels.map((v,i)=>node(s,v,80+i*298,300,i===4?240:250,112));
 for(let i=0;i<4;i++)ar(s,a[i],a[i+1]);ar(s,a[4],a[2],'bottom','bottom',true);
 let wm=node(s,'可选世界模型\n提供未来预测',685,560,380,100);ar(s,wm,a[2],'top','bottom',true);
 tx(s,'高层规划：较低更新频率\n底层控制：按控制周期持续运行',85,535,530,130,30,C.gray);
 tx(s,'调度机制\n协调推理时机与动作执行',1120,550,395,125,30,C.teal,true);
 tx(s,'系统能力依赖模型，也依赖感知、预测、控制及反馈之间的配合',85,737,1430,63,32,C.navy,true);
}
{
 let s=sl('重点文献带来的认识','六篇代表工作，分别对应动作、效率、执行与评测','原始来源：arxiv.org/abs/2303.04137、2502.19645、2506.01844、2506.07339、2512.01031、2611.03103');
 tab(s,[['论文','解决问题','我得到的启发'],['Diffusion Policy','生成式动作策略与滚动执行','动作质量与执行窗口需分别分析'],['OpenVLA-OFT','优化解码、动作表示与微调配方','模型效率需要结合任务表现判断'],['SmolVLA','轻量模型与异步执行','适合优先考察，兼容性仍待验证'],['RTC','动作块衔接与推理时修补','平滑衔接已有明确方法'],['VLASH','预测与执行时刻的状态错位','未来状态对齐也已有研究'],['VLALagBench','延迟与闭环成功率的关系','先识别任务与策略的敏感性']],[300,555,585],{h:530,font:29,header:28});
 tx(s,'前四篇：动作与执行基础　　后两篇：状态陈旧与延迟敏感性',85,773,1430,35,25,C.gray);
 s.speakerNotes.text+='\n贡献与方法以本地对应论文PDF方法/实验段落核查。DP非语言条件VLA。出处与完整作者见独立证据文档。';
}
{
 let s=sl('模型推理更快，机器人一定表现更好吗？','这是待验证的问题，尚无本人的实验结论','依据：OpenVLA-OFT、SmolVLA、RTC、VLASH、VLALagBench原文；具体效应依赖任务与实现');
 tab(s,[['需要分开测量','对应问题'],['模型计算效率','每次推理多快？显存与GPU开销多少？'],['动作生成质量','计算预算变化后，动作是否仍有效？'],['端到端时延','观测有多旧？等待、通信与队列占多少？'],['闭环任务表现','成功率、完整任务耗时、扰动响应如何？']],[400,1040],{h:370,font:31});
 tx(s,'视觉上平滑，不等于遇到环境变化时响应及时',85,630,1430,65,36,C.navy,true);
 tx(s,'异步、衔接与阈值机制已有先例，不能直接作为独创贡献\n当前打算：先定位实际瓶颈，再选择模型加速或系统优化',85,710,1430,95,29,C.amber);
}
{
 let s=sl('本科毕设的初步方向','当前首选 A，B / C 仅在基线和瓶颈证据明确后进入','个人初步方案，非已批准题目或已验证创新点；与VLALagBench、SmolVLA、RTC、VLASH比较');
 tab(s,[['备选方向','问题与最小实验','复杂度 / 重叠风险','与 OpenFleX 的联系'],['A 高效推理与\n闭环性能评测\n当前首选','固定策略与任务\n延迟扫描、成功率\n及全任务计时','中 / 高\n已有延迟评测\n需明确工程范围','复用日志、计时\n与评测接口'],['B 动作生成预算\n与执行策略联合研究','固定权重，改采样步数\n再对比执行长度\n分离动作质量影响','中高 / 高\n已有预算与分块研究\n需更稳定基线','复用预算与\n控制周期分析'],['C 闭环推理与\n执行协同优化','对实际瓶颈做\n一种改进及消融\n与官方强基线比较','高 / 高\n异步和状态对齐\n已有多项先例','复用队列、接口\n与失效处理']],[340,440,350,310],{h:465,font:28});
 tx(s,'本科先做一个可复核的小问题，不预设一定有加速收益',85,728,1430,50,32,C.navy,true);
 tx(s,'Agent、世界模型、RSI 为长期兴趣，不列为本科最低交付',85,783,1430,34,25,C.gray);
}
{
 let s=sl('下一步最小可执行路线','所有环节均为计划，尚未宣称模型或代码已经跑通','官方候选：huggingface.co/docs/lerobot/smolvla；具体checkpoint、仿真接口与设备环境待验证');
 let labels=['选择开源模型','核对权重与\n仿真兼容性','基础推理与\n完整任务','性能与闭环\n评测'];
 let a=labels.map((v,i)=>node(s,v,80+i*370,235,310,100));for(let i=0;i<3;i++)ar(s,a[i],a[i+1]);
 let b=['定位实际瓶颈','一种小规模优化','对照实验与失败分析'].map((v,i)=>node(s,v,80+i*495,420,450,88));
 ar(s,a[3],b[0],'bottom','top',true);for(let i=0;i<2;i++)ar(s,b[i],b[i+1]);
 tx(s,'模型',85,570,170,50,32,C.teal,true);tx(s,'优先考察 SmolVLA；ACT 为非 VLA 模仿学习基线',260,570,1250,55,31);
 tx(s,'资源',85,650,170,50,32,C.teal,true);tx(s,'RTX 5070 12GB 本地开发，A100 40GB 服务器实验\n配额与兼容性待核实；OpenFleX 为未来平台',260,650,1250,100,30);
 tx(s,'两三周建议：兼容与任务跑通 → 真实基线 → 小规模延迟对照',85,778,1430,36,27,C.amber);
}
{
 let s=sl('本科毕业要求与课题适配','八项 H 支撑已核原始培养方案，以下均为拟交付成果','原始培养方案：PDF第53—56、59页（印刷50—53、56页），机械工程（机器人试验班）');
 tab(s,[['指标点','拟交付成果'],['2.3 方案分析与比较','文献研究、同步/异步与部署方案比较、选型依据'],['3.3 机械系统设计','控制系统、接口时序、约束与失效处理、闭环验收'],['6.2 / 7.2 工程影响与可持续性','风险及责任分析，资源/能耗测量与生命周期讨论'],['8.2 / 10.3 规范与跨文化交流','真实日志及许可记录，英文摘要与专业交流证据'],['11.2 / 12.2 决策与终身学习','成本性能决策、学习记录与失败复盘']],[570,870],{y:215,h:415,font:29});
 tx(s,'3.3 是否可通过机器人控制系统与软硬件协同设计满足，\n需要导师与学院确认',85,664,1430,92,32,C.amber,true);
 tx(s,'是否必须新增机械结构/零部件设计，待学院确认；目前不主动作为研究主线',85,778,1430,36,25,C.gray);
}
{
 let s=sl('希望老师指导的问题','请帮助判断主线、工作量、专业要求与近期优先级');
 const q=['本科阶段是否应继续以 VLA 高效推理作为主线？','先复现并建立闭环评测，再选择优化问题，工作量是否合理？','机器人控制与系统设计能否满足 3.3，是否需补机械设计？','结合组内模型、服务器和 OpenFleX，两三周最应该先做什么？'];
 q.forEach((v,i)=>{tx(s,String(i+1).padStart(2,'0'),85,245+i*119,90,65,39,C.teal,true);tx(s,v,205,245+i*119,1300,90,35,C.ink);});
 tx(s,'以上为初步调研和个人设想，希望老师批评指正',85,767,1430,45,31,C.gray);
}
{
 let s=sl('附录 A　八项毕业要求的完整对应表','培养方案要求为压缩转述，逐字原文与详细核查见独立 Markdown','原始培养方案PDF第53—56及59页；H为课程支撑强度，不等于个人课题已经达成','A');
 tab(s,[['指标','培养方案要求','本课题可能支撑','可提交证据','确认需求'],['2.3','文献、多方案、影响因素\n与可持续发展','方案比较与瓶颈分析','文献矩阵、取舍依据','常规审核'],['3.3','机械系统或制造工艺\n流程设计，体现创新意识','控制系统、接口时序\n约束与失效处理','架构/时序/接口\n验收数据','重点确认口径\n及仿真/实物要求'],['6.2','工程社会影响及\n安全、法律责任','风险、影响与责任分析','保护停止试验\n风险与许可清单','需补证据'],['7.2','环保、可持续性\n及产品周期损害隐患','资源能耗与部署分析','功率测量边界\n生命周期讨论','需补测量'],['8.2','诚实公正职业规范\n公众福祉与法律责任','诚信、许可和可复现','版本/日志/负结果\n许可与数据来源','常规审核'],['10.3','跨文化语言、书面\n及专业沟通能力','外文理解、专业表达','英文摘要与\n专业介绍/问答','确认交流形式'],['11.2','系统工程、项目管理\n经济评价与决策','成本性能与资源选型','时数/成本假设\n进度风险与决策','需补成本依据'],['12.2','自学、归纳、提问\n严谨批判与新技术学习','精读、复现与问题复盘','周记、版本适配\n代码与更正记录','常规审核']],[100,380,365,335,260],{y:215,h:540,font:21,header:22,pad:2});
 tx(s,'3.2 为机械单元/零部件设计，毕业设计行为空；是否要求补机械交付物仍需学院确认',85,786,1430,31,23,C.amber);
}
{
 let s=sl('附录 B　论文、模型资源与风险','出处与可用性按本轮核查边界说明，不作为已经运行的证据','完整作者、日期、出处链接与原文定位见《重点论文证据与引用》；软件方法披露见Skills调用记录','B');
 tab(s,[['重点论文 / 资源','核查状态与使用边界'],['Diffusion Policy　2303.04137','RSS 2023 / 扩展版 IJRR 2024（作者页）；非语言条件 VLA'],['OpenVLA-OFT　2502.19645','RSS 2025（arXiv作者记录）；不将配方收益全归单一组件'],['SmolVLA　2506.01844','2025预印本；轻量与异步已有研究，checkpoint/仿真待适配'],['RTC　2506.07339','NeurIPS 2025正式论文集已核验；动作块衔接已有先例'],['VLASH　2512.01031','2025首投 / 2026修订，正式出处未核实；需区分训练适配'],['VLALagBench　2611.03103','2026-11预印本，代码注明将发布；与本方向高度接近']],[610,830],{y:210,h:418,font:26,header:27});
 tx(s,'SmolVLA：候选主模型　　ACT：非 VLA 管线基线\n5070 12GB / A100 40GB：环境与配额待核实　　OpenFleX：未来平台',85,663,1430,95,28);
 tx(s,'主要风险：兼容权重缺口、仿真计时失真、已有工作重叠、3.3交付口径未确认',85,784,1430,31,24,C.amber);
 s.speakerNotes.text+='\n软件方法：Kassis et al. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. 2026. https://arxiv.org/abs/2609.00065';
}
await fs.mkdir(path.join(B,'previews'),{recursive:true});
await fs.writeFile(path.join(B,'text_geometry.json'),JSON.stringify(geometry,null,2));
const candidate=path.join(B,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
console.log('Exported',p.slides.items.length);
for(let i=0;i<p.slides.items.length;i++){const r=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(B,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await r.arrayBuffer()));console.log('Rendered',i+1);}
const {finalizePresentation}=await import(pathToFileURL(path.join(SK,'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(path.join(B,'validated'),{recursive:true});
let final=path.join(B,'validated',`validated-${Date.now()}.pptx`);
const result=await finalizePresentation({workspaceDir:W,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:path.join(SK,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SK,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[4,5,6,8,10,11],layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-heading-fit',...[4,5,6,8,10,11].flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(B,`validation-${Date.now()}.json`)});
console.log(JSON.stringify(result));await fs.copyFile(final,path.join(O,'本科毕设方向交流_精简版.pptx'));



