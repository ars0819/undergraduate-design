import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {META,REFS,NOTES} from './content.mjs';

const SRC=path.dirname(fileURLToPath(import.meta.url));
const OUT=path.dirname(SRC),WORK=path.dirname(OUT),BUILD=path.join(WORK,'.pptx_build_20261106');
const SKILL='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations';
const PY='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const FONT='Microsoft YaHei';
const C={navy:'#17375E',blue:'#245A81',ink:'#20334B',gray:'#596879',muted:'#758190',line:'#D4DDE6',pale:'#EDF3F8',white:'#FFFFFF',teal:'#197B82',amber:'#9A5A16'};
const p=Presentation.create({slideSize:{width:1600,height:900}});
const tracked=[];
function text(s,t,x,y,w,h,size=32,color=C.ink,bold=false,align='left',name=''){
 const z=s.shapes.add({geometry:'textbox',name:name||t.slice(0,28),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=t;z.text.style={typeface:FONT,fontSize:size,bold,color,alignment:align,verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0};
 tracked.push({slide:p.slides.items.length,text:t,x,y,w,h,size});return z;
}
function rect(s,x,y,w,h,fill=C.pale,stroke=C.line){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1}})}
function rule(s,x,y,w,color=C.line){return rect(s,x,y,w,2,color,color)}
function box(s,t,x,y,w,h,{size=30,fill=C.pale,color=C.navy,border=C.line}={}){
 const z=s.shapes.add({geometry:'rect',name:t.slice(0,24),position:{left:x,top:y,width:w,height:h},fill,line:{fill:border,width:1.5}});
 z.text=t;z.text.style={typeface:FONT,fontSize:size,color,bold:false,alignment:'center',verticalAlignment:'middle',wrap:'square',autoFit:'none',insets:12};return z;
}
function arrow(s,a,b,from='right',to='left',dashed=false){return s.shapes.connect(a,b,{kind:from==='right'&&to==='left'?'straight':'elbow',fromSide:from,toSide:to,line:{fill:C.blue,width:2.5,style:dashed?'dashed':'solid'},tail:{type:'triangle',width:'med',length:'med'}})}
function slide(title,sub='',refs=[],n){
 const s=p.slides.add();s.background.fill=C.white;const idx=p.slides.items.length;
 text(s,title,78,58,1440,72,48,C.navy,true);
 if(sub)text(s,sub,80,143,1420,52,27,C.gray);
 rule(s,80,824,1440);
 let src=refs.length?refs.map(k=>{const r=REFS[k];return `${r[0]} ${r[1]} · arXiv:${r[2]}`}).join('    '):'前期探索与导师交流，题目及实验路线尚待讨论';
 text(s,src,80,844,1350,32,17.5,C.gray);
 text(s,n||String(idx).padStart(2,'0'),1450,844,70,32,18,C.gray,false,'right');
 const note=NOTES[idx-1];s.speakerNotes.text=(note?.text||'')+'\n\n资料来源：\n'+refs.map(k=>REFS[k][0]+' '+REFS[k][3]).join('\n')+'\n本地研究报告：D:/Desktop/bishe/具身智能系统架构与协同调度_深度研究报告.md';
 return s;
}
function table(s,values,widths,{x=80,y=215,h=515,font=27,header=27,rowHeights}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:widths.reduce((a,b)=>a+b,0),height:h,columnWidths:widths,values});
 t.borders.assign({fill:C.line,width:1,style:'solid'});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=rowHeights?rowHeights[r]:h/values.length;
  for(let c=0;c<values[0].length;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?C.navy:C.white;
   cell.text.style={typeface:FONT,fontSize:r===0?header:font,color:r===0?C.white:C.ink,bold:r===0};
  }
 }
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:14,right:14,top:6,bottom:6},anchor:'center'});
 return t;
}

// 01 Cover. All text remains editable.
{
const s=p.slides.add();s.background.fill=C.white;
text(s,'本科毕业设计方向初步探索\n与文献调研汇报',100,166,1390,200,65,C.navy,true);
text(s,'从 VLA 模型加速到机器人闭环执行与系统协同',105,406,1370,68,35,C.gray);
text(s,`${META.school}\n${META.major}`,105,570,1050,100,30,C.ink);
text(s,`汇报人：${META.name}\n${META.date}`,105,707,1130,100,28,C.gray);
s.speakerNotes.text=NOTES[0].text+'\n姓名来源：D:/Desktop/dickseep/research/thesis/2026-11-04_导师交流初稿/11_两份表格待补充信息与审批清单.md，学生姓名行。未复用其他个人联系信息。';
}
// 02 Interests and discussion scope.
{
const s=slide('研究兴趣与本次交流目的','本科阶段先形成可验证的小问题，保留向轮式双臂扩展的接口');
text(s,'本科阶段',90,230,600,55,36,C.blue,true);
text(s,'可实现、可评估\n符合机械工程毕设要求\n积累复现与实验能力',90,317,650,220,35);
text(s,'博士阶段',865,230,600,55,36,C.blue,true);
text(s,'面向 OpenFleX 轮式双臂\nVLA 高效推理与具身系统\n复用接口、日志与评测方法',865,317,650,220,35);
rule(s,90,590,1420);
text(s,'目前尚未确定题目，希望老师帮助判断方向与研究边界',90,658,1420,95,37,C.navy,true);
}
// 03 Honest progress accounting.
{
const s=slide('文献整理已形成框架，个人理解仍需验证','资料完成情况与本人实际掌握程度分开说明');
text(s,'文献调研工作',85,220,700,50,35,C.blue,true);
text(s,'24 篇结构化笔记',85,300,700,53,37,C.navy,true);
text(s,'每篇 5 类笔记，另有总索引与横向比较',85,365,700,92,30);
text(s,'36 篇研究条目的系统调研报告',85,497,700,55,33,C.navy,true);
text(s,'其中 15 篇重点核读方法与实验\n上述为 AI 辅助核读与整理范围',85,560,700,120,30);
text(s,'当前学习进度与不足',865,220,640,50,35,C.blue,true);
text(s,'基础 Python\n正在学习机器学习与机器人学习',865,303,640,110,32);
text(s,'本人实际精读数量没有明确记录\n代码复现与闭环实验尚未完成',865,465,640,145,32);
text(s,'两组文献有交叉，不合计为 60 篇',85,731,1430,52,29,C.amber);
}
// 04 Native editable architecture diagram.
{
const s=slide('具身智能系统中的模块分工','不同模块采用不同更新频率，模型输出仍需经过控制与安全约束',['H','G']);
const labels=['自然语言任务','Agent／高层任务规划','VLA／动作策略','机器人控制与执行','环境反馈'];
const ns=labels.map((t,i)=>box(s,t,415,210+i*108,600,75,{size:31}));
for(let i=0;i<ns.length-1;i++)arrow(s,ns[i],ns[i+1],'bottom','top');
arrow(s,ns[4],ns[2],'left','left',true);
const wm=box(s,'可选世界模型\n预测行动后果',1115,280,390,122,{size:29,fill:'#F5F7F9'});
arrow(s,wm,ns[1],'left','right',true);
const sc=box(s,'调度器\n推理时机、队列、有效期',1115,500,390,122,{size:28,fill:'#F5F7F9'});
arrow(s,sc,ns[2],'left','right',true);arrow(s,sc,ns[3],'left','right',true);
text(s,'WAM：联合建模\n未来世界与动作\n是一类策略实现路线',80,252,280,150,27,C.gray);
text(s,'反馈进入规划与策略\n控制器执行本地保护',80,552,280,150,27,C.gray);
s.speakerNotes.text+='\nWAM 参考：'+REFS.Z[3];
}
// 05 Knowledge map, four native editable branches.
{
const s=slide('与研究兴趣最相关的四条技术主线','目前更关注模型推理效率与机器人闭环执行的交叉',['O','S','R']);
const root=box(s,'VLA 高效推理\n与闭环执行',570,205,460,110,{size:35,fill:C.pale});
const defs=[
 ['VLA 动作生成','RT-2 / OpenVLA\nDiffusion Policy\nπ₀ / SmolVLA',80],
 ['模型高效推理','OpenVLA-OFT\nTinyVLA\n量化、剪枝、蒸馏、缓存',455],
 ['执行与时序协同','SmolVLA 异步 / RTC\nVLASH / PACE / PTS\nVLALagBench',830],
 ['具身系统架构','Hi Robot / ARP\nLeap-VLA\nAgent 与世界模型协同',1205]
];
for(const [h,b,x] of defs){const a=box(s,h,x,425,315,80,{size:29});arrow(s,root,a,'bottom','top');text(s,b,x,555,315,180,28);}
}
// 06 Timing is the bridge from literature to the research question.
{
const s=slide('动作生成足够快，机器人就能执行得足够好吗？','抓取杯子的教学示例：推理期间，目标位置可能已经变化',['S','R','L']);
const x=[90,370,650,930,1210],w=230;
const labels=['采集杯子观测','VLA 推理开始','杯子被移动','新动作生成完成','按旧位置抓取'];
const nodes=labels.map((v,i)=>box(s,v,x[i],245,w,82,{size:27,fill:i===2?'#FBF2E7':C.pale}));
for(let i=0;i<4;i++)arrow(s,nodes[i],nodes[i+1]);
text(s,'0 ms',90,354,230,40,26,C.gray);text(s,'示意 200 ms 后',930,354,300,40,26,C.gray);
const a=box(s,'推理：根据旧观测生成一段未来动作',370,445,790,76,{size:30});
const b=box(s,'异步执行：机器人继续消耗上一动作块',370,550,790,76,{size:30});
text(s,'Action Chunking：一次预测多个控制步\n同步会等待推理，异步继续执行旧块',90,680,860,105,30);
text(s,'关键：延迟、衔接与有效期',995,690,520,90,31,C.navy,true);
}
// 07 Required evidence comparison table.
{
const s=slide('最相关工作已经覆盖了哪些问题','线程、队列、历史延迟估计与动态执行长度都有先例');
const literatureTable=table(s,[
 ['论文与年份','核心问题','主要方法','对候选方向的启发'],
 ['SmolVLA 2025\n2506.01844','推理与执行等待','队列阈值与观测过滤','先复现官方异步'],
 ['RTC 2025\n2506.07339','动作块衔接','已承诺前缀与补全','连续性已有强方法'],
 ['VLASH 2025 首发\n2512.01031','预测与执行错位','未来本体状态条件','需要区分训练适配'],
 ['PACE 2026\n2606.00537','执行多少步再更新','速度谷选择执行前缀','动态长度已有研究'],
 ['VLALagBench 2026\n2611.03103','延迟何时影响成功','受控延迟与动态任务','评价必须包括闭环'],
 ['ARP 2026\n2611.05155','语义与动作频率耦合','慢语义与快动作分离','多频率涉及模型设计']
],[330,310,380,420],{y:209,h:550,font:27,header:27});
for(let r=1;r<7;r++)literatureTable.getCell(r,0).text.style={typeface:FONT,fontSize:24,color:C.ink};
text(s,'目前尚不能把增加异步线程、队列或阈值直接称为创新',85,776,1420,38,27,C.amber);
s.speakerNotes.text+='\n'+['S','R','V','P','L','A'].map(k=>REFS[k][0]+' '+REFS[k][3]).join('\n');
}
// 08 Explicit knowledge boundary.
{
const s=slide('我想进一步验证的几个问题','围绕固定模型、真实时延和公平预算缩小研究边界',['S','R','L']);
const ys=[230,398,566],hs=['已有研究覆盖','仍需实验验证','可能进一步探索'];
const bodies=['异步执行、延迟历史估计、动作衔接、动态执行长度',
 '延迟波动时怎样选择请求时机？\n队列仍有动作时，如何判断它们是否已经过时？',
 '相同模型与计算预算下，调度是否改善闭环成功？\nGPU 计算效率与实际任务效率的差异有多大？'];
ys.forEach((y,i)=>{text(s,hs[i],85,y,330,55,34,C.blue,true);text(s,bodies[i],435,y,1070,120,33);if(i<2)rule(s,85,y+141,1420);});
text(s,'这些是待验证的问题，尚不能据此认定原创空白',85,759,1420,47,28,C.gray);
}
// 09 Four candidates, concise table.
{
const s=slide('本科毕设的四个初步候选','暂以 A 为探索重点，D 为稳妥备选');
table(s,[
 ['候选方向','预计工作','可行性与主要困难','博士衔接'],
 ['A 动态请求提前量\n同时考虑动作有效期','请求触发、接纳与日志','优先评估\n与现有方法接近，收益待验','时效感知执行接口'],
 ['B 阶段与外部变化\n联合触发推理','阶段管理与变化检测','可选扩展\n误触发、漏触发和检测开销','感知与执行协同'],
 ['C 共享 GPU\n协同调度','优先级与作业接纳','长期兴趣\n显存竞争、长 kernel 阻塞','多模型资源管理'],
 ['D 闭环时序\n与故障注入评测','统一基线与延迟轨迹','稳妥备选\n需确保时间与基线公平','可迁移评测框架']
],[420,310,440,270],{y:220,h:495,font:27});
text(s,'仅为初步备选，希望老师帮助判断',85,765,1430,50,31,C.navy,true);
}
// 10 Minimum viable experiment, native flowchart.
{
const s=slide('最小实验路线','先验证兼容性并冻结策略，暂不从零训练大型 VLA',['S','L']);
const top=['选择开放模型','配置兼容仿真','跑通单个任务','同步与官方异步'];
const bot=['分析失败与价值','比较调度方式','设置延迟条件','建立统一时序日志'];
const xx=[80,455,830,1205],ww=315;
const nt=top.map((t,i)=>box(s,t,xx[i],235,ww,92,{size:29}));
const nb=bot.map((t,i)=>box(s,t,xx[i],425,ww,92,{size:29}));
for(let i=0;i<3;i++)arrow(s,nt[i],nt[i+1]);arrow(s,nt[3],nb[3],'bottom','top');
for(let i=3;i>0;i--)arrow(s,nb[i],nb[i-1],'left','right');
text(s,'拟用工具：SmolVLA + LeRobot，仿真以权重与动作接口兼容为前提',85,568,1430,58,29,C.gray);
text(s,'成功率与完整任务时间',85,670,680,52,34,C.navy,true);
text(s,'端到端延迟与观测年龄',865,670,650,52,34,C.navy,true);
text(s,'队列耗尽与异常恢复',85,739,680,48,31);
text(s,'请求次数与计算开销',865,739,650,48,31);
}
// 11 Beginner-friendly milestones.
{
const s=slide('近期推进按可验收成果设置门槛','每一步先形成证据，再决定是否增加复杂度');
table(s,[
 ['阶段','学习或实验任务','预期成果','继续条件'],
 ['一 理解接口','Action Chunking、SmolVLA\n基本机器人动作接口','清楚的执行时间线\n动作单位与坐标约定','能解释生成与执行的区别'],
 ['二 跑通基线','单个模型与兼容仿真任务','固定权重与可运行配置','低延迟下任务能力可靠'],
 ['三 记录时序','同步／异步、日志与延迟分析','可重建的一次执行 trace','时间推进与指标可信'],
 ['四 缩小选题','受控对照和小规模改进','收益、失败与适用边界','有稳定现象或转评测备选']
],[215,440,420,365],{y:220,h:470,font:27});
text(s,'当前不足：原理学习、工程复现、真实平台与创新性核查仍需推进',85,742,1430,72,29,C.gray);
}
// 12 Meeting close, no defense language.
{
const s=slide('希望老师帮助判断的几个问题');
const qs=[
 'VLA 推理与机器人闭环执行协同，是否适合作为本科主线？',
 '机械工程属性是否需要加强运动学、控制或系统工程设计？',
 '动态请求与动作有效期是否值得研究，边界还应怎样缩小？',
 '启动时优先复现模型，还是先搭统一评测与控制接口？',
 '为未来 OpenFleX 平台，哪些技术成果值得提前准备？'];
qs.forEach((q,i)=>{text(s,String(i+1).padStart(2,'0'),85,205+i*94,75,55,33,C.blue,true);text(s,q,185,205+i*94,1310,70,32);});
text(s,'以上是初步学习、调研与设想，理解仍有不足\n希望老师批评指正，帮助明确研究问题与实施路线',85,708,1430,96,30,C.navy);
}
// 13 Appendix reference list.
{
const s=slide('A1 重点文献与阅读入口','先读 SmolVLA 与 RTC，再看延迟评测和多频率架构',[],'A1');
const keys=['S','R','V','P','L','A','O','H','E','T'];
keys.forEach((k,i)=>{let col=i<5?0:1,row=i%5,x=85+col*760,y=225+row*106;const r=REFS[k];text(s,`${String(i+1).padStart(2,'0')}  ${r[0]}`,x,y,680,43,31,C.navy,true);const z=text(s,`${r[1]}   arXiv:${r[2]}`,x,y+48,680,39,25,C.gray);z.text.get(`arXiv:${r[2]}`).link={uri:r[3],isExternal:true};});
s.speakerNotes.text=NOTES[12].text+'\n'+keys.map(k=>REFS[k][0]+' '+REFS[k][3]).join('\n');
text(s,'条目可点击跳转原文。较新论文保留预印本身份，不用未经核实的发表等级',85,774,1430,36,24,C.gray);
}
// 14 Appendix existing 24-note collection, one category per paper.
{
const s=slide('A2 24 篇文献笔记的分类与整理情况','分类用于说明资料覆盖，各篇只计一次；不代表本人已掌握全部内容',[],'A2');
table(s,[
 ['主题','对应笔记','篇数'],
 ['基础与动作生成','P003 RT-2、P004 OpenVLA、P005 π₀、P006 Diffusion Policy','4'],
 ['高效推理与模型适配','P001 TinyVLA、P002 SPAR、P008 Cue the Flow\nP009 TrajWAM、P010 DiPLO','5'],
 ['系统架构与交互','P007 SayCan、P011 MAVP、P012 Cracking MM\nP013 OmniVTLA、P020 euROBIN','5'],
 ['规划与运动控制','P014 YOPO、P015 ATE、P016 Neuromorphic QP、P017 AC-MPC','4'],
 ['理论与奖励学习','P018 FLORA、P019 Koopman 综述','2'],
 ['综述与应用地图','P021 VLA、P022 灵巧操作、P023 生成模型、P024 采矿自动化综述','4']
],[330,1010,100],{y:216,h:505,font:25.5,header:27});
text(s,'已找到 24 个论文目录及 120 个分篇笔记，另有索引和横向比较',85,749,1430,45,27,C.navy);
text(s,'旧笔记中的部分年份、代码状态与绝对判断待复核，主讲未使用其性能排名',85,794,1430,29,22,C.gray);
s.speakerNotes.text=NOTES[13].text+'\n来源：D:/Desktop/library/Literature_Notes/00_文献总索引.md；00_横向比较与总结.md；本次目录清点。';
}
// 15 Appendix hypotheses and falsification.
{
const s=slide('A3 四个候选的最小实验与失败条件','创新性尚待最近邻核查，以可证伪问题代替预设正收益',[],'A3');
table(s,[
 ['候选','最接近工作','最低实验','何时假设不成立','可复用成果'],
 ['A 请求提前量\n与动作有效期','SmolVLA / RTC\nReflex / VLALagBench','固定权重\n固定与突变延迟','调优固定阈值已不劣\n或年龄增加抵消收益','带时效的请求接口\n有效覆盖与失效规则'],
 ['B 阶段与变化\n联合触发','PACE / SmolVLA\nLeap-VLA','静态任务\n一次目标位移','漏检或检测开销\n抵消调用节省','阶段与事件监控'],
 ['C 共享 GPU\n资源接纳','Clockwork / Paella\nARP','动作模型\n固定后台负载','不可抢占阻塞主导\n或后台长期饿死','资源测量与接纳接口'],
 ['D 时序评测\n与故障注入','VLALagBench\nVLA-HWBench','统一日志\n可重放延迟轨迹','时间推进不正确\n或基线不公平','复现配置与评测框架']
],[235,315,245,345,300],{y:222,h:520,font:24.5,header:26});
text(s,'A／B／D 的调度研究可先固定策略；任务适配本身仍可能需要一次微调',85,775,1430,39,25,C.gray);
}
// 16 Appendix timestamp contract.
{
const s=slide('A4 延迟、观测年龄与动作队列','采集、推理、接收与执行应分别记录，示例数字不代表实测',[],'A4');
const names=['采集观测','发送请求','推理开始','结果接收','首次执行'];
const vals=['t_cap','t_send','t_start','t_arrive','t_cmd'];
const ns=names.map((n,i)=>box(s,n+'\n'+vals[i],80+i*300,235,240,105,{size:28}));
for(let i=0;i<4;i++)arrow(s,ns[i],ns[i+1]);
text(s,'端到端系统延迟 = t_arrive − t_cap',85,397,1380,55,35,C.navy,true);
text(s,'动作观测年龄 = t_cmd − 源观测 t_cap',85,480,1380,55,35,C.navy,true);
text(s,'队列覆盖时间约为连续有效步数 × 控制周期',85,567,1400,55,33);
text(s,'教学量级：30 Hz 下，200 ms 约消耗 6 个控制步\n新结果需核对任务版本、时间轴与有效期，再接入未来队列',85,655,1410,108,29);
text(s,'跨主机时钟需同步与偏差估计。CUDA Event 不能替代完整闭环计时',85,780,1430,35,23,C.gray);
}
// 17 Appendix resource truth and fallback.
{
const s=slide('A5 实验资源、风险与备选路线','尚未验证的兼容性和设备条件不写成已经跑通',[],'A5');
table(s,[
 ['条件','当前状态','实施与降级方式'],
 ['RTX 5070 12GB','可用资源\n推理兼容与峰值显存未实测','先测单样本推理\n记录版本、显存与全链路延迟'],
 ['A100 40GB','实验室可能可用\n配额与共享情况待确认','必要时微调或远程服务\n不依赖独占承诺'],
 ['SO-101 / OpenFleX','SO-101 尚未采购，OpenFleX 接口待确认','先完成兼容仿真，真机到位后再做平台适配'],
 ['模型／仿真匹配','SmolVLA + LeRobot 为拟议起点','若能力不足先检查数据与校准\n必要时用 ACT 验证管线'],
 ['调度没有正收益','允许出现的实验结论','转候选 D，保留失败分析、负结果与可复现配置']
],[320,495,625],{y:222,h:490,font:26});
text(s,'当前最小系统：单个策略 + 轻量状态机 + 调度与日志',85,751,1430,53,31,C.navy,true);
}

await fs.mkdir(path.join(BUILD,'previews'),{recursive:true});
await fs.writeFile(path.join(BUILD,'text_geometry.json'),JSON.stringify(tracked,null,2));
await fs.writeFile(path.join(BUILD,'presentation.json'),JSON.stringify(p.toProto()));
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
console.log('PPTX exported',candidate,'slides',p.slides.items.length);
for(let i=0;i<p.slides.items.length;i++){
 try{const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(BUILD,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));console.log('Rendered',i+1);}
 catch(e){console.error('Render failed',i+1,e.stack);throw e;}
}
try{const b=await p.export({format:'pdf'});await fs.writeFile(path.join(BUILD,'deck-preview.pdf'),new Uint8Array(await b.arrayBuffer()));console.log('PDF exported');}catch(e){console.warn('PDF export unavailable',e.message);}
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(path.join(BUILD,'validated'),{recursive:true});
await fs.mkdir(path.join(BUILD,'receipts'),{recursive:true});
const final=path.join(BUILD,'validated',`deck-${Date.now()}.pptx`);
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:final,pythonExecutable:PY,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:17,requiredNativeTableOwnerSlides:[7,9,11,14,15,17],
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit',...[7,9,11,14,15,17].flatMap(n=>['--require-native-table-slide',String(n)])],
 fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'receipts',path.basename(final)+'.json')});
console.log(JSON.stringify(result,null,2));
await fs.copyFile(final,path.join(OUT,'本科毕设方向与文献调研汇报.pptx'));
try{await fs.copyFile(path.join(BUILD,'deck-preview.pdf'),path.join(OUT,'本科毕设方向与文献调研汇报.pdf'));}catch{}
await fs.writeFile(path.join(OUT,'讲解稿.md'),'# 本科毕设方向初步汇报讲解稿\n\n主讲12页，计划约13分45秒。附录仅在追问时展开。讲解中的技术理解和计划不代表已经完成复现。\n\n'+NOTES.map((n,i)=>`## ${i+1} ${n.title}${i<12?'（约'+n.seconds+'秒）':'（备查）'}\n\n${n.text}`).join('\n\n')+'\n');
console.log('Done');
