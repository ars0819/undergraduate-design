# 源码与复现说明
prepare.py读取本地材料、生成配套稿件并实际调用experimental-design的随机顺序函数。build.mjs使用Codex自带Artifact Tool生成11页原生可编辑PPTX、渲染和finalizePresentation校验。finish.py生成预览PDF、单页提词卡、候选登记与交付记录。

使用Codex load_workspace_dependencies返回的Node/Python；环境路径写在build.mjs及实际调用记录中。构建中间件与回执位于D:/Desktop/bishe/.精简版_build，最终输出位于本目录上级。重新运行prepare.py会重建生成文档，finish.py负责更新具体段落定位。原始任务书、培养方案和文献笔记只读。

PDF为已核查渲染页的阅读预览，无可搜索文字；可编辑版本为PPTX。提词卡PDF使用微软雅黑，独立一页。没有PowerPoint原生打开验证。
