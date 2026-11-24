# 生成源码使用说明

`build_deck.mjs`以`@oai/artifact-tool`生成17页可编辑PPTX，`content.mjs`保存讲解与参考条目，`prepare_materials.py`记录来源清点和配套说明，`make_preview.py`生成PDF预览。

本机使用Codex自带Node与Python，未安装或修改全局依赖。`node_modules`为指向现有运行库的目录联接，不是复制的依赖。移到其他电脑时需提供同等Artifact Tool环境，源码中的SKILL、PY与运行库路径需对应修改。

构建过程的候选PPTX、17页PNG、渲染数据与校验回执在工作区`.pptx_build_20261106/`，不属于演示内容。重新构建会生成新的验证副本，再复制到交付文件名，不修改原始调研资料。
