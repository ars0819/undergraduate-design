# paper/

这里只保留辅助脚本和公开配置，**不包含论文原文**（PDF / HTML / 全文转换件）：一是版权，二是仓库体积。

- 论文原文：[arXiv:2506.01844](https://arxiv.org/abs/2506.01844)（HTML 版：<https://arxiv.org/html/2506.01844v1>）
- `smolvla_base/`：HuggingFace Hub 上 `lerobot/smolvla_base` 的公开 `config.json`、`policy_preprocessor.json`、`README.md`，用于对比 base 与 libero checkpoint 的配置差异（见 [`notes/05`](../notes/05-代码对照与复现.md) §5）
- `html2md.py` / `extract_tables.py`：把 arXiv HTML 转成 Markdown、按 `<figure>` 抽取表格的小工具（需先把 HTML 下载到本地再运行）
