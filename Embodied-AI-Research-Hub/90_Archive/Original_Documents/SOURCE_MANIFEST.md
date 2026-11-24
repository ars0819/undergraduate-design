# 原始文件清单与完整性记录

生成时间：2026-11-06（Asia/Shanghai）。本清单只记录实际读取、哈希与复制结果。

| 编号 | 类型 | 原路径 | 归档副本 | 字节 | 原文件修改时间 | SHA256 | 可提取内容 |
|---|---|---|---|---:|---|---|---|
| S1 | 学习路线 V3.0 | `D:\Desktop\机器人学习与具身智能科研路线_V3.0_简化执行版.docx` | `90_Archive/Original_Documents/机器人学习与具身智能科研路线_V3.0_简化执行版.docx` | 115004 | 2026-10-13T15:38:59+08:00 | `8c1e4c095422fcd83f55c1e76af805cf9ff8ce7170460543367d9d5e20154ee3` | DOCX；413 段、17 表、0 图；文档内部作者 刘鑫宇；内部修改属性 2026-10-13 07:38:59+00:00 |
| S2 | 原始学习笔记 | `D:\Desktop\926150.docx` | `90_Archive/Original_Documents/926150.docx` | 477995 | 2026-10-21T22:59:04+08:00 | `05c1e7311cbc24a274dbb2fae12bf405b5e30ffae9145bb6f0ffc6286536d1e0` | DOCX；473 段、1 表、5 图；文档内部作者 Lenovo；内部修改属性 2026-10-21 14:59:04+00:00 |
| S3 | 进度表 | `D:\Desktop\机器人学习与具身智能科研路线_表格版.xlsx` | `90_Archive/Original_Documents/机器人学习与具身智能科研路线_表格版.xlsx` | 47997 | 2026-10-13T12:23:13+08:00 | `00c5460cacb220e317b75683fd03326234da0e72220bde8206946f790db82dec` | XLSX；7 个工作表：总览、32周主线、前14天、课程资源、数学与MLS、框架与CS课程、验收与自测；文件系统修改时间见左列；内部 core properties 不作为修改证据 |

## 路径与版本核对

- 任务说明中写作 `926150(1).docx`，实际附件与磁盘中只找到 `D:\Desktop\926150.docx`；未发现同名 `926150(1).docx`。本库按实际文件处理，不伪造第二份副本。
- 对 `D:\` 与用户目录做了文件名检索；未发现 `926150` 的第二个 DOCX 版本。三份源文件均未被修改。
- Word 文档中的日期是文档内容日期或文件保存时间，不等于实际学习完成日期；正文日期另见学习笔记。
- Excel 的 `最后修改时间` 与文件系统 mtime 已记录；原始工作簿没有公式，七个工作表均为 `visible`。openpyxl 读取时可能用默认值填充缺失的内部 core properties，因此**内部 modified 属性不作为修改证据**，以文件系统 mtime + SHA256 为准。
- 每张 Excel 工作表页眉有 1 个装饰图锚点，工作簿包里共用 1 个 PNG；已一并保存到 `_extracted_media/xlsx/`。

## 提取媒体

| 来源 | 包内路径 | 归档路径 | 字节 | SHA256 |
|---|---|---|---:|---|
| notes | `word/media/image1.png` | `90_Archive/Original_Documents/_extracted_media/notes/image1.png` | 77385 | `f0c67154eadc0d2e0b79d1cffa194937ee1ae23ac84d34744a669b880de41412` |
| notes | `word/media/image2.png` | `90_Archive/Original_Documents/_extracted_media/notes/image2.png` | 62924 | `a1627e637875abe873d133b5848d1cf69f2eff11a8cac6dbf59628b85eb049e1` |
| notes | `word/media/image3.png` | `90_Archive/Original_Documents/_extracted_media/notes/image3.png` | 62255 | `3909b7a2f7ed1ef355a716914bcf420725ac1f715965b03cb23b7346cd2f737f` |
| notes | `word/media/image4.png` | `90_Archive/Original_Documents/_extracted_media/notes/image4.png` | 90039 | `f14f53f117c8a0d27826fabf204c997775f3d560d3df4e26c824b7324545b641` |
| notes | `word/media/image5.png` | `90_Archive/Original_Documents/_extracted_media/notes/image5.png` | 106829 | `726c9e3461fd72cc65dc0556282647e49d3b438776802a0fc6809d31c6dee559` |
| xlsx | `xl/media/image1.png` | `90_Archive/Original_Documents/_extracted_media/xlsx/image1.png` | 7316 | `73c751afe7ce3cd1501265a2606992c45f0a1cfb72287b5872592a88534dcb3f` |

## 原文件引用位置

- `90_Archive/Original_Documents/926150_原始笔记逐段提取.md`：逐段机械提取，保留原文。
- `90_Archive/Old_Roadmaps/V3.0_完整机械提取.md`：原 32 周路线机械转换。
- `90_Archive/Old_Roadmaps/Excel_七工作表结构化提取.md`：七表逐单元格提取。
- `99_Assets/Images/IMAGE_INDEX.md`：5 张学习图片的来源、段落与哈希。

## 未改动声明

- 未修改三份源文件、原 Excel 工作簿、Zotero 数据库或现有 Obsidian Vault。
- 未从既有本科毕设项目复制、修改或启动任何代码。
- 大模型权重、数据集、Conda 环境与运行缓存均未放入本库。
