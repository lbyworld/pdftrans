# PDF Translation Pipeline

> 使用 MinerU (Pipeline 后端) + OpenAI API 进行 PDF 文档翻译，保留图片和题注。

## 项目概述

该 pipeline 将 PDF 翻译分为三步（均在单进程中完成，无需启动服务器）：

1. **提取 (Extract)** — 使用 MinerU Pipeline API 直接解析 PDF，提取文字、图片、文档结构，输出 middle JSON + 图片
2. **翻译 (Translate)** — 使用 OpenAI 兼容 API 翻译文字，保护数学公式、URL、引用
3. **生成 (Generate)** — 使用 reportlab 生成翻译后 PDF，含图片和题注

**特点：**
- 进程内提取（无 subprocess，避免 Windows 权限问题）
- 仅支持 `pipeline` 后端（快速、稳定、通用）
- 自动检测系统 CJK 字体
- `--quick` 调试模式：2 页、120s 超时、不翻译

## 环境

- Python 环境：Conda env `pdftrans`（`D:\anaconda3\envs\pdftrans`，Python 3.12）
- 关键依赖：
  - `mineru` 3.2.3 — PDF 解析（pipeline 后端，进程内调用）
  - `openai` 2.41+ — 翻译 API 客户端
  - `reportlab` 4.x — PDF 生成
  - `pillow` — 图片处理
  - `loguru` — 日志
  - `python-dotenv` — 环境配置
- **设备**：由 `.env` 的 `MINERU_DEVICE_MODE` 决定（默认 `cpu`，未设置时回退 CPU）
  - 已安装 `torch 2.11.0+cu128`，支持 RTX 5060（sm_120 / Blackwell）等新架构；设置 `MINERU_DEVICE_MODE=cuda` 即可启用 GPU 加速

## 项目结构

```
.
├── .env                      # API 密钥等配置
├── .env.example              # 环境变量模板
├── input/                    # 待翻译的 PDF 文件
├── output/                   # 翻译后的 PDF 输出
├── temp/                     # 临时提取文件（ext_* 目录，--keep-temp 时保留）
├── src/                      # 源代码
│   ├── __init__.py
│   ├── extract.py            # PDF 提取模块 (MinerU Pipeline API)
│   ├── translate.py          # 翻译模块 (OpenAI)
│   ├── generate_pdf.py       # PDF 生成模块 (reportlab)
│   └── pipeline.py           # 主流程编排
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
└── CLAUDE.md                 # 本文档
```

## 使用方法

### 1. 配置 API 密钥

编辑 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
SOURCE_LANG=en
TARGET_LANG=chinese
MINERU_METHOD=auto
MINERU_DEVICE_MODE=cpu   # 或 cuda（需 CUDA 可用）
```

> `SOURCE_LANG`、`TARGET_LANG`、`MINERU_METHOD`、`MINERU_DEVICE_MODE`、
> `MINERU_BACKEND`、`MINERU_TIMEOUT` 均会被 pipeline 读取生效。

### 2. 常用命令

> 先激活环境：`conda activate pdftrans`（或将下面的 `python` 换成 `D:\anaconda3\envs\pdftrans\python.exe`）。

```bash
# 快速调试（2 页，不翻译，120s 超时）
python src/pipeline.py input/paper.pdf --quick

# 翻译单个 PDF（全部页面）
python src/pipeline.py input/paper.pdf

# 只提取不翻译（测试提取步骤）
python src/pipeline.py input/paper.pdf --no-translate

# 指定页面范围
python src/pipeline.py input/paper.pdf --start 0 --end 4

# 快速模式（跳过公式/表格解析）
python src/pipeline.py input/paper.pdf --fast

# 使用已有提取结果
python src/pipeline.py input/paper.pdf --skip-extraction

# 查看详细日志
python src/pipeline.py input/paper.pdf -v
```

### 3. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入 PDF 文件或目录 | (必需) |
| `-o, --output` | 输出路径 | `output/<name>_translated.pdf` |
| `--temp` | 临时目录 | `temp/`（项目根目录） |
| `-l, --lang` | 源文档语言 | `SOURCE_LANG` env 或 `en` |
| `--target` | 目标语言 | `TARGET_LANG` env 或 `chinese` |
| `-m, --model` | OpenAI 模型 | `gpt-4o` (env) |
| `--delay` | API 调用间隔(秒) | `0.5` |
| `--start` | 起始页 (0-indexed) | `0` |
| `--end` | 结束页 | 全部 |
| `--timeout` | 提取超时(秒) | `600` (env) |
| `--quick` | 调试模式 | — |
| `--fast` | 跳过公式表格 | — |
| `--no-translate` | 跳过翻译 | — |
| `--skip-extraction` | 跳过提取 | — |
| `--keep-temp` | 保留临时文件 | — |

## 提取流程 (extract.py)

### 架构

使用 MinerU Pipeline API（`doc_analyze_streaming`）在进程内直接调用，无需启动 mineru-api 服务器：

```python
from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
from mineru.data.data_reader_writer.filebase import FileBasedDataWriter

image_writer = FileBasedDataWriter(str(images_dir))
doc_analyze_streaming(
    pdf_bytes_list=[pdf_bytes],
    image_writer_list=[image_writer],
    lang_list=[lang],
    on_doc_ready=callback,
    parse_method='auto',
    formula_enable=True,
    table_enable=True,
)
```

### 输出结构

```
temp/ext_{timestamp}_{hash}/
├── middle.json       # 结构化文档数据
├── images/           # 提取的图片 (SHA256 哈希名)
│   └── <hash>.jpg
└── translation_map.json  # 翻译对照表（翻译步骤生成）
```

### middle.json 结构

```json
{
  "_backend": "pipeline",
  "pdf_info": [
    {
      "page_idx": 0,
      "page_size": [612.0, 792.0],
      "para_blocks": [
        {
          "type": "text",
          "lines": [{ "spans": [{ "type": "text", "content": "Text..." }] }]
        },
        {
          "type": "image",
          "blocks": [
            { "type": "image_body", "lines": [{ "spans": [{ "image_path": "hash.jpg" }] }] },
            { "type": "image_caption", "lines": [{ "spans": [{ "content": "Fig. 1: ..." }] }] }
          ]
        }
      ]
    }
  ]
}
```

### BlockType 关键类型

| 类型 | 说明 | 翻译 |
|------|------|------|
| `text` | 正文 | ✓ |
| `title`, `paragraph_title` | 标题 | ✓ |
| `abstract` | 摘要 | ✓ |
| `image` | 图片（含 image_body + image_caption 子块）| — |
| `image_caption` | 图片题注 | ✓ |
| `table_caption` | 表格题注 | ✓ |
| `interline_equation` | 行间公式 | 保留 |
| `ref_text` | 参考文献 | 保留 |
| `code` | 代码 | 保留 |

## 翻译策略 (translate.py)

- **保护**：数学公式 `$...$`, `$$...$$`、URL、引用 `[1]` 使用占位符保护
- **跳过**：纯公式行、纯数字行、URL 行
- **批量**：逐块翻译，间隔 0.5s 防限速
- **回退**：API 调用异常时保留原文

## PDF 生成 (generate_pdf.py)

使用 reportlab 生成新 PDF：
- **字体**：自动检测系统 CJK 字体（Windows: 微软雅黑→宋体→黑体）
- **布局**：Paragraph 流式排版，自动换行/分页
- **图片**：KeepTogether 保持图+题注在同一页，缩放至 150×120mm
- **表格**：支持简单 HTML 表格 → reportlab Table
- **公式**：保留原样输出
- **样式**：正文 10pt、标题 14-18pt、题注 9pt、参考文献 8pt

## 常见问题

### Q: 首次运行很慢？
A: 首次需下载 Pipeline 模型（~1-2GB）。使用 `mineru-models-download -m pipeline -s huggingface` 预下载。

### Q: 中文显示为空白？
A: 需要中文字体。Windows 已有微软雅黑；Linux: `apt install fonts-noto-cjk`。

### Q: CUDA error: no kernel image is available for execution on the device？
A: 该错误通常由 PyTorch 版本过旧、不支持新 GPU 架构（如 RTX 5060 sm_120）引起。本项目已使用 `torch 2.11.0+cu128`（支持 sm_120），CUDA 可用。若仍遇到，可回退 `MINERU_DEVICE_MODE=cpu`。

### Q: 运行时间太长？
A: 使用 `--quick` 调试（2 页、120s 超时）；或用 `--start/--end` 限制页面；正常 CPU 提取 ~6s/页。

### Q: 翻译后排版不完美？
A: reportlab 是新排版，不完全复刻原版。重点保留文字顺序、图片位置和题注关联。
