# PDF 翻译流水线（PDF Translation Pipeline）

> 使用 MinerU（Pipeline 后端）+ OpenAI 兼容 API，将 PDF 文档翻译为中文，保留图片与题注。

## 概述

该流水线将 PDF 翻译分为三步，均在单进程内完成，无需启动服务器：

1. **提取（Extract）** — 使用 MinerU Pipeline API 解析 PDF，提取文字、图片与文档结构，输出结构化 JSON 与图片
2. **翻译（Translate）** — 使用 OpenAI 兼容 API 翻译文字，保护数学公式、URL、引用
3. **生成（Generate）** — 使用 reportlab 生成翻译后 PDF，含图片与题注

## 特性

- 进程内提取（无 subprocess，避免 Windows 权限问题）
- 仅支持 `pipeline` 后端（快速、稳定、通用）
- 自动检测系统 CJK 字体（微软雅黑 → 宋体 → 黑体）
- `--quick` 调试模式（2 页 / 120 秒超时 / 不翻译）
- GPU 加速（CUDA 12.8，`torch 2.11.0+cu128` 支持 RTX 5060 等新架构）

## 环境要求

- Python 3.12
- （可选）NVIDIA GPU + CUDA 12.8
- 首次运行自动下载 MinerU Pipeline 模型（约 1–2 GB）

## 安装

### 1. 创建并激活 Conda 环境

```bash
conda create -n pdftrans python=3.12 -y
conda activate pdftrans
```

### 2. 安装 PyTorch（CUDA 12.8）

> 必须先于 requirements.txt 安装，否则 `mineru[pipeline]` 会拉取 CPU 版 torch。

```bash
pip install torch==2.11.0+cu128 --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128
```

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型（约 1–2 GB）

```bash
mineru-models-download -m pipeline -s huggingface
```

> 模型默认缓存在 `~/.cache/huggingface`。可移动至任意位置（如 `D:\models\huggingface`），
> 再通过目录联接（`mklink /J`）或 `HF_HOME` 环境变量指向它。

### 5. 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
```

## 使用

```bash
# 快速调试（2 页、不翻译、120 秒超时）
python src/pipeline.py input/paper.pdf --quick

# 翻译单个 PDF（全部页面）
python src/pipeline.py input/paper.pdf

# 只提取不翻译（测试提取步骤）
python src/pipeline.py input/paper.pdf --no-translate

# 指定页面范围（0-indexed）
python src/pipeline.py input/paper.pdf --start 0 --end 4
```

完整参数见 `python src/pipeline.py --help`。

## 项目结构

```
.
├── src/
│   ├── extract.py        # PDF 提取（MinerU Pipeline API）
│   ├── translate.py      # 翻译（OpenAI 兼容 API）
│   ├── generate_pdf.py   # PDF 生成（reportlab）
│   └── pipeline.py       # 主流程编排
├── requirements.txt      # Python 依赖
├── .env.example          # 环境变量模板
└── CLAUDE.md             # 开发者文档
```

## 配置（.env）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | API 密钥（必需） | — |
| `OPENAI_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 模型 | `gpt-4o` |
| `SOURCE_LANG` | 源文档语言 | `en` |
| `TARGET_LANG` | 目标语言 | `chinese` |
| `MINERU_BACKEND` | MinerU 后端 | `pipeline` |
| `MINERU_DEVICE_MODE` | `cpu` 或 `cuda` | `cpu` |
| `MINERU_TIMEOUT` | 提取超时（秒） | `600` |

## 常见问题

- **首次运行很慢？** 首次需下载模型（1–2 GB），可预下载：`mineru-models-download -m pipeline -s huggingface`
- **中文显示空白？** 需要中文字体。Windows 自带微软雅黑；Linux：`apt install fonts-noto-cjk`
- **CUDA 报错 `no kernel image is available`？** 多为 PyTorch 版本过旧，请用 `torch 2.11.0+cu128`；或回退 `MINERU_DEVICE_MODE=cpu`

更多细节见 [CLAUDE.md](CLAUDE.md)。
