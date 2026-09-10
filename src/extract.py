"""
PDF extraction module using MinerU Pipeline API directly.

Uses MinerU's doc_analyze_streaming (pipeline backend) to parse PDFs
in-process — avoiding subprocess overhead and Windows permission issues.
Outputs structured middle.json and extracted images.
"""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

# Load project-root .env BEFORE applying any defaults, so settings such as
# MINERU_DEVICE_MODE (cpu/cuda) in .env are honored.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

# Workaround Windows symlink permission issues with HuggingFace cache
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
# Avoid PyTorch CUDA warnings on some configs
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Default device mode: CPU unless explicitly set (e.g. MINERU_DEVICE_MODE=cuda
# to use the GPU). The default is only applied when no value is present.
os.environ.setdefault("MINERU_DEVICE_MODE", "cpu")


def check_models_downloaded(backend: str = "pipeline") -> bool:
    """Check if the required models are already downloaded."""
    cache_dir = Path.home() / ".cache"
    hf_cache = cache_dir / "huggingface" / "hub"
    ms_cache = cache_dir / "modelscope" / "hub"

    if backend.startswith("pipeline"):
        for cache in (hf_cache, ms_cache):
            if cache.exists():
                for item in cache.iterdir():
                    if "pdf-extract-kit" in item.name.lower():
                        logger.info(f"Pipeline models found: {item.name}")
                        return True
        logger.warning("Pipeline models not found; will be auto-downloaded (~1-2GB)")
        return False

    logger.warning(f"Model check not implemented for backend: {backend}")
    return True


def extract_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    backend: str = "pipeline",
    method: str = "auto",
    lang: str = "en",
    formula_enable: bool = True,
    table_enable: bool = True,
    start_page: int = 0,
    end_page: Optional[int] = None,
    timeout: Optional[int] = None,
    show_progress: bool = True,
) -> Path:
    """
    Extract text, images, and structure from a PDF using MinerU Pipeline API (in-process).

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to store extraction results.
        backend: Only 'pipeline' is supported for direct (subprocess-free) extraction.
        method: Parse method (auto, txt, ocr).
        lang: Document language for OCR (ch, en, etc.).
        formula_enable: Enable formula parsing.
        table_enable: Enable table parsing.
        start_page: Starting page (0-indexed).
        end_page: Ending page (0-indexed, None = all pages).
        timeout: Ignored (kept for API compatibility).
        show_progress: Show extraction progress.

    Returns:
        Path to the extraction output directory (contains middle.json, images/, etc.).
    """
    if backend not in ("pipeline",):
        raise ValueError(
            f"Backend '{backend}' requires subprocess mode (Windows issues). "
            f"Use 'pipeline' for reliable in-process extraction."
        )

    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected .pdf file, got: {pdf_path.suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"MinerU Pipeline extraction (in-process)")
    logger.info(f"  PDF: {pdf_path.name}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Language: {lang}, Method: {method}")

    # Check models
    check_models_downloaded(backend)

    # Read PDF bytes
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Set up the image writer — images go to output_dir/images/
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    from mineru.data.data_reader_writer.filebase import FileBasedDataWriter
    image_writer = FileBasedDataWriter(str(images_dir))

    # Collect results here
    results = []

    def on_doc_ready(doc_index, model_list, middle_json, ocr_enable):
        """Callback: MinerU calls this when a document is fully processed."""
        results.append({
            "doc_index": doc_index,
            "middle_json": middle_json,
            "ocr_enable": ocr_enable,
        })
        logger.info(f"  Doc {doc_index} ready — {len(middle_json.get('pdf_info', []))} pages")

    # ── Run the pipeline (with hard timeout) ──────────────────────────
    start_time = time.time()
    hard_timeout = timeout if timeout else 600

    # Thread-based hard timeout for Windows (signals not reliable)
    timed_out = threading.Event()
    _current_thread_id = threading.get_ident()

    def _hard_timeout_handler():
        timed_out.set()
        logger.error(
            f"⏰ HARD TIMEOUT after {hard_timeout}s! "
            f"Forcing exit. Use --quick for debug, or --timeout N."
        )
        # Force exit the process — brutal but effective for debugging
        os._exit(1)

    timer = threading.Timer(hard_timeout, _hard_timeout_handler)
    timer.daemon = True
    timer.start()

    try:
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming

        doc_analyze_streaming(
            pdf_bytes_list=[pdf_bytes],
            image_writer_list=[image_writer],
            lang_list=[lang],
            on_doc_ready=on_doc_ready,
            parse_method=method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            client_side_output_generation=False,
        )
    finally:
        timer.cancel()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline extraction completed in {elapsed:.0f}s")

    if not results:
        raise RuntimeError("No results from MinerU pipeline")

    result = results[0]
    middle_json = result["middle_json"]

    # ── Page filtering (if start/end specified) ─────────────────────
    pdf_info = middle_json.get("pdf_info", [])
    if start_page > 0 or end_page is not None:
        if end_page is None:
            end_page = len(pdf_info) - 1
        pdf_info = pdf_info[start_page:end_page + 1]
        middle_json["pdf_info"] = pdf_info
        logger.info(f"  Filtered to pages {start_page}-{end_page} ({len(pdf_info)} pages)")

    # ── Save middle JSON ─────────────────────────────────────────────
    middle_json_path = output_dir / "middle.json"
    # Sanitize surrogate characters (MinerU occasionally produces them)
    middle_json = _sanitize_surrogates(middle_json)
    with open(middle_json_path, "w", encoding="utf-8") as f:
        json.dump(middle_json, f, ensure_ascii=False, indent=2)
    logger.info(f"  Middle JSON saved: {middle_json_path}")

    logger.info(f"Extraction complete → {output_dir}")
    return output_dir


def find_middle_json(parse_dir: str | Path) -> Path:
    """Locate the middle JSON file in a parse directory.

    Tries middle.json first, then *_middle.json (legacy naming). Returns the
    resolved path so callers don't have to re-derive the filename.
    """
    parse_dir = Path(parse_dir)
    for pattern in ["middle.json", "*_middle.json"]:
        matches = list(parse_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No middle.json found in {parse_dir}")


def load_middle_json(parse_dir: str | Path) -> dict:
    """Load the middle JSON from a MinerU parse directory."""
    with open(find_middle_json(parse_dir), "r", encoding="utf-8") as f:
        return json.load(f)


def get_images_dir(parse_dir: str | Path) -> Path:
    """Get the images directory path."""
    parse_dir = Path(parse_dir)
    images_dir = parse_dir / "images"
    if images_dir.exists():
        return images_dir
    for d in parse_dir.iterdir():
        if d.is_dir() and d.name.lower() == "images":
            return d
    raise FileNotFoundError(f"No images directory in {parse_dir}")


def extract_translatable_blocks(middle_json: dict) -> list[dict]:
    """
    Extract all translatable text blocks from a MinerU middle JSON.

    Returns a list of dicts with keys:
        - block_type: e.g. 'text', 'title', 'image_caption', etc.
        - text: the original text to translate
        - page_idx: page number
        - block_idx: block index on the page
        - image_path: associated image path (for captions), or None
    """
    translatable_types = {
        "text", "title", "paragraph_title", "doc_title",
        "abstract", "ref_text", "index", "list",
        "image_caption", "table_caption", "chart_caption",
        "algorithm_caption", "code_caption",
        "image_footnote", "table_footnote", "chart_footnote",
        "header", "footer",
    }

    blocks = []
    pdf_info = middle_json.get("pdf_info", [])

    for page in pdf_info:
        page_idx = page.get("page_idx", 0)
        para_blocks = page.get("para_blocks", [])

        for block_idx, block in enumerate(para_blocks):
            block_type = block.get("type", "")
            if block_type not in translatable_types:
                continue

            text = _extract_block_text(block)
            if not text or not text.strip():
                continue

            image_path = _find_image_in_block(block)

            blocks.append({
                "block_type": block_type,
                "text": text.strip(),
                "page_idx": page_idx,
                "block_idx": block_idx,
                "image_path": image_path,
            })

    return blocks


def _sanitize_surrogates(obj):
    """Recursively remove lone surrogate characters (U+D800-U+DFFF) from strings.

    MinerU OCR occasionally produces text with unpaired surrogates, which are
    invalid UTF-8 and cause json.dump to fail. This sanitizes all strings in a
    nested dict/list/str structure.
    """
    if isinstance(obj, str):
        # Remove lone surrogates: high surrogates U+D800-U+DBFF,
        # low surrogates U+DC00-U+DFFF
        return re.sub(r'[\ud800-\udfff]', '', obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_surrogates(item) for item in obj]
    return obj


def _normalize_latex(latex: str) -> str:
    """Compact LaTeX math by removing unnecessary spaces.

    '\\\\varepsilon _ { r } ~ \\\\approx ~ 1 1 . 7'
    → '\\\\varepsilon_{r} \\\\approx 11.7'
    """
    s = str(latex)
    # Remove tildes (~) used as LaTeX spacing
    s = s.replace(' ~ ', ' ').replace('~ ', ' ').replace(' ~', ' ')
    # Compact subscript/superscript braces: "_ { r }" → "_{r}"
    s = re.sub(r'([_^])\s*\{\s*([^}]*?)\s*\}', r'\1{\2}', s)
    # Remove spaces around dots in numbers (loop until stable)
    for _ in range(5):
        prev = s
        s = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', s)
        # Merge adjacent digits separated by space: "4 0 0" → "400"
        s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
        if s == prev:
            break
    # Clean up math operators
    s = re.sub(r'\s*\\approx\s*', r' \\approx ', s)
    s = re.sub(r'\s*\\mu\s*', r' \\mu ', s)
    # Collapse multiple spaces
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()


def _extract_block_text(block: dict) -> str:
    """Extract full text from a block, handling span types intelligently.

    - text spans: joined with spaces (normal words)
    - inline_equation spans: compacted LaTeX wrapped in $...$
      so the translator treats them as math placeholders
    - other spans (equation, hyperlink, etc.): left as-is
    """
    lines = block.get("lines", [])
    result_parts = []
    for line in lines:
        spans = line.get("spans", [])
        for span in spans:
            content = span.get("content", "")
            if not content:
                continue
            span_type = span.get("type", "text")
            if span_type == "inline_equation":
                # Compact LaTeX spacing, wrap in $...$ for translation protection
                compact = _normalize_latex(content)
                result_parts.append(f"${compact}$")
            elif span_type == "text":
                result_parts.append(str(content))
            else:
                result_parts.append(str(content))
    return " ".join(result_parts)


def _find_image_in_block(block: dict) -> Optional[str]:
    """Find image path in a block's sub-blocks."""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("image_path"):
                return span["image_path"]

    for sub_block in block.get("blocks", []):
        for line in sub_block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("image_path"):
                    return span["image_path"]

    return None
