"""
Main PDF translation pipeline.

Orchestrates the end-to-end workflow:
  1. Extract PDF content and structure with MinerU
  2. Translate text blocks via OpenAI API
  3. Generate a translated PDF with images and captions

Usage:
    python src/pipeline.py input/paper.pdf

Or with custom options:
    python src/pipeline.py input/paper.pdf -o output/ -l en --target zh -m gpt-4o
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load project-root .env BEFORE importing modules, so env-driven defaults
# (MINERU_DEVICE_MODE, SOURCE_LANG, TARGET_LANG, MINERU_METHOD, ...) are set
# before any module-level initialization reads them.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (_PROJECT_ROOT / ".env").exists():
    load_dotenv(_PROJECT_ROOT / ".env")

from extract import (
    extract_pdf,
    load_middle_json,
    find_middle_json,
    get_images_dir,
    extract_translatable_blocks,
)
from translate import PDFTranslator
from generate_pdf import PDFGenerator


def _setup_logging(verbose: bool = False):
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def _load_env(env_path: str = None):
    """Load environment variables.

    The project-root .env is already loaded at module import time. If a
    different env file is passed via --env, load it with override so it
    takes precedence over the default.
    """
    if env_path:
        load_dotenv(env_path, override=True)


def main():
    parser = argparse.ArgumentParser(
        description="Translate a PDF document using MinerU + OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py input/paper.pdf
  python src/pipeline.py input/paper.pdf --fast              # fastest extraction
  python src/pipeline.py input/paper.pdf -o output/translated.pdf
  python src/pipeline.py input/paper.pdf -l en --target chinese -m gpt-4o
  python src/pipeline.py input/paper.pdf --timeout 1200       # 20 min timeout
  python src/pipeline.py input/ -o output/                    # process all PDFs
        """,
    )

    # Input/Output
    parser.add_argument(
        "input", type=str,
        help="Path to input PDF file or directory of PDFs",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output path (PDF file or directory). Default: output/<pdf_name>_translated.pdf",
    )
    parser.add_argument(
        "--temp", type=str, default="temp",
        help="Temporary directory for extraction results. Default: temp/",
    )

    # MinerU settings
    parser.add_argument(
        "-b", "--backend", type=str,
        default=os.getenv("MINERU_BACKEND", "pipeline"),
        choices=["pipeline"],
        help="MinerU backend. Currently only 'pipeline' is supported "
             "(direct in-process extraction, no server needed).",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Timeout in seconds for MinerU extraction. Default: from MINERU_TIMEOUT env or 600",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: pipeline backend, skip formula/table parsing for speed",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Debug mode: 2 pages max, 120s timeout, no-translate, keep temp",
    )
    parser.add_argument(
        "--method", type=str, default=os.getenv("MINERU_METHOD", "auto"),
        choices=["auto", "txt", "ocr"],
        help="PDF parse method. Default: MINERU_METHOD env or auto",
    )
    parser.add_argument(
        "-l", "--lang", type=str, default=os.getenv("SOURCE_LANG", "en"),
        help="Source document language for OCR. Default: SOURCE_LANG env or en",
    )
    parser.add_argument(
        "--no-formula", action="store_true",
        help="Disable formula parsing",
    )
    parser.add_argument(
        "--no-table", action="store_true",
        help="Disable table parsing",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start page (0-indexed). Default: 0",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End page (0-indexed). Default: all",
    )

    # Translation settings
    parser.add_argument(
        "--target", "--target-lang", type=str,
        default=os.getenv("TARGET_LANG", "chinese"),
        dest="target_lang",
        help="Target language. Default: TARGET_LANG env or chinese",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help="OpenAI model. Default: from env OPENAI_MODEL or gpt-4o",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API key. Default: from env OPENAI_API_KEY",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="OpenAI base URL. Default: from env OPENAI_BASE_URL or https://api.openai.com/v1",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds. Default: 0.5",
    )

    # Options
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip MinerU extraction (use existing temp results)",
    )
    parser.add_argument(
        "--no-translate", action="store_true",
        help="Skip translation (generate PDF with original text)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary extraction files",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Path to .env file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)
    _load_env(args.env)

    # Resolve timeout: CLI arg > .env > default 600
    if args.timeout is None:
        args.timeout = int(os.getenv("MINERU_TIMEOUT", "600"))

    # Apply --quick mode overrides (for debugging)
    if args.quick:
        args.backend = "pipeline"
        args.no_translate = True
        args.keep_temp = True
        args.no_formula = True
        args.no_table = True
        args.timeout = 120
        if args.end is None:
            args.end = 1  # 2 pages (0, 1)
        args.verbose = True
        logger.info("🐛 Quick/Debug mode: 2 pages, 120s timeout, no translate")

    # Apply --fast mode overrides
    if args.fast:
        args.backend = "pipeline"
        args.no_formula = True
        args.no_table = True
        if args.timeout > 300:
            args.timeout = 300
        logger.info("⚡ Fast mode: pipeline backend, no formula/table parsing")

    project_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input).resolve()
    temp_dir = Path(args.temp)
    if not temp_dir.is_absolute():
        temp_dir = (project_root / temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve input PDFs ──────────────────────────────────────────────────
    if input_path.is_dir():
        pdf_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() == ".pdf"
        )
        if not pdf_files:
            logger.error(f"No PDF files found in {input_path}")
            sys.exit(1)
    else:
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        if input_path.suffix.lower() != ".pdf":
            logger.error(f"Expected .pdf file, got: {input_path}")
            sys.exit(1)
        pdf_files = [input_path]

    logger.info(f"Found {len(pdf_files)} PDF(s) to process")

    # ── Process each PDF ───────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_dirs = []  # extraction dirs created this run (for cleanup)

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        logger.info(f"\n{'='*60}\nProcessing: {pdf_path.name}\n{'='*60}")

        # Determine output path
        if args.output:
            out = Path(args.output)
            if out.suffix == ".pdf":
                output_pdf = out.resolve()
            else:
                output_pdf = (out / f"{stem}_translated.pdf").resolve()
        else:
            # Default output lives in the project-root output/ dir (same
            # convention as temp/), not the process CWD.
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_pdf = (output_dir / f"{stem}_translated.pdf").resolve()

        # ── Step 1: Extract with MinerU ─────────────────────────────────────
        import hashlib
        doc_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        extract_temp = temp_dir / f"ext_{ts}_{doc_hash}"

        if args.skip_extraction:
            # Search for an existing extraction dir matching this doc_hash.
            # Dir names sort chronologically, so the last one is the latest.
            existing = sorted(temp_dir.glob(f"ext_*_{doc_hash}"))
            if existing:
                parse_dir = existing[-1]
                logger.info(f"Using existing extraction: {parse_dir}")
            else:
                logger.error(f"No existing extraction found for hash {doc_hash}. Remove --skip-extraction.")
                continue
        else:
            logger.info(f"Step 1/3: Extracting with MinerU ({args.backend})...")
            parse_dir = extract_pdf(
                pdf_path=pdf_path,
                output_dir=extract_temp,
                backend=args.backend,
                method=args.method,
                lang=args.lang,
                formula_enable=not args.no_formula,
                table_enable=not args.no_table,
                start_page=args.start,
                end_page=args.end,
                timeout=args.timeout,
            )
            created_dirs.append(parse_dir)

        # Load structured data
        middle_json = load_middle_json(parse_dir)
        blocks = extract_translatable_blocks(middle_json)
        logger.info(f"Found {len(blocks)} translatable blocks")

        try:
            images_dir = get_images_dir(parse_dir)
        except FileNotFoundError:
            logger.warning("No images directory found — images won't be included")
            images_dir = parse_dir  # fallback

        # ── Step 2: Translate ──────────────────────────────────────────────
        if args.no_translate:
            logger.info("Step 2/3: Skipping translation (--no-translate)")
            translations = {}
        else:
            logger.info(f"Step 2/3: Translating {len(blocks)} blocks...")
            translator = PDFTranslator(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                source_lang=args.lang if args.lang != "ch" else "zh",
                target_lang=args.target_lang,
            )

            translated_blocks = translator.translate_batch(
                blocks,
                batch_delay=args.delay,
            )

            # Build translation map: original_text → translated_text
            translations = {}
            for block in translated_blocks:
                orig = block.get("text", "")
                trans = block.get("translated_text", orig)
                if orig.strip():
                    translations[orig.strip()] = trans

            logger.info(f"Translation map has {len(translations)} entries")

            # Save translation map for reference
            map_path = parse_dir / "translation_map.json"
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            logger.info(f"Translation map saved: {map_path}")

        # ── Step 3: Generate PDF ───────────────────────────────────────────
        logger.info("Step 3/3: Generating translated PDF...")
        middle_json_path = find_middle_json(parse_dir)

        generator = PDFGenerator(
            middle_json_path=middle_json_path,
            images_dir=images_dir,
            output_path=output_pdf,
            title=None,  # Uses original title (translated)
        )
        generator.generate(translations=translations)

        logger.info(f"\n✅ Translated PDF saved: {output_pdf}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    if not args.keep_temp and not args.skip_extraction:
        for d in created_dirs:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"Cleaned up temporary files: {d}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
