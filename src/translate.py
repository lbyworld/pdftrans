"""
Translation module using OpenAI-compatible API.

Translates extracted text blocks while preserving document structure,
equation formatting, and scientific notation.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI


def _get_project_root() -> Path:
    """Find the project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def _load_env():
    """Load .env from project root if not already loaded."""
    env_path = _get_project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)


class PDFTranslator:
    """Translate PDF text content using OpenAI API."""

    # Patterns to protect during translation
    MATH_PATTERN = re.compile(
        r'(\$\$[^$]+\$\$|\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\))',
        re.DOTALL,
    )
    URL_PATTERN = re.compile(r'https?://\S+')
    CITATION_PATTERN = re.compile(r'\[\d+(?:,\s*\d+)*\]')  # [1,2,3] or [42]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "chinese",
    ):
        _load_env()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.source_lang = source_lang
        self.target_lang = target_lang

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Provide api_key, set env var, "
                "or create .env file with OPENAI_API_KEY=..."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(
            f"PDFTranslator initialized: model={self.model}, "
            f"base_url={self.base_url}, "
            f"translate: {source_lang} → {target_lang}"
        )

    def translate_block(self, text: str, block_type: str = "text") -> str:
        """Translate a single text block."""
        # Short or empty text
        if not text or len(text.strip()) < 2:
            return text

        # Protect math, URLs, and citations
        placeholders = {}
        protected = self._protect_special(text, placeholders)

        system_prompt = self._build_system_prompt(block_type)
        user_prompt = protected

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this {self.source_lang} text to {self.target_lang}:\n\n{user_prompt}"},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            translated = response.choices[0].message.content.strip()
            translated = self._restore_special(translated, placeholders)
            return translated

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    def translate_batch(
        self,
        blocks: list[dict],
        batch_delay: float = 0.5,
    ) -> list[dict]:
        """
        Translate a list of text blocks.

        Args:
            blocks: List of dicts with 'text' and 'block_type' keys.
            batch_delay: Seconds between API calls to avoid rate limits.

        Returns:
            The same list with 'translated_text' added to each block.
        """
        translated_blocks = []
        total = len(blocks)

        for i, block in enumerate(blocks):
            text = block.get("text", "")
            block_type = block.get("block_type", "text")

            if not text.strip() or self._should_skip(text):
                block["translated_text"] = text
                translated_blocks.append(block)
                continue

            try:
                logger.info(f"[{i+1}/{total}] Translating [{block_type}]: {text[:60]}...")
                translated = self.translate_block(text, block_type)
                block["translated_text"] = translated
                translated_blocks.append(block)

                if i < total - 1 and batch_delay > 0:
                    time.sleep(batch_delay)

            except Exception as e:
                logger.error(f"Translation error for block {i}: {e}")
                block["translated_text"] = text  # Fallback to original
                translated_blocks.append(block)

        logger.info(f"Translated {total} blocks")
        return translated_blocks

    def _build_system_prompt(self, block_type: str) -> str:
        """Build system prompt with translation instructions."""
        base = (
            f"You are a professional {self.source_lang}-to-{self.target_lang} translator "
            f"specializing in scientific and technical documents. "
            f"Translate the text accurately and fluently. "
            f"Preserve all LaTeX math expressions (marked with PLACEHOLDER_MATH_N). "
            f"Preserve all URLs (marked with PLACEHOLDER_URL_N). "
            f"Preserve citation numbers like [1], [2,3] (marked with PLACEHOLDER_CITE_N). "
            f"Output ONLY the translation, no explanations."
        )

        if block_type in ("title", "doc_title", "paragraph_title"):
            base += " This is a section TITLE — maintain concise, formal academic style."
        elif "caption" in block_type:
            base += " This is a figure/table CAPTION — keep it concise."
        elif block_type == "ref_text":
            base += " This is a REFERENCE — preserve author names, publication details exactly."
        elif block_type == "abstract":
            base += " This is an ABSTRACT — maintain formal academic style."

        return base

    def _protect_special(self, text: str, placeholders: dict) -> str:
        """Replace math, URLs, and citations with placeholders."""
        counter = [0]

        def _replace_math(m):
            key = f"PLACEHOLDER_MATH_{counter[0]}"
            placeholders[key] = m.group(0)
            counter[0] += 1
            return key

        def _replace_url(m):
            key = f"PLACEHOLDER_URL_{counter[0]}"
            placeholders[key] = m.group(0)
            counter[0] += 1
            return key

        def _replace_cite(m):
            key = f"PLACEHOLDER_CITE_{counter[0]}"
            placeholders[key] = m.group(0)
            counter[0] += 1
            return key

        text = self.MATH_PATTERN.sub(_replace_math, text)
        text = self.URL_PATTERN.sub(_replace_url, text)
        text = self.CITATION_PATTERN.sub(_replace_cite, text)
        return text

    def _restore_special(self, text: str, placeholders: dict) -> str:
        """Restore placeholders back to original content."""
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text

    def _should_skip(self, text: str) -> bool:
        """Check if text should be skipped (pure math, numbers, etc.)."""
        stripped = text.strip()
        # Skip pure math expressions
        if stripped.startswith("$$") and stripped.endswith("$$"):
            return True
        if stripped.startswith("$") and stripped.endswith("$"):
            return True
        # Skip URLs
        if self.URL_PATTERN.fullmatch(stripped):
            return True
        # Skip numeric-only
        if re.fullmatch(r'[\d\s.,+\-*/=()\[\]]+', stripped):
            return True
        return False


if __name__ == "__main__":
    translator = PDFTranslator()
    test_texts = [
        "The rapid advancement of heterogeneous integration technologies.",
        "Fig. 1: Schematic of the proposed die-to-die bridge architecture.",
        "We demonstrate a novel approach to $\\lambda/4$ impedance matching.",
    ]
    for t in test_texts:
        result = translator.translate_block(t)
        print(f"  EN: {t}")
        print(f"  ZH: {result}")
        print()
