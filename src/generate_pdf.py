"""
PDF generation module using reportlab.

Rebuilds a translated PDF from the MinerU middle JSON structure,
replacing original text with translations while preserving images,
captions, and document layout.
"""

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# ── Font Setup ──────────────────────────────────────────────────────────────
# Register CJK fonts for Chinese text rendering.
# Reportlab's built-in fonts don't support CJK — we need a TrueType font.

_FONT_REGISTERED = False


def _register_fonts():
    """Register fonts for CJK support. Looks for system fonts."""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return

    # Try common Windows CJK font paths
    cjk_candidates = [
        ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"),  # 微软雅黑
        ("C:/Windows/Fonts/simsun.ttc", "SimSun"),         # 宋体
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),         # 黑体
        ("C:/Windows/Fonts/msmincho.ttc", "MS Mincho"),    # Japanese
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK"),
        ("/System/Library/Fonts/PingFang.ttc", "PingFang"),  # macOS
    ]

    for font_path, font_name in cjk_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("CJK", font_path))
                pdfmetrics.registerFont(TTFont("CJK-Bold", font_path))  # fallback
                _FONT_REGISTERED = True
                logger.info(f"Registered CJK font: {font_name} ({font_path})")
                return
            except Exception as e:
                logger.warning(f"Failed to register {font_path}: {e}")
                continue

    # Fallback: use built-in Helvetica (no CJK support)
    logger.warning("No CJK font found. Chinese chars will render as blanks. "
                   "Install a Chinese font or set FONT_PATH env var.")
    _FONT_REGISTERED = True


def _get_font_name():
    """Get the best available font name."""
    _register_fonts()
    # Check if CJK was registered
    from reportlab.pdfbase.pdfmetrics import _fonts
    if "CJK" in _fonts:
        return "CJK"
    return "Helvetica"


# ── Style Definitions ───────────────────────────────────────────────────────

def _build_styles(font_name: str) -> dict:
    """Build paragraph styles for the translated PDF."""
    return {
        "text": ParagraphStyle(
            "TranslatedText",
            fontName=font_name,
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,  # 左对齐避免 CJK/拉丁混排时拉伸空白
            spaceAfter=6,
            wordSpace=0,
            firstLineIndent=20,  # 首行缩进 2 字符 (10pt × 2)
        ),
        "title": ParagraphStyle(
            "TranslatedTitle",
            fontName=font_name,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=12,
            spaceBefore=6,
        ),
        "heading1": ParagraphStyle(
            "TranslatedH1",
            fontName=font_name,
            fontSize=14,
            leading=18,
            spaceAfter=8,
            spaceBefore=12,
        ),
        "heading1_centered": ParagraphStyle(
            "TranslatedH1Centered",
            fontName=font_name,
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=8,
            spaceBefore=12,
        ),
        "heading2": ParagraphStyle(
            "TranslatedH2",
            fontName=font_name,
            fontSize=12,
            leading=16,
            spaceAfter=6,
            spaceBefore=10,
        ),
        "heading2_centered": ParagraphStyle(
            "TranslatedH2Centered",
            fontName=font_name,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=6,
            spaceBefore=10,
        ),
        "caption": ParagraphStyle(
            "TranslatedCaption",
            fontName=font_name,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=12,
            spaceBefore=4,
        ),
        "text_centered": ParagraphStyle(
            "TranslatedTextCentered",
            fontName=font_name,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "abstract": ParagraphStyle(
            "TranslatedAbstract",
            fontName=font_name,
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
        ),
        "reference": ParagraphStyle(
            "TranslatedRef",
            fontName=font_name,
            fontSize=8,
            leading=11,
            spaceAfter=2,
        ),
        "header": ParagraphStyle(
            "TranslatedHeader",
            fontName=font_name,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666666"),
        ),
    }


# ── PDF Generator ───────────────────────────────────────────────────────────

class PDFGenerator:
    """Generate a translated PDF from MinerU middle JSON and translations."""

    def __init__(
        self,
        middle_json_path: str | Path,
        images_dir: str | Path,
        output_path: str | Path,
        title: Optional[str] = None,
    ):
        self.middle_json_path = Path(middle_json_path)
        self.images_dir = Path(images_dir)
        self.output_path = Path(output_path)
        self.custom_title = title

        if not self.middle_json_path.exists():
            raise FileNotFoundError(f"Middle JSON not found: {self.middle_json_path}")

        with open(self.middle_json_path, "r", encoding="utf-8") as f:
            self.middle_json = json.load(f)

        self.font_name = _get_font_name()
        self.styles = _build_styles(self.font_name)

    def generate(self, translations: Optional[dict[str, str]] = None):
        """
        Generate the translated PDF.

        Args:
            translations: Mapping of original text → translated text.
                          If None, generate PDF with original text (testing mode).
        """
        if translations is None:
            translations = {}

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            title=self.custom_title or "Translated Document",
        )

        story = self._build_story(translations)
        doc.build(story)
        logger.info(f"PDF generated: {self.output_path}")

    def _build_story(self, translations: dict[str, str]) -> list:
        """Build the flowable story from the middle JSON."""
        story = []

        # Add custom title if provided
        if self.custom_title:
            story.append(Paragraph(self.custom_title, self.styles["title"]))
            story.append(Spacer(1, 6 * mm))

        pdf_info = self.middle_json.get("pdf_info", [])

        for page in pdf_info:
            para_blocks = page.get("para_blocks", [])
            page_idx = page.get("page_idx", 0)
            page_size = page.get("page_size", [612, 792])
            page_width = page_size[0]

            for block in para_blocks:
                elements = self._render_block(block, translations, page_width)
                if elements:
                    story.extend(elements)

        return story

    @staticmethod
    def _is_block_centered(block: dict, page_width: float, threshold: float = 0.12) -> bool:
        """Check if a block's bbox is centered on the page.

        Uses the block's horizontal bbox center relative to the page center.
        A block is considered centered if its center is within `threshold`
        fraction of page width from the page center.

        Args:
            block: The para_block dict with a 'bbox' key [x0, y0, x1, y1].
            page_width: Page width in points.
            threshold: Fraction of page width tolerance (default 0.12 = ±12%).
        """
        bbox = block.get("bbox")
        if not bbox or len(bbox) < 4:
            return False
        block_center_x = (bbox[0] + bbox[2]) / 2
        page_center_x = page_width / 2
        offset = abs(block_center_x - page_center_x) / page_width
        return offset < threshold

    def _render_block(
        self,
        block: dict,
        translations: dict[str, str],
        page_width: float = 612,
    ) -> list:
        """Render a single para_block into a list of flowable elements."""
        block_type = block.get("type", "")
        elements = []

        # ── Handle compound blocks (image + caption, table + caption, etc.) ──
        if block_type in ("image", "chart"):
            # Image/chart block has sub-blocks: image_body, image_caption, image_footnote
            return self._render_compound_block(block, translations)

        if block_type == "table":
            return self._render_compound_block(block, translations)

        # ── Handle simple text blocks ──
        if block_type in ("text", "list", "index", "ref_text", "abstract"):
            text = self._extract_text_from_block(block)
            translated = translations.get(text, text)
            style = self.styles.get(block_type, self.styles["text"])
            if block_type == "abstract":
                style = self.styles["abstract"]
            elif block_type == "ref_text":
                style = self.styles["reference"]
            elif block_type == "text":
                # Narrow + bbox-centered → author/affiliation/etc.
                bbox = block.get("bbox")
                if bbox and len(bbox) >= 4:
                    block_w = bbox[2] - bbox[0]
                    if (block_w / page_width < 0.5 and
                            self._is_block_centered(block, page_width)):
                        style = self.styles["text_centered"]
            if translated.strip():
                elements.append(Paragraph(
                    self._escape_xml_preserve_tags(self._process_inline_formulas(translated)), style
                ))

        elif block_type in ("title", "doc_title", "paragraph_title"):
            text = self._extract_text_from_block(block)
            translated = translations.get(text, text)
            level = block.get("level", 1)
            # title/doc_title are always centered (chapter/section headings).
            # paragraph_title uses bbox detection (some are left-aligned).
            if block_type == "paragraph_title":
                is_centered = self._is_block_centered(block, page_width)
            else:
                is_centered = True  # title, doc_title → always centered
            if level <= 1:
                style_key = "heading1_centered" if is_centered else "heading1"
            else:
                style_key = "heading2_centered" if is_centered else "heading2"
            style = self.styles[style_key]
            if translated.strip():
                elements.append(Paragraph(
                    self._escape_xml_preserve_tags(self._process_inline_formulas(translated)), style
                ))

        elif block_type == "interline_equation":
            # Render display equation as MinerU-extracted image (no reconstruction)
            img = self._render_image(block)
            if img:
                img.hAlign = 'CENTER'
                elements.append(Spacer(1, 3 * mm))
                elements.append(img)
                elements.append(Spacer(1, 3 * mm))
            else:
                logger.warning(
                    f"No image found for interline_equation on page "
                    f"{block.get('page_idx', '?')} — equation skipped"
                )

        elif block_type == "header":
            text = self._extract_text_from_block(block)
            translated = translations.get(text, text)
            if translated.strip():
                elements.append(Paragraph(
                    self._escape_xml_preserve_tags(self._process_inline_formulas(translated)), self.styles["header"]
                ))

        elif block_type == "footer":
            text = self._extract_text_from_block(block)
            translated = translations.get(text, text)
            if translated.strip():
                elements.append(Paragraph(
                    self._escape_xml_preserve_tags(self._process_inline_formulas(translated)), self.styles["header"]
                ))

        elif block_type == "code":
            code_text = self._extract_text_from_block(block)
            if code_text.strip():
                elements.append(Paragraph(
                    f"<pre>{self._escape_xml_preserve_tags(code_text)}</pre>",
                    self.styles["reference"],
                ))

        # Default: render as text
        else:
            text = self._extract_text_from_block(block)
            if text.strip():
                translated = translations.get(text, text)
                elements.append(Paragraph(
                    self._escape_xml_preserve_tags(self._process_inline_formulas(translated)), self.styles["text"]
                ))

        return elements

    def _render_compound_block(
        self,
        block: dict,
        translations: dict[str, str],
    ) -> list:
        """Render a compound block (image/table with caption)."""
        elements = []
        sub_elements = []

        for sub_block in block.get("blocks", []):
            sub_type = sub_block.get("type", "")

            if sub_type in ("image_body", "chart_body"):
                img = self._render_image(sub_block)
                if img:
                    sub_elements.append(img)

            elif sub_type == "table_body":
                # Render table as MinerU-extracted image (no HTML reconstruction)
                img = self._render_image(sub_block)
                if img:
                    sub_elements.append(img)
                else:
                    logger.warning(
                        f"No image found for table_body — table skipped"
                    )

            elif sub_type in (
                "image_caption", "chart_caption", "table_caption",
                "algorithm_caption", "code_caption",
            ):
                text = self._extract_text_from_block(sub_block)
                translated = translations.get(text, text)
                if translated.strip():
                    sub_elements.append(Paragraph(
                        self._escape_xml_preserve_tags(self._process_inline_formulas(translated)), self.styles["caption"]
                    ))

            elif sub_type in (
                "image_footnote", "table_footnote", "chart_footnote",
                "code_footnote",
            ):
                text = self._extract_text_from_block(sub_block)
                translated = translations.get(text, text)
                if translated.strip():
                    sub_elements.append(Paragraph(
                        self._escape_xml_preserve_tags(self._process_inline_formulas(translated)), self.styles["caption"]
                    ))

            elif sub_type == "code_body":
                code_text = self._extract_text_from_block(sub_block)
                if code_text.strip():
                    sub_elements.append(Paragraph(
                        f"<pre>{self._escape_xml_preserve_tags(code_text)}</pre>",
                        self.styles["reference"],
                    ))

        # Keep the whole figure (image + caption) together when possible
        if sub_elements:
            elements.append(KeepTogether(sub_elements))
            elements.append(Spacer(1, 4 * mm))

        return elements

    def _render_image(self, sub_block: dict):
        """Render an image from an image_body sub-block."""
        for line in sub_block.get("lines", []):
            for span in line.get("spans", []):
                image_path = span.get("image_path", "")
                if not image_path:
                    continue

                # Determine full path
                full_path = self.images_dir / image_path
                if not full_path.exists():
                    # Try without images_dir prefix
                    alt_path = self.images_dir / Path(image_path).name
                    if alt_path.exists():
                        full_path = alt_path
                    else:
                        logger.warning(f"Image not found: {full_path}")
                        return None

                try:
                    # Get image dimensions
                    with PILImage.open(full_path) as pil_img:
                        img_w, img_h = pil_img.size

                    # Scale to fit within page
                    max_w = 150 * mm  # ~6 inches
                    max_h = 120 * mm  # ~4.7 inches
                    scale = min(max_w / img_w, max_h / img_h, 1.0)
                    display_w = img_w * scale
                    display_h = img_h * scale

                    return Image(
                        str(full_path),
                        width=display_w,
                        height=display_h,
                    )
                except Exception as e:
                    logger.warning(f"Cannot render image {full_path}: {e}")
                    return None

        return None

    def _extract_text_from_block(self, block: dict) -> str:
        """Extract text from a block, matching extract.py's logic for translation lookup.

        - text spans: joined with spaces
        - inline_equation spans: compacted LaTeX wrapped in $...$
        - other spans: left as-is
        """
        import re as _re
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
                    # Compact LaTeX (same logic as extract.py)
                    s = str(content)
                    s = s.replace(' ~ ', ' ').replace('~ ', ' ').replace(' ~', ' ')
                    s = _re.sub(r'([_^])\s*\{\s*([^}]*?)\s*\}', r'\1{\2}', s)
                    for _ in range(5):
                        prev = s
                        s = _re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', s)
                        s = _re.sub(r'(\d)\s+(\d)', r'\1\2', s)
                        if s == prev:
                            break
                    s = _re.sub(r'\s{2,}', ' ', s).strip()
                    result_parts.append(f"${s}$")
                elif span_type == "text":
                    result_parts.append(str(content))
                else:
                    result_parts.append(str(content))
        return " ".join(result_parts)

    # ── LaTeX → Unicode Math Converter ─────────────────────────────────────────
    # Converts inline LaTeX math formulas to Unicode for clean PDF display.
    # Display (interline) equations are rendered as images via matplotlib.

    # Mapping: LaTeX command → Unicode character
    _GREEK_UNICODE = {
        # Lowercase
        'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
        'epsilon': 'ϵ', 'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η',
        'theta': 'θ', 'vartheta': 'ϑ', 'iota': 'ι', 'kappa': 'κ',
        'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ',
        'pi': 'π', 'varpi': 'ϖ', 'rho': 'ρ', 'varrho': 'ϱ',
        'sigma': 'σ', 'varsigma': 'ς', 'tau': 'τ', 'upsilon': 'υ',
        'phi': 'φ', 'varphi': 'ϕ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
        # Uppercase (only those that differ from Latin)
        'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
        'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'ϒ',
        'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
    }
    _MATH_SYMBOLS = {
        'approx': '≈', 'times': '×', 'pm': '±', 'mp': '∓',
        'cdot': '·', 'div': '÷', 'leq': '≤', 'geq': '≥',
        'le': '≤', 'ge': '≥',
        'neq': '≠', 'equiv': '≡', 'sim': '∼', 'simeq': '≃',
        'propto': '∝', 'infty': '∞', 'partial': '∂', 'nabla': '∇',
        'int': '∫', 'sum': '∑', 'prod': '∏', 'sqrt': '√',
        'angle': '∠', 'perp': '⊥', 'parallel': '∥',
        'rightarrow': '→', 'to': '→', 'leftarrow': '←',
        'Rightarrow': '⇒', 'Leftarrow': '⇐',
        'leftrightarrow': '↔', 'uparrow': '↑', 'downarrow': '↓',
        'langle': '⟨', 'rangle': '⟩', 'lceil': '⌈', 'rceil': '⌉',
        'lfloor': '⌊', 'rfloor': '⌋', 'ldots': '…', 'cdots': '⋯',
        'vdots': '⋮', 'ddots': '⋱', 'circ': '∘', 'bullet': '•',
        'oplus': '⊕', 'otimes': '⊗', 'ominus': '⊖', 'odot': '⊙',
        'wedge': '∧', 'vee': '∨', 'neg': '¬', 'forall': '∀',
        'exists': '∃', 'in': '∈', 'notin': '∉', 'subset': '⊂',
        'supset': '⊃', 'subseteq': '⊆', 'supseteq': '⊇',
        'cup': '∪', 'cap': '∩', 'setminus': '∖', 'emptyset': '∅',
        'cong': '≅', 'triangle': '△', 'Box': '□',
    }
    @classmethod
    def _latex_to_unicode(cls, latex: str) -> str:
        """Convert a LaTeX math expression to Unicode approximation.

        Handles Greek letters, math symbols, subscripts, superscripts,
        fractions, and basic math formatting.
        """
        import re as _re
        s = str(latex).strip()

        # Remove outer $$ or $ if present
        s = _re.sub(r'^\$\$?(.*?)\$\$?$', r'\1', s)

        # Replace \operatorname{...} with just the content
        s = _re.sub(r'\\operatorname\s*\{([^}]*)\}', r'\1', s)
        s = _re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', s)
        s = _re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)

        # Strip \tag{...} before processing (not renderable in Unicode)
        s = _re.sub(r'\\tag\s*\{[^}]*\}', '', s)
        # \stackrel{a}{b} → a/b
        s = _re.sub(r'\\stackrel\s*\{([^}]*)\}\s*\{([^}]*)\}',
                     r'\1/\2', s)

        # Greek letters (longest names first to avoid partial matches)
        # Use (?![a-zA-Z]) instead of \b — \b fails before _ since _ is a word char
        greek_sorted = sorted(cls._GREEK_UNICODE.items(), key=lambda x: -len(x[0]))
        for cmd, char in greek_sorted:
            s = _re.sub(r'\\' + cmd + r'(?![a-zA-Z])', char, s)

        # Math symbols (longest names first)
        math_sorted = sorted(cls._MATH_SYMBOLS.items(), key=lambda x: -len(x[0]))
        for cmd, char in math_sorted:
            s = _re.sub(r'\\' + cmd + r'(?![a-zA-Z])', char, s)

        # Fractions: \frac{a}{b} → (a)/(b)  (simple inline rendering)
        s = _re.sub(r'\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}',
                     r'(\1)/(\2)', s)

        # Square roots: \sqrt[n]{x} or \sqrt{x}
        s = _re.sub(r'\\sqrt\s*\[([^\]]*)\]\s*\{([^}]*)\}',
                     r'\1⎯√\2', s)  # nth root
        s = _re.sub(r'\\sqrt\s*\{([^}]*)\}',
                     r'√(\1)', s)

        # Subscripts: _{abc} → <sub>abc</sub> (reportlab XML tags)
        # We use XML tags instead of Unicode subscripts because CJK fonts
        # (Microsoft YaHei, etc.) lack Unicode subscript characters.
        def _matched_brace_group(s, start):
            """Find content inside matching braces starting at 'start'.
            Returns (content, end_pos) or (None, -1) if not on '{'.
            """
            if start >= len(s) or s[start] != '{':
                return None, -1
            depth = 0
            for i in range(start, len(s)):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start+1:i], i
            return s[start+1:], len(s)  # unclosed — return rest

        def _replace_sub_sup_with_braces(s, prefix, tag):
            """Replace _ { ... } or ^ { ... } with <tag>...</tag>,
            handling nested braces correctly."""
            result = []
            i = 0
            while i < len(s):
                # Look for _ or ^ followed by optional spaces and {
                if s[i] == prefix and i + 1 < len(s):
                    j = i + 1
                    # Skip whitespace between prefix and brace
                    while j < len(s) and s[j] == ' ':
                        j += 1
                    if j < len(s) and s[j] == '{':
                        inner, end = _matched_brace_group(s, j)
                        if inner is not None:
                            # Clean LaTeX commands from inner
                            inner = _re.sub(r'\\[a-zA-Z]+', '', inner)
                            result.append(f'<{tag}>{inner.strip()}</{tag}>')
                            i = end + 1
                            continue
                result.append(s[i])
                i += 1
            return ''.join(result)

        s = _replace_sub_sup_with_braces(s, '_', 'sub')
        s = _replace_sub_sup_with_braces(s, '^', 'super')

        # Simple subscript/superscript without braces: _a → <sub>a</sub>
        def _simple_sub_repl(m):
            return f'<sub>{m.group(1)}</sub>'
        s = _re.sub(r'_\s*(\S)', _simple_sub_repl, s)

        def _simple_sup_repl(m):
            return f'<super>{m.group(1)}</super>'
        s = _re.sub(r'\^\s*(\S)', _simple_sup_repl, s)

        # Remove remaining { } grouping braces
        s = s.replace('{', '').replace('}', '')

        # Clean up: remove spaces around parentheses and punctuation
        s = _re.sub(r'\(\s+', '(', s)
        s = _re.sub(r'\s+\)', ')', s)
        s = _re.sub(r'\[\s+', '[', s)
        s = _re.sub(r'\s+\]', ']', s)

        # Remove stray backslashes before letters
        s = _re.sub(r'\\(?=[a-zA-Z])', '', s)

        # Collapse multiple spaces to one
        s = _re.sub(r'\s{2,}', ' ', s)

        return s.strip()

    @classmethod
    def _process_inline_formulas(cls, text: str) -> str:
        """Convert $...$ LaTeX formulas in text to Unicode math.

        Also strips residual HTML tags and inserts non-breaking spaces
        between digits and Latin/Greek letters to prevent line breaks
        between numbers and their units (e.g., "400 μm").
        """
        import re as _re

        s = str(text)

        # Strip residual HTML tags (MinerU sometimes outputs <sup>/<sub>)
        s = _re.sub(r'<[^>]+>', '', s)

        NBSP = '\xa0'  # non-breaking space, prevents line breaks

        def _convert_formula(m):
            latex = m.group(1)
            unicode_formula = cls._latex_to_unicode(latex)
            # Replace ALL spaces within a formula with NBSP to prevent
            # line breaks inside number-unit pairs (e.g., "400 μm")
            unicode_formula = unicode_formula.replace(' ', NBSP)
            return unicode_formula

        # Replace $...$ formulas with Unicode math
        result = _re.sub(r'\$([^$]+?)\$', _convert_formula, s)

        # Also prevent breaks between digits and adjacent units in
        # non-formula translated text (e.g., "4-5 μm", "170 µm")
        result = _re.sub(
            r'(\d)\s+([a-zA-Zα-ωμµε])',
            r'\1' + NBSP + r'\2', result,
        )

        # Prevent breaks within short English phrases (2-4 words, ≥2 chars each).
        # Joins "fused silica", "low temperature cofired ceramic", etc.
        # Stops at punctuation, digits, or single-letter words.
        result = _re.sub(
            r'\b([a-zA-Z]{2,}(?:\s+[a-zA-Z]{2,}){1,3})\b',
            lambda m: m.group(0).replace(' ', NBSP),
            result,
        )

        return result

    @staticmethod
    def _escape_xml_preserve_tags(text: str) -> str:
        """Escape XML special chars but preserve reportlab tags: <sub>, <super>.

        After _latex_to_unicode converts LaTeX subscripts/superscripts to
        <sub>/<super> XML tags, this method escapes everything else so the
        tags survive into the reportlab Paragraph renderer.
        """
        import re as _re
        if not text:
            return ""

        placeholders = {}
        counter = [0]

        def _protect(m):
            key = f'\x00TAG{counter[0]}\x00'
            counter[0] += 1
            placeholders[key] = m.group(0)
            return key

        # Protect <sub>, </sub>, <super>, </super> tags
        text = _re.sub(r'</?su(?:b|per)>', _protect, text)

        # Escape XML
        text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

        # Restore protected tags
        for key, tag in placeholders.items():
            text = text.replace(key, tag)

        return text

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape XML special characters for reportlab Paragraph."""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


if __name__ == "__main__":
    # Test: generate a PDF using original text (no translation)
    import sys
    parse_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("temp/extract_test")
    output = sys.argv[2] if len(sys.argv) > 2 else "output/test.pdf"

    # Find files
    middle_jsons = list(parse_dir.glob("middle.json")) or list(parse_dir.glob("*_middle.json"))
    if not middle_jsons:
        print(f"No middle.json found in {parse_dir}")
        sys.exit(1)

    images_dir = parse_dir / "images"
    if not images_dir.exists():
        # Try subdirectory
        for d in parse_dir.iterdir():
            if d.is_dir() and d.name == "images":
                images_dir = d
                break

    gen = PDFGenerator(
        middle_json_path=middle_jsons[0],
        images_dir=images_dir,
        output_path=output,
        title="Test Generated PDF",
    )
    gen.generate()  # Uses original text
