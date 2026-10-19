"""Turn a PDF or text file into page-tagged text. Never silently drops pages."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

PAGE_MARK = re.compile(r"\[\[PAGE (\d+)\]\]")
MIN_TEXT_CHARS = 30  # below this a PDF page is treated as having no text layer


@dataclass
class Document:
    source: str
    pages: list[str]
    warnings: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)


def _ocr_page(pdf_path: Path, page_index: int):
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return None
    try:
        images = convert_from_path(
            str(pdf_path), first_page=page_index + 1, last_page=page_index + 1, dpi=300
        )
        return pytesseract.image_to_string(images[0]) if images else None
    except Exception:  # missing tesseract/poppler binaries, corrupt page, ...
        return None


def _load_pdf(path: Path) -> Document:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber is required to read PDFs") from exc

    doc = Document(source=str(path), pages=[])
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = (page.extract_text() or "").strip()
            if len(txt) < MIN_TEXT_CHARS:
                ocr = _ocr_page(path, i)
                if ocr and len(ocr.strip()) >= MIN_TEXT_CHARS:
                    txt = ocr.strip()
                    doc.ocr_pages.append(i + 1)
                else:
                    doc.warnings.append(
                        f"page {i + 1}: no text layer and OCR unavailable/empty; content missing"
                    )
            doc.pages.append(f"[[PAGE {i + 1}]]\n{txt}")
    if not doc.pages:
        doc.warnings.append("PDF has no pages")
    return doc


def _load_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parts = PAGE_MARK.split(raw)
    if len(parts) > 1:
        # split() yields [pre, num, body, num, body, ...]
        pages = []
        pre = parts[0].strip()
        if pre:
            pages.append(f"[[PAGE 0]]\n{pre}")
        for num, body in zip(parts[1::2], parts[2::2]):
            pages.append(f"[[PAGE {num}]]\n{body.strip()}")
        return Document(source=str(path), pages=pages)
    return Document(source=str(path), pages=[f"[[PAGE 1]]\n{raw.strip()}"])


def load_document(path) -> Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md", ".text"}:
        return _load_text(path)
    raise ValueError(f"unsupported file type: {suffix} (use .pdf or .txt)")
