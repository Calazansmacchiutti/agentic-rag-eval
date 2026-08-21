"""OCR para PDFs digitalizados (sem texto extraivel).

Renderiza cada pagina em imagem (PyMuPDF) e roda Tesseract (pytesseract). Devolve um
Block de texto por pagina. Sem info de fonte, entao docs OCR usam recorte por pagina/bloco.

Dependencias: `pytesseract` + `pillow` (extra `ocr` no pyproject) MAIS o binario do
**Tesseract** instalado no sistema (nao vem via pip). `available()` degrada gracioso.
"""
from __future__ import annotations

import logging

from agentic_rag.pdf.schemas import Block

logger = logging.getLogger(__name__)


def available() -> bool:
    """True se pytesseract E o binario do Tesseract estiverem acessiveis."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        logger.debug("Tesseract indisponivel; OCR desligado", exc_info=True)
        return False


def ocr_blocks(path: str, dpi: int = 200, lang: str = "por+eng") -> list[Block]:
    """Um Block de texto por pagina, via OCR. Requer `available()` True."""
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(path)
    blocks: list[Block] = []
    try:
        for pno, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = pytesseract.image_to_string(img, lang=lang).strip()
            if text:
                blocks.append(
                    Block(
                        page=pno,
                        type="text",
                        text=text,
                        bbox=(0.0, 0.0, float(pix.width), float(pix.height)),
                        n_lines=text.count("\n") + 1,
                    )
                )
    finally:
        doc.close()
    return blocks
