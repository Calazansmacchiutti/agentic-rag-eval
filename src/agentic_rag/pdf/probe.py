"""Base analitica deterministica: le sinais de layout reais do PDF.

Roda ANTES de qualquer LLM. Extrai blocos (texto/tabela/imagem) com fonte/posicao,
agrega em LayoutSignals e deriva um DocProfile (o "comportamento" do PDF) com a
estrategia de recorte recomendada. O agente usa esse perfil como ponto de partida e
o LLM so refina depois.

Dependencia: PyMuPDF (`pip install pymupdf`). Import preguicoso (pesado p/ subir).
"""
from __future__ import annotations

from collections import Counter

from agentic_rag.pdf.schemas import Block, DocProfile, LayoutSignals

_BOLD_FLAG = 1 << 4  # bit de negrito nos span["flags"] do PyMuPDF


def _open(path: str):
    import fitz  # PyMuPDF

    return fitz.open(path)


def _dominant_font(spans: list[dict]) -> tuple[float | None, bool]:
    """Tamanho de fonte dominante (ponderado por nro de chars) e se o bloco e negrito."""
    if not spans:
        return None, False
    sizes: Counter[float] = Counter()
    bold_chars = 0
    total_chars = 0
    for s in spans:
        n = len(s.get("text", ""))
        if n == 0:
            continue
        sizes[round(float(s.get("size", 0)), 1)] += n
        total_chars += n
        if int(s.get("flags", 0)) & _BOLD_FLAG:
            bold_chars += n
    if not sizes:
        return None, False
    dominant = sizes.most_common(1)[0][0]
    return dominant, (total_chars > 0 and bold_chars / total_chars > 0.6)


def _center_inside(bbox: tuple, region: tuple) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _page_tables(page) -> list[tuple[tuple, str]]:
    """Tabelas da pagina como (bbox, texto_em_markdown). Degrada p/ [] sem find_tables."""
    out = []
    try:
        for t in page.find_tables().tables:
            rows = t.extract()
            text = "\n".join(" | ".join((c or "").strip() for c in row) for row in rows)
            if text.strip():
                out.append((tuple(t.bbox), text))
    except Exception:
        pass
    return out


def extract_blocks(path: str) -> list[Block]:
    """Extrai blocos de layout de todas as paginas (texto, tabela e imagem).

    Tabelas viram Blocks proprios (type='table'); blocos de texto que caem DENTRO de uma
    tabela sao suprimidos para nao duplicar conteudo.
    """
    doc = _open(path)
    blocks: list[Block] = []
    try:
        for pno, page in enumerate(doc):
            tables = _page_tables(page)
            for bbox, ttext in tables:
                blocks.append(
                    Block(page=pno, type="table", text=ttext, bbox=bbox,
                          n_lines=ttext.count("\n") + 1)
                )
            data = page.get_text("dict")
            for b in data.get("blocks", []):
                if b.get("type") == 1:  # imagem
                    blocks.append(Block(page=pno, type="image", bbox=tuple(b["bbox"]), text=""))
                    continue
                bbox = tuple(b["bbox"])
                if any(_center_inside(bbox, reg) for reg, _ in tables):
                    continue  # texto dentro de tabela ja foi capturado acima
                lines = b.get("lines", [])
                spans = [s for ln in lines for s in ln.get("spans", [])]
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size, bold = _dominant_font(spans)
                blocks.append(
                    Block(page=pno, type="text", text=text, bbox=bbox,
                          font_size=size, bold=bold, n_lines=len(lines))
                )
    finally:
        doc.close()
    return blocks


def _count_tables(path: str) -> int:
    """Conta tabelas via PyMuPDF find_tables (disponivel em versoes recentes)."""
    doc = _open(path)
    n = 0
    try:
        for page in doc:
            try:
                n += len(page.find_tables().tables)
            except Exception:
                pass  # versao sem find_tables: tabelas ficam como 0 (heuristica degrada graciosamente)
    finally:
        doc.close()
    return n


def _multi_column(blocks: list[Block], page_widths: dict[int, float]) -> bool:
    """Heuristica simples de 2 colunas: blocos consistentemente a esquerda E a direita."""
    left = right = 0
    for b in blocks:
        w = page_widths.get(b.page)
        if not w or b.type != "text":
            continue
        cx = (b.bbox[0] + b.bbox[2]) / 2
        if cx < w * 0.45:
            left += 1
        elif cx > w * 0.55:
            right += 1
    return left >= 3 and right >= 3 and min(left, right) / max(left, right) > 0.4


def analyze(path: str) -> LayoutSignals:
    """Agrega os sinais de layout do documento inteiro (deterministico)."""
    doc = _open(path)
    n_pages = doc.page_count
    page_widths = {i: doc[i].rect.width for i in range(n_pages)}
    has_toc = bool(doc.get_toc())
    doc.close()

    blocks = extract_blocks(path)
    text_blocks = [b for b in blocks if b.type == "text"]
    total_chars = sum(len(b.text) for b in text_blocks)

    # fonte de corpo = tamanho mais frequente ponderado por chars; headings = acima dela
    size_weight: Counter[float] = Counter()
    for b in text_blocks:
        if b.font_size:
            size_weight[b.font_size] += len(b.text)
    body = size_weight.most_common(1)[0][0] if size_weight else None
    heading_sizes = sorted(
        {s for s in size_weight if body and s > body * 1.15}, reverse=True
    )[:4]

    n_tables = _count_tables(path)

    return LayoutSignals(
        n_pages=n_pages,
        n_blocks=len(blocks),
        has_toc=has_toc,
        has_tables=n_tables > 0,
        n_tables=n_tables,
        is_scanned=(n_pages > 0 and total_chars / n_pages < 50),  # quase sem texto -> OCR
        multi_column=_multi_column(text_blocks, page_widths),
        body_font_size=body,
        heading_font_sizes=heading_sizes,
        avg_blocks_per_page=(len(blocks) / n_pages if n_pages else 0.0),
        text_density=(total_chars / n_pages if n_pages else 0.0),
    )


def load_blocks(path: str, ocr: str = "auto") -> list[Block]:
    """Blocos de texto, usando OCR quando o PDF for digitalizado.

    ocr: 'off' (nunca), 'on' (sempre), 'auto' (OCR so se o doc parecer digitalizado E o
    Tesseract estiver disponivel). Degrada para o texto nativo se o OCR nao rolar.
    """
    if ocr == "off":
        return extract_blocks(path)

    want = ocr == "on" or (ocr == "auto" and analyze(path).is_scanned)
    if want:
        from agentic_rag.pdf import ocr as ocr_mod

        if ocr_mod.available():
            blks = ocr_mod.ocr_blocks(path)
            if blks:
                return blks
    return extract_blocks(path)


def profile(path: str) -> DocProfile:
    """Deriva o comportamento do PDF + a estrategia de recorte recomendada (heuristica).

    Heuristica auditavel (registrada em `rationale`). O agente pode sobrescrever via LLM,
    mas comeca daqui para nao chunkar no escuro.
    """
    s = analyze(path)

    richness = min(len(s.heading_font_sizes) / 3.0, 1.0)
    table_heaviness = min(s.n_tables / max(s.n_pages, 1), 1.0)

    if s.is_scanned:
        dtype, strat, why = "scanned", "by_page", "texto extraivel quase nulo: precisa de OCR antes"
        return DocProfile(
            doc_type=dtype, structure_richness=richness, table_heaviness=table_heaviness,
            recommended_strategy=strat, needs_ocr=True, rationale=why, signals=s,
        )
    if table_heaviness >= 0.5:
        dtype, strat, why = "table_heavy", "table_aware", "alta densidade de tabelas por pagina"
    elif s.avg_blocks_per_page <= 6 and s.heading_font_sizes and s.text_density < 600:
        dtype, strat, why = "slides", "by_page", "poucos blocos/pagina + titulos grandes + baixa densidade"
    elif richness >= 0.66 and s.text_density >= 800:
        dtype, strat, why = "article", "by_heading", "hierarquia de headings clara e texto denso"
    elif s.has_tables and s.heading_font_sizes:
        dtype, strat, why = "report", "table_aware", "secoes com headings + tabelas presentes"
    else:
        dtype, strat, why = "unknown", "by_block", "sem sinal forte de estrutura: cortar por bloco"

    return DocProfile(
        doc_type=dtype,
        structure_richness=richness,
        table_heaviness=table_heaviness,
        recommended_strategy=strat,
        needs_ocr=False,
        rationale=why,
        signals=s,
    )
