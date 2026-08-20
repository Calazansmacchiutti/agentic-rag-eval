"""Aplica um CutPlan aos blocos do PDF e devolve os Segmentos.

Deterministico. Implementa as estrategias de recorte; respeita headings e tamanho.
table_aware/by_section caem em by_heading/by_block nesta v1 (ver TODO).
"""
from __future__ import annotations

from agentic_rag.pdf.schemas import Block, CutPlan, DocProfile, Segment

_TOL = 0.3  # tolerancia de comparacao de tamanho de fonte


def _heading_level(b: Block, heading_sizes: list[float], body: float | None) -> int | None:
    """Nivel de heading do bloco (1 = maior). None se for corpo."""
    if b.type != "text" or not b.font_size or not heading_sizes:
        return None
    for lvl, hs in enumerate(heading_sizes, start=1):  # heading_sizes vem ordenado desc
        if abs(b.font_size - hs) <= _TOL:
            return lvl
    # negrito notavelmente maior que o corpo tambem conta como heading de ultimo nivel
    if body and b.bold and b.font_size > body * 1.15:
        return len(heading_sizes) + 1
    return None


def _apply_overlap(segments: list[Segment], overlap: int) -> None:
    """Prefixa o rabo do segmento anterior quando ambos sao da MESMA secao (heading_path)."""
    if overlap <= 0:
        return
    for i in range(1, len(segments)):
        prev, cur = segments[i - 1], segments[i]
        if prev.heading_path == cur.heading_path and prev.type == "text" == cur.type:
            tail = prev.text[-overlap:]
            cur.text = tail + " " + cur.text
            cur.n_chars = len(cur.text)


def segment(blocks: list[Block], profile: DocProfile, plan: CutPlan) -> list[Segment]:
    """Recorta os blocos conforme o plano. Retorna segmentos em ordem de leitura."""
    body = profile.signals.body_font_size
    heading_sizes = profile.signals.heading_font_sizes
    segments: list[Segment] = []
    buf: list[str] = []
    pages: list[int] = []
    stack: list[tuple[int, str]] = []  # (nivel, titulo) ativo
    idx = 0

    def flush() -> None:
        nonlocal idx, buf, pages
        text = "\n".join(p for p in buf if p).strip()
        if text:
            segments.append(
                Segment(
                    index=idx,
                    text=text,
                    type="text",
                    page_start=min(pages),
                    page_end=max(pages),
                    heading_path=[t for _, t in stack],
                    n_chars=len(text),
                )
            )
            idx += 1
        buf, pages = [], []

    last_page = None
    for b in blocks:
        if b.type == "image":
            continue  # imagens ignoradas nesta v1

        # tabela: vira chunk proprio (nunca rachada) quando keep_tables_whole
        if b.type == "table" and plan.keep_tables_whole:
            flush()
            segments.append(
                Segment(
                    index=idx,
                    text=b.text,
                    type="table",
                    page_start=b.page,
                    page_end=b.page,
                    heading_path=[t for _, t in stack],
                    n_chars=len(b.text),
                )
            )
            idx += 1
            last_page = b.page
            continue
        # tabela sem keep_tables_whole cai no fluxo de texto abaixo

        # by_page: fecha o segmento ao trocar de pagina
        if plan.strategy == "by_page" and last_page is not None and b.page != last_page:
            flush()
        last_page = b.page

        lvl = _heading_level(b, heading_sizes, body)
        if lvl and plan.respect_headings and plan.strategy in ("by_heading", "table_aware", "by_section"):
            flush()  # fecha a secao anterior na fronteira do heading
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, b.text.strip()))
            buf.append(b.text.strip())  # titulo abre a nova secao (contexto p/ autocontido)
            pages.append(b.page)
            continue

        # bloco de corpo: estoura o teto? fecha antes de adicionar
        cur_len = sum(len(x) for x in buf)
        if cur_len and cur_len + len(b.text) > plan.max_chars:
            flush()
        buf.append(b.text)
        pages.append(b.page)

        # atingiu o alvo em estrategias de empacotamento -> fecha
        if plan.strategy in ("by_block", "table_aware", "by_section"):
            if sum(len(x) for x in buf) >= plan.target_chars:
                flush()

    flush()
    _apply_overlap(segments, plan.overlap_chars)
    return segments
