"""Agente B: extrai campos estruturados de um PDF.

Loop curto e orcado (<= 2 chamadas de LLM, como o RAG):
  1. Passada cheia: contexto do doc (texto por pagina) + specs dos campos -> ExtractionResult.
  2. Valida quais OBRIGATORIOS ficaram vazios.
  3. Se faltou, 2a passada FOCADA: so nas paginas que mencionam o campo, so os campos faltantes.
  4. Funde (valor preenchido vence vazio).

Structured output via Pydantic (llm.complete(schema=...)). Page grounding: o modelo
devolve a pagina de onde tirou cada campo. `dry_run=True` monta o contexto sem chamar LLM.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from agentic_rag.config import settings
from agentic_rag.pdf import probe
from agentic_rag.pdf.schemas import Block, ExtractedField, ExtractionResult, FieldSpec

_CTX_BUDGET = 12000  # chars de contexto enviados ao LLM (doc grande e truncado nesta v1)


class _Extracted(BaseModel):
    """Schema enxuto que o LLM devolve (montamos o ExtractionResult final no codigo)."""

    fields: list[ExtractedField]


def _page_texts(blocks: list[Block]) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    for b in blocks:
        if b.type == "text" and b.text:
            pages.setdefault(b.page, []).append(b.text)
    return {p: "\n".join(parts) for p, parts in pages.items()}


def _context(page_texts: dict[int, str], pages: list[int] | None, budget: int) -> str:
    """Texto rotulado por pagina, truncado ao orcamento."""
    out, used = [], 0
    for p in sorted(pages if pages is not None else page_texts):
        t = page_texts.get(p, "")
        if not t:
            continue
        block = f"[pagina {p}]\n{t}"
        if used + len(block) > budget:
            block = block[: max(0, budget - used)]
        out.append(block)
        used += len(block)
        if used >= budget:
            break
    return "\n\n".join(out)


def _pages_mentioning(field: FieldSpec, page_texts: dict[int, str]) -> list[int]:
    """Paginas cujo texto cita palavras do nome/descricao do campo (filtro p/ 2a passada)."""
    terms = [w for w in re.findall(r"\w{4,}", f"{field.name} {field.description}".lower())]
    hits = []
    for p, t in page_texts.items():
        low = t.lower()
        if any(term in low for term in terms):
            hits.append(p)
    return sorted(hits)


def _call(context: str, fields: list[FieldSpec]) -> list[ExtractedField]:
    from agentic_rag import llm

    spec = "\n".join(
        f"- {f.name}: {f.description}" + (" (OBRIGATORIO)" if f.required else "")
        for f in fields
    )
    prompt = (
        "Extraia os campos abaixo do documento. Para cada campo devolva: name (igual ao "
        "pedido), value (texto extraido ou null se nao achar), page (numero da pagina de "
        "onde veio, ou null) e confidence de 0 a 1. NAO invente: se nao estiver no texto, "
        "value=null e confidence=0.\n\n"
        f"CAMPOS:\n{spec}\n\nDOCUMENTO:\n{context}"
    )
    return llm.complete(prompt, schema=_Extracted, model=settings.llm_model).fields


def _merge(base: list[ExtractedField], extra: list[ExtractedField]) -> list[ExtractedField]:
    """Funde: um valor preenchido sobrescreve o vazio do mesmo campo."""
    by_name = {f.name: f for f in base}
    for f in extra:
        cur = by_name.get(f.name)
        if cur is None or (not cur.value and f.value):
            by_name[f.name] = f
    return list(by_name.values())


def extract(
    path: str, fields: list[FieldSpec], use_llm: bool = True, ocr: str = "auto"
) -> ExtractionResult:
    """Extrai os campos do PDF. Retorna ExtractionResult com os obrigatorios faltantes."""
    blocks = probe.load_blocks(path, ocr=ocr)
    page_texts = _page_texts(blocks)
    required = [f.name for f in fields if f.required]

    if not use_llm:  # dry run: so mostra o que iria pro LLM (sem custo/sem key)
        return ExtractionResult(
            fields=[],
            missing_required=required,
            notes=f"dry-run: {len(page_texts)} paginas, contexto ~{min(sum(len(t) for t in page_texts.values()), _CTX_BUDGET)} chars",
        )

    # 1a passada: documento inteiro (truncado)
    result = _call(_context(page_texts, None, _CTX_BUDGET), fields)

    # 2a passada focada: so os obrigatorios ainda vazios
    filled = {f.name for f in result if f.value}
    missing = [f for f in fields if f.required and f.name not in filled]
    if missing:
        pages = sorted({p for f in missing for p in _pages_mentioning(f, page_texts)})
        if pages:
            focused = _call(_context(page_texts, pages, _CTX_BUDGET), missing)
            result = _merge(result, focused)

    filled = {f.name for f in result if f.value}
    return ExtractionResult(
        fields=result,
        missing_required=[n for n in required if n not in filled],
        notes=f"{len(page_texts)} paginas; {len(required)} obrigatorios; loop {'2 passadas' if missing else '1 passada'}",
    )
