"""Indexa os segmentos do Agente A no Qdrant, preservando o recorte e os metadados.

Nao re-chunka: usa `Retriever.index_chunks`. Cada ponto leva texto + heading_path, pagina,
tipo e doc_id, para busca filtravel depois.

Requer Qdrant no ar (`docker run -p 6333:6333 qdrant/qdrant`) e o modelo de embedding.
"""
from __future__ import annotations

from agentic_rag.pdf.schemas import ChunkResult


def to_points(doc_id: str, result: ChunkResult) -> tuple[list[str], list[dict]]:
    """Converte os segmentos em (chunks, metadados) paralelos para o Retriever."""
    chunks, metas = [], []
    for s in result.segments:
        chunks.append(s.text)
        metas.append(
            {
                "doc_id": doc_id,
                "chunk_index": s.index,
                "type": s.type,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "heading_path": s.heading_path,
                "strategy": result.plan.strategy,
            }
        )
    return chunks, metas


def index_result(doc_id: str, result: ChunkResult, retriever=None) -> int:
    """Indexa um ChunkResult no Qdrant. Devolve o numero de pontos inseridos."""
    if retriever is None:
        from agentic_rag.retriever import Retriever

        retriever = Retriever()
    chunks, metas = to_points(doc_id, result)
    return retriever.index_chunks(chunks, metas)
