"""Reranking com cross-encoder (deterministico, sem LLM).

Estrategia (ADR 0005, item 5): a busca vetorial recupera um pool maior (rerank_fetch_k) e
aqui reordenamos por relevancia real query-trecho e cortamos nos top_k. Ao contrario de baixar
o top_k, isto melhora a PRECISAO sem perder recall — pega os melhores k de um conjunto maior,
preservando a completude da resposta. Modelo carregado preguicosamente (pesado p/ subir).
"""
from functools import lru_cache

from agentic_rag.config import settings


@lru_cache(maxsize=1)
def _model():
    """Carrega o CrossEncoder uma vez (import pesado adiado, como o embedder)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Reordena `hits` por relevancia (cross-encoder) e devolve os top_k.

    Anexa `rerank_score` a cada hit devolvido. Sem hits, no-op. Deterministico: mesma
    entrada => mesma saida (o cross-encoder nao amostra).
    """
    if not hits:
        return hits
    scores = _model().predict([(query, h.get("text", "")) for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [{**h, "rerank_score": float(s)} for h, s in ranked[:top_k]]
