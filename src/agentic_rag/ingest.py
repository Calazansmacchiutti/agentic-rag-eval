"""Ingestao: chunking + embeddings -> vector DB."""
from collections.abc import Iterable
from functools import lru_cache

from agentic_rag.config import settings


def chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Divide texto em chunks com sobreposicao."""
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


@lru_cache(maxsize=1)
def _model():
    """Carrega o SentenceTransformer uma vez (import preguicoso: pesado p/ subir)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed(chunks: Iterable[str]):
    """Embeddings normalizados (cosine-ready) para uma lista de textos.

    Retorna np.ndarray shape (n, dim). normalize_embeddings=True deixa o produto
    interno == similaridade de cosseno, que e a metrica que usamos no Qdrant.
    """
    texts = list(chunks)
    if not texts:
        import numpy as np

        return np.empty((0, embedding_dim()), dtype="float32")
    return _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def embedding_dim() -> int:
    """Dimensao do vetor (p/ criar a collection no Qdrant com o tamanho certo)."""
    model = _model()
    # sentence-transformers 5.x renomeou get_sentence_embedding_dimension -> get_embedding_dimension
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return getter()
