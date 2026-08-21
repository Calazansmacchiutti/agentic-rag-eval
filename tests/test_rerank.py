from typing import ClassVar

"""Testes do reranking sem carregar o cross-encoder (mockado) nem tocar Qdrant."""
import numpy as np

from agentic_rag import ingest, rerank
from agentic_rag.retriever import Retriever


class _FakeCE:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.scores


def test_rerank_reordena_por_score_e_corta(monkeypatch):
    hits = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    fake = _FakeCE([0.1, 0.9, 0.5])
    monkeypatch.setattr(rerank, "_model", lambda: fake)

    out = rerank.rerank("q", hits, top_k=2)

    assert [h["text"] for h in out] == ["b", "c"]          # ordena por relevancia desc
    assert out[0]["rerank_score"] == 0.9                    # score anexado
    assert fake.pairs == [("q", "a"), ("q", "b"), ("q", "c")]  # pares query-trecho


def test_rerank_vazio_nao_carrega_modelo(monkeypatch):
    def _boom():
        raise AssertionError("nao deveria carregar o cross-encoder p/ hits vazio")

    monkeypatch.setattr(rerank, "_model", _boom)
    assert rerank.rerank("q", [], top_k=3) == []


def test_retriever_search_recupera_pool_maior_e_chama_reranker(monkeypatch):
    monkeypatch.setattr(ingest, "embed", lambda xs: np.zeros((len(xs), 3), dtype="float32"))
    monkeypatch.setattr(rerank.settings, "rerank", True)  # liga o rerank
    called = {}

    def spy_rerank(query, hits, top_k):
        called["args"] = (query, len(hits), top_k)
        return hits[:top_k]

    monkeypatch.setattr(rerank, "rerank", spy_rerank)

    captured = {}

    class _Res:
        points: ClassVar[list] = [
            type("P", (), {"payload": {"text": f"t{i}"}, "score": 1.0})() for i in range(20)
        ]

    class _Cli:
        def query_points(self, **kw):
            captured.update(kw)
            return _Res()

    r = Retriever(top_k=5)
    r._client = _Cli()

    out = r.search("q")

    assert captured["limit"] == rerank.settings.rerank_fetch_k  # recuperou o pool maior
    assert called["args"] == ("q", 20, 5)                       # rerank recebeu o pool, corta em top_k
    assert len(out) == 5


def test_retriever_search_sem_rerank_usa_top_k(monkeypatch):
    monkeypatch.setattr(ingest, "embed", lambda xs: np.zeros((len(xs), 3), dtype="float32"))
    monkeypatch.setattr(rerank.settings, "rerank", False)  # rerank desligado (default)
    captured = {}

    class _Res:
        points: ClassVar[list] = []

    class _Cli:
        def query_points(self, **kw):
            captured.update(kw)
            return _Res()

    r = Retriever(top_k=4)
    r._client = _Cli()
    r.search("q")

    assert captured["limit"] == 4  # sem rerank, recupera exatamente top_k
