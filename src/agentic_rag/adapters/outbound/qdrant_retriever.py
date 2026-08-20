"""Adapter de recuperacao: envolve o Retriever (Qdrant) e devolve entidades do dominio.

O `Retriever` legado fala em `dict`; o dominio fala em `Trecho`. A traducao acontece aqui,
que e exatamente o papel de um adapter - o dominio nunca ve o payload cru do Qdrant.
"""
from __future__ import annotations

from agentic_rag.domain.entities import Trecho
from agentic_rag.retriever import Retriever


class RecuperacaoQdrant:
    """Implementa PortaRecuperacao sobre o Retriever existente."""

    def __init__(self, retriever: Retriever | None = None):
        self._r = retriever or Retriever()

    def buscar(self, consulta: str, filtros: dict | None = None) -> list[Trecho]:
        achados = self._r.search(consulta, filters=filtros)
        return [self._para_trecho(h) for h in achados]

    @staticmethod
    def _para_trecho(h: dict) -> Trecho:
        """Separa texto e score do resto do payload, que vira metadado de citacao."""
        meta = {k: v for k, v in h.items() if k not in ("text", "score")}
        return Trecho(texto=h.get("text", ""), score=float(h.get("score") or 0.0), meta=meta)
