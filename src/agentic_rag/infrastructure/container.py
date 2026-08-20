"""Composition root: o unico lugar que sabe qual adapter concreto atende cada porta.

Toda a fiacao mora aqui. Se amanha o LLM virar Bedrock e o vetor virar FAISS, muda este
arquivo e mais nada - `domain/` nao tem ideia de quem entrega.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from agentic_rag.adapters.outbound.anthropic_llm import LLMAnthropic
from agentic_rag.adapters.outbound.auditoria_jsonl import AuditoriaJSONL
from agentic_rag.adapters.outbound.qdrant_retriever import RecuperacaoQdrant
from agentic_rag.domain.ports.outbound import PortaAuditoria, PortaLLM, PortaRecuperacao
from agentic_rag.infrastructure import prompts
from agentic_rag.infrastructure.config import settings


@dataclass
class Servicos:
    """Conjunto de dependencias ja resolvidas, pronto para o caso de uso."""

    recuperacao: PortaRecuperacao
    llm: PortaLLM
    auditoria: PortaAuditoria
    sistema: str
    versao_prompt: str
    max_chamadas: int


@lru_cache(maxsize=1)
def servicos() -> Servicos:
    """Monta os servicos de producao (cacheado: o cliente de LLM e compartilhado)."""
    p = prompts.carregar("system")
    return Servicos(
        recuperacao=RecuperacaoQdrant(),
        llm=LLMAnthropic(),
        auditoria=AuditoriaJSONL(settings.audit_log_path),
        sistema=p.texto,
        versao_prompt=p.versao,
        max_chamadas=settings.max_llm_calls_per_item,
    )
