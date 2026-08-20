"""Portas de saida: o que o dominio EXIGE do mundo externo, sem saber quem entrega.

Sao Protocols (tipagem estrutural): um adapter nao precisa herdar nada, basta ter os
metodos. Trocar Anthropic por Bedrock, ou Qdrant por FAISS, e escrever outro adapter -
nenhum arquivo de dominio muda.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentic_rag.domain.entities import EventoAuditoria, Trecho


@runtime_checkable
class PortaRecuperacao(Protocol):
    """Busca trechos relevantes. `filtros` restringe por metadado ANTES do LLM ver o texto."""

    def buscar(self, consulta: str, filtros: dict | None = None) -> list[Trecho]: ...


@runtime_checkable
class PortaLLM(Protocol):
    """Geracao de texto. O dominio so pede resposta; quem fala com a API e o adapter."""

    def completar(self, prompt: str, *, schema: Any = None, modelo: str | None = None,
                  temperatura: float | None = None) -> Any: ...

    def loop_ferramentas(self, *, sistema: str, mensagens: list[dict], ferramentas: list[dict],
                         modelo: str | None = None) -> Any: ...

    def resposta_estruturada(self, *, sistema: str, mensagens: list[dict], schema: Any,
                             modelo: str | None = None) -> Any: ...

    @property
    def modelo_corrente(self) -> str: ...


@runtime_checkable
class PortaEmbedding(Protocol):
    """Vetorizacao de texto."""

    def vetorizar(self, textos: list[str]) -> Any: ...

    def dimensao(self) -> int: ...


@runtime_checkable
class PortaAuditoria(Protocol):
    """Trilha de auditoria. Append-only por contrato: registrar nunca sobrescreve."""

    def registrar(self, evento: EventoAuditoria) -> None: ...

    def listar(self, usuario: str | None = None, limite: int = 100) -> list[dict]: ...
