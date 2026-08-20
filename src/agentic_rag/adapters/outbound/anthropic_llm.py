"""Adapter de LLM: Anthropic (Claude) por tras da PortaLLM.

Trocar de fornecedor = escrever outro arquivo aqui. Nenhum arquivo de `domain/` muda.
Ver ADR 0003 para a escolha do SDK oficial em vez de camada generica.
"""
from __future__ import annotations

from typing import Any

from agentic_rag.infrastructure.config import settings


def _anthropic():
    """Cliente vindo do gateway `agentic_rag.llm`.

    O adapter NAO instancia o SDK direto de proposito: o gateway concentra a criacao do
    cliente (chave, preguica) e e o ponto onde teste e ferramenta substituem o SDK. Uma
    segunda instanciacao aqui duplicaria a costura e quebraria essa substituicao.
    """
    from agentic_rag import llm as gateway

    return gateway.client()


class LLMAnthropic:
    """Implementa PortaLLM. Sem estado proprio alem do cliente compartilhado."""

    def __init__(self, modelo: str | None = None):
        self._modelo = modelo or settings.llm_model

    @property
    def modelo_corrente(self) -> str:
        return self._modelo

    def completar(self, prompt: str, *, schema: Any = None, modelo: str | None = None,
                  temperatura: float | None = None) -> Any:
        """Texto simples ou instancia Pydantic validada, quando `schema` e informado."""
        cli = _anthropic()
        kwargs: dict = {
            "model": modelo or self._modelo,
            "max_tokens": settings.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperatura is not None:
            kwargs["temperature"] = temperatura
        if schema is not None:
            return cli.messages.parse(**kwargs, output_format=schema).parsed_output
        resp = cli.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def loop_ferramentas(self, *, sistema: str, mensagens: list[dict], ferramentas: list[dict],
                         modelo: str | None = None) -> Any:
        """Uma volta do loop tool-use; devolve a resposta crua (o caso de uso le os blocos)."""
        return _anthropic().messages.create(
            model=modelo or self._modelo,
            max_tokens=settings.max_output_tokens,
            system=sistema,
            tools=ferramentas,
            messages=mensagens,
        )

    def resposta_estruturada(self, *, sistema: str, mensagens: list[dict], schema: Any,
                             modelo: str | None = None) -> Any:
        """Ultima chamada: saida validada contra o schema Pydantic."""
        return _anthropic().messages.parse(
            model=modelo or self._modelo,
            max_tokens=settings.max_output_tokens,
            system=sistema,
            messages=mensagens,
            output_format=schema,
        ).parsed_output
