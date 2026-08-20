"""Compatibilidade + fachada: o loop agentic agora mora em `domain/use_cases`.

Este modulo continua sendo o ponto de entrada historico (`answer_question`), mas a regra
foi para o dominio, atras de portas. Aqui so acontece a fiacao com os adapters concretos.

Ver `domain/use_cases/responder_pergunta.py` para o fluxo e
`docs/architecture.md` para o porque da separacao.
"""
from __future__ import annotations

from agentic_rag import llm  # noqa: F401  (costura historica: testes trocam agent.llm.client)
from agentic_rag.config import settings  # costura historica: testes fazem monkeypatch do orcamento
from agentic_rag.domain.entities import Escopo, Resposta
from agentic_rag.domain.use_cases import responder_pergunta as uc
from agentic_rag.retriever import Retriever

# Nome historico mantido: varios testes e o README importam `Answer`.
Answer = Resposta

SEARCH_TOOL = uc.FERRAMENTA_BUSCA

# Mantido como constante para compatibilidade; a fonte de verdade e `prompts/system.md`.
SYSTEM = (
    "Voce e um assistente de QA sobre uma base de documentos. "
    "Use a tool `search` para recuperar contexto antes de responder. "
    "Responda APENAS com base no contexto recuperado e cite os indices [n] dos trechos usados. "
    "Se o contexto nao sustentar a resposta, defina grounded=false e seja explicito."
)


def _format_hits(hits: list[dict], start: int) -> str:
    """Compat (nivel dict): numera trechos a partir de `start`.

    O caso de uso trabalha com `Trecho`; estes helpers seguem operando em `dict` porque
    e o contrato historico exercitado pelos testes e por chamadores antigos.
    """
    linhas = [f"[{i}] {h.get('text', '')}" for i, h in enumerate(hits, start=start)]
    return "\n".join(linhas) if linhas else "(sem resultados)"


def _dedup(hits: list[dict], seen: set[str]) -> list[dict]:
    """Compat (nivel dict): descarta texto repetido no batch e entre rodadas."""
    saida = []
    for h in hits:
        chave = (h.get("text") or "").strip()
        if not chave or chave in seen:
            continue
        seen.add(chave)
        saida.append(h)
    return saida


def consultar(pergunta: str, *, escopo: Escopo | None = None) -> uc.ResultadoConsulta:
    """Caminho recomendado: resposta + trechos + veredito dos guardrails + auditoria."""
    from agentic_rag.infrastructure.container import servicos

    s = servicos()
    return uc.responder(
        pergunta,
        recuperacao=s.recuperacao,
        llm=s.llm,
        sistema=s.sistema,
        versao_prompt=s.versao_prompt,
        escopo=escopo,
        auditoria=s.auditoria,
        max_chamadas=s.max_chamadas,
    )


def answer_question(question: str, retriever: Retriever) -> tuple[Answer, list[dict]]:
    """Assinatura historica: recebe um Retriever cru e devolve (resposta, contextos em dict).

    Mantida para nao quebrar chamadores e testes existentes. Novo codigo deve usar
    `consultar()`, que aplica guardrails, escopo e auditoria.
    """
    from agentic_rag.adapters.outbound.anthropic_llm import LLMAnthropic
    from agentic_rag.adapters.outbound.qdrant_retriever import RecuperacaoQdrant
    from agentic_rag.infrastructure import prompts

    resultado = uc.responder(
        question,
        recuperacao=RecuperacaoQdrant(retriever),
        llm=LLMAnthropic(),
        sistema=prompts.carregar("system").texto,
        versao_prompt=prompts.carregar("system").versao,
        max_chamadas=settings.max_llm_calls_per_item,
    )
    contextos = [{"text": t.texto, "score": t.score, **t.meta} for t in resultado.trechos]
    return resultado.resposta, contextos
