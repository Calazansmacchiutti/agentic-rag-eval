"""Guardrails: quando o sistema DEVE se recusar a responder.

Premissa, herdada do ADR 0005: alegacao sem fonte rastreavel nao vale nada em contexto
regulado. Uma recusa explicita e um resultado correto; uma resposta plausivel e sem
lastro e um defeito - mesmo que soe bem.

Dominio puro: sem LLM, sem I/O. Todas as regras sao deterministicas e testaveis.
"""
from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.domain.entities import Resposta, Trecho


@dataclass(frozen=True)
class Veredito:
    """Resultado da checagem. `motivo` e o texto que vai para o usuario E para a auditoria."""

    permitido: bool
    decisao: str = "respondido"
    motivo: str = ""


# Categorias que o sistema nao responde por politica, nao por incapacidade.
# Recomendacao de investimento/credito e ato regulado - explicar != recomendar.
PADROES_FORA_DE_POLITICA = (
    "devo investir", "vale a pena investir", "me recomenda", "qual acao comprar",
    "devo comprar", "devo vender", "garante retorno", "sem risco",
    "aprova o credito", "libera o limite", "posso emprestar",
)


def checar_pergunta(pergunta: str) -> Veredito:
    """Barra pedido de recomendacao ANTES de gastar chamada de LLM."""
    p = (pergunta or "").strip().lower()
    if not p:
        return Veredito(False, "recusado_pergunta_vazia", "Pergunta vazia.")
    for padrao in PADROES_FORA_DE_POLITICA:
        if padrao in p:
            return Veredito(
                False,
                "recusado_fora_de_politica",
                "Este assistente explica dados e decisoes existentes, mas nao emite "
                "recomendacao de investimento ou de concessao de credito.",
            )
    return Veredito(True)


def checar_resposta(resposta: Resposta, trechos: list[Trecho]) -> Veredito:
    """Sem citacao valida, nao passa. Esta e a regra central do modo auditavel.

    Tres formas de reprovar:
      1. o proprio modelo declarou que o contexto nao sustenta (grounded=False);
      2. afirmou algo sem citar trecho nenhum;
      3. citou indice que nao existe na lista entregue (citacao fabricada).
    """
    if not resposta.grounded:
        return Veredito(
            False, "recusado_sem_fundamento",
            "O contexto recuperado nao sustenta uma resposta a esta pergunta.",
        )
    if not resposta.citations:
        return Veredito(
            False, "recusado_sem_citacao",
            "A resposta nao indicou qual trecho a sustenta.",
        )
    fora = [i for i in resposta.citations if i < 0 or i >= len(trechos)]
    if fora:
        return Veredito(
            False, "recusado_citacao_invalida",
            f"A resposta citou trechos inexistentes: {fora}.",
        )
    return Veredito(True)
