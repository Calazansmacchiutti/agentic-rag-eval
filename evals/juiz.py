"""Juiz LLM do harness, reusando a rubrica ja calibrada em `agentic_rag.evaluate`.

Por que importar de la em vez de reescrever: a rubrica (escalas discretas, precisao
deterministica, temperatura baixa) e o resultado do ADR 0005 - foi ela que domou a variancia
do juiz. Duplicar o texto do prompt aqui criaria duas rubricas que divergem em silencio, e a
comparacao entre rodadas passaria a medir a diferenca entre elas, nao o sistema.

O acoplamento e proposital e esta declarado. Se a rubrica mudar, muda nos dois lugares de uma
vez porque so existe um lugar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean

from agentic_rag.evaluate import (
    _FAITH_MAP,
    _JUDGE_PROMPT,
    _RELEV_MAP,
    METRIC_KEYS,
    _ItemScore,
    _precision,
)
from agentic_rag.infrastructure.config import settings


@dataclass
class ItemJulgado:
    """Julgamento de um item, com o rationale preservado para auditoria do proprio eval."""

    id: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    rationale: str = ""


@dataclass
class Julgamento:
    """Resultado do juiz sobre o lote, com proveniencia junto das metricas.

    `grader_model` e `scorer` viajam com o numero de proposito (ADR 0005): metrica sem
    proveniencia nao e comparavel entre rodadas - uma troca de juiz se passaria por ganho.
    """

    metricas: dict[str, float]
    itens: list[ItemJulgado] = field(default_factory=list)
    grader_model: str = ""
    scorer: str = ""
    n_julgados: int = 0
    n_pulados: int = 0


class JuizStub:
    """Juiz deterministico para CI: nao chama LLM.

    Nao emite opiniao sobre qualidade - so devolve valores fixos para o harness poder rodar
    ponta a ponta sem chave. Os numeros que ele produz NAO significam qualidade e estao
    marcados como `scorer="stub"` para nunca serem confundidos com medicao real.
    """

    grader_model = "stub"

    def julgar(self, itens: list[dict]) -> Julgamento:
        julgados = [
            ItemJulgado(id=i["id"], faithfulness=1.0, answer_relevancy=1.0,
                        context_precision=1.0 if i.get("contextos") else 0.0,
                        rationale="stub: sem julgamento real")
            for i in itens
        ]
        metricas = (
            {
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_precision": fmean(j.context_precision for j in julgados),
            }
            if julgados
            else {k: 0.0 for k in METRIC_KEYS}
        )
        return Julgamento(metricas=metricas, itens=julgados, grader_model="stub",
                          scorer="stub", n_julgados=len(julgados))


class JuizLLM:
    """Juiz real: 1 chamada por item, rubrica analitica do ADR 0005."""

    def __init__(self, modelo: str | None = None, temperatura: float | None = None):
        self.grader_model = modelo or settings.grader_model
        self.temperatura = settings.grader_temperature if temperatura is None else temperatura

    def julgar(self, itens: list[dict]) -> Julgamento:
        """`itens`: [{id, pergunta, resposta, contextos: list[str], ground_truth}]."""
        from agentic_rag import llm

        if not itens:
            return Julgamento(metricas={k: 0.0 for k in METRIC_KEYS},
                              grader_model=self.grader_model, scorer="llm_judge_score")

        julgados: list[ItemJulgado] = []
        for i in itens:
            contextos = i.get("contextos") or []
            ctx = "\n".join(f"[{n}] {c}" for n, c in enumerate(contextos)) or "(nenhum)"
            prompt = _JUDGE_PROMPT.format(
                question=i["pergunta"],
                answer=i["resposta"],
                contexts=ctx,
                ground_truth=i.get("ground_truth") or "(nao fornecida)",
                n=len(contextos),
                last=max(len(contextos) - 1, 0),
            )
            s = llm.complete(prompt, schema=_ItemScore, model=self.grader_model,
                             temperature=self.temperatura)
            julgados.append(ItemJulgado(
                id=i["id"],
                faithfulness=_FAITH_MAP[s.faithfulness],
                answer_relevancy=_RELEV_MAP[s.answer_relevancy],
                context_precision=_precision(s.context_relevance, len(contextos)),
                rationale=s.rationale,
            ))

        return Julgamento(
            metricas={
                "faithfulness": fmean(j.faithfulness for j in julgados),
                "answer_relevancy": fmean(j.answer_relevancy for j in julgados),
                "context_precision": fmean(j.context_precision for j in julgados),
            },
            itens=julgados,
            grader_model=self.grader_model,
            scorer="llm_judge_score",   # mesmo nome usado por evaluate.py: baselines comparaveis
            n_julgados=len(julgados),
        )


def selecionar_para_juiz(itens_golden, resultados_oraculo, respostas) -> list[dict]:
    """Escolhe o que vale julgar: apenas o que foi RESPONDIDO.

    Recusa nao vai ao juiz. Ela ja foi verificada exatamente pelo oraculo (a decisao bateu ou
    nao bateu) - pedir a um LLM que opine sobre a qualidade de "nao encontrei base para
    responder" gasta chamada e nao acrescenta informacao.

    Item respondido que REPROVOU no oraculo continua indo ao juiz: saber *quao* ruim ficou a
    resposta errada e informacao util para diagnostico.
    """
    por_id = {i.id: i for i in itens_golden}
    resp_por_id = {r["id"]: r for r in respostas}
    selecionados = []
    for r in resultados_oraculo:
        resposta = resp_por_id.get(r.id, {})
        if resposta.get("decisao") != "respondido":
            continue
        item = por_id[r.id]
        selecionados.append({
            "id": r.id,
            "pergunta": item.pergunta,
            "resposta": resposta.get("resposta", ""),
            "contextos": resposta.get("contextos_texto", []),
            "ground_truth": item.ground_truth,
        })
    return selecionados
