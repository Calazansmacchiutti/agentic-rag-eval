"""Runner do eval: oraculos deterministicos primeiro, juiz LLM so no que passou.

    python -m evals.harness --stub              # sem LLM: exercita o harness inteiro
    python -m evals.harness                     # com LLM real (precisa de chave + Qdrant)
    python -m evals.harness --json saida.json

Ordem deliberada: o oraculo e barato, estavel e binario; o juiz e caro e ruidoso. Rodar o
juiz sobre uma recusa nao produz informacao - a recusa ja foi verificada exatamente.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evals import juiz as juiz_mod
from evals import oraculos

RAIZ = Path(__file__).resolve().parents[1]
GOLDEN = RAIZ / "evals" / "golden_set.jsonl"
CORPUS = RAIZ / "data" / "financeiro" / "corpus.jsonl"


def _carregar_corpus() -> list[dict]:
    linhas = CORPUS.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in linhas if x.strip()]


class RespondedorStub:
    """Responde por casamento de palavra sobre o corpus, sem LLM.

    Serve para exercitar o harness inteiro (oraculos, escopo, agregacao) em CI, sem chave
    nem Qdrant. NAO e um modelo: e um piso deterministico. Se o sistema com LLM nao superar
    o stub nas categorias que dependem de compreensao, isso e um achado, nao um detalhe.
    """

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus

    def responder(self, item: oraculos.ItemGolden) -> tuple[str, str, list[dict]]:
        from agentic_rag.domain import guardrails

        veredito = guardrails.checar_pergunta(item.pergunta)
        if not veredito.permitido:
            return veredito.decisao, veredito.motivo, []

        # escopo vira filtro ANTES da busca - mesma regra do caso de uso real
        docs = self.corpus
        gestora = item.escopo.get("gestora")
        if gestora:
            docs = [d for d in docs if d.get("gestora") in (gestora, "comum")]

        termos = {t for t in oraculos._normalizar(item.pergunta).split() if len(t) > 4}
        marcados = []
        for d in docs:
            texto = oraculos._normalizar(d["text"])
            marcados.append((sum(1 for t in termos if t in texto), d))
        marcados.sort(key=lambda x: -x[0])
        melhores = [d for n, d in marcados[:3] if n > 0]

        if not melhores:
            return "recusado_sem_fundamento", "Nao encontrei base para responder.", []
        return "respondido", " ".join(d["text"] for d in melhores), melhores


def _respondedor_real():
    """Liga no caso de uso de verdade (precisa de chave e Qdrant no ar)."""
    from agentic_rag.domain.entities import Escopo
    from agentic_rag.domain.use_cases import responder_pergunta as uc
    from agentic_rag.infrastructure.container import servicos

    s = servicos()

    def responder(item: oraculos.ItemGolden):
        r = uc.responder(
            item.pergunta, recuperacao=s.recuperacao, llm=s.llm, sistema=s.sistema,
            versao_prompt=s.versao_prompt, auditoria=s.auditoria, max_chamadas=s.max_chamadas,
            escopo=Escopo(usuario="eval", papel="avaliador", filtros=item.escopo),
        )
        contextos = [{"text": t.texto, **t.meta} for t in r.trechos]
        return r.decisao, r.resposta.answer, contextos

    return type("RespondedorReal", (), {"responder": staticmethod(responder)})()


def rodar(*, stub: bool = False, com_juiz: bool = True) -> dict:
    """Duas camadas: oraculo deterministico em tudo, juiz LLM so no que foi respondido."""
    itens = oraculos.carregar_golden(GOLDEN)
    respondedor = RespondedorStub(_carregar_corpus()) if stub else _respondedor_real()

    resultados, respostas = [], []
    for item in itens:
        decisao, texto, contextos = respondedor.responder(item)
        resultados.append(oraculos.avaliar(item, decisao=decisao, texto_resposta=texto))
        respostas.append({
            "id": item.id, "pergunta": item.pergunta, "decisao": decisao,
            "resposta": texto[:400], "n_contextos": len(contextos),
            # guardado para o juiz; nao entra no relatorio impresso
            "contextos_texto": [c.get("text", "") for c in contextos],
        })

    relatorio = {
        "modo": "stub" if stub else "llm",
        "resumo": oraculos.resumir(resultados),
        "itens": [asdict(r) for r in resultados],
        "respostas": [{k: v for k, v in r.items() if k != "contextos_texto"} for r in respostas],
    }

    if com_juiz:
        selecionados = juiz_mod.selecionar_para_juiz(itens, resultados, respostas)
        juiz = juiz_mod.JuizStub() if stub else juiz_mod.JuizLLM()
        j = juiz.julgar(selecionados)
        j.n_pulados = len(itens) - len(selecionados)
        relatorio["juiz"] = {
            "metricas": {k: round(v, 4) for k, v in j.metricas.items()},
            "grader_model": j.grader_model,
            "scorer": j.scorer,
            "n_julgados": j.n_julgados,
            "n_pulados_por_serem_recusa": j.n_pulados,
            "itens": [asdict(i) for i in j.itens],
        }

    return relatorio


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval harness: oraculos deterministicos + juiz LLM.")
    ap.add_argument("--stub", action="store_true", help="sem LLM (piso deterministico, roda em CI)")
    ap.add_argument("--sem-juiz", action="store_true",
                    help="so os oraculos deterministicos (nao gasta chamada de LLM)")
    ap.add_argument("--json", metavar="ARQUIVO", help="grava o relatorio completo")
    args = ap.parse_args()

    rel = rodar(stub=args.stub, com_juiz=not args.sem_juiz)
    r = rel["resumo"]
    print(f"modo: {rel['modo']} | {r['passaram']}/{r['total']} itens ({r['taxa_geral']:.0%})\n")
    print(f"{'categoria':<22} {'passaram':>10} {'taxa':>8}")
    for cat, d in sorted(r["por_categoria"].items()):
        print(f"{cat:<22} {d['passaram']:>4}/{d['total']:<5} {d['taxa']:>7.0%}")

    falhos = [i for i in rel["itens"] if not i["passou"]]
    if falhos:
        print("\nfalhas:")
        for f in falhos:
            print(f"  [{f['id']}] {f['categoria']}: " + "; ".join(f["falhas"]))

    if r["houve_vazamento"]:
        print(f"\n*** VAZAMENTO DE ESCOPO em {r['vazamentos']} - reprova o conjunto ***")

    if "juiz" in rel:
        j = rel["juiz"]
        print(f"\njuiz ({j['scorer']} / {j['grader_model']}): {j['n_julgados']} julgados, "
              f"{j['n_pulados_por_serem_recusa']} pulados por serem recusa")
        for k, v in sorted(j["metricas"].items()):
            print(f"  {k:<20} {v:.3f}")
        if j["scorer"] == "stub":
            print("  (stub: valores fixos, NAO sao medicao de qualidade)")

    if args.json:
        Path(args.json).write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nrelatorio -> {args.json}")

    # vazamento reprova sozinho, independentemente da taxa geral
    return 1 if (r["houve_vazamento"] or falhos) else 0


if __name__ == "__main__":
    raise SystemExit(main())
