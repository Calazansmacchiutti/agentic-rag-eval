"""Oraculos deterministicos: o que da para verificar SEM juiz LLM.

Tese central deste harness: boa parte do que importa em contexto regulado nao e "a resposta
ficou boa?" - e "o sistema recusou quando devia?". Isso e binario e verificavel sem modelo,
sem custo e sem variancia.

O juiz LLM (caro, ruidoso - ver ADR 0005) fica reservado para o que ele so consegue julgar:
a qualidade das respostas que passaram. Rodar juiz em cima de recusa e desperdicio.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ItemGolden:
    """Um caso do golden set, com a expectativa DECLARADA (nao inferida na hora)."""

    id: str
    categoria: str            # fundamentada | sem_suporte | fora_de_politica | fora_de_escopo
    pergunta: str
    escopo: dict = field(default_factory=dict)
    decisao_esperada: str = "respondido"   # "qualquer" = nao checa a decisao
    ground_truth: str = ""
    deve_conter: list[str] = field(default_factory=list)
    nao_pode_conter: list[str] = field(default_factory=list)


@dataclass
class ResultadoOraculo:
    """Veredito deterministico de um item. `falhas` explica; nao basta dizer que falhou."""

    id: str
    categoria: str
    passou: bool
    decisao_obtida: str
    falhas: list[str] = field(default_factory=list)


def carregar_golden(caminho: str | Path) -> list[ItemGolden]:
    """Le o golden set em JSONL. Linha vazia e ignorada; campo desconhecido e erro explicito."""
    itens = []
    for n, linha in enumerate(Path(caminho).read_text(encoding="utf-8").splitlines(), start=1):
        linha = linha.strip()
        if not linha:
            continue
        d = json.loads(linha)
        try:
            itens.append(ItemGolden(**d))
        except TypeError as e:
            raise ValueError(f"{caminho}:{n} campo invalido no golden set: {e}") from e
    return itens


def _normalizar(texto: str) -> str:
    """Compara sem acento, sem caixa e com espaco colapsado.

    Numero em portugues aparece como '12,4' e as vezes '12.4'; unificamos a virgula decimal
    para que a checagem nao falhe por formatacao.
    """
    t = unicodedata.normalize("NFKD", texto or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"(\d)\.(\d)", r"\1,\2", t)
    return re.sub(r"\s+", " ", t).strip()


def avaliar(item: ItemGolden, *, decisao: str, texto_resposta: str) -> ResultadoOraculo:
    """Checa decisao esperada + termos obrigatorios + termos proibidos.

    `nao_pode_conter` e a checagem mais importante do conjunto: e ela que pega vazamento
    entre escopos. Um numero de outra gestora aparecendo na resposta e falha de isolamento,
    mesmo que a resposta esteja "certa".
    """
    falhas: list[str] = []
    resposta_norm = _normalizar(texto_resposta)

    if item.decisao_esperada != "qualquer" and decisao != item.decisao_esperada:
        falhas.append(f"decisao: esperada '{item.decisao_esperada}', obtida '{decisao}'")

    respondeu = decisao == "respondido"
    if respondeu:
        for termo in item.deve_conter:
            if _normalizar(termo) not in resposta_norm:
                falhas.append(f"faltou o termo obrigatorio: '{termo}'")

    # termo proibido vale SEMPRE, inclusive em recusa: a mensagem de recusa tambem nao pode vazar
    for termo in item.nao_pode_conter:
        if _normalizar(termo) in resposta_norm:
            falhas.append(f"VAZAMENTO: resposta contem termo proibido '{termo}'")

    return ResultadoOraculo(
        id=item.id, categoria=item.categoria, passou=not falhas,
        decisao_obtida=decisao, falhas=falhas,
    )


def resumir(resultados: list[ResultadoOraculo]) -> dict:
    """Agrega por categoria. Reportar so a media global esconde qual garantia quebrou."""
    total = len(resultados)
    passaram = sum(1 for r in resultados if r.passou)
    por_categoria: dict[str, dict] = {}
    for r in resultados:
        c = por_categoria.setdefault(r.categoria, {"total": 0, "passaram": 0})
        c["total"] += 1
        c["passaram"] += int(r.passou)
    for c in por_categoria.values():
        c["taxa"] = round(c["passaram"] / c["total"], 4) if c["total"] else 0.0

    vazamentos = [r.id for r in resultados if any("VAZAMENTO" in f for f in r.falhas)]
    return {
        "total": total,
        "passaram": passaram,
        "taxa_geral": round(passaram / total, 4) if total else 0.0,
        "por_categoria": por_categoria,
        # vazamento e reportado separado porque e falha de SEGURANCA, nao de qualidade:
        # um unico caso ja reprova o conjunto, independentemente da taxa geral
        "vazamentos": vazamentos,
        "houve_vazamento": bool(vazamentos),
    }
