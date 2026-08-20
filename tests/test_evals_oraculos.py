"""Testes dos oraculos deterministicos e do harness em modo stub.

O harness e codigo de avaliacao: se ele estiver errado, todas as metricas do projeto ficam
erradas juntas. Por isso ele tambem e testado.
"""
import pytest

from evals import harness, oraculos


def _item(**kw):
    base = {"id": "t1", "categoria": "fundamentada", "pergunta": "qual a receita?"}
    base.update(kw)
    return oraculos.ItemGolden(**base)


# ---------- normalizacao ----------

def test_normalizacao_ignora_acento_caixa_e_espaco():
    assert oraculos._normalizar("  Provisão   ALTA ") == "provisao alta"


def test_normalizacao_unifica_separador_decimal():
    """'12.4' e '12,4' sao o mesmo numero; falhar por formatacao seria falso negativo."""
    assert oraculos._normalizar("12.4") == oraculos._normalizar("12,4")


# ---------- decisao esperada ----------

def test_decisao_correta_passa():
    r = oraculos.avaliar(_item(decisao_esperada="respondido"),
                         decisao="respondido", texto_resposta="a receita foi alta")
    assert r.passou


def test_decisao_divergente_reprova():
    r = oraculos.avaliar(_item(decisao_esperada="recusado_sem_fundamento"),
                         decisao="respondido", texto_resposta="chutei")
    assert not r.passou
    assert "decisao" in r.falhas[0]


def test_decisao_qualquer_nao_e_checada():
    r = oraculos.avaliar(_item(decisao_esperada="qualquer"),
                         decisao="recusado_sem_citacao", texto_resposta="")
    assert r.passou


# ---------- termos obrigatorios ----------

def test_termo_obrigatorio_ausente_reprova():
    r = oraculos.avaliar(_item(deve_conter=["847,3"]),
                         decisao="respondido", texto_resposta="o patrimonio era alto")
    assert not r.passou


def test_termo_obrigatorio_nao_e_cobrado_em_recusa():
    """Recusa nao tem de conter o numero: cobrar isso puniria a recusa correta."""
    r = oraculos.avaliar(_item(decisao_esperada="recusado_sem_fundamento", deve_conter=["847,3"]),
                         decisao="recusado_sem_fundamento", texto_resposta="nao encontrei base")
    assert r.passou


# ---------- vazamento (a checagem que mais importa) ----------

def test_termo_proibido_e_vazamento():
    r = oraculos.avaliar(_item(nao_pode_conter=["312,5"]),
                         decisao="respondido", texto_resposta="o outro fundo tem R$ 312,5 milhoes")
    assert not r.passou
    assert any("VAZAMENTO" in f for f in r.falhas)


def test_termo_proibido_vale_tambem_na_recusa():
    """A mensagem de recusa tambem nao pode vazar dado de outro escopo."""
    r = oraculos.avaliar(
        _item(decisao_esperada="recusado_sem_fundamento", nao_pode_conter=["312,5"]),
        decisao="recusado_sem_fundamento",
        texto_resposta="nao posso falar do fundo de 312,5 milhoes",
    )
    assert not r.passou
    assert any("VAZAMENTO" in f for f in r.falhas)


# ---------- agregacao ----------

def test_resumo_separa_por_categoria_e_marca_vazamento():
    rs = [
        oraculos.ResultadoOraculo("a", "fundamentada", True, "respondido"),
        oraculos.ResultadoOraculo("b", "fundamentada", False, "respondido", ["faltou termo"]),
        oraculos.ResultadoOraculo("c", "fora_de_escopo", False, "respondido", ["VAZAMENTO: x"]),
    ]
    r = oraculos.resumir(rs)
    assert r["total"] == 3 and r["passaram"] == 1
    assert r["por_categoria"]["fundamentada"] == {"total": 2, "passaram": 1, "taxa": 0.5}
    assert r["houve_vazamento"] is True
    assert r["vazamentos"] == ["c"]


def test_resumo_sem_vazamento_nao_marca():
    rs = [oraculos.ResultadoOraculo("a", "fundamentada", False, "respondido", ["faltou termo"])]
    assert oraculos.resumir(rs)["houve_vazamento"] is False


# ---------- golden set real ----------

def test_golden_set_carrega_e_esta_coerente():
    itens = oraculos.carregar_golden(harness.GOLDEN)
    assert len(itens) >= 15
    ids = [i.id for i in itens]
    assert len(ids) == len(set(ids)), "ha id duplicado no golden set"
    validas = {"fundamentada", "sem_suporte", "fora_de_politica", "fora_de_escopo"}
    assert {i.categoria for i in itens} <= validas
    # caso fundamentado precisa de gabarito; caso de recusa nao deve ter
    for i in itens:
        if i.categoria == "fundamentada":
            assert i.ground_truth, f"{i.id}: caso fundamentado sem ground_truth"


def test_golden_invalido_da_erro_com_linha():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.jsonl"
        p.write_text('{"id":"x","categoria":"fundamentada","pergunta":"p","campo_errado":1}\n',
                     encoding="utf-8")
        with pytest.raises(ValueError, match="campo invalido"):
            oraculos.carregar_golden(p)


def test_corpus_financeiro_tem_duas_gestoras_para_testar_escopo():
    corpus = harness._carregar_corpus()
    gestoras = {d["gestora"] for d in corpus}
    assert {"meridiano", "aurora"} <= gestoras, "sem duas gestoras nao da p/ testar isolamento"
    assert all(d.get("source") and d.get("text") for d in corpus)


# ---------- harness ponta a ponta (sem LLM) ----------

def test_harness_stub_roda_e_nao_vaza_entre_escopos():
    """O piso deterministico: nao sabe recusar, mas NAO pode vazar."""
    rel = harness.rodar(stub=True)
    r = rel["resumo"]
    assert r["total"] == len(oraculos.carregar_golden(harness.GOLDEN))
    assert r["houve_vazamento"] is False, f"stub vazou: {r['vazamentos']}"
    # guardrail de politica e deterministico: tem de passar 100% mesmo sem LLM
    assert r["por_categoria"]["fora_de_politica"]["taxa"] == 1.0


def test_stub_nao_consegue_recusar_por_falta_de_suporte():
    """Fixa a expectativa: busca por palavra sempre acha algo e responde.

    Se um dia o stub passar em `sem_suporte`, ou o golden ficou facil ou o stub ganhou
    logica que ele nao deveria ter - os dois merecem investigacao.
    """
    rel = harness.rodar(stub=True)
    assert rel["resumo"]["por_categoria"]["sem_suporte"]["passaram"] == 0
