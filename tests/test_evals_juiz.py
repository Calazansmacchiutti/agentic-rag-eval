"""Testes da integracao do juiz LLM ao harness.

O que importa fixar aqui nao e a nota que o juiz da - e a DISCIPLINA em torno dele:
quem vai a julgamento, o que nunca vai, e se a proveniencia viaja junto da metrica.
"""
from evals import harness, oraculos
from evals import juiz as juiz_mod

# ---------- selecao: quem vai ao juiz ----------

def _cenario():
    itens = [
        oraculos.ItemGolden(id="a", categoria="fundamentada", pergunta="p-a", ground_truth="g-a"),
        oraculos.ItemGolden(id="b", categoria="sem_suporte", pergunta="p-b"),
        oraculos.ItemGolden(id="c", categoria="fora_de_politica", pergunta="p-c"),
        oraculos.ItemGolden(id="d", categoria="fundamentada", pergunta="p-d", ground_truth="g-d"),
    ]
    resultados = [
        oraculos.ResultadoOraculo("a", "fundamentada", True, "respondido"),
        oraculos.ResultadoOraculo("b", "sem_suporte", True, "recusado_sem_fundamento"),
        oraculos.ResultadoOraculo("c", "fora_de_politica", True, "recusado_fora_de_politica"),
        oraculos.ResultadoOraculo("d", "fundamentada", False, "respondido", ["faltou termo"]),
    ]
    respostas = [
        {"id": "a", "decisao": "respondido", "resposta": "r-a", "contextos_texto": ["ctx"]},
        {"id": "b", "decisao": "recusado_sem_fundamento", "resposta": "nao achei",
         "contextos_texto": []},
        {"id": "c", "decisao": "recusado_fora_de_politica", "resposta": "nao recomendo",
         "contextos_texto": []},
        {"id": "d", "decisao": "respondido", "resposta": "r-d errada", "contextos_texto": ["ctx"]},
    ]
    return itens, resultados, respostas


def test_recusa_nunca_vai_ao_juiz():
    """Recusa ja foi verificada exatamente pelo oraculo; julga-la gastaria chamada a toa."""
    sel = juiz_mod.selecionar_para_juiz(*_cenario())
    assert {s["id"] for s in sel} == {"a", "d"}


def test_resposta_errada_VAI_ao_juiz():
    """Saber *quao* ruim ficou a resposta errada e diagnostico, nao desperdicio."""
    sel = juiz_mod.selecionar_para_juiz(*_cenario())
    assert "d" in {s["id"] for s in sel}


def test_selecao_carrega_ground_truth_e_contextos():
    sel = {s["id"]: s for s in juiz_mod.selecionar_para_juiz(*_cenario())}
    assert sel["a"]["ground_truth"] == "g-a"
    assert sel["a"]["contextos"] == ["ctx"]
    assert sel["a"]["pergunta"] == "p-a"


# ---------- proveniencia ----------

def test_juiz_stub_se_identifica_como_stub():
    """Numero de stub jamais pode ser confundido com medicao real."""
    j = juiz_mod.JuizStub().julgar([{"id": "x", "pergunta": "p", "resposta": "r",
                                     "contextos": ["c"], "ground_truth": ""}])
    assert j.scorer == "stub" and j.grader_model == "stub"


def test_juiz_llm_usa_o_mesmo_nome_de_scorer_do_evaluate():
    """Baseline so e comparavel entre rodadas do MESMO scorer (ADR 0005)."""
    j = juiz_mod.JuizLLM(modelo="modelo-x")
    assert j.grader_model == "modelo-x"
    vazio = j.julgar([])
    assert vazio.scorer == "llm_judge_score"   # mesmo nome usado por evaluate.llm_judge_score


def test_lote_vazio_nao_quebra():
    j = juiz_mod.JuizLLM().julgar([])
    assert set(j.metricas) == {"faithfulness", "answer_relevancy", "context_precision"}
    assert all(v == 0.0 for v in j.metricas.values())


def test_rubrica_vem_de_evaluate_e_nao_esta_duplicada():
    """Se a rubrica for copiada, as duas versoes divergem em silencio (ADR 0005)."""
    from agentic_rag import evaluate

    assert juiz_mod._JUDGE_PROMPT is evaluate._JUDGE_PROMPT
    assert juiz_mod._FAITH_MAP is evaluate._FAITH_MAP
    assert juiz_mod._RELEV_MAP is evaluate._RELEV_MAP


# ---------- harness com juiz ----------

def test_harness_com_juiz_stub_reporta_proveniencia_e_contagem():
    rel = harness.rodar(stub=True)
    j = rel["juiz"]
    assert j["scorer"] == "stub"
    assert j["n_julgados"] + j["n_pulados_por_serem_recusa"] == rel["resumo"]["total"]
    assert set(j["metricas"]) == {"faithfulness", "answer_relevancy", "context_precision"}


def test_sem_juiz_nao_produz_secao_de_juiz():
    rel = harness.rodar(stub=True, com_juiz=False)
    assert "juiz" not in rel


def test_relatorio_nao_vaza_contextos_crus():
    """`contextos_texto` alimenta o juiz mas nao entra no relatorio impresso/gravado."""
    rel = harness.rodar(stub=True)
    assert all("contextos_texto" not in r for r in rel["respostas"])
