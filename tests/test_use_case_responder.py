"""O caso de uso inteiro testado sem rede, so com dubles das portas.

Este arquivo e a prova de que a separacao hexagonal funciona: se fosse preciso subir
Qdrant ou chamar a API da Anthropic para testar a regra, a separacao seria decorativa.
"""
from types import SimpleNamespace

from agentic_rag.adapters.outbound.auditoria_jsonl import AuditoriaJSONL, AuditoriaMemoria
from agentic_rag.domain.entities import Escopo, Resposta, Trecho
from agentic_rag.domain.use_cases import responder_pergunta as uc


class RecuperacaoFake:
    """Devolve trechos fixos e registra os filtros recebidos (p/ checar o escopo)."""

    def __init__(self, trechos):
        self.trechos = trechos
        self.filtros_vistos = []

    def buscar(self, consulta, filtros=None):
        self.filtros_vistos.append(filtros)
        return list(self.trechos)


class LLMFake:
    """Emula o loop tool-use: primeira volta pede busca, ultima devolve a estrutura."""

    def __init__(self, resposta, *, buscar_uma_vez=True):
        self.resposta = resposta
        self.buscar_uma_vez = buscar_uma_vez
        self.chamadas_loop = 0

    modelo_corrente = "modelo-de-teste"

    def loop_ferramentas(self, *, sistema, mensagens, ferramentas, modelo=None):
        self.chamadas_loop += 1
        if self.buscar_uma_vez and self.chamadas_loop == 1:
            uso = SimpleNamespace(type="tool_use", id="t1", input={"query": "consulta"})
            return SimpleNamespace(content=[uso])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    def resposta_estruturada(self, *, sistema, mensagens, schema, modelo=None):
        return self.resposta

    def completar(self, prompt, *, schema=None, modelo=None, temperatura=None):
        return "nao usado"


TRECHOS = [
    Trecho(texto="A receita de 2025 foi de R$ 10 milhoes.", score=0.9, meta={"pagina": 2}),
    Trecho(texto="O custo operacional caiu 4%.", score=0.7, meta={"pagina": 3}),
]


def _rodar(resposta, trechos=TRECHOS, **kw):
    rec = RecuperacaoFake(trechos)
    llm = LLMFake(resposta)
    aud = AuditoriaMemoria()
    res = uc.responder(
        kw.pop("pergunta", "qual foi a receita em 2025?"),
        recuperacao=rec, llm=llm, sistema="sistema de teste",
        versao_prompt="v-teste", auditoria=aud, max_chamadas=2, **kw,
    )
    return res, rec, aud


def test_resposta_fundamentada_passa_e_e_auditada():
    resposta = Resposta(answer="R$ 10 milhoes.", citations=[0], grounded=True, confidence=0.9)
    res, _, aud = _rodar(resposta)

    assert res.permitido is True
    assert res.decisao == "respondido"
    assert res.fontes == ["p.2"]
    ev = aud.eventos[-1]
    assert ev["versao_prompt"] == "v-teste"
    assert ev["modelo"] == "modelo-de-teste"
    # a trilha guarda o ID do trecho, nunca o texto do corpus
    assert ev["trechos"] == [TRECHOS[0].id_conteudo, TRECHOS[1].id_conteudo]
    assert "R$ 10 milhoes" not in " ".join(ev["trechos"])


def test_sem_citacao_e_recusado():
    resposta = Resposta(answer="Foi alto.", citations=[], grounded=True, confidence=0.8)
    res, _, aud = _rodar(resposta)
    assert res.permitido is False
    assert res.decisao == "recusado_sem_citacao"
    assert aud.eventos[-1]["decisao"] == "recusado_sem_citacao"


def test_citacao_fabricada_e_recusada():
    resposta = Resposta(answer="Consta na pagina 9.", citations=[7], grounded=True, confidence=0.9)
    res, _, _ = _rodar(resposta)
    assert res.permitido is False
    assert res.decisao == "recusado_citacao_invalida"


def test_modelo_admitindo_falta_de_contexto_e_recusa_honesta():
    resposta = Resposta(answer="Nao encontrei.", citations=[], grounded=False, confidence=0.1)
    res, _, _ = _rodar(resposta)
    assert res.permitido is False
    assert res.decisao == "recusado_sem_fundamento"


def test_pedido_de_recomendacao_nao_gasta_chamada_de_llm():
    rec = RecuperacaoFake(TRECHOS)
    llm = LLMFake(Resposta(answer="x", citations=[0], grounded=True, confidence=0.9))
    res = uc.responder(
        "devo investir nesse fundo?", recuperacao=rec, llm=llm,
        sistema="s", versao_prompt="v", max_chamadas=2,
    )
    assert res.permitido is False
    assert res.decisao == "recusado_fora_de_politica"
    assert llm.chamadas_loop == 0          # barrou ANTES de gastar LLM
    assert rec.filtros_vistos == []        # e antes de tocar o retriever


def test_escopo_vira_filtro_no_retriever():
    """A restricao acontece na recuperacao, nao numa instrucao de prompt."""
    resposta = Resposta(answer="ok", citations=[0], grounded=True, confidence=0.9)
    escopo = Escopo(usuario="ana", papel="analista", filtros={"gestora": "G1"})
    _, rec, aud = _rodar(resposta, escopo=escopo)
    assert rec.filtros_vistos == [{"gestora": "G1"}]
    assert aud.eventos[-1]["usuario"] == "ana"
    assert aud.eventos[-1]["papel"] == "analista"


def test_dedup_nao_repete_trecho_entre_rodadas():
    repetidos = [TRECHOS[0], TRECHOS[0], TRECHOS[1]]
    resposta = Resposta(answer="ok", citations=[0], grounded=True, confidence=0.9)
    res, _, _ = _rodar(resposta, trechos=repetidos)
    assert len(res.trechos) == 2


def test_orcamento_de_turnos_e_respeitado():
    resposta = Resposta(answer="ok", citations=[0], grounded=True, confidence=0.9)
    res, _, _ = _rodar(resposta)
    assert res.turnos_usados <= 2


def test_auditoria_jsonl_e_append_only(tmp_path):
    from agentic_rag.domain.entities import EventoAuditoria

    caminho = tmp_path / "logs" / "auditoria.jsonl"
    aud = AuditoriaJSONL(caminho)
    for i in range(3):
        aud.registrar(EventoAuditoria(
            pergunta=f"p{i}", resposta="r", grounded=True, confidence=1.0, trechos=["abc"],
            versao_prompt="v", modelo="m", usuario="ana", papel="analista",
        ))
    assert len(aud.listar()) == 3
    assert len(caminho.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert aud.listar(usuario="joao") == []


def test_linha_corrompida_nao_derruba_a_leitura(tmp_path):
    caminho = tmp_path / "auditoria.jsonl"
    caminho.write_text('{"usuario":"ana"}\nlixo nao-json\n{"usuario":"ana"}\n', encoding="utf-8")
    assert len(AuditoriaJSONL(caminho).listar()) == 2
