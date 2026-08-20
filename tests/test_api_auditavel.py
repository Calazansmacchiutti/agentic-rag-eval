"""API auditavel: recusa e HTTP 200 com motivo, e toda resposta entra na trilha."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agentic_rag.adapters.inbound.api import criar_app, obter_servicos
from agentic_rag.adapters.outbound.auditoria_jsonl import AuditoriaMemoria
from agentic_rag.domain.entities import Resposta, Trecho
from agentic_rag.infrastructure.container import Servicos

TRECHOS = [Trecho(texto="A receita de 2025 foi de R$ 10 milhoes.", score=0.9, meta={"pagina": 2})]


class RecFake:
    def __init__(self):
        self.filtros_vistos = []

    def buscar(self, consulta, filtros=None):
        self.filtros_vistos.append(filtros)
        return list(TRECHOS)


class LLMFake:
    modelo_corrente = "modelo-de-teste"

    def __init__(self, resposta):
        self.resposta = resposta
        self.voltas = 0

    def loop_ferramentas(self, *, sistema, mensagens, ferramentas, modelo=None):
        self.voltas += 1
        if self.voltas == 1:
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", id="t1", input={"query": "q"})]
            )
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    def resposta_estruturada(self, *, sistema, mensagens, schema, modelo=None):
        return self.resposta

    def completar(self, prompt, **kw):
        return ""


@pytest.fixture
def cliente():
    """Monta o app com dubles; devolve (client, servicos) p/ inspecionar a auditoria."""
    def _montar(resposta):
        s = Servicos(
            recuperacao=RecFake(), llm=LLMFake(resposta), auditoria=AuditoriaMemoria(),
            sistema="sistema de teste", versao_prompt="v-teste", max_chamadas=2,
        )
        app = criar_app()
        app.dependency_overrides[obter_servicos] = lambda: s
        return TestClient(app), s
    return _montar


def test_health(cliente):
    c, _ = cliente(Resposta(answer="x", citations=[0], grounded=True, confidence=0.9))
    assert c.get("/health").json() == {"status": "ok"}


def test_resposta_fundamentada_traz_fontes_e_versao(cliente):
    c, s = cliente(Resposta(answer="R$ 10 milhoes.", citations=[0], grounded=True, confidence=0.9))
    r = c.post("/perguntar", json={"pergunta": "qual a receita?", "usuario": "ana"})
    assert r.status_code == 200
    d = r.json()
    assert d["permitido"] is True
    assert d["fontes"] == ["p.2"]
    assert d["versao_prompt"] == "v-teste"
    assert d["modelo"] == "modelo-de-teste"
    assert len(s.auditoria.listar()) == 1


def test_recusa_e_200_com_motivo_nao_erro(cliente):
    """Recusar nao e falha de servidor: e o guardrail funcionando."""
    c, _ = cliente(Resposta(answer="acho que sim", citations=[], grounded=True, confidence=0.7))
    r = c.post("/perguntar", json={"pergunta": "qual a receita?"})
    assert r.status_code == 200
    d = r.json()
    assert d["permitido"] is False
    assert d["decisao"] == "recusado_sem_citacao"
    assert d["motivo"]


def test_escopo_do_request_chega_ao_retriever(cliente):
    c, s = cliente(Resposta(answer="ok", citations=[0], grounded=True, confidence=0.9))
    c.post("/perguntar", json={
        "pergunta": "qual a receita?", "usuario": "ana", "papel": "analista",
        "filtros": {"gestora": "G1"},
    })
    assert s.recuperacao.filtros_vistos == [{"gestora": "G1"}]


def test_endpoint_de_auditoria_filtra_por_usuario(cliente):
    c, _ = cliente(Resposta(answer="ok", citations=[0], grounded=True, confidence=0.9))
    c.post("/perguntar", json={"pergunta": "p1", "usuario": "ana"})
    c.post("/perguntar", json={"pergunta": "p2", "usuario": "joao"})
    assert len(c.get("/auditoria").json()) == 2
    so_ana = c.get("/auditoria", params={"usuario": "ana"}).json()
    assert len(so_ana) == 1 and so_ana[0]["usuario"] == "ana"


def test_pergunta_vazia_e_rejeitada_pelo_schema(cliente):
    c, _ = cliente(Resposta(answer="x", citations=[0], grounded=True, confidence=0.9))
    assert c.post("/perguntar", json={"pergunta": ""}).status_code == 422
