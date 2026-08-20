"""Testes do loop agentic (agent.answer_question) sem rede nem SDK real.

Fixam o comportamento *estrutural* do nucleo RAG:
- respeita o orcamento de chamadas de LLM (max_llm_calls_per_item);
- reserva a ultima chamada para a resposta estruturada (parse);
- chama a tool `search`, acumula os contextos e casa a numeracao das citacoes;
- para cedo se o modelo responde sem pedir busca.

Estrategia: injeta um FakeClient em llm.client() e um FakeRetriever, entao
nenhuma dependencia pesada (anthropic/qdrant) e exercitada.
"""
from types import SimpleNamespace

from agentic_rag import agent
from agentic_rag.agent import Answer


def _tool_use(query, tid="t1"):
    """Resposta do modelo pedindo a tool `search`."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tid, input={"query": query})]
    )


def _text(text="pronto"):
    """Resposta do modelo sem tool_use (responde direto)."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeMessages:
    def __init__(self, create_responses, parsed):
        self._create = list(create_responses)
        self._parsed = parsed
        self.create_calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._create.pop(0)

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return SimpleNamespace(parsed_output=self._parsed)


class FakeClient:
    def __init__(self, messages):
        self.messages = messages


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.queries = []

    def search(self, query, filters=None):
        self.queries.append(query)
        return list(self._hits)


def _wire(monkeypatch, messages, budget):
    monkeypatch.setattr(agent.llm, "client", lambda: FakeClient(messages))
    monkeypatch.setattr(agent.settings, "max_llm_calls_per_item", budget)


def test_retrieves_then_structured_answer(monkeypatch):
    """Budget 2: 1 rodada de retrieval (create) + 1 resposta estruturada (parse)."""
    parsed = Answer(answer="42", citations=[0], grounded=True, confidence=0.8)
    msgs = FakeMessages([_tool_use("qual a resposta")], parsed)
    _wire(monkeypatch, msgs, budget=2)
    retriever = FakeRetriever([{"text": "a resposta e 42", "score": 0.9}])

    result, contexts = agent.answer_question("qual a resposta", retriever)

    assert result is parsed
    assert retriever.queries == ["qual a resposta"]          # a tool foi chamada
    assert contexts == [{"text": "a resposta e 42", "score": 0.9}]
    assert len(msgs.create_calls) == 1                        # 1 rodada de retrieval
    assert len(msgs.parse_calls) == 1                         # 1 saida estruturada


def test_respects_call_budget(monkeypatch):
    """Budget 1: budget-1 = 0 rodadas de retrieval; so a resposta estruturada."""
    parsed = Answer(answer="sem contexto", citations=[], grounded=False, confidence=0.1)
    msgs = FakeMessages([], parsed)  # nenhuma resposta de create disponivel de proposito
    _wire(monkeypatch, msgs, budget=1)
    retriever = FakeRetriever([{"text": "nao deveria ser usado"}])

    result, contexts = agent.answer_question("q", retriever)

    assert msgs.create_calls == []       # nenhuma chamada de retrieval
    assert retriever.queries == []       # nenhuma busca
    assert contexts == []
    assert len(msgs.parse_calls) == 1    # ainda produz resposta estruturada
    assert result is parsed


def test_stops_when_model_skips_search(monkeypatch):
    """Se o modelo responde sem pedir a tool, o loop para e vai direto ao parse."""
    parsed = Answer(answer="ok", citations=[], grounded=True, confidence=0.5)
    msgs = FakeMessages([_text("nao preciso buscar")], parsed)
    _wire(monkeypatch, msgs, budget=3)  # daria 2 rodadas, mas para na 1a
    retriever = FakeRetriever([{"text": "x"}])

    result, contexts = agent.answer_question("q", retriever)

    assert len(msgs.create_calls) == 1   # entrou 1x; sem tool_use -> break
    assert retriever.queries == []
    assert contexts == []
    assert result is parsed


def test_accumulates_contexts_across_rounds(monkeypatch):
    """Budget 3: duas rodadas de busca acumulam contextos DISTINTOS na ordem recuperada."""
    parsed = Answer(answer="combinado", citations=[0, 1], grounded=True, confidence=0.7)
    msgs = FakeMessages([_tool_use("a", "t1"), _tool_use("b", "t2")], parsed)
    _wire(monkeypatch, msgs, budget=3)

    class _PerQueryRetriever:  # cada busca traz um trecho diferente
        def __init__(self):
            self.queries = []

        def search(self, query, filters=None):
            self.queries.append(query)
            return [{"text": f"hit-{query}", "score": 0.5}]

    retriever = _PerQueryRetriever()
    _, contexts = agent.answer_question("q", retriever)

    assert retriever.queries == ["a", "b"]
    assert len(msgs.create_calls) == 2                    # budget-1 = 2 rodadas de retrieval
    assert [c["text"] for c in contexts] == ["hit-a", "hit-b"]  # acumulados na ordem


def test_dedup_helper_remove_repetidos_no_batch_e_entre_chamadas():
    seen: set[str] = set()
    out1 = agent._dedup([{"text": "a"}, {"text": "a"}, {"text": "b"}, {"text": ""}], seen)
    assert [h["text"] for h in out1] == ["a", "b"]   # dedup no batch + ignora vazio
    out2 = agent._dedup([{"text": "a"}, {"text": "c"}], seen)
    assert [h["text"] for h in out2] == ["c"]         # "a" ja visto na rodada anterior


def test_agente_deduplica_contexto_entre_rodadas(monkeypatch):
    """Duas buscas que re-recuperam o mesmo trecho nao devem inflar `contexts`."""
    parsed = Answer(answer="x", citations=[0], grounded=True, confidence=0.6)
    msgs = FakeMessages([_tool_use("a", "t1"), _tool_use("b", "t2")], parsed)
    _wire(monkeypatch, msgs, budget=3)
    # o retriever devolve o MESMO conjunto nas duas rodadas
    retriever = FakeRetriever([{"text": "dup", "score": 0.9}, {"text": "unico", "score": 0.5}])

    _, contexts = agent.answer_question("q", retriever)

    texts = [c["text"] for c in contexts]
    assert texts.count("dup") == 1     # duplicata entre rodadas removida
    assert "unico" in texts
    assert len(contexts) == 2          # sem o curador seriam 4


def test_format_hits_numbers_from_start():
    """A numeracao dos trechos comeca em `start` p/ casar com as citacoes globais."""
    hits = [{"text": "a"}, {"text": "b"}]
    assert agent._format_hits(hits, 3) == "[3] a\n[4] b"


def test_format_hits_empty():
    assert agent._format_hits([], 0) == "(sem resultados)"
