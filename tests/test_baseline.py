"""Testes do RAG ingenuo (baseline do ADR 0001) sem rede.

Fixa o contrato do baseline: exatamente 1 retrieval + 1 chamada de LLM, contexto numerado
no prompt e mesma forma de saida usada pelo eval harness.
"""
from agentic_rag import baseline
from agentic_rag import llm as llm_mod


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.queries = []

    def search(self, query, filters=None):
        self.queries.append(query)
        return list(self._hits)


def test_naive_faz_um_retrieval_e_uma_chamada(monkeypatch):
    prompts = []
    monkeypatch.setattr(llm_mod, "complete", lambda prompt, **k: prompts.append(prompt) or "resposta X")
    r = FakeRetriever([{"text": "ctx1", "score": 0.9}, {"text": "ctx2", "score": 0.8}])

    text, contexts = baseline.answer_question("pergunta?", r)

    assert text == "resposta X"
    assert len(prompts) == 1           # 1 chamada de LLM (ingenuo, sem loop)
    assert r.queries == ["pergunta?"]  # 1 retrieval (top-k fixo)
    assert contexts[0]["text"] == "ctx1"
    assert "[0] ctx1" in prompts[0] and "[1] ctx2" in prompts[0]  # contexto numerado no prompt


def test_prompt_lida_com_contexto_vazio(monkeypatch):
    prompts = []
    monkeypatch.setattr(llm_mod, "complete", lambda prompt, **k: prompts.append(prompt) or "nao sei")
    text, contexts = baseline.answer_question("q", FakeRetriever([]))

    assert contexts == []
    assert "(sem resultados)" in prompts[0]


def test_eval_answer_fn_devolve_texto_e_lista_de_contextos(monkeypatch):
    monkeypatch.setattr(llm_mod, "complete", lambda prompt, **k: "ans")
    r = FakeRetriever([{"text": "c1", "score": 0.5}, {"text": "c2", "score": 0.4}])

    fn = baseline.eval_answer_fn(r)
    ans, ctxs = fn("q")

    assert ans == "ans"
    assert ctxs == ["c1", "c2"]  # forma esperada pelo build_records do eval
