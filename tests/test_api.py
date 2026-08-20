from fastapi.testclient import TestClient

from agentic_rag import api
from agentic_rag.agent import Answer


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_ask_wires_agent(monkeypatch):
    """/ask deve chamar o agente e serializar resposta estruturada + fontes."""
    contexts = [{"text": "a resposta e 42", "score": 0.8}]
    fake = Answer(answer="42", citations=[0], grounded=True, confidence=0.9)

    def fake_answer_question(question, retriever):
        assert question == "qual a resposta?"
        return fake, contexts

    monkeypatch.setattr(api.agent_mod, "answer_question", fake_answer_question)
    # evita instanciar o Retriever real (Qdrant) no teste
    api.app.dependency_overrides[api.get_retriever] = lambda: object()
    try:
        resp = TestClient(api.app).post("/ask", json={"question": "qual a resposta?"})
    finally:
        api.app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "42"
    assert body["citations"] == [0]
    assert body["grounded"] is True
    assert body["confidence"] == 0.9
    assert body["sources"] == contexts


def test_ask_rejects_empty_question():
    resp = TestClient(api.app).post("/ask", json={"question": ""})
    assert resp.status_code == 422
