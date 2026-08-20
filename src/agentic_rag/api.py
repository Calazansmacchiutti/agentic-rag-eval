"""Serving: FastAPI expoe /ask ligado ao loop agentic (agent.answer_question)."""
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from agentic_rag import agent as agent_mod
from agentic_rag.retriever import Retriever

app = FastAPI(title="agentic-rag-eval")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Pergunta em linguagem natural.")


class AskResponse(BaseModel):
    """Espelha a saida estruturada do agente + o contexto recuperado (p/ auditoria)."""

    answer: str
    citations: list[int]
    grounded: bool
    confidence: float
    sources: list[dict]


def get_retriever() -> Retriever:
    """Dependencia do retriever; sobrescrevivel em teste via app.dependency_overrides."""
    return Retriever()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, retriever: Retriever = Depends(get_retriever)) -> AskResponse:
    """Roda o agente sobre a pergunta e devolve resposta estruturada + fontes citaveis."""
    result, contexts = agent_mod.answer_question(req.question, retriever)
    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        grounded=result.grounded,
        confidence=result.confidence,
        sources=contexts,
    )
