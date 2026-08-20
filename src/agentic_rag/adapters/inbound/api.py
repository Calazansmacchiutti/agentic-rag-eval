"""Adapter de entrada HTTP: expoe o caso de uso, sem regra propria.

Diferenca para o `api.py` legado: aqui a resposta carrega o VEREDITO dos guardrails e as
fontes legiveis, e existe endpoint de auditoria. Uma recusa e HTTP 200 com
`permitido=false` e motivo - nao e erro de servidor, e o sistema funcionando.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, Query
from pydantic import BaseModel, Field

from agentic_rag.domain.entities import Escopo
from agentic_rag.domain.use_cases import responder_pergunta as uc
from agentic_rag.infrastructure.container import Servicos, servicos

router = APIRouter()


class PerguntaIn(BaseModel):
    pergunta: str = Field(min_length=1, description="Pergunta em linguagem natural.")
    usuario: str = Field(default="anonimo", description="Quem pergunta (trilha de auditoria).")
    papel: str = Field(default="leitor", description="Papel do usuario (controle de escopo).")
    filtros: dict = Field(
        default_factory=dict,
        description="Restricao de escopo aplicada na RECUPERACAO, antes do LLM ver o texto.",
    )


class TrechoOut(BaseModel):
    texto: str
    score: float
    fonte: str
    id_conteudo: str


class RespostaOut(BaseModel):
    permitido: bool = Field(description="False = o sistema recusou; veja `motivo`.")
    decisao: str
    motivo: str
    answer: str
    citations: list[int]
    grounded: bool
    confidence: float
    fontes: list[str] = Field(description="Rotulos das fontes efetivamente citadas.")
    trechos: list[TrechoOut]
    versao_prompt: str
    modelo: str
    turnos_usados: int


def obter_servicos() -> Servicos:
    """Dependencia sobrescrivivel em teste via app.dependency_overrides."""
    return servicos()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/perguntar", response_model=RespostaOut)
def perguntar(entrada: PerguntaIn, s: Servicos = Depends(obter_servicos)) -> RespostaOut:  # noqa: B008 (Depends em default e o idioma do FastAPI)
    """Responde com fundamentacao auditavel; recusa explicitamente quando falta lastro."""
    resultado = uc.responder(
        entrada.pergunta,
        recuperacao=s.recuperacao,
        llm=s.llm,
        sistema=s.sistema,
        versao_prompt=s.versao_prompt,
        escopo=Escopo(usuario=entrada.usuario, papel=entrada.papel, filtros=entrada.filtros),
        auditoria=s.auditoria,
        max_chamadas=s.max_chamadas,
    )
    return RespostaOut(
        permitido=resultado.permitido,
        decisao=resultado.decisao,
        motivo=resultado.motivo,
        answer=resultado.resposta.answer,
        citations=resultado.resposta.citations,
        grounded=resultado.resposta.grounded,
        confidence=resultado.resposta.confidence,
        fontes=resultado.fontes,
        trechos=[
            TrechoOut(texto=t.texto, score=t.score, fonte=t.fonte, id_conteudo=t.id_conteudo)
            for t in resultado.trechos
        ],
        versao_prompt=s.versao_prompt,
        modelo=s.llm.modelo_corrente,
        turnos_usados=resultado.turnos_usados,
    )


@router.get("/auditoria")
def auditoria(
    usuario: str | None = Query(default=None, description="Filtra por usuario."),
    limite: int = Query(default=50, ge=1, le=1000),
    s: Servicos = Depends(obter_servicos),  # noqa: B008 (idioma do FastAPI)
) -> list[dict]:
    """Trilha das respostas ja dadas. Em producao isto fica atras de autenticacao."""
    return s.auditoria.listar(usuario=usuario, limite=limite)


def criar_app() -> FastAPI:
    app = FastAPI(
        title="agentic-rag-eval",
        description="RAG agentic com fundamentacao auditavel: guardrails, escopo e trilha.",
    )
    app.include_router(router)
    return app
