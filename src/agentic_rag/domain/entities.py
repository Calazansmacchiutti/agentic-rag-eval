"""Entidades do dominio. Sem dependencia de fornecedor, framework ou I/O.

Regra da camada: nada aqui importa anthropic, qdrant, fastapi ou sentence-transformers.
Se um import desses aparecer neste arquivo, a separacao hexagonal foi quebrada.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Trecho:
    """Um pedaco de documento recuperado, com a origem preservada para citacao.

    `score` e a similaridade do retriever; `meta` carrega o payload da fonte
    (pagina, heading_path, tipo) sem que o dominio precise conhecer o formato.
    """

    texto: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def id_conteudo(self) -> str:
        """Hash do texto: identidade estavel do trecho para dedup e trilha de auditoria."""
        return hashlib.sha256(self.texto.strip().encode("utf-8")).hexdigest()[:16]

    @property
    def fonte(self) -> str:
        """Rotulo legivel da origem, para citacao humana."""
        m = self.meta or {}
        partes = [str(m[k]) for k in ("documento", "arquivo", "source", "heading_path") if m.get(k)]
        if m.get("pagina"):
            partes.append(f"p.{m['pagina']}")
        return " · ".join(partes) or "fonte nao identificada"


class Resposta(BaseModel):
    """Saida estruturada do agente, com auto-checagem de fundamentacao embutida.

    `citations` sao indices na lista de trechos entregue ao modelo; `grounded=False`
    e uma recusa honesta, nao um erro - o guardrail depende dela.
    """

    answer: str = Field(description="Resposta fundamentada apenas no contexto recuperado.")
    citations: list[int] = Field(
        default_factory=list, description="Indices dos trechos de contexto efetivamente usados."
    )
    grounded: bool = Field(description="True se o contexto sustenta a resposta; False se insuficiente.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianca de 0 a 1.")


@dataclass(frozen=True)
class Escopo:
    """Quem esta perguntando e o que essa pessoa pode ver.

    Existe porque em contexto regulado a mesma pergunta tem respostas diferentes conforme
    o papel de quem pergunta - e responder fora do escopo e vazamento, nao imprecisao.
    `filtros` viram filtro de metadado no retriever: a restricao acontece ANTES do LLM.
    """

    usuario: str = "anonimo"
    papel: str = "leitor"
    filtros: dict = field(default_factory=dict)

    @property
    def anonimo(self) -> bool:
        return self.usuario == "anonimo"


@dataclass(frozen=True)
class EventoAuditoria:
    """Registro imutavel de uma resposta, suficiente para reconstruir a decisao depois.

    Guarda o QUE foi respondido, COM QUE trechos, POR QUAL versao de prompt e modelo.
    Sem os tres, "o sistema disse isso" nao e auditavel - e so uma alegacao.
    """

    pergunta: str
    resposta: str
    grounded: bool
    confidence: float
    trechos: list[str]          # ids de conteudo, nao o texto: o log nao replica o corpus
    versao_prompt: str
    modelo: str
    usuario: str
    papel: str
    decisao: str = "respondido"  # respondido | recusado_sem_citacao | recusado_fora_de_escopo
    motivo: str = ""
    em: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)
