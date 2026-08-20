"""Carregamento de prompt a partir de arquivo versionado.

Por que o prompt nao mora no codigo: mudar prompt e mudar comportamento do sistema. Como
arquivo, a alteracao aparece como diff legivel na revisao, e o hash do conteudo vira a
"versao" gravada na auditoria - dando para saber exatamente qual texto produziu qual
resposta, meses depois.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# raiz do repo: .../src/agentic_rag/infrastructure/prompts.py -> sobe 4 niveis
RAIZ = Path(__file__).resolve().parents[3]
DIR_PROMPTS = RAIZ / "prompts"

PADRAO = (
    "Voce e um assistente de QA sobre uma base de documentos. "
    "Use a tool `search` para recuperar contexto antes de responder. "
    "Responda APENAS com base no contexto recuperado e cite os indices [n] dos trechos usados. "
    "Se o contexto nao sustentar a resposta, defina grounded=false e seja explicito."
)


@dataclass(frozen=True)
class Prompt:
    """Texto + identidade. `versao` e o hash curto do conteudo, nao um numero manual."""

    nome: str
    texto: str
    versao: str

    def __str__(self) -> str:
        return self.texto


@lru_cache(maxsize=8)
def carregar(nome: str = "system") -> Prompt:
    """Le `prompts/<nome>.md`. Se o arquivo nao existir, cai no texto embutido.

    O fallback existe para o pacote seguir funcionando instalado como wheel (sem a pasta
    `prompts/` do repo) e para nao quebrar teste que nao monta o diretorio.
    """
    caminho = DIR_PROMPTS / f"{nome}.md"
    if caminho.exists():
        texto = caminho.read_text(encoding="utf-8").strip()
        origem = "arquivo"
    else:
        texto = PADRAO
        origem = "embutido"
    versao = hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]
    return Prompt(nome=f"{nome}:{origem}", texto=texto, versao=versao)
