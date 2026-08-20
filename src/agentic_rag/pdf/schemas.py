"""Contratos de dados (Pydantic) do estruturador de PDF.

Tres camadas:
  1. Sinais de layout brutos (Block, LayoutSignals) - saida do probe deterministico.
  2. Perfil do documento (DocProfile) - o "comportamento" do PDF + estrategia recomendada.
     Compartilhado pelos dois agentes.
  3. Saidas dos agentes (CutPlan/Segment/ChunkResult/ChunkEval p/ RAG; ExtractionResult
     p/ campos).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------- #
# 1. Sinais de layout brutos (probe deterministico)
# ----------------------------------------------------------------------------- #

BlockType = Literal["text", "table", "image"]


class Block(BaseModel):
    """Um bloco de layout extraido da pagina (texto, tabela ou imagem)."""

    page: int
    type: BlockType = "text"
    text: str = ""
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float | None = None           # tamanho dominante do bloco (texto)
    bold: bool = False
    n_lines: int = 0


class LayoutSignals(BaseModel):
    """Agregado deterministico do documento inteiro. Base para o perfil."""

    n_pages: int
    n_blocks: int
    has_toc: bool = False                     # bookmarks/sumario nativo
    has_tables: bool = False
    n_tables: int = 0
    is_scanned: bool = False                  # quase sem texto extraivel -> provavel OCR
    multi_column: bool = False
    body_font_size: float | None = None       # tamanho de fonte mais comum (corpo)
    heading_font_sizes: list[float] = Field(default_factory=list)  # > corpo, ordenado desc
    avg_blocks_per_page: float = 0.0
    text_density: float = 0.0                 # chars por pagina (proxy de densidade)


# ----------------------------------------------------------------------------- #
# 2. Perfil do documento (o "comportamento") - compartilhado pelos agentes
# ----------------------------------------------------------------------------- #

DocType = Literal[
    "article",      # artigo/paper: headings claros, denso, 1-2 colunas
    "report",       # relatorio: secoes + tabelas
    "form",         # formulario: campos rotulados, pouco texto corrido
    "slides",       # apresentacao: pouco texto por pagina, titulos grandes
    "table_heavy",  # planilhao/financeiro: dominado por tabelas
    "scanned",      # digitalizado: precisa de OCR antes
    "unknown",
]

CutStrategy = Literal[
    "by_heading",     # cortar na hierarquia de titulos detectada
    "by_section",     # secoes semanticas (LLM ajuda a achar fronteira)
    "by_page",        # 1 chunk por pagina (slides, formularios)
    "by_block",       # bloco a bloco (denso sem heading claro)
    "table_aware",    # isola tabelas como chunks proprios, texto a parte
]


class DocProfile(BaseModel):
    """O comportamento do PDF + estrategia recomendada (heuristica, refinavel por LLM)."""

    doc_type: DocType = "unknown"
    structure_richness: float = 0.0  # 0..1: quao clara e a hierarquia de headings
    table_heaviness: float = 0.0     # 0..1: fracao de area/blocos em tabela
    recommended_strategy: CutStrategy = "by_block"
    needs_ocr: bool = False
    rationale: str = ""              # por que esse perfil (auditavel)
    signals: LayoutSignals


# ----------------------------------------------------------------------------- #
# 3a. Agente A - chunking adaptativo para RAG
# ----------------------------------------------------------------------------- #


class CutPlan(BaseModel):
    """Estrategia de recorte que o agente propoe (e refina no loop)."""

    strategy: CutStrategy
    target_chars: int = 800          # tamanho alvo do chunk
    max_chars: int = 1500            # teto duro
    min_chars: int = 200             # piso (evita chunk-cocoruto)
    overlap_chars: int = 100
    respect_headings: bool = True    # nao cruzar fronteira de heading
    keep_tables_whole: bool = True   # nunca rachar tabela
    notes: str = ""


class Segment(BaseModel):
    """Um chunk resultante do recorte."""

    index: int
    text: str
    type: BlockType = "text"
    page_start: int
    page_end: int
    heading_path: list[str] = Field(default_factory=list)  # ex.: ["3. Metodos", "3.1 Dados"]
    n_chars: int = 0


class ChunkEval(BaseModel):
    """Avaliacao do recorte. Composto deterministico + LLM-judge. Objetivo do loop."""

    coverage: float = 0.0            # 0..1: fracao do texto-fonte preservada (sem perda)
    boundary_integrity: float = 0.0  # 0..1: chunks que NAO racham frase/tabela
    size_fitness: float = 0.0        # 0..1: chunks dentro de [min, max]
    self_contained: float = 0.0      # 0..1: LLM-judge, chunk entendivel sozinho
    topical_coherence: float = 0.0   # 0..1: LLM-judge, um topico por chunk
    score: float = 0.0               # composto ponderado
    issues: list[str] = Field(default_factory=list)


class ChunkResult(BaseModel):
    """Saida do agente A: o melhor recorte encontrado + plano + avaliacao + trilha do loop."""

    plan: CutPlan
    segments: list[Segment]
    evaluation: ChunkEval
    iterations: int = 1
    history: list[ChunkEval] = Field(default_factory=list)  # score por iteracao (auditavel)


# ----------------------------------------------------------------------------- #
# 3b. Agente B - extracao de campos estruturados
# ----------------------------------------------------------------------------- #


class FieldSpec(BaseModel):
    """Um campo que o agente B deve extrair."""

    name: str
    description: str
    required: bool = False


class ExtractedField(BaseModel):
    name: str
    value: str | None = None
    page: int | None = None
    confidence: float = 0.0          # 0..1 (auto-reportado pelo modelo, calibrar com eval)


class ExtractionResult(BaseModel):
    """Saida do agente B: campos extraidos + cobertura dos obrigatorios."""

    fields: list[ExtractedField]
    missing_required: list[str] = Field(default_factory=list)
    notes: str = ""
