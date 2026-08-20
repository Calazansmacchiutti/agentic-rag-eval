"""Smoke test da fundação: o grafo de modulos importa limpo, sem dep pesada.

Como todos os imports pesados (torch/qdrant/pymupdf/anthropic) sao preguiçosos,
importar cada modulo valida o esqueleto (config + schemas + fiacao interna) e
pega import circular/quebrado cedo, sem precisar do ambiente completo.
"""
import importlib

import pytest

# Camadas da fundacao para a borda. Se um destes falhar ao importar,
# a estrutura esta quebrada (nao e so uma feature faltando).
CORE_MODULES = [
    "agentic_rag.config",        # root: single source of truth
    "agentic_rag.ingest",
    "agentic_rag.llm",           # gateway de LLM
    "agentic_rag.retriever",
    "agentic_rag.rerank",        # reranker cross-encoder (opt-in)
    "agentic_rag.agent",         # loop agentic RAG
    "agentic_rag.baseline",      # RAG ingenuo (referencia do ADR 0001)
    "agentic_rag.api",           # borda de serving
    "agentic_rag.evaluate",
]

PDF_MODULES = [
    "agentic_rag.pdf.schemas",   # root do subsistema PDF
    "agentic_rag.pdf.probe",
    "agentic_rag.pdf.segmenter",
    "agentic_rag.pdf.evaluator",
    "agentic_rag.pdf.extract_agent",
    "agentic_rag.pdf.chunk_agent",
    "agentic_rag.pdf.indexer",
    "agentic_rag.pdf.ocr",
    "agentic_rag.pdf.cli",
    "agentic_rag.pdf.extract_cli",
]


@pytest.mark.parametrize("module", CORE_MODULES + PDF_MODULES)
def test_module_imports_clean(module):
    assert importlib.import_module(module) is not None


def test_config_is_the_root_singleton():
    """config.settings e o ponto unico de verdade que todo o grafo consome."""
    from agentic_rag.config import Settings, settings

    assert isinstance(settings, Settings)
    assert settings.llm_provider == "anthropic"  # default do ADR 0003
