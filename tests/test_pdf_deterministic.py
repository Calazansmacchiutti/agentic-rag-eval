from typing import ClassVar

"""Testes deterministicos do estruturador de PDF (sem LLM, sem rede)."""
from agentic_rag.pdf import evaluator, extract_agent, indexer, probe, segmenter
from agentic_rag.pdf.schemas import (
    Block,
    ChunkEval,
    ChunkResult,
    CutPlan,
    DocProfile,
    FieldSpec,
    LayoutSignals,
    Segment,
)


def _profile(strategy="by_block", headings=None):
    sig = LayoutSignals(
        n_pages=1, n_blocks=1, body_font_size=10.0,
        heading_font_sizes=headings or [], avg_blocks_per_page=1.0, text_density=1000.0,
    )
    return DocProfile(
        doc_type="article", structure_richness=1.0, table_heaviness=0.0,
        recommended_strategy=strategy, signals=sig,
    )


# --------------------------------------------------------------------------- evaluator
def test_coverage_identico_e_alto():
    source = "palavra " * 60
    segs = [Segment(index=0, text=source, page_start=0, page_end=0, n_chars=len(source))]
    assert evaluator._coverage(source, segs) > 0.95


def test_coverage_parcial_cai():
    # conteudo DISTINTO (nao repetido), com o segmento cobrindo so ~40%
    source = " ".join(f"w{i}" for i in range(100))
    segs = [Segment(index=0, text=" ".join(f"w{i}" for i in range(40)), page_start=0, page_end=0)]
    assert evaluator._coverage(source, segs) < 0.95


def test_eval_sem_llm_usa_pesos_renormalizados():
    plan = CutPlan(strategy="by_block", min_chars=5, max_chars=1000)
    source = "uma frase completa que termina bem."
    segs = [Segment(index=0, text=source, page_start=0, page_end=0, n_chars=len(source))]
    ev = evaluator.evaluate(segs, plan, source, use_llm=False)
    assert ev.self_contained == 0.0          # LLM desligado
    assert ev.boundary_integrity == 1.0      # termina com "."
    assert 0.0 < ev.score <= 1.0


# --------------------------------------------------------------------------- segmenter
def test_by_block_respeita_o_teto():
    blocks = [Block(page=0, text="x" * 300, bbox=(0, i, 1, i + 1), font_size=10.0) for i in range(6)]
    plan = CutPlan(strategy="by_block", target_chars=800, max_chars=1500, min_chars=1, overlap_chars=0)
    segs = segmenter.segment(blocks, _profile("by_block"), plan)
    assert len(segs) >= 2
    assert all(s.n_chars <= plan.max_chars for s in segs)


def test_by_heading_cria_fronteira_e_path():
    blocks = [
        Block(page=0, text="Introducao", bbox=(0, 0, 1, 1), font_size=16.0, bold=True),
        Block(page=0, text="corpo da introducao. " * 5, bbox=(0, 1, 1, 2), font_size=10.0),
        Block(page=0, text="Metodos", bbox=(0, 2, 1, 3), font_size=16.0, bold=True),
        Block(page=0, text="corpo dos metodos. " * 5, bbox=(0, 3, 1, 4), font_size=10.0),
    ]
    plan = CutPlan(strategy="by_heading", target_chars=5000, max_chars=9000, min_chars=1, overlap_chars=0)
    segs = segmenter.segment(blocks, _profile("by_heading", headings=[16.0]), plan)
    paths = {tuple(s.heading_path) for s in segs}
    assert ("Introducao",) in paths and ("Metodos",) in paths


def test_tabela_vira_chunk_proprio():
    blocks = [
        Block(page=0, text="texto antes", bbox=(0, 0, 1, 1), font_size=10.0),
        Block(page=0, type="table", text="a | b\n1 | 2", bbox=(0, 1, 1, 2)),
    ]
    plan = CutPlan(strategy="table_aware", min_chars=1, max_chars=9000, keep_tables_whole=True)
    segs = segmenter.segment(blocks, _profile("table_aware"), plan)
    assert any(s.type == "table" for s in segs)


# --------------------------------------------------------------------------- probe (PDF real)
def test_probe_em_pdf_gerado(tmp_path):
    import fitz

    p = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Titulo Grande", fontsize=22)
    page.insert_text((72, 110), "Paragrafo de corpo com bastante texto. " * 8, fontsize=10)
    doc.save(p)
    doc.close()

    blocks = probe.extract_blocks(str(p))
    assert any(b.type == "text" and b.text for b in blocks)
    prof = probe.profile(str(p))
    assert prof.signals.n_pages == 1
    assert prof.recommended_strategy in {"by_heading", "by_block", "by_page", "table_aware"}


# --------------------------------------------------------------------------- extract dry-run
def test_extract_dryrun_lista_obrigatorios(tmp_path):
    import fitz

    p = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "conteudo qualquer", fontsize=11)
    doc.save(p)
    doc.close()

    fields = [
        FieldSpec(name="titulo", description="titulo do doc", required=True),
        FieldSpec(name="data", description="data", required=False),
    ]
    res = extract_agent.extract(str(p), fields, use_llm=False)
    assert res.missing_required == ["titulo"]
    assert res.fields == []


# --------------------------------------------------------------------------- indexer
class _StubRetriever:
    def __init__(self):
        self.chunks = None
        self.metas = None

    def index_chunks(self, chunks, metadatas=None):
        self.chunks, self.metas = chunks, metadatas
        return len(chunks)


def test_indexer_preserva_recorte_e_metadados():
    result = ChunkResult(
        plan=CutPlan(strategy="by_heading"),
        segments=[
            Segment(index=0, text="intro", page_start=0, page_end=0, heading_path=["Intro"], n_chars=5),
            Segment(index=1, text="a | b", type="table", page_start=1, page_end=1, n_chars=5),
        ],
        evaluation=ChunkEval(),
    )
    stub = _StubRetriever()
    n = indexer.index_result("doc.pdf", result, retriever=stub)
    assert n == 2
    assert stub.chunks == ["intro", "a | b"]               # nao re-chunkou
    assert stub.metas[0]["heading_path"] == ["Intro"]      # metadado preservado
    assert stub.metas[1]["type"] == "table"
    assert all(m["doc_id"] == "doc.pdf" and m["strategy"] == "by_heading" for m in stub.metas)


# --------------------------------------------------------------------------- busca filtrada
def test_search_monta_filtro_por_metadado(monkeypatch):
    import numpy as np

    from agentic_rag import ingest
    from agentic_rag.retriever import Retriever

    monkeypatch.setattr(ingest, "embed", lambda xs: np.zeros((len(xs), 3), dtype="float32"))

    class _Res:
        points: ClassVar[list] = []

    class _Cli:
        def __init__(self):
            self.kwargs = None

        def query_points(self, **kw):
            self.kwargs = kw
            return _Res()

    r = Retriever(top_k=3)
    r._client = _Cli()

    r.search("q", filters={"type": "table", "heading_path": "Metodos"})
    qf = r._client.kwargs["query_filter"]
    assert qf is not None and len(qf.must) == 2     # uma condicao por campo

    r.search("q")                                    # sem filtro -> None
    assert r._client.kwargs["query_filter"] is None


# --------------------------------------------------------------------------- OCR (wiring)
def _pdf_com_texto(tmp_path, txt="texto nativo aqui"):
    import fitz

    p = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), txt, fontsize=11)
    doc.save(p)
    doc.close()
    return str(p)


def test_ocr_available_retorna_bool():
    from agentic_rag.pdf import ocr

    assert isinstance(ocr.available(), bool)


def test_load_blocks_off_nunca_chama_ocr(tmp_path, monkeypatch):
    from agentic_rag.pdf import ocr as ocr_mod

    def _boom(*a, **k):
        raise AssertionError("OCR nao deveria ser chamado com ocr='off'")

    monkeypatch.setattr(ocr_mod, "available", lambda: True)
    monkeypatch.setattr(ocr_mod, "ocr_blocks", _boom)
    blocks = probe.load_blocks(_pdf_com_texto(tmp_path), ocr="off")
    assert any("nativo" in b.text for b in blocks)


def test_load_blocks_on_usa_ocr_quando_disponivel(tmp_path, monkeypatch):
    from agentic_rag.pdf import ocr as ocr_mod

    monkeypatch.setattr(ocr_mod, "available", lambda: True)
    monkeypatch.setattr(ocr_mod, "ocr_blocks", lambda *a, **k: [Block(page=0, text="TEXTO OCR", bbox=(0, 0, 1, 1))])
    blocks = probe.load_blocks(_pdf_com_texto(tmp_path), ocr="on")
    assert any(b.text == "TEXTO OCR" for b in blocks)


def test_load_blocks_cai_para_nativo_sem_tesseract(tmp_path, monkeypatch):
    from agentic_rag.pdf import ocr as ocr_mod

    monkeypatch.setattr(ocr_mod, "available", lambda: False)  # Tesseract ausente
    blocks = probe.load_blocks(_pdf_com_texto(tmp_path, "so nativo"), ocr="on")
    assert any("nativo" in b.text for b in blocks)
