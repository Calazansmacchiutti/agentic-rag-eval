"""CLI do estruturador de PDF (agente A).

Uso:
  python -m agentic_rag.pdf.cli caminho.pdf            # loop completo (com LLM)
  python -m agentic_rag.pdf.cli caminho.pdf --no-llm   # so deterministico (gratis)
  python -m agentic_rag.pdf.cli caminho.pdf --json      # despeja o ChunkResult inteiro
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Estruturador de PDF: descobre o recorte ideal p/ RAG.")
    ap.add_argument("pdf", help="caminho do PDF")
    ap.add_argument("--no-llm", action="store_true", help="roda 100%% deterministico (sem custo)")
    ap.add_argument("--json", action="store_true", help="imprime o ChunkResult completo em JSON")
    ap.add_argument("--index", action="store_true", help="indexa os chunks no Qdrant (precisa do servidor no ar)")
    ap.add_argument("--ocr", choices=["auto", "on", "off"], default="auto",
                    help="OCR p/ digitalizado (auto=so se preciso e Tesseract disponivel)")
    args = ap.parse_args(argv)

    from agentic_rag.pdf import chunk_agent, probe

    prof = probe.profile(args.pdf)
    print(f"# perfil: {prof.doc_type}  | estrategia inicial: {prof.recommended_strategy}")
    print(f"  richness={prof.structure_richness:.2f} table_heaviness={prof.table_heaviness:.2f} "
          f"paginas={prof.signals.n_pages} tabelas={prof.signals.n_tables} "
          f"ocr={prof.needs_ocr}")
    print(f"  motivo: {prof.rationale}")

    res = chunk_agent.structure(args.pdf, use_llm=not args.no_llm, ocr=args.ocr)

    if args.json:
        print(res.model_dump_json(indent=2))
        return 0

    ev = res.evaluation
    print(f"\n# recorte: {len(res.segments)} chunks em {res.iterations} iteracao(oes) | "
          f"estrategia final: {res.plan.strategy}")
    print(f"  score={ev.score:.2f}  cobertura={ev.coverage:.2f} autocontido={ev.self_contained:.2f} "
          f"fronteira={ev.boundary_integrity:.2f} tamanho={ev.size_fitness:.2f}")
    print(f"  trilha de score: {[round(h.score, 2) for h in res.history]}")
    if ev.issues:
        print("  issues:", "; ".join(ev.issues))
    print("\n# primeiros chunks:")
    for s in res.segments[:3]:
        head = " > ".join(s.heading_path) or "(sem heading)"
        print(f"  [{s.index}] p{s.page_start}-{s.page_end} | {head} | {s.n_chars} chars")
        print(f"      {s.text[:160].replace(chr(10), ' ')}...")

    if args.index:
        import os

        from agentic_rag.pdf import indexer

        doc_id = os.path.basename(args.pdf)
        n = indexer.index_result(doc_id, res)
        print(f"\n# indexado no Qdrant: {n} pontos (doc_id={doc_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
