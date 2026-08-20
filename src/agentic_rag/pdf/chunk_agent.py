"""Agente A: descobre o recorte ideal de um PDF para RAG.

Loop: profile (deterministico) -> propor CutPlan -> cortar -> avaliar -> refinar,
ate o score passar do limiar ou esgotar o orcamento de iteracoes. Guarda o melhor.

Por iteracao gasta no maximo 2 chamadas de LLM: 1 no judge (dentro de evaluate) e 1
no refinamento do plano. `use_llm=False` roda tudo deterministico (gratis).
"""
from __future__ import annotations

from agentic_rag.config import settings
from agentic_rag.pdf import evaluator, probe, segmenter
from agentic_rag.pdf.schemas import ChunkResult, CutPlan, DocProfile, ChunkEval


def _initial_plan(profile: DocProfile) -> CutPlan:
    """Plano inicial a partir do comportamento do PDF (ponto de partida, nao chute)."""
    table_aware = profile.recommended_strategy == "table_aware"
    return CutPlan(
        strategy=profile.recommended_strategy,
        target_chars=800,
        max_chars=1500,
        min_chars=200,
        overlap_chars=100,
        respect_headings=profile.structure_richness >= 0.33,
        keep_tables_whole=table_aware or profile.signals.has_tables,
        notes=f"inicial a partir do perfil {profile.doc_type} ({profile.rationale})",
    )


def _refine_plan(plan: CutPlan, ev: ChunkEval, profile: DocProfile):
    """LLM propoe um CutPlan melhor dadas as issues do recorte atual (1 chamada)."""
    from agentic_rag import llm

    prompt = (
        "Voce ajusta a estrategia de recorte de um PDF para RAG. Documento do tipo "
        f"'{profile.doc_type}' (richness={profile.structure_richness:.2f}, "
        f"tabelas={profile.signals.n_tables}).\n\n"
        f"Plano atual: {plan.model_dump_json()}\n"
        f"Avaliacao: score={ev.score:.2f}; problemas: {ev.issues}\n\n"
        "Proponha um CutPlan melhor que ataque esses problemas. Regras: se chunks racham "
        "frases, reduza target/aumente respeito a fronteira; se ficam grandes demais, baixe "
        "max_chars; se pouco autocontidos, ligue respect_headings e suba overlap. Mantenha "
        "min_chars<=target_chars<=max_chars."
    )
    return llm.complete(prompt, schema=CutPlan, model=settings.llm_model)


def structure(
    path: str,
    use_llm: bool = True,
    max_iter: int | None = None,
    threshold: float | None = None,
    ocr: str = "auto",
) -> ChunkResult:
    """Roda o loop e devolve o melhor ChunkResult (com a trilha de scores)."""
    max_iter = max_iter or settings.max_structure_iterations
    threshold = threshold if threshold is not None else settings.structure_score_threshold

    prof = probe.profile(path)
    blocks = probe.load_blocks(path, ocr=ocr)
    source = "\n".join(b.text for b in blocks if b.type == "text")

    plan = _initial_plan(prof)
    best: ChunkResult | None = None
    history: list[ChunkEval] = []

    for it in range(1, max_iter + 1):
        segs = segmenter.segment(blocks, prof, plan)
        ev = evaluator.evaluate(segs, plan, source, use_llm=use_llm)
        history.append(ev)

        if best is None or ev.score > best.evaluation.score:
            best = ChunkResult(plan=plan, segments=segs, evaluation=ev, iterations=it)

        if ev.score >= threshold or it == max_iter:
            break
        if it >= 2 and ev.score <= history[-2].score + 1e-3:
            break  # estagnou: nao adianta gastar mais LLM
        if use_llm:
            plan = _refine_plan(plan, ev, prof)
        else:
            break  # sem LLM nao ha como refinar a estrategia; fica no plano inicial

    assert best is not None
    best.history = history
    best.iterations = len(history)
    return best
