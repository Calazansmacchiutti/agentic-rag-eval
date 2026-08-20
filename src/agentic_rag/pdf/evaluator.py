"""Avalia um recorte. Composto deterministico + LLM-judge. E o objetivo do loop.

Pesos (equilibrio cobertura x autocontido, definido pelo usuario):
  cobertura 0.30, autocontido 0.30, integridade de fronteira 0.20,
  tamanho 0.10, coerencia topica 0.10.

`use_llm=False` roda 100% deterministico (gratis, p/ smoke test); o score e
renormalizado sobre as metricas disponiveis.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agentic_rag.config import settings
from agentic_rag.pdf.schemas import ChunkEval, CutPlan, Segment

_W = {
    "coverage": 0.30,
    "self_contained": 0.30,
    "boundary_integrity": 0.20,
    "size_fitness": 0.10,
    "topical_coherence": 0.10,
}
_LLM_KEYS = {"self_contained", "topical_coherence"}
_SENT_END = tuple(".!?:)]”\"'")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def _shingles(t: str, n: int = 8) -> set[str]:
    """Shingles de n palavras (independentes de posicao) p/ medir cobertura.

    n-gramas de palavra casam o MESMO trecho no source e no segmento, mesmo em offsets
    diferentes (ao contrario de janela de char por offset, que nao alinha).
    """
    words = _norm(t).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _coverage(source: str, segments: list[Segment]) -> float:
    src = _shingles(source)
    if not src:
        return 1.0
    seg = set()
    for s in segments:
        seg |= _shingles(s.text)
    return len(src & seg) / len(src)


def _boundary_integrity(segments: list[Segment]) -> tuple[float, int]:
    """Fracao de segmentos que NAO terminam no meio de uma frase."""
    if not segments:
        return 0.0, 0
    ok = 0
    broken = 0
    for s in segments:
        t = s.text.rstrip()
        if not t or t[-1] in _SENT_END:
            ok += 1
        else:
            broken += 1
    return ok / len(segments), broken


def _size_fitness(segments: list[Segment], plan: CutPlan) -> tuple[float, int]:
    if not segments:
        return 0.0, 0
    bad = sum(1 for s in segments if not (plan.min_chars <= s.n_chars <= plan.max_chars))
    return (len(segments) - bad) / len(segments), bad


# --------------------------------------------------------------------------- #
# LLM-judge (batelado: 1 chamada por avaliacao)
# --------------------------------------------------------------------------- #


class _SegJudge(BaseModel):
    index: int
    self_contained: int = Field(ge=1, le=5)    # da p/ entender o chunk sozinho?
    topical_coherence: int = Field(ge=1, le=5)  # um topico so?


class _JudgeBatch(BaseModel):
    judgements: list[_SegJudge]


def _sample(segments: list[Segment], k: int) -> list[Segment]:
    if len(segments) <= k:
        return segments
    step = len(segments) / k
    return [segments[int(i * step)] for i in range(k)]


def _llm_judge(segments: list[Segment], k: int = 5) -> tuple[float, float]:
    """Mede autocontido e coerencia numa amostra, em UMA chamada (grader_model)."""
    from agentic_rag import llm

    sample = _sample(segments, k)
    if not sample:
        return 0.0, 0.0
    body = "\n\n".join(
        f"[chunk {s.index}]\n{s.text[:1200]}" for s in sample
    )
    prompt = (
        "Voce avalia a QUALIDADE DE RECORTE de chunks para RAG. Para cada chunk, "
        "de duas notas de 1 a 5:\n"
        "- self_contained: da para entender o chunk sozinho, sem o resto do documento?\n"
        "- topical_coherence: o chunk trata de um topico so (nao mistura assuntos)?\n"
        "Responda APENAS no schema, um item por chunk.\n\n" + body
    )
    batch = llm.complete(prompt, schema=_JudgeBatch, model=settings.grader_model)
    if not batch.judgements:
        return 0.0, 0.0
    sc = sum(j.self_contained for j in batch.judgements) / len(batch.judgements)
    tc = sum(j.topical_coherence for j in batch.judgements) / len(batch.judgements)
    return (sc - 1) / 4, (tc - 1) / 4  # normaliza 1..5 -> 0..1


def evaluate(
    segments: list[Segment], plan: CutPlan, source: str, use_llm: bool = True
) -> ChunkEval:
    """Calcula o ChunkEval (metricas + score composto + issues acionaveis)."""
    cov = _coverage(source, segments)
    bnd, n_broken = _boundary_integrity(segments)
    siz, n_badsize = _size_fitness(segments, plan)

    if use_llm:
        sc, tc = _llm_judge(segments)
        weights = _W
    else:
        sc = tc = 0.0
        weights = {k: v for k, v in _W.items() if k not in _LLM_KEYS}

    metrics = {
        "coverage": cov,
        "self_contained": sc,
        "boundary_integrity": bnd,
        "size_fitness": siz,
        "topical_coherence": tc,
    }
    wsum = sum(weights.values())
    score = sum(metrics[k] * w for k, w in weights.items()) / wsum

    issues: list[str] = []
    if cov < 0.97:
        issues.append(f"cobertura {cov:.2f}: parte do texto-fonte ficou de fora")
    if n_broken:
        issues.append(f"{n_broken} chunk(s) terminam no meio de uma frase")
    if n_badsize:
        issues.append(f"{n_badsize} chunk(s) fora de [{plan.min_chars},{plan.max_chars}] chars")
    if use_llm and sc < 0.6:
        issues.append("chunks pouco autocontidos: faltam contexto/heading")

    return ChunkEval(
        coverage=cov,
        boundary_integrity=bnd,
        size_fitness=siz,
        self_contained=sc,
        topical_coherence=tc,
        score=score,
        issues=issues,
    )
