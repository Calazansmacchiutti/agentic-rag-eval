"""Eval harness automatizado: fecha o ciclo agentic-RAG + eval.

Fluxo (ADR 0001): roda o sistema sobre um conjunto dourado (perguntas + resposta de
referencia), pontua as metricas faithfulness/answer_relevancy/context_precision e aplica
um GATE de promocao: so adota o novo estado se bater o baseline versionado.

Dois scorers atras da mesma interface: o default `llm_judge_score` (juiz Claude nativo,
structured output, roda out-of-the-box) e `ragas_score` (opt-in, quando o stack ragas/
langchain estiver compativel). Ver docs/adr e o README ("Ciclo de eval").

Desenho testavel: as bordas pesadas/nao-deterministicas (rodar o agente, pontuar) entram
por injecao (answer_fn, scorer). Os testes passam fakes e nao tocam LLM/Qdrant/ragas.
Imports pesados adiados como no resto do projeto.
"""
import json
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from agentic_rag.config import settings

# Metricas ragas usadas como criterio (ordem estavel p/ relatorio/baseline).
METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision")

# Assinaturas das bordas injetaveis.
AnswerFn = Callable[[str], tuple[str, list[str]]]        # pergunta -> (resposta, contextos)
Scorer = Callable[[list["EvalRecord"]], dict[str, float]]  # registros -> {metrica: media}


class EvalItem(BaseModel):
    """Uma linha do golden set (data/eval_set.jsonl)."""

    question: str
    ground_truth: str = ""
    adversarial: bool = False  # pergunta sem resposta no corpus (testa o guardrail de grounding)


class EvalRecord(BaseModel):
    """O que vai para o scorer: pergunta, resposta gerada, contextos recuperados e referencia."""

    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str = ""


class EvalReport(BaseModel):
    """Resultado do harness: metricas, comparacao com baseline e decisao do gate."""

    metrics: dict[str, float]
    baseline: dict[str, float] = Field(default_factory=dict)
    deltas: dict[str, float] = Field(default_factory=dict)
    passed: bool
    promoted: bool = False
    provenance_ok: bool = True  # False => baseline veio de outro scorer/grader (incomparavel)
    n_items: int


def load_dataset(path: str) -> list[EvalItem]:
    """Le o golden set em JSONL (uma pergunta por linha). Erra se vazio/ausente."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"golden set nao encontrado: {path}")
    items = [
        EvalItem.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise ValueError(f"golden set vazio: {path}")
    return items


def _default_answer_fn() -> AnswerFn:
    """Liga no stack real: um Retriever compartilhado + o loop agentic."""
    from agentic_rag.agent import answer_question
    from agentic_rag.retriever import Retriever

    retriever = Retriever()

    def _fn(question: str) -> tuple[str, list[str]]:
        result, contexts = answer_question(question, retriever)
        return result.answer, [c.get("text", "") for c in contexts]

    return _fn


def build_records(items: list[EvalItem], answer_fn: AnswerFn | None = None) -> list[EvalRecord]:
    """Roda o sistema por item e monta os registros pontuaveis."""
    answer_fn = answer_fn or _default_answer_fn()
    records = []
    for it in items:
        answer, contexts = answer_fn(it.question)
        records.append(
            EvalRecord(
                question=it.question,
                answer=answer,
                contexts=contexts,
                ground_truth=it.ground_truth,
            )
        )
    return records


def ragas_score(records: list[EvalRecord]) -> dict[str, float]:
    """Pontua com ragas (import pesado adiado; exige LLM configurado p/ o grader).

    Isolado atras da interface Scorer: o resto do harness e deterministico e testavel
    sem chamar ragas nem rede.
    """
    from datasets import Dataset  # noqa: PLC0415  (dep transitiva do ragas)
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [r.question for r in records],
            "answer": [r.answer for r in records],
            "contexts": [r.contexts for r in records],
            "ground_truth": [r.ground_truth for r in records],
        }
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    return {k: float(v) for k, v in dict(result).items()}


# Rubrica analitica (ADR 0005): escalas discretas mapeadas p/ numero -> menos variancia que
# pedir um float holistico. context_precision e computado DETERMINISTICAMENTE a partir de um
# julgamento booleano por contexto (a metrica que mais sofria com ruido do juiz).
_FAITH_MAP = {"grounded": 1.0, "partial": 0.5, "unsupported": 0.0}
_RELEV_MAP = {"complete": 1.0, "partial": 0.5, "off_topic": 0.0}


class _ItemScore(BaseModel):
    """Julgamento por item do juiz LLM: rationale (CoT) + rotulos discretos + booleanos por contexto."""

    rationale: str = Field(description="Analise curta ANTES dos rotulos (chain-of-thought).")
    faithfulness: Literal["grounded", "partial", "unsupported"] = Field(
        description="grounded=toda afirmacao sustentada pelos contextos; partial=parte; unsupported=alucina."
    )
    answer_relevancy: Literal["complete", "partial", "off_topic"] = Field(
        description="complete=responde direto e completo; partial=parcial; off_topic=nao responde."
    )
    context_relevance: list[bool] = Field(
        default_factory=list,
        description="Um booleano por trecho recuperado, na ORDEM [0..n-1]: e relevante p/ responder?",
    )


_JUDGE_PROMPT = """Voce e um avaliador rigoroso de QA/RAG. Primeiro escreva um rationale curto, depois classifique.

Pergunta:
{question}

Resposta gerada:
{answer}

Contextos recuperados (numerados [0..{last}]):
{contexts}

Resposta de referencia (pode faltar):
{ground_truth}

Instrucoes:
- faithfulness: 'grounded' se TODA afirmacao da resposta e sustentada pelos contextos; 'partial' se parte;
  'unsupported' se alucina ou o contexto nao sustenta.
- answer_relevancy: 'complete' se responde direta e completamente; 'partial' se parcial; 'off_topic' se nao responde.
- context_relevance: uma lista com EXATAMENTE {n} booleanos, um por contexto na ordem [0..{last}],
  True se aquele trecho e relevante para responder, False se e ruido.
- Nao favoreca respostas longas nem a ordem dos contextos. Se faltar suporte, seja explicito ('unsupported')."""


def _precision(flags: list[bool], n: int) -> float:
    """context_precision deterministico: fracao de contextos marcados relevantes.

    Robusto a desalinhamento do juiz: trunca flags extras e conta flags faltantes como False
    (denominador = nro real de contextos). Sem contexto => 0.0.
    """
    if n <= 0:
        return 0.0
    return sum(1 for f in flags[:n] if f) / n


def llm_judge_score(records: list[EvalRecord]) -> dict[str, float]:
    """Juiz nativo com Claude (rubrica analitica, ADR 0005): 1 chamada/item, medias no lote.

    Escalas discretas + precisao deterministica + temperatura baixa (settings.grader_temperature)
    reduzem a variancia. Usa grader_model (default claude-opus-4-8; haiku p/ baratear).
    Alternativa ao ragas quando o stack langchain/ragas nao esta disponivel.
    """
    from statistics import fmean

    from agentic_rag import llm

    if not records:
        return {k: 0.0 for k in METRIC_KEYS}

    faith, relev, prec = [], [], []
    for r in records:
        ctx = "\n".join(f"[{i}] {c}" for i, c in enumerate(r.contexts)) or "(nenhum)"
        prompt = _JUDGE_PROMPT.format(
            question=r.question,
            answer=r.answer,
            contexts=ctx,
            ground_truth=r.ground_truth or "(nao fornecida)",
            n=len(r.contexts),
            last=max(len(r.contexts) - 1, 0),
        )
        s = llm.complete(
            prompt,
            schema=_ItemScore,
            model=settings.grader_model,
            temperature=settings.grader_temperature,
        )
        faith.append(_FAITH_MAP[s.faithfulness])
        relev.append(_RELEV_MAP[s.answer_relevancy])
        prec.append(_precision(s.context_relevance, len(r.contexts)))

    return {
        "faithfulness": fmean(faith),
        "answer_relevancy": fmean(relev),
        "context_precision": fmean(prec),
    }


def load_baseline(path: str) -> dict:
    """Le o baseline versionado (payload {metrics, scorer, grader_model}); ausente => {}."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_baseline(metrics: dict[str, float], path: str, *, scorer: str, grader_model: str) -> None:
    """Persiste o baseline COM proveniencia (ADR 0005): scorer + grader_model junto das metricas.

    Assim uma troca de rubrica/juiz nao se passa por ganho do modelo — a comparacao so vale
    entre rodadas do mesmo scorer e mesmo grader_model.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "scorer": scorer, "grader_model": grader_model}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def gate(
    metrics: dict[str, float], baseline: dict[str, float], tolerance: float = 0.0
) -> tuple[bool, dict[str, float]]:
    """Decide promocao: passa se toda metrica do baseline for >= baseline - tolerancia.

    Sem baseline (primeira vez), passa por definicao — a rodada estabelece a referencia.
    """
    if not baseline:
        return True, {k: 0.0 for k in metrics}
    deltas = {k: metrics.get(k, 0.0) - baseline.get(k, 0.0) for k in baseline}
    passed = all(metrics.get(k, 0.0) >= v - tolerance for k, v in baseline.items())
    return passed, deltas


def run_eval(
    dataset_path: str | None = None,
    *,
    answer_fn: AnswerFn | None = None,
    scorer: Scorer | None = None,
    promote: bool = False,
) -> EvalReport:
    """Orquestra o harness: carrega -> roda -> pontua -> gate (-> promove se pedido).

    Scorer default = juiz Claude nativo (llm_judge_score), que roda out-of-the-box no stack
    Anthropic-first. Passe scorer=ragas_score para usar o ragas quando o ambiente suportar.
    """
    dataset_path = dataset_path or settings.eval_dataset_path
    scorer = scorer or llm_judge_score
    scorer_name = getattr(scorer, "__name__", str(scorer))

    items = load_dataset(dataset_path)
    records = build_records(items, answer_fn)
    metrics = scorer(records)

    stored = load_baseline(settings.baseline_metrics_path)
    baseline_metrics = stored.get("metrics", {})
    # proveniencia (ADR 0005): so compara se o baseline veio do MESMO scorer + grader_model
    provenance_ok = not stored or (
        stored.get("scorer") == scorer_name and stored.get("grader_model") == settings.grader_model
    )
    # baseline incomparavel (rubrica/juiz diferente) => trata como se nao houvesse baseline
    compare = baseline_metrics if provenance_ok else {}
    passed, deltas = gate(metrics, compare, settings.eval_promotion_tolerance)

    promoted = False
    if promote and passed:
        save_baseline(
            metrics,
            settings.baseline_metrics_path,
            scorer=scorer_name,
            grader_model=settings.grader_model,
        )
        promoted = True

    return EvalReport(
        metrics=metrics,
        baseline=baseline_metrics,
        deltas=deltas,
        passed=passed,
        promoted=promoted,
        provenance_ok=provenance_ok,
        n_items=len(items),
    )


if __name__ == "__main__":
    # `make eval`: roda o gate real e promove o baseline se bater.
    print(run_eval(promote=True).model_dump_json(indent=2))
