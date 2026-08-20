"""RAG ingenuo — o baseline do ADR 0001 (agentic vs naive).

1 retrieval (top-k fixo) + 1 chamada de LLM, SEM loop de tool-use e SEM self-check de
grounding. E a referencia honesta do projeto: o agente so "vence" se bater estas metricas
no eval harness. Mesma assinatura de saida do caminho agentic p/ ser drop-in no eval.
"""
from agentic_rag import llm
from agentic_rag.retriever import Retriever

NAIVE_SYSTEM = (
    "Responda a pergunta usando APENAS o contexto fornecido e cite os indices [n] dos "
    "trechos usados. Se o contexto nao sustentar a resposta, diga explicitamente que nao sabe."
)


def _format_contexts(contexts: list[dict]) -> str:
    """Numera os trechos [0..] p/ o modelo poder cita-los (igual ao caminho agentic)."""
    lines = [f"[{i}] {c.get('text', '')}" for i, c in enumerate(contexts)]
    return "\n".join(lines) if lines else "(sem resultados)"


def answer_question(question: str, retriever: Retriever) -> tuple[str, list[dict]]:
    """RAG ingenuo: recupera top-k fixo e responde em 1 unica chamada.

    Devolve (texto_da_resposta, contextos) — sem structured output nem auto-checagem,
    justamente o que o caminho agentic acrescenta e o que o eval deve mostrar valer a pena.
    """
    contexts = retriever.search(question)
    prompt = f"{NAIVE_SYSTEM}\n\nContexto:\n{_format_contexts(contexts)}\n\nPergunta: {question}"
    return llm.complete(prompt), contexts


def eval_answer_fn(retriever: Retriever | None = None):
    """answer_fn p/ o eval harness: (pergunta) -> (resposta, contextos_texto).

    Uso: `evaluate.run_eval(answer_fn=baseline.eval_answer_fn(), ...)` para pontuar o
    baseline ingenuo e salva-lo como referencia contra a qual o agente e comparado.
    """
    retriever = retriever or Retriever()

    def _fn(question: str) -> tuple[str, list[str]]:
        text, contexts = answer_question(question, retriever)
        return text, [c.get("text", "") for c in contexts]

    return _fn
