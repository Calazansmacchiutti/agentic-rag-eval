"""Gateway de LLM.

Caminho padrao = Claude via SDK oficial da Anthropic (provider "anthropic").
Outros providers (openai/bedrock) ficam atras do LiteLLM como escape hatch
vendor-agnostico, ainda nao implementado (ver ADR 0001).
"""
import sys

from agentic_rag.config import settings

_client = None

# Modelos que ja responderam 400 rejeitando `temperature`. Preenchido em tempo de execucao:
# manter uma tabela estatica de "quem aceita o que" envelhece a cada lancamento de modelo.
_SEM_TEMPERATURE: set[str] = set()


def _temperatura_rejeitada(erro: Exception) -> bool:
    """True quando o 400 e especificamente sobre `temperature`.

    Deliberadamente estreito: qualquer outro 400 (schema invalido, modelo inexistente,
    contexto estourado) tem de continuar subindo. Engolir 400 generico esconderia defeito.
    """
    return getattr(erro, "status_code", None) == 400 and "temperature" in str(erro).lower()


def _anthropic():
    """Cliente Anthropic preguicoso (le ANTHROPIC_API_KEY do .env via Settings)."""
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    return _client


def client():
    """Cliente Anthropic compartilhado (p/ quem precisa do loop tool-use cru, ex.: agent.py)."""
    return _anthropic()


def complete(prompt: str, schema=None, model: str | None = None, temperature: float | None = None):
    """Chama o LLM e devolve texto; se `schema` (classe Pydantic), devolve a instancia validada.

    `temperature` opcional (None = default do SDK); usada pelo eval p/ um juiz de baixa
    variancia (ADR 0005). Sem extended thinking por padrao: este projeto orca <= 2 chamadas/item,
    entao a geracao roda enxuta. Suba para thinking adaptativo so onde o raciocinio compensar.
    """
    model = model or settings.llm_model

    if settings.llm_provider == "anthropic":
        client = _anthropic()
        # so passa temperature quando definida, p/ nao sobrescrever o default do SDK
        extra = {} if (temperature is None or model in _SEM_TEMPERATURE) else {"temperature": temperature}

        def _chamar(kw: dict):
            if schema is not None:
                # structured output: valida a resposta contra o schema Pydantic
                r = client.messages.parse(
                    model=model,
                    max_tokens=settings.max_output_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    output_format=schema,
                    **kw,
                )
                return r.parsed_output
            r = client.messages.create(
                model=model,
                max_tokens=settings.max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kw,
            )
            return next(b.text for b in r.content if b.type == "text")

        try:
            return _chamar(extra)
        except Exception as e:  # reclassificado logo abaixo: so trata o 400 de temperature
            if not (extra and _temperatura_rejeitada(e)):
                raise
            # A familia Opus 4.7+ removeu `temperature` (400 "deprecated for this model").
            # Descobrimos por resposta do servidor em vez de manter tabela de modelos, que
            # envelhece a cada lancamento. Memoriza p/ nao repetir a chamada perdida.
            _SEM_TEMPERATURE.add(model)
            print(
                f"[llm] modelo {model} rejeita `temperature`; repetindo sem o parametro. "
                "Determinismo do juiz (ADR 0005) fica por conta do modelo.",
                file=sys.stderr,
            )
            return _chamar({})

    raise NotImplementedError(
        f"provider {settings.llm_provider!r} ainda nao implementado; use 'anthropic'."
    )
