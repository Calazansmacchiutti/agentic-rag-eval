"""Gateway de LLM.

Caminho padrao = Claude via SDK oficial da Anthropic (provider "anthropic").
Outros providers (openai/bedrock) ficam atras do LiteLLM como escape hatch
vendor-agnostico, ainda nao implementado (ver ADR 0001).
"""
from agentic_rag.config import settings

_client = None


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
        extra = {} if temperature is None else {"temperature": temperature}
        if schema is not None:
            # structured output: valida a resposta contra o schema Pydantic
            resp = client.messages.parse(
                model=model,
                max_tokens=settings.max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                **extra,
            )
            return resp.parsed_output
        resp = client.messages.create(
            model=model,
            max_tokens=settings.max_output_tokens,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        return next(b.text for b in resp.content if b.type == "text")

    raise NotImplementedError(
        f"provider {settings.llm_provider!r} ainda nao implementado; use 'anthropic'."
    )
