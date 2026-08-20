# ADR 0003 - LLM via SDK oficial da Anthropic por padrao

## Contexto
O scaffold inicial colocava o LiteLLM como gateway primario (vendor-agnostico). Na pratica o
projeto roda com a API do Claude, e o caminho que sera executado/testado precisa ser real.

## Decisao
Caminho padrao = **SDK oficial da Anthropic** (Claude). `llm.complete()` chama
`messages.create`; com schema Pydantic usa `messages.parse` (structured output validado).
`llm.client()` expoe o cliente p/ o loop de tool-use do agente (ver ADR 0001). Modelos via
`LLM_MODEL` (default `claude-opus-4-8`) e `GRADER_MODEL` (eval; trocavel p/ `claude-haiku-4-5`).
Sem extended thinking por padrao, coerente com o orcamento de <=2 chamadas de LLM por item.
LiteLLM permanece como **escape hatch** p/ outros providers (openai/bedrock), ainda nao implementado.

## Trade-offs
- (+) caminho real e testavel, structured output nativo, controle fino de custo/modelo.
- (-) a abstracao vendor-agnostica deixa de ser o caminho padrao; mitigado pelo escape hatch documentado.

## Status
Aceito.
