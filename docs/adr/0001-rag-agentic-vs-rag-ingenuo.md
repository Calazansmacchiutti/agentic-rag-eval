# ADR 0001 - RAG agentic com eval gate, em vez de RAG ingenuo

## Contexto
RAG de 1 chamada e barato mas aluciana e nao da pra auditar. Precisamos de fidelidade,
rastreabilidade e custo previsivel.

## Decisao
Loop de tool-use com o SDK oficial da Anthropic (o modelo decide quando buscar via a tool
`search`), resposta com structured output (Pydantic) e self-check de grounding, com um eval harness
(ragas) como GATE: so promove mudanca se bater o baseline. Orcamento de <=2 chamadas de LLM por item.

LangGraph foi considerado (orquestracao multi-no, keyword de mercado) mas preterido: para este
fluxo linear o loop cru e mais simples, sem dependencia extra e totalmente auditavel. O design fica
portavel a LangGraph se a orquestracao crescer.

## Trade-offs
- (+) fidelidade/auditabilidade/custo controlado; menos deps; (-) mais complexidade que RAG ingenuo.
- (-) sem a keyword "LangGraph" no repo; (+) controle total do loop e do orcamento de chamadas.
- LiteLLM fica como escape hatch p/ outros providers (elimina lock-in) sem virar caminho padrao.

## Status
Aceito.
