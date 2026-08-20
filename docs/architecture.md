# Arquitetura

Diagramas em [Mermaid](https://mermaid.js.org) (renderizam direto no GitHub). Para exportar
SVG standalone, ver "Gerar SVG" no fim.

## Visao geral (duas capacidades, mesmos principios)

```mermaid
flowchart TB
  core["SDK Anthropic (Claude) + structured output (Pydantic)<br/>orcamento &lt;= 2 chamadas de LLM por item"]

  subgraph RAG["RAG agentic + eval"]
    Q([query]) --> API["FastAPI /ask"]
    API --> AG["Agente tool-use<br/>retrieve - reason - answer - self-check"]
    AG <--> RET["Retriever<br/>Qdrant + embeddings"]
    EV["Eval harness (ragas)<br/>gate de promocao"] -.-> AG
  end

  subgraph PDFS["Estruturador de PDF (multi-agente)"]
    P([PDF]) --> PROBE["probe (deterministico)<br/>sinais de layout para DocProfile"]
    PROBE --> A["Agente A<br/>chunking adaptativo"]
    PROBE --> B["Agente B<br/>extracao de campos"]
    A --> CH([chunks p/ RAG])
    B --> FI([campos estruturados])
  end

  core -.-> RAG
  core -.-> PDFS
```

## Agente A: loop de chunking adaptativo

```mermaid
flowchart TB
  PROF["DocProfile (probe)"] --> PLAN["propor CutPlan"]
  PLAN --> SEG["segmentar (segmenter)"]
  SEG --> EVAL["avaliar (evaluator)<br/>cobertura .30 + autocontido .30<br/>+ fronteira .20 + tamanho .10 + coerencia .10"]
  EVAL --> DEC{"score no limiar<br/>ou orcamento esgotado?"}
  DEC -- sim --> OUT(["ChunkResult<br/>melhor recorte + trilha de score"])
  DEC -- nao --> REF["refinar plano (LLM)"]
  REF --> SEG
```

## Agente B: extracao de campos com 2 passadas

```mermaid
flowchart TB
  P([PDF]) --> PT["texto por pagina (probe)"]
  PT --> C1["passada cheia: extrair campos"]
  C1 --> V{"obrigatorios<br/>faltando?"}
  V -- nao --> R(["ExtractionResult"])
  V -- sim --> C2["2a passada focada<br/>paginas que citam o campo"]
  C2 --> M["merge (preenchido vence vazio)"]
  M --> R
```

Decisoes em [`docs/adr/`](adr/). Principios: structured output (Pydantic), orcamento <=2
chamadas/item, deterministico-first, evals como gate de promocao.

## Gerar SVG (opcional)

Os diagramas acima ja renderizam no GitHub. Para exportar SVG standalone (ex.: slides, site):

```bash
# requer Node (npx resolve sem instalar global)
npx -y -p @mermaid-js/mermaid-cli mmdc -i docs/architecture.md -o docs/diagrams/arch.svg
```

Na primeira execucao o mermaid-cli baixa um Chromium headless. Saida em `docs/diagrams/`.
