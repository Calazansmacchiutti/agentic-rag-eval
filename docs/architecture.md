# Arquitetura

Diagramas em [Mermaid](https://mermaid.js.org) (renderizam direto no GitHub). Para exportar
SVG standalone, ver "Gerar SVG" no fim.

## Visao geral

Duas capacidades sobre o mesmo nucleo, e uma camada de garantia em volta das duas.

```mermaid
flowchart TB
  core["SDK Anthropic (Claude) + structured output (Pydantic)<br/>orcamento &lt;= 2 chamadas de LLM por item"]

  subgraph RAG["RAG agentic com fundamentacao auditavel"]
    Q([pergunta + escopo]) --> API["FastAPI /perguntar"]
    API --> GIN{"guardrail de entrada<br/>pedido de recomendacao?"}
    GIN -- barra --> REC(["recusa com motivo<br/>0 chamadas de LLM"])
    GIN -- segue --> AG["caso de uso: responder<br/>retrieve - reason - answer - self-check"]
    AG <--> RET["recuperacao<br/>Qdrant + embeddings<br/>filtro de escopo por metadado"]
    AG --> GOUT{"guardrail de saida<br/>tem citacao valida?"}
    GOUT -- nao --> REC
    GOUT -- sim --> OK(["resposta + fontes"])
    REC --> AUD[("trilha de auditoria<br/>append-only JSONL")]
    OK --> AUD
  end

  subgraph PDFS["Estruturador de PDF (multi-agente)"]
    P([PDF]) --> PROBE["probe (deterministico)<br/>sinais de layout para DocProfile"]
    PROBE --> A["Agente A<br/>chunking adaptativo"]
    PROBE --> B["Agente B<br/>extracao de campos"]
    A --> CH([chunks p/ RAG])
    B --> FI([campos estruturados])
  end

  EV["eval harness<br/>oraculos + juiz"] -.-> AG
  core -.-> RAG
  core -.-> PDFS
```

## Fluxo de uma pergunta (o que garante o que)

O ponto central: **o escopo restringe na recuperacao, nao por instrucao de prompt** — o trecho
fora do escopo nunca chega ao modelo. E **nenhuma resposta sai sem citacao valida**.

```mermaid
sequenceDiagram
  autonumber
  participant U as Usuario (escopo)
  participant API as adapters/inbound/api
  participant UC as domain/use_cases
  participant G as domain/guardrails
  participant R as PortaRecuperacao
  participant L as PortaLLM
  participant A as PortaAuditoria

  U->>API: pergunta + usuario/papel/filtros
  API->>UC: responder(...)
  UC->>G: checar_pergunta
  alt fora de politica
    G-->>UC: recusa
    UC->>A: registra decisao + motivo
    UC-->>API: permitido=false (HTTP 200)
  else segue
    loop ate o teto de turnos
      UC->>L: loop de tool-use
      L-->>UC: pede busca
      UC->>R: buscar(consulta, filtros do escopo)
      R-->>UC: trechos ja filtrados + dedup
    end
    UC->>L: resposta estruturada (Pydantic)
    UC->>G: checar_resposta (grounded? citou? indice existe?)
    UC->>A: registra id dos trechos + versao do prompt + modelo
    UC-->>API: resposta + fontes + veredito
  end
```

## Camadas (ports & adapters)

O dominio nao conhece fornecedor. Trocar Anthropic por outro provedor, ou Qdrant por outro
banco vetorial, e escrever um adapter — nenhum arquivo de `domain/` muda.

```mermaid
flowchart LR
  subgraph IN["adapters/inbound"]
    HTTP["api.py<br/>FastAPI + schemas"]
  end

  subgraph DOM["domain (sem I/O, sem fornecedor)"]
    UC["use_cases/responder_pergunta"]
    ENT["entities<br/>Trecho · Resposta · Escopo · EventoAuditoria"]
    GR["guardrails<br/>4 decisoes de recusa"]
    LIM["limites<br/>LimiteDeTurnos · PortaoDeEscrita"]
    PORT{{"ports/outbound<br/>PortaLLM · PortaRecuperacao<br/>PortaEmbedding · PortaAuditoria"}}
  end

  subgraph OUT["adapters/outbound"]
    ANT["anthropic_llm"]
    QDR["qdrant_retriever"]
    AUD["auditoria_jsonl"]
  end

  subgraph INFRA["infrastructure"]
    CFG["config"]
    PR["prompts<br/>prompts/system.md + hash = versao"]
    CT["container<br/>composition root"]
  end

  HTTP --> UC
  UC --> ENT
  UC --> GR
  UC --> LIM
  UC --> PORT
  PORT -.implementado por.-> ANT
  PORT -.implementado por.-> QDR
  PORT -.implementado por.-> AUD
  CT --> ANT
  CT --> QDR
  CT --> AUD
  CT --> PR
  CT --> CFG
```

## Eval em duas camadas

O oraculo e barato, binario e estavel; o juiz e caro e ruidoso (ADR 0005). Por isso o juiz
roda **so no que foi respondido** — recusa ja foi verificada exatamente.

```mermaid
flowchart TB
  GS([golden set<br/>18 casos, 4 categorias]) --> SYS["sistema sob teste"]
  SYS --> OR["oraculos deterministicos<br/>decisao esperada · deve_conter · nao_pode_conter"]
  OR --> VAZ{"vazamento<br/>de escopo?"}
  VAZ -- sim --> REP(["REPROVA o conjunto<br/>falha de seguranca, nao de qualidade"])
  VAZ -- nao --> SEL{"foi respondido?"}
  SEL -- "nao (recusa)" --> PULA(["pulado: ja verificado<br/>0 chamadas de juiz"])
  SEL -- sim --> J["juiz LLM<br/>rubrica do ADR 0005"]
  J --> MET(["faithfulness · answer relevancy<br/>context precision<br/>+ proveniencia (scorer, grader_model)"])
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
chamadas/item, deterministico-first, evals como gate de promocao, e **nenhuma alegacao sem
fonte rastreavel** (ver [`GOVERNANCA_IA.md`](GOVERNANCA_IA.md)).

## Gerar SVG (opcional)

**O Mermaid acima e a unica fonte de verdade** - inclusive no README, que passou a embutir o
diagrama inline em vez de apontar para um arquivo. Isso e deliberado: SVG exportado envelhece
em silencio, e foi o que aconteceu (`docs/diagrams/arch-*.svg`, gerados em 2026-06-30, ficaram
desenhando uma arquitetura que nao existe mais - sem guardrails, escopo nem trilha de
auditoria). Diagrama que precisa de passo manual para acompanhar o codigo acaba mentindo.

Se precisar de SVG standalone para slide ou site, gere na hora e trate como descartavel:

```bash
# requer Node (npx resolve sem instalar global)
npx -y -p @mermaid-js/mermaid-cli mmdc -i docs/architecture.md -o docs/diagrams/arch.svg
```

Na primeira execucao o mermaid-cli baixa um Chromium headless.
