# ADR 0004 - Estruturador de PDF: probe deterministico + loop eval-driven

## Contexto
PDFs nao tem estrutura uniforme: um artigo, um relatorio com tabelas, slides e um formulario
pedem recortes diferentes. Chunking de tamanho fixo racha frases e tabelas e produz chunks que
nao se sustentam sozinhos. Por outro lado, jogar o PDF inteiro num LLM para "estruturar" e caro,
nao auditavel e nao garante cobertura. Precisamos de um recorte que se adapte ao documento e
que seja mensuravel.

## Decisao
Um estruturador de PDF **multi-agente** (`src/agentic_rag/pdf/`) com tres camadas:

1. **Probe deterministico (`probe.py`)**: le sinais de layout reais (fontes, blocos, tabelas via
   `find_tables`, TOC, densidade, multi-coluna) ANTES de qualquer LLM, e deriva um `DocProfile`
   (o "comportamento" do PDF: tipo + estrategia de recorte recomendada + `rationale` auditavel).
   E a base analitica de como quebrar cada PDF.

2. **Agente A - chunking adaptativo para RAG (`chunk_agent.py`)**: parte do perfil e roda um loop
   **propor `CutPlan` -> cortar (`segmenter.py`) -> avaliar (`evaluator.py`) -> refinar** ate o
   score passar do limiar ou esgotar o orcamento de iteracoes (guarda o melhor). O score equilibra
   **cobertura (0.30)** e **autocontido (0.30)**, mais integridade de fronteira (0.20), tamanho
   (0.10) e coerencia topica (0.10). Cobertura e integridade sao deterministicas; autocontido e
   coerencia vem de um LLM-judge batelado (1 chamada). Orcamento <= 2 chamadas/iteracao.

3. **Agente B - extracao de campos (`extract_agent.py`)**: extrai campos estruturados (Pydantic)
   com loop de <= 2 chamadas: passada cheia -> valida obrigatorios -> 2a passada FOCADA so nas
   paginas que mencionam os campos faltantes. Page grounding e anti-alucinacao explicita (campo
   ausente => value=null).

Os dois agentes compartilham o probe. `use_llm=False` roda tudo deterministico (gratis, testavel
offline e na CI).

## Trade-offs
- (+) recorte se adapta ao documento; metrica de **cobertura** evita perda silenciosa de texto.
- (+) deterministico-first: barato, auditavel, testavel sem key; LLM so onde agrega (propor/julgar).
- (+) mesmo ethos do RAG do repo (orcamento de chamadas, structured output, eval-driven).
- (-) `find_tables` depende de versao do PyMuPDF (degrada para sem-tabela); OCR ainda nao executado
  (PDF digitalizado so sinaliza `needs_ocr`); contexto truncado em docs muito grandes (v1).

## Status
Aceito. Camada deterministica e testes verdes; caminhos com LLM implementados, pendente rodar ao vivo.
