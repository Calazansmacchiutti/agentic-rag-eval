# ADR 0005 - Eval do RAG: estabilizar o LLM-judge e versionar o baseline

## Contexto
Ao rodar a comparacao agentic-vs-ingenuo ao vivo (golden set de 3 perguntas, juiz Claude
haiku) o gate reportou o agente PERDENDO em context_precision. Uma verificacao deterministica
derrubou esse diagnostico: para as 3 perguntas, agentic e ingenuo recuperam **conjuntos de
contexto identicos** (o modelo faz 1 busca/pergunta). Se o contexto e o mesmo, a precisao
verdadeira e igual — logo a diferenca observada (0.583 vs 0.4) era **ruido do juiz**, nao
regressao. Conclusao: o gargalo nao e o agente, e a **metodologia de eval** (n=3 + juiz
nao-deterministico + nota holistica continua = sem sinal). Pesquisamos prior art para trazer
solucoes testadas antes de continuar a evoluir o agente.

## Prior art (resumo da pesquisa)
- **Estabilidade de LLM-as-judge** (mer.vin 2025; arXiv 2606.19544): temperatura baixa (0.0-0.2);
  **rubrica analitica decomposta** (pontuar criterio-a-criterio) em vez de nota holistica; escalas
  **discretas** (binario / 3 pontos) mais confiaveis que 0-100; **rationale antes da nota** (CoT);
  **few-shot** com exemplos por nivel; instrucoes **anti-vies** (nao favorecer resposta longa/ordem);
  opcao explicita "nao sei". Frontier judges para calibracao/gate; checks deterministicos para
  monitoramento barato.
- **Frameworks de eval** (genai.qa; deepeval.com): **DeepEval** e "o pytest dos evals" — nativo em
  pytest, juiz Claude plugavel, gate em PR; calcula faithfulness por verificacao de claims (a tal
  rubrica analitica). **ragas** e forte em RAG mas defaulta a OpenAI e roda melhor como job agendado.
  **Promptfoo** e CLI/YAML, multi-modelo, forte em red-team. Consenso: um framework leve para gate
  em CI + uma plataforma para anotacao humana/regressao.
- **Golden set e baseline** (Statsig; vectara/open-rag-eval): golden dataset **versionado** como
  fonte de verdade; **proteger o baseline historico para que uma mudanca de rubrica nao se passe por
  ganho do modelo**; open-rag-eval avalia sem "golden answers" (curar referencias e caro).
- **Retrieval agentic** (NirDiamant/RAG_Techniques; Deep_Research; Contextual-RAG): **hybrid search
  (vetorial + BM25) + reranker** (lever deterministico de precisao); query decomposition/rewriting
  para multi-hop (respeitando nosso orcamento <=2 chamadas).

## Decisao
1. **Estabilizar o `llm_judge_score`** (ataca o gargalo provado): temperatura baixa no grader
   (`grader_temperature`, default 0.0); **rubrica analitica** — faithfulness/answer_relevancy em
   escala discreta de 3 pontos mapeada para {0.0, 0.5, 1.0}, e **context_precision computada de
   forma deterministica** a partir de um julgamento booleano por contexto (relevante sim/nao);
   **rationale (CoT) antes das notas** e instrucoes anti-vies no prompt.
2. **Versionar a proveniencia no baseline**: `models/baseline_metrics.json` passa a gravar o
   `scorer` e o `grader_model` junto das metricas; se a proveniencia da rodada divergir do baseline,
   a comparacao e tratada como incomparavel (nao deixa troca de rubrica se passar por ganho).
3. **DeepEval como `Scorer` opcional** (futuro): melhor encaixe que ragas no nosso ambiente
   (Claude plugavel, pytest-native); entra pela mesma interface `Scorer`, ao lado de `ragas_score`.
4. **Golden set maior** (10-20 perguntas, com adversariais) para ter significancia — pre-requisito
   para qualquer conclusao agentic-vs-ingenuo.
5. **Hybrid + rerank parkeados**: sao o lever real de precisao em escala, mas so entram DEPOIS de
   termos juiz estavel + golden set — senao repetimos o erro de "consertar sem sinal".

## Trade-offs
- (+) rubrica analitica + temperatura baixa reduzem a variancia que provamos dominar o resultado;
  precisao deterministica remove o juiz da metrica mais ruidosa.
- (+) proveniencia no baseline evita conclusao falsa ao trocar juiz/modelo.
- (+) mantem o ethos do repo: determinismo-first onde da, LLM so onde agrega, tudo testavel offline.
- (-) juiz LLM continua com variancia residual (so diminui); escala discreta perde granularidade
  fina; per-contexto booleano depende de o juiz alinhar a lista aos contextos numerados (tratamos
  desalinhamento de tamanho de forma robusta).

## Status
Aceito. Itens 1 e 2 implementados. Item 4 (golden set) expandido para 17 itens (14 respondiveis
+ 3 adversariais) com corpus de referencia versionado (`data/corpus.jsonl`). Item 5 (rerank)
implementado como opt-in (`src/agentic_rag/rerank.py`, cross-encoder deterministico), mas o A/B ao
vivo (top_k=5, n=17) NAO mostrou ganho de precisao — deltas dentro do ruido. Causas provaveis:
corpus sintetico pequeno demais (16 docs) para reranking render, teto estrutural de precisao e
cross-encoder EN em corpus PT. Fica OFF por padrao. Aprendizado: no corpus sintetico atual,
melhorias de retrieval (curador de dedup, rerank) nao sao mensuraveis — o proximo gargalo real e
um CORPUS realista maior, nao mais componentes. Item 3 (DeepEval) segue pendente.

## Referencias
- LLM-as-a-Judge best practices: https://mer.vin/2025/11/llm-as-a-judge-best-practices-for-consistent-evaluation/
- Reliability without Validity (LLM-as-judge): https://arxiv.org/pdf/2606.19544
- Promptfoo vs DeepEval vs RAGAS: https://genai.qa/blog/promptfoo-vs-deepeval-vs-ragas/
- DeepEval alternatives comparados: https://deepeval.com/blog/deepeval-alternatives-compared
- RAG_Techniques (NirDiamant): https://github.com/NirDiamant/RAG_Techniques
- Deep_Research (decomposition+rerank): https://github.com/dHiebl/Deep_Research
- Contextual-RAG (hybrid+rerank): https://github.com/chatterjeesaurabh/Contextual-RAG-System-with-Hybrid-Search-and-Reranking
- open-rag-eval (sem golden answers): https://github.com/vectara/open-rag-eval
- Golden datasets (Statsig): https://www.statsig.com/perspectives/golden-datasets-evaluation-standards
