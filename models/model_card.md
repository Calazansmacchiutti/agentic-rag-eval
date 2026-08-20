# Model Card - Agentic RAG

- **Uso pretendido:** QA sobre base de documentos com respostas rastreaveis.
- **Baseline:** RAG ingenuo (1 chamada, top-k fixo), implementado em `src/agentic_rag/baseline.py`;
  pontue-o no eval (`run_eval(answer_fn=baseline.eval_answer_fn())`) e salve como referencia — o
  caminho agentic so e promovido se bater essas metricas.
- **Metricas:** faithfulness, answer_relevancy, context_precision. (TODO: numeros reais)
- **Limitacoes:** depende da qualidade do corpus; nao recomendado para dominios sem documentos-fonte;
  custo cresce com nº de agentes (mitigado pelo orcamento <=2 chamadas/item).
- **Etica/risco:** guardrails contra alucinacao; respostas citam fontes; sem PII no corpus publico.

## Avaliacao e gate de promocao
Processo automatizado em `src/agentic_rag/evaluate.py` (`make eval`); detalhes no README, secao
"Ciclo de eval".

- **Dados de avaliacao:** conjunto dourado em `data/eval_set.jsonl` (pergunta + `ground_truth`).
  Curado a mao; deve cobrir os casos de uso pretendidos e alguns adversariais (sem resposta no corpus).
- **Procedimento:** para cada item, roda o agente (retrieve -> answer) e coleta `answer` + `contexts`;
  pontua o lote com o `grader_model` (default `claude-opus-4-8`; troque p/ `claude-haiku-4-5` para
  baratear).
- **Scorer:** default e o juiz Claude nativo (`llm_judge_score`, structured output, 1 chamada/item),
  que roda out-of-the-box; `ragas_score` fica como opcao (opt-in) quando o stack ragas/langchain
  estiver compativel.
- **Criterio de promocao (gate):** um estado so e adotado se **toda** metrica for >= baseline menos
  `eval_promotion_tolerance`. O baseline aprovado fica versionado em `models/baseline_metrics.json`;
  uma regressao bloqueia a promocao e preserva o baseline anterior. Sem baseline, a 1a rodada
  estabelece a referencia.
- **Reprodutibilidade:** metricas dependem do corpus indexado, do `llm_model`/`grader_model` e do
  golden set; registre essas versoes ao publicar numeros. Nao publicar resultado sem numeros honestos
  de uma rodada real.
