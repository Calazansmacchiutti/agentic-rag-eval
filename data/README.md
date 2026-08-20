# Dados do eval harness

Dados crus de terceiros vao em `raw/` (gitignored). Os dois arquivos abaixo sao **versionados**
e formam um par auto-contido e reproduzivel para o ciclo de eval (ver README raiz, "Ciclo de eval").

## `corpus.jsonl` (corpus de referencia)
Uma linha por documento: `{"id", "source", "text"}`. E um mini-corpus sintetico sobre a
**arquitetura do proprio projeto** (decisoes dos ADRs, componentes), suficiente para o `/ask` e o
eval harness terem o que recuperar sem depender de dados externos.

Indexar no Qdrant (recria a collection do zero):
```bash
make qdrant                # sobe o Qdrant em :6333
make seed                  # python scripts/seed_corpus.py -> indexa data/corpus.jsonl
```

## `eval_set.jsonl` (golden set)
Uma linha por pergunta: `{"question", "ground_truth", "adversarial?}`. Curado a mao e alinhado ao
`corpus.jsonl`. Inclui:
- **respondiveis**: a resposta esta no corpus (mede faithfulness/relevancy/precision);
- **adversariais** (`"adversarial": true`): a resposta NAO esta no corpus — testa o guardrail de
  grounding (o sistema deve recusar em vez de alucinar). Puxam a precisao media para baixo de
  proposito, o que e o comportamento honesto esperado.

Ao trocar de corpus real, mantenha o `eval_set.jsonl` alinhado a ele e registre a fonte aqui.
