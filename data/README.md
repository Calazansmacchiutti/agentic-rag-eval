# Dados do eval harness

> **Procedencia, em uma frase: todo dado versionado neste repositorio e SINTETICO.**
> Nenhum documento aqui vem de cliente, instituicao ou base de terceiro. Dados crus de
> terceiros vao em `raw/`, que e **gitignored** e contem apenas um `.gitkeep` vazio.

Sao dois conjuntos independentes, cada um com seu proposito:

| Conjunto | Serve a | Mede |
|---|---|---|
| `corpus.jsonl` + `eval_set.jsonl` | `src/agentic_rag/evaluate.py` | **qualidade** da resposta (faithfulness, relevancy, precision) com juiz LLM |
| `financeiro/corpus.jsonl` + `evals/golden_set.jsonl` | `evals/harness.py` | **garantia**: recusa correta e isolamento de escopo, deterministico |

O primeiro par forma um conjunto auto-contido e reproduzivel para o ciclo de eval
(ver README raiz, "Ciclo de eval").

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

---

## `financeiro/corpus.jsonl` (corpus de dominio regulado)

17 documentos **inteiramente ficticios**, escritos para este repositorio. Nao derivam de
relatorio, politica ou ata de nenhuma instituicao real.

Uma linha por documento:
`{"id", "source", "gestora", "tipo", "pagina", "data", "text"}`.

**Duas gestoras ficticias**, e isso nao e enfeite:

| Gestora | Fundo | Documentos |
|---|---|---|
| `meridiano` | Meridiano Absoluto FIM | relatorio trimestral, regulamento, politica de credito, ata de comite de risco, fato relevante |
| `aurora` | Aurora Yield FIRF | relatorio trimestral, regulamento |
| `comum` | — | glossario (FPD, VaR, marca d'agua, CDI) |

**Sem duas gestoras nao ha como testar isolamento de escopo.** Com uma so, "o sistema nao
vazou" seria vacuo: nao havia o que vazar. O campo `gestora` vira filtro de metadado na
recuperacao (`Escopo.filtros`), entao o documento fora do escopo nunca chega ao modelo.
Ha um teste que trava essa propriedade do corpus
(`tests/test_evals_oraculos.py::test_corpus_financeiro_tem_duas_gestoras_para_testar_escopo`).

Os numeros sao **internamente consistentes** de proposito: cada pergunta do golden set tem
resposta inequivoca e verificavel por casamento de string, sem juiz. Exemplos: rentabilidade
2025 de 12,4% (Meridiano) contra 9,8% (Aurora); PL de R$ 847,3 mi contra R$ 312,5 mi; FPD
maximo de 3,5%; limite de VaR de 2,5% do PL.

O par deste corpus e o `evals/golden_set.jsonl` (18 casos em quatro categorias:
`fundamentada`, `sem_suporte`, `fora_de_politica`, `fora_de_escopo`). Metodologia,
baseline e limitacoes conhecidas em [`evals/README.md`](../evals/README.md).

### Por que um dominio financeiro ficticio, e nao um corpus publico real

Um corpus real (relatorios da CVM, por exemplo) daria realismo, mas **impediria o teste de
isolamento**: seria preciso inventar a segregacao por gestora de qualquer jeito, e os numeros
mudariam a cada revisao do documento-fonte, quebrando o golden set. Aqui a escolha foi
priorizar **verificabilidade** sobre realismo — o corpus existe para provar comportamento do
sistema, nao para representar o mercado.

Trocar por corpus real depois e direto: manter o campo `gestora` (ou equivalente de
segregacao) e realinhar o `golden_set.jsonl`.
