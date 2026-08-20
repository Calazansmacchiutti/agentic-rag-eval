# Eval harness

Como este projeto mede se o sistema está funcionando — e por que a métrica não é "a resposta
ficou boa".

---

## A tese

Em contexto regulado, a pergunta que importa não é só *"a resposta está correta?"*. É:

1. **O sistema recusou quando devia?** (sem lastro, fora de política, fora de escopo)
2. **Vazou dado de outro escopo?**
3. E só então: a resposta que passou está boa?

As duas primeiras são **binárias e verificáveis sem modelo**. A terceira precisa de juiz — que
é caro e ruidoso (ver [ADR 0005](../docs/adr/0005-eval-estabilizar-llm-judge-e-versionar-baseline.md)).

Daí a ordem do harness: **oráculo determinístico primeiro, juiz depois, e só no que passou.**
Rodar juiz em cima de uma recusa não produz informação — a recusa já foi verificada
exatamente.

## O golden set

`golden_set.jsonl`, 18 casos em quatro categorias:

| Categoria | Casos | O que verifica |
|---|---|---|
| `fundamentada` | 9 | responde certo, com o número correto |
| `sem_suporte` | 3 | recusa quando o corpus não sustenta |
| `fora_de_politica` | 3 | recusa pedido de recomendação |
| `fora_de_escopo` | 3 | não vaza dado de outra gestora |

Cada caso declara a expectativa **antes** de rodar: `decisao_esperada`, `deve_conter`,
`nao_pode_conter`. Nada é inferido na hora.

`nao_pode_conter` é a checagem mais importante do conjunto — é ela que pega vazamento entre
escopos. E vale **também na mensagem de recusa**: negar a resposta citando o dado que deveria
proteger continua sendo vazamento.

## O corpus

`data/financeiro/corpus.jsonl` — 17 documentos **sintéticos** de duas gestoras fictícias
(Meridiano Asset e Aurora Capital): relatório trimestral, política de crédito, ata de comitê
de risco, regulamento, fato relevante e glossário.

Duas gestoras não é enfeite: **sem duas, não há como testar isolamento de escopo.** Há um teste
que trava essa propriedade do corpus.

Nenhum dado é real. Os números são internamente consistentes para que cada pergunta tenha
resposta inequívoca.

## Como rodar

```bash
python -m evals.harness --stub      # sem LLM, sem Qdrant, sem chave — roda em CI
python -m evals.harness             # sistema real
python -m evals.harness --json relatorio.json
```

Saída de processo: **1** se houver qualquer falha **ou** qualquer vazamento. Vazamento reprova
sozinho, independentemente da taxa geral — é falha de segurança, não de qualidade.

## O baseline determinístico

`baseline_deterministico.json` guarda o desempenho de um **piso sem LLM**: casamento de
palavra sobre o corpus, com o mesmo filtro de escopo.

| Categoria | Piso | Leitura |
|---|---|---|
| `fora_de_politica` | **3/3 (100%)** | o guardrail é determinístico; não precisa de modelo |
| `fundamentada` | 7/9 (78%) | busca por palavra acerta boa parte das factuais |
| `sem_suporte` | **0/3 (0%)** | **um buscador sempre acha algo e responde** |
| `fora_de_escopo` | 2/3 (67%) | **zero vazamentos** — o filtro funciona |

Duas conclusões que este piso entrega de graça:

- **Recusar exige compreensão, não busca.** O 0/3 em `sem_suporte` é a evidência. É contra
  isso que o sistema com LLM tem de provar valor.
- **O isolamento de escopo é estrutural, não depende do modelo.** Zero vazamentos com um
  respondedor burro, porque a restrição acontece na recuperação — antes de qualquer LLM.

O baseline existe para que "o LLM melhorou" seja uma afirmação verificável, não uma impressão.

## Relação com o eval anterior

`src/agentic_rag/evaluate.py` continua responsável pelas métricas de **qualidade de resposta**
(faithfulness, answer relevancy, context precision) com juiz LLM calibrado, e pelo gate de
promoção contra `models/baseline_metrics.json`. Ele não foi substituído.

Este harness cobre o que aquele não cobria: **comportamento de recusa e isolamento**. São
camadas complementares — uma mede qualidade, a outra mede garantia.

## Limitações conhecidas

- O corpus é pequeno (17 documentos). Ele testa comportamento, não escala de recuperação.
- `deve_conter` casa string normalizada; uma resposta correta que expresse o número por
  extenso ("doze vírgula quatro por cento") falharia. É um falso negativo conhecido, aceito
  em troca de uma checagem sem modelo.
- O stub não é modelo nem baseline competitivo: é piso. Comparação justa contra um RAG
  ingênuo exigiria o mesmo retriever vetorial sem o loop agentic.
- O juiz LLM ainda não foi integrado a este harness — hoje ele vive em `evaluate.py`. Unificar
  os dois é trabalho pendente.
