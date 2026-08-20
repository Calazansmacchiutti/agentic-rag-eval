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

## O juiz LLM

Depois dos oráculos, o harness roda o juiz — **apenas nos itens que foram respondidos**.

Recusa não vai a julgamento. Ela já foi verificada exatamente pelo oráculo (a decisão bateu ou
não bateu); pedir a um LLM que opine sobre a qualidade de *"não encontrei base para responder"*
gasta chamada e não acrescenta informação. Na rodada atual são **15 julgados e 3 pulados**.

Resposta que **reprovou** no oráculo continua indo ao juiz: saber *quão* ruim ficou a resposta
errada é diagnóstico útil.

A rubrica **não é reescrita aqui** — `evals/juiz.py` importa `_JUDGE_PROMPT`, `_FAITH_MAP`,
`_RELEV_MAP` e `_precision` de `agentic_rag.evaluate`. Duplicar o texto criaria duas rubricas
que divergem em silêncio, e a comparação entre rodadas passaria a medir a diferença entre elas
em vez do sistema. O acoplamento é proposital, está declarado no módulo e há um teste que trava
a identidade dos objetos.

Toda métrica sai com **proveniência junto** (`scorer` + `grader_model`), pela mesma razão do
ADR 0005: métrica sem proveniência não é comparável entre rodadas — uma troca de juiz se
passaria por ganho do sistema. O `JuizStub` se identifica como `scorer="stub"` e imprime um
aviso, para que seus números nunca sejam confundidos com medição real.

## Como rodar

```bash
python -m evals.harness --stub      # sem LLM, sem Qdrant, sem chave — roda em CI
python -m evals.harness             # sistema real (oráculos + juiz)
python -m evals.harness --sem-juiz  # só oráculos: não gasta chamada de LLM
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

## Relação com `evaluate.py`

Os dois convivem, com papéis distintos:

| | `evaluate.py` | `evals/harness.py` |
|---|---|---|
| Conjunto | `data/eval_set.jsonl` (17 perguntas sobre o próprio projeto) | `evals/golden_set.jsonl` (18 casos, domínio financeiro) |
| Mede | qualidade da resposta | **garantia**: recusa, escopo — e qualidade, via o mesmo juiz |
| Gate | promoção contra `models/baseline_metrics.json` | falha ou vazamento reprova |

A **rubrica do juiz é uma só**, morando em `evaluate.py` e importada pelo harness. Os
`scorer` names coincidem (`llm_judge_score`), então os baselines dos dois são comparáveis
entre si quando o `grader_model` for o mesmo.

O que este harness acrescenta e aquele não cobria: **comportamento de recusa e isolamento de
escopo**.

## Limitações conhecidas

- O corpus é pequeno (17 documentos). Ele testa comportamento, não escala de recuperação.
- `deve_conter` casa string normalizada; uma resposta correta que expresse o número por
  extenso ("doze vírgula quatro por cento") falharia. É um falso negativo conhecido, aceito
  em troca de uma checagem sem modelo.
- O stub não é modelo nem baseline competitivo: é piso. Comparação justa contra um RAG
  ingênuo exigiria o mesmo retriever vetorial sem o loop agentic.
- O `baseline_deterministico.json` cobre **só a camada de oráculos**. As métricas do juiz
  ainda não têm baseline versionado neste harness — o gate de promoção segue em `evaluate.py`,
  contra `models/baseline_metrics.json`.
- O juiz nunca foi executado sobre o corpus financeiro com LLM real: os números de juiz que
  aparecem hoje vêm do `JuizStub` e **não são medição de qualidade**. Rodar `--stub` sem a
  flag produz valores fixos, marcados como tal na saída.
