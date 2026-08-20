# Governança de IA

Este documento descreve as garantias que o sistema oferece — e, com igual importância, as
que **não** oferece. Ele existe porque em contexto regulado a pergunta não é "o modelo é
bom?", e sim "você consegue explicar, meses depois, por que ele respondeu aquilo?".

---

## 1. Princípio: o modelo explica, não decide

O LLM **não calcula número e não toma decisão**. Ele recupera trechos, redige uma
explicação e cita a origem de cada afirmação. Qualquer valor que apareça na resposta tem de
existir literalmente num trecho recuperado.

A consequência prática: se amanhã o modelo for trocado, os **números não mudam** — apenas a
redação. Um sistema em que a troca de modelo altera valores não é auditável.

Isto é a aplicação direta do ADR 0005: seleção determinística por restrições, LLM fora do
laço crítico.

## 2. Recusar é um resultado correto

O sistema recusa em quatro situações, todas registradas com código próprio:

| Decisão | Quando |
|---|---|
| `recusado_fora_de_politica` | pedido de recomendação de investimento/crédito |
| `recusado_sem_fundamento` | o próprio modelo declarou que o contexto não sustenta |
| `recusado_sem_citacao` | afirmou algo sem apontar trecho |
| `recusado_citacao_invalida` | citou índice inexistente (citação fabricada) |

Uma recusa retorna **HTTP 200** com `permitido: false` e o motivo. Não é erro de servidor: é
o guardrail funcionando. Tratar recusa como exceção incentiva a contorná-la.

O primeiro caso é barrado **antes de qualquer chamada de LLM** — não gasta token nem toca
no acervo.

## 3. O que fica registrado

Cada resposta gera um evento append-only em `logs/auditoria.jsonl`:

```json
{"pergunta":"...","resposta":"...","grounded":true,"confidence":0.9,
 "trechos":["c0fc3c713f09a433"],"versao_prompt":"d8def792448c",
 "modelo":"claude-opus-4-8","usuario":"ana","papel":"analista",
 "decisao":"respondido","motivo":"","em":"2026-08-19T18:22:41+00:00"}
```

Três decisões deliberadas:

- **Guarda o ID do trecho, não o texto.** A trilha não replica o acervo (que pode ser
  confidencial) e ainda permite reconstruir a decisão.
- **`versao_prompt` é o hash do conteúdo de `prompts/system.md`**, não um número escrito à
  mão. Mudou o prompt, muda a versão — sem depender de alguém lembrar de incrementar.
- **Append-only por construção**: o arquivo é aberto em modo `a`. Registrar nunca sobrescreve.

## 4. Escopo: a restrição acontece antes do modelo

`Escopo.filtros` vira **filtro de metadado na recuperação**. O trecho fora do escopo do
usuário nunca entra no contexto — o modelo não o vê e portanto não pode vazá-lo.

Restringir por instrução de prompt ("não fale sobre X") não é controle de acesso: é pedido.
O controle tem de ser estrutural.

## 5. Travas do agente

- **`LimiteDeTurnos`** — teto explícito de chamadas de LLM por pergunta. Estoura com erro
  claro em vez de truncar em silêncio; um agente que para de buscar sem avisar produz
  resposta pior sem que ninguém perceba.
- **`PortaoDeEscrita`** — o agente é *read-only* por padrão. O portão existe para que essa
  escolha seja explícita no código e no log, e para que habilitar escrita amanhã seja
  decisão consciente, com lista de permissão.

## 6. O que este sistema NÃO garante

Dito sem rodeio, porque uma seção de governança que só lista virtudes não serve:

- **Não garante que a resposta esteja correta.** Garante que ela é rastreável até um trecho
  e que citações fabricadas são barradas. Trecho errado no acervo produz resposta errada
  com citação válida.
- **Não faz verificação de fato contra o mundo.** A fonte de verdade é o acervo indexado.
- **Não é controle de acesso completo.** `Escopo` filtra a recuperação; não há autenticação
  embutida. Em produção, `/perguntar` e `/auditoria` precisam ficar atrás de autenticação
  real.
- **A trilha não é assinada nem imutável no nível de armazenamento.** É append-only por
  convenção do adapter; quem tiver acesso de escrita ao arquivo pode alterá-lo. Para
  garantia forte, trocar por armazenamento WORM ou banco com log de auditoria.
- **A calibragem do juiz vale para o conjunto avaliado**, não é uma garantia universal de
  qualidade (ver ADR 0005 e `docs/METRICAS_RAG.md`).

## 7. Como verificar as afirmações acima

Nenhuma dessas garantias precisa ser aceita na palavra:

```bash
pytest tests/test_use_case_responder.py   # guardrails, escopo, auditoria, orçamento
pytest tests/test_api_auditavel.py        # recusa como 200, trilha, filtro por usuário
```

Os testes rodam **sem rede, sem Qdrant e sem chave de API** — o que só é possível porque o
domínio depende de portas, não de fornecedores.
