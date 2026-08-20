# Segurança

Escopo deste documento: o que o repositório protege hoje, o que assume do ambiente, e o que
precisa ser resolvido antes de qualquer uso real com dados de terceiros.

---

## Segredos

- `ANTHROPIC_API_KEY` e demais chaves vêm **do ambiente**, via `pydantic-settings`. Não há
  chave no código.
- `.env` está no `.gitignore`; só `.env.example` é versionado, com placeholders.
- O repositório não contém dado real: `data/corpus.jsonl` e `data/eval_set.jsonl` são
  sintéticos e servem para o pipeline e o eval rodarem sem expor nada.

## Superfície exposta

| Rota | O que faz | Risco |
|---|---|---|
| `GET /health` | liveness | nenhum |
| `POST /perguntar` | consulta o acervo | **sem autenticação hoje** |
| `GET /auditoria` | lê a trilha | **sem autenticação hoje; expõe perguntas de outros usuários** |

**Antes de expor em rede:** as duas últimas precisam de autenticação e autorização reais. O
campo `usuario` do request é **declarado pelo cliente** — serve para a trilha, não é
identidade verificada. Tratar como identidade seria falha de controle de acesso.

## Isolamento entre usuários

`Escopo.filtros` restringe a recuperação por metadado, então o trecho fora do escopo não
entra no contexto do modelo. Isso é isolamento **de conteúdo**, e depende de o acervo estar
indexado com o metadado correto: se o chunk não tem o campo de segregação, o filtro não o
alcança.

Não há hoje: multi-tenancy no banco vetorial, criptografia por tenant, nem rotação de chave.

## Dados na trilha de auditoria

A trilha guarda **hash do trecho**, não o texto — o log não replica o acervo. Mas guarda
**a pergunta e a resposta em texto**, que podem conter dado sensível informado pelo usuário.
Em ambiente regulado, avaliar mascaramento ou retenção limitada dessas duas colunas.

## Dependências

- `pip-audit` / `safety` não estão no CI hoje — lacuna conhecida.
- `.pre-commit-config.yaml` roda lint e formatação; **não** roda varredura de segredo.
  Adicionar `gitleaks` é o próximo passo natural (o repo-molde usado como referência tem
  `.gitleaks.toml`).

## Modelo de ameaça, resumido

| Ameaça | Situação |
|---|---|
| Vazamento de chave via commit | mitigado (`.gitignore` + `.env.example`) |
| Vazamento de acervo via resposta | mitigado por filtro de escopo na recuperação |
| Citação fabricada passando como fato | mitigado por `guardrails.checar_resposta` |
| Prompt injection no conteúdo indexado | **não mitigado** — trecho malicioso no acervo pode influenciar a resposta |
| Acesso indevido à trilha | **não mitigado** — `/auditoria` está aberta |
| Adulteração da trilha | **não mitigado** — append-only por convenção, não por armazenamento |
| Exaustão de custo por laço do agente | mitigado por `LimiteDeTurnos` |

## Reportar problema

Abra uma issue **sem** incluir dado sensível, chave ou trecho de acervo real.
