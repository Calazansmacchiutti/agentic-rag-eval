# ADR 0002 - Retrieval: Qdrant + embeddings locais

## Contexto
Precisamos de busca vetorial reprodutivel, gratuita e que rode offline (dev/CI), sem custo de
API de embedding por documento.

## Decisao
Vector DB: **Qdrant** (Docker/local). Embeddings: **SentenceTransformers all-MiniLM-L6-v2**
(dim 384), normalizados, com distancia de **cosseno**. `ingest.embed()` gera os vetores e
`ingest.embedding_dim()` dimensiona a collection. O `Retriever` faz chunk -> embed -> upsert
(`index`) e busca via `query_points` (`search`), devolvendo trechos com texto, score e metadados.

## Trade-offs
- (+) gratuito, offline, leve (384 dims), reprodutivel; modelo trocavel via `EMBEDDING_MODEL`.
- (-) qualidade abaixo de embeddings grandes (OpenAI/Voyage); aceitavel p/ baseline, trocavel depois.
- (+) Qdrant local elimina dependencia de servico gerenciado no portfolio.

## Status
Aceito.
