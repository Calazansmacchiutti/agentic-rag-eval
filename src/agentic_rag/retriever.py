"""Retrieval: busca vetorial (Qdrant) com embeddings normalizados (cosine)."""
import uuid

from agentic_rag import ingest
from agentic_rag.config import settings

COLLECTION = "documents"


class Retriever:
    def __init__(self, url: str | None = None, top_k: int | None = None, collection: str = COLLECTION):
        self.url = url or settings.vector_db_url
        self.top_k = top_k if top_k is not None else settings.top_k  # default via config (tunavel)
        self.collection = collection
        self._client = None

    def _qdrant(self):
        """Cliente Qdrant preguicoso (import pesado adiado p/ nao onerar o import do modulo)."""
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url)
        return self._client

    def ensure_collection(self) -> None:
        """Cria a collection com o tamanho de vetor do modelo de embedding, se nao existir."""
        from qdrant_client.models import Distance, VectorParams

        cli = self._qdrant()
        if not cli.collection_exists(self.collection):
            cli.create_collection(
                self.collection,
                vectors_config=VectorParams(size=ingest.embedding_dim(), distance=Distance.COSINE),
            )

    def index_chunks(self, chunks: list[str], metadatas: list[dict] | None = None) -> int:
        """Embed + upsert de chunks JA prontos (sem re-chunkar), um metadado por chunk.

        E o caminho do estruturador de PDF: o Agente A ja decidiu o recorte ideal, entao
        aqui so vetorizamos e indexamos, preservando o payload (heading_path, pagina, tipo).
        """
        from qdrant_client.models import PointStruct

        if not chunks:
            return 0
        self.ensure_collection()
        vectors = ingest.embed(chunks)
        metas = metadatas or [{} for _ in chunks]
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload={"text": c, **m})
            for c, vec, m in zip(chunks, vectors, metas)
        ]
        self._qdrant().upsert(self.collection, points=points)
        return len(points)

    def index(self, text: str, metadata: dict | None = None) -> int:
        """Chunking ingenuo (tamanho fixo) + embed + upsert. Devolve nro de chunks."""
        chunks = ingest.chunk(text)
        return self.index_chunks(chunks, [dict(metadata or {}) for _ in chunks])

    @staticmethod
    def _build_filter(filters: dict | None):
        """Monta um Filter do Qdrant a partir de {campo: valor} (match exato; AND entre campos).

        Em campo de array (ex.: heading_path), MatchValue casa se o array CONTEM o valor,
        entao da p/ buscar "so na secao Metodos" ou "so chunks type=table".
        """
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        )

    def search(self, query: str, filters: dict | None = None) -> list[dict]:
        """Busca os top_k trechos mais similares, opcionalmente filtrando por metadado.

        Ex.: search("perda", filters={"type": "table"}) ou {"heading_path": "Metodos"}.
        """
        # com rerank, recupera um pool maior p/ o cross-encoder escolher os melhores top_k
        limit = settings.rerank_fetch_k if settings.rerank else self.top_k
        vec = ingest.embed([query])[0]
        res = self._qdrant().query_points(
            collection_name=self.collection,
            query=vec.tolist(),
            limit=limit,
            with_payload=True,
            query_filter=self._build_filter(filters),
        )
        out = []
        for point in res.points:
            payload = point.payload or {}
            out.append({"text": payload.get("text", ""), "score": point.score, **payload})
        if settings.rerank:
            from agentic_rag import rerank as rerank_mod

            out = rerank_mod.rerank(query, out, self.top_k)
        return out
