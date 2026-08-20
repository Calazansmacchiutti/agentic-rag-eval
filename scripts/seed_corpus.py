"""Indexa o corpus de referencia (data/corpus.jsonl) na collection padrao do Retriever.

Reproduzivel e versionado (substitui o script de scratchpad). Uso:

    make qdrant                 # sobe o Qdrant em :6333
    python scripts/seed_corpus.py   # (ou: make seed)

Cada linha do corpus e um JSON {"id", "source", "text"}. O texto vai para o Qdrant com
os demais campos como metadados, de modo que o /ask e o eval harness tenham o que recuperar.
"""
import json
import sys
from pathlib import Path

# permite rodar sem instalar o pacote (mesma estrategia do pytest: src no path)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.retriever import Retriever  # noqa: E402


def load_corpus(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def main(path: str = "data/corpus.jsonl") -> None:
    docs = load_corpus(ROOT / path)
    if not docs:
        raise SystemExit(f"corpus vazio: {path}")

    retriever = Retriever()
    # comeca limpo p/ o seed ser deterministico (recria a collection)
    if retriever._qdrant().collection_exists(retriever.collection):
        retriever._qdrant().delete_collection(retriever.collection)

    chunks = [d["text"] for d in docs]
    metas = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    n = retriever.index_chunks(chunks, metas)
    print(f"indexados {n} documentos na collection '{retriever.collection}'")


if __name__ == "__main__":
    main(*sys.argv[1:])
