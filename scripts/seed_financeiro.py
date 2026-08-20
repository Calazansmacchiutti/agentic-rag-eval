"""Indexa o corpus financeiro sintetico numa collection PROPRIA.

    python scripts/seed_financeiro.py            # indexa em "financeiro"
    python scripts/seed_financeiro.py --recriar  # apaga e recria antes

Collection separada de proposito: o corpus do projeto (`data/corpus.jsonl`, collection
"documents") e o financeiro medem coisas diferentes. Misturar os dois na mesma collection
faria o eval de um recuperar trecho do outro - e as metricas passariam a medir contaminacao.

Cada linha vira um ponto com o texto no payload e TODOS os metadados preservados, incluindo
`gestora`, que e o campo usado como filtro de escopo (`Escopo.filtros`).
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from agentic_rag.retriever import Retriever

CORPUS = RAIZ / "data" / "financeiro" / "corpus.jsonl"
COLLECTION = "financeiro"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recriar", action="store_true", help="apaga a collection antes de indexar")
    args = ap.parse_args()

    docs = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = Retriever(collection=COLLECTION)

    if args.recriar:
        cli = r._qdrant()
        if cli.collection_exists(COLLECTION):
            cli.delete_collection(COLLECTION)
            print(f"collection '{COLLECTION}' apagada")

    # index_chunks: os documentos ja sao a unidade de recuperacao (curtos e auto-contidos),
    # entao NAO re-chunkamos - preserva a citacao apontando para o documento inteiro.
    textos = [d["text"] for d in docs]
    metas = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    n = r.index_chunks(textos, metas)

    gestoras = {}
    for d in docs:
        gestoras[d["gestora"]] = gestoras.get(d["gestora"], 0) + 1
    print(f"indexados {n} documentos em '{COLLECTION}': {gestoras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
