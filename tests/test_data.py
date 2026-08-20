"""Integridade do par corpus + golden set (versionados). Sem rede.

Guarda contra JSON quebrado, campos faltando, ids duplicados e golden set pequeno demais
para ter sinal (motivacao do ADR 0005 para expandir o conjunto).
"""
import json
from pathlib import Path

from agentic_rag.evaluate import load_dataset

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.jsonl"
GOLDEN = ROOT / "data" / "eval_set.jsonl"

MIN_GOLDEN = 15  # abaixo disso o eval nao tem significancia (ADR 0005)


def _read_jsonl(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_corpus_bem_formado():
    docs = _read_jsonl(CORPUS)
    assert len(docs) >= 10
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))  # ids unicos
    for d in docs:
        assert d["text"].strip()      # texto nao-vazio
        assert d["source"].strip()


def test_golden_set_carrega_e_tem_tamanho_minimo():
    items = load_dataset(str(GOLDEN))
    assert len(items) >= MIN_GOLDEN
    assert all(it.question.strip() for it in items)


def test_golden_set_tem_respondiveis_e_adversariais():
    items = load_dataset(str(GOLDEN))
    adversarial = [it for it in items if it.adversarial]
    answerable = [it for it in items if not it.adversarial]
    assert len(adversarial) >= 2       # cobre o guardrail de grounding
    assert len(answerable) >= 10       # maioria respondivel
    # adversarial deve deixar claro que nao consta no corpus
    assert all("nao consta" in it.ground_truth.lower() for it in adversarial)
