"""Testes do eval harness sem tocar ragas/LLM/Qdrant.

Injeta answer_fn e scorer fakes e aponta o baseline p/ um arquivo temporario, entao
exercita a logica deterministica: carga do golden set, montagem de registros, gate de
promocao e persistencia do baseline.
"""
import json

import pytest

from agentic_rag import evaluate
from agentic_rag import llm as llm_mod
from agentic_rag.evaluate import EvalItem, EvalRecord, _ItemScore


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_dataset_parses_and_skips_blanks(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"question": "q1", "ground_truth": "a1"}\n\n{"question": "q2"}\n',
        encoding="utf-8",
    )
    items = evaluate.load_dataset(str(p))
    assert items == [
        EvalItem(question="q1", ground_truth="a1"),
        EvalItem(question="q2", ground_truth=""),
    ]


def test_load_dataset_errors_on_missing_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.load_dataset(str(tmp_path / "nao_existe.jsonl"))
    empty = tmp_path / "vazio.jsonl"
    empty.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate.load_dataset(str(empty))


def test_build_records_runs_system_per_item():
    items = [EvalItem(question="q1", ground_truth="ref1")]

    def fake_answer_fn(question):
        assert question == "q1"
        return "resposta gerada", ["ctx a", "ctx b"]

    records = evaluate.build_records(items, fake_answer_fn)
    assert records == [
        EvalRecord(
            question="q1",
            answer="resposta gerada",
            contexts=["ctx a", "ctx b"],
            ground_truth="ref1",
        )
    ]


def test_llm_judge_score_agrega_rubrica_analitica(monkeypatch):
    records = [
        EvalRecord(question="q1", answer="a1", contexts=["c0", "c1"], ground_truth="g"),
        EvalRecord(question="q2", answer="a2", contexts=[], ground_truth=""),
    ]
    seq = [
        # faith grounded=1.0 | relev partial=0.5 | prec 1/2=0.5
        _ItemScore(rationale="r", faithfulness="grounded", answer_relevancy="partial",
                   context_relevance=[True, False]),
        # faith unsupported=0.0 | relev complete=1.0 | prec 0.0 (sem contexto)
        _ItemScore(rationale="r", faithfulness="unsupported", answer_relevancy="complete",
                   context_relevance=[]),
    ]
    calls = []

    def fake_complete(prompt, schema=None, model=None, temperature=None):
        assert schema is _ItemScore
        calls.append(temperature)
        return seq[len(calls) - 1]

    monkeypatch.setattr(llm_mod, "complete", fake_complete)  # 1 chamada por item, sem rede

    out = evaluate.llm_judge_score(records)
    assert len(calls) == 2
    assert calls[0] == evaluate.settings.grader_temperature  # juiz roda com temperatura baixa
    assert out["faithfulness"] == pytest.approx(0.5)          # mean(1.0, 0.0)
    assert out["answer_relevancy"] == pytest.approx(0.75)     # mean(0.5, 1.0)
    assert out["context_precision"] == pytest.approx(0.25)    # mean(0.5, 0.0)


def test_precision_helper_robusto_a_desalinhamento():
    assert evaluate._precision([True, True, True], 2) == pytest.approx(1.0)  # trunca flags extras
    assert evaluate._precision([True], 2) == pytest.approx(0.5)              # faltante conta False
    assert evaluate._precision([], 0) == 0.0                                  # sem contexto


def test_llm_judge_score_empty_returns_zeros():
    assert evaluate.llm_judge_score([]) == {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
    }


def test_gate_passes_without_baseline():
    passed, deltas = evaluate.gate({"faithfulness": 0.7}, baseline={})
    assert passed is True
    assert deltas == {"faithfulness": 0.0}


def test_gate_passes_when_beating_baseline():
    passed, deltas = evaluate.gate(
        {"faithfulness": 0.8, "context_precision": 0.6},
        baseline={"faithfulness": 0.7, "context_precision": 0.6},
    )
    assert passed is True
    assert deltas["faithfulness"] == pytest.approx(0.1)
    assert deltas["context_precision"] == pytest.approx(0.0)


def test_gate_fails_when_below_baseline():
    passed, _ = evaluate.gate({"faithfulness": 0.5}, baseline={"faithfulness": 0.7})
    assert passed is False


def test_gate_tolerance_allows_small_regression():
    passed, _ = evaluate.gate(
        {"faithfulness": 0.68}, baseline={"faithfulness": 0.70}, tolerance=0.05
    )
    assert passed is True


def test_run_eval_first_run_establishes_baseline(tmp_path, monkeypatch):
    ds = tmp_path / "eval.jsonl"
    _write_jsonl(ds, [{"question": "q1", "ground_truth": "a1"}])
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(evaluate.settings, "baseline_metrics_path", str(baseline_path))

    report = evaluate.run_eval(
        str(ds),
        answer_fn=lambda q: ("ans", ["ctx"]),
        scorer=lambda records: {"faithfulness": 0.9},
        promote=True,
    )

    assert report.passed is True
    assert report.promoted is True
    assert report.provenance_ok is True
    assert report.baseline == {}          # nao havia baseline nesta rodada
    assert report.n_items == 1
    # baseline persistido COM proveniencia (metrics + scorer + grader_model)
    saved = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert saved["metrics"] == {"faithfulness": 0.9}
    assert saved["scorer"] == "<lambda>"
    assert saved["grader_model"] == evaluate.settings.grader_model


def test_run_eval_gate_blocks_regression(tmp_path, monkeypatch):
    ds = tmp_path / "eval.jsonl"
    _write_jsonl(ds, [{"question": "q1"}])
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({
            "metrics": {"faithfulness": 0.8},
            "scorer": "<lambda>",  # mesma proveniencia do scorer da rodada -> comparavel
            "grader_model": evaluate.settings.grader_model,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate.settings, "baseline_metrics_path", str(baseline_path))

    report = evaluate.run_eval(
        str(ds),
        answer_fn=lambda q: ("ans", ["ctx"]),
        scorer=lambda records: {"faithfulness": 0.6},  # regressao
        promote=True,
    )

    assert report.passed is False
    assert report.promoted is False
    assert report.provenance_ok is True
    assert report.deltas["faithfulness"] == pytest.approx(-0.2)
    # baseline NAO foi sobrescrito
    assert json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"] == {"faithfulness": 0.8}


def test_run_eval_baseline_de_outro_scorer_e_incomparavel(tmp_path, monkeypatch):
    """Proveniencia divergente (outro scorer/juiz) => trata como sem baseline (nao vira 'ganho falso')."""
    ds = tmp_path / "eval.jsonl"
    _write_jsonl(ds, [{"question": "q1"}])
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({
            "metrics": {"faithfulness": 0.9},
            "scorer": "ragas_score",  # scorer diferente do desta rodada
            "grader_model": evaluate.settings.grader_model,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate.settings, "baseline_metrics_path", str(baseline_path))

    report = evaluate.run_eval(
        str(ds),
        answer_fn=lambda q: ("ans", ["ctx"]),
        scorer=lambda records: {"faithfulness": 0.3},  # abaixo do baseline antigo, mas incomparavel
        promote=False,
    )

    assert report.provenance_ok is False
    assert report.passed is True                      # nao falha contra baseline incomparavel
    assert report.deltas == {"faithfulness": 0.0}     # gate rodou como se nao houvesse baseline
