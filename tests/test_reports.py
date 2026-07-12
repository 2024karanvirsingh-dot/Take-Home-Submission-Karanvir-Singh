import csv, json, os, subprocess, sys
import pytest
from eval.build_report import build as build_report, failure_category

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "outputs")


def _rec(refused=False, cited=("118",), first_rank=1, all_gold=True,
         answerable=True, gold=("118",), retrieved=("118", "43")):
    return {
        "answerable_from_corpus": answerable,
        "gold_articles": list(gold),
        "retrieved": [{"article": a} for a in retrieved],
        "retrieval": ({"scored": True, "first_gold_rank": first_rank,
                       "all_gold_retrieved": all_gold}
                      if answerable else {"scored": False}),
        "answer": {"cited_articles": list(cited)},
    }


def _score(refused=False, hit=True, cov=1.0, support=True, refusal_ok=True):
    return {"refused": refused, "citation_hit": hit, "content_coverage": cov,
            "support_ok": support, "refusal_correct": refusal_ok,
            "citation_precision": 1.0}


def test_failure_category_success():
    cat, _ = failure_category(_rec(), _score())
    assert cat == "success"


def test_failure_category_false_refusal_from_miss():
    cat, why = failure_category(_rec(refused=True, first_rank=0),
                                _score(refused=True, hit=None, cov=None))
    assert cat == "false_refusal_after_retrieval_miss"
    assert "118" in why


def test_failure_category_misattribution():
    cat, _ = failure_category(_rec(cited=("25a",), first_rank=0, all_gold=False),
                              _score(hit=False))
    assert cat == "misattribution_after_retrieval_miss"


def test_failure_category_partial_coverage():
    cat, why = failure_category(_rec(), _score(cov=0.5))
    assert cat == "partial" and "coverage" in why


def test_failure_category_controls():
    ok, _ = failure_category(_rec(answerable=False, gold=()),
                             _score(refused=True, hit=None, cov=None))
    assert ok == "control_refused_correctly"
    bad, _ = failure_category(_rec(answerable=False, gold=()),
                              _score(refused=False, refusal_ok=False))
    assert bad == "control_answered"


def test_report_generation_schema(tmp_path):
    # generated from the committed bm25 run, written to a temp dir so the
    # committed artifacts are not touched
    rows = build_report("bm25_baseline", out_dir=str(tmp_path))
    assert len(rows) == 18
    expected = {"id", "type", "question", "answerable_from_corpus",
                "gold_articles", "retrieved_articles", "first_gold_rank",
                "refused", "cited_articles", "citation_hit",
                "citation_precision", "content_coverage", "support_ok",
                "numbers_supported", "category", "interpretation", "answer"}
    assert expected <= set(rows[0])
    with open(tmp_path / "question_results.csv") as f:
        assert len(list(csv.DictReader(f))) == 18
    report = (tmp_path / "evaluation_report.md").read_text()
    assert report.count("## q") == 18
    assert "**Category.**" in report


def test_before_after_examples_generated(tmp_path):
    from eval.build_before_after import build as build_ba
    n = build_ba(out_dir=str(tmp_path))
    assert n >= 3, "at least three complete before/after examples"
    text = (tmp_path / "before_after.md").read_text()
    for qid in ("q05", "q14", "q04", "q17"):
        assert qid in text
    assert text.count("**Before**") + text.count("**Before** (") >= 3
    assert "Answer:" in text and "Retrieved" in text


def test_committed_artifacts_match_schema():
    # the committed evaluator facing artifacts exist and parse
    with open(os.path.join(OUT, "question_results.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18
    with open(os.path.join(OUT, "summary.json")) as f:
        summary = json.load(f)
    assert summary["default_system"]["run"] == "bm25_baseline"
    assert summary["best_measured_configuration"]["run"] == "hybrid_baseline"
    assert "retrieval" in summary and "answers" in summary


def test_cli_pipeline_answers(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "rag.pipeline",
         "What are the three types of courts-martial?"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0
    assert "Answer:" in out.stdout and "Retrieved:" in out.stdout
