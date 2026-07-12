import json, os
import pytest
from eval.validate_dataset import validate
from eval.run_eval import gold_ranks, retrieval_record
from rag.chunk import load_manifest

HERE = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def questions():
    with open(os.path.join(HERE, "..", "eval", "questions.json")) as f:
        return json.load(f)


def test_dataset_is_valid(questions):
    assert validate(questions, load_manifest()) == []


def test_unique_ids(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_answerable_have_gold_articles(questions):
    for q in questions:
        if q["answerable_from_corpus"]:
            assert q["gold_articles"], q["id"]
        else:
            assert q["gold_articles"] == [], q["id"]


def test_accounting_16_answerable_2_controls(questions):
    answerable = [q for q in questions if q["answerable_from_corpus"]]
    controls = [q for q in questions if not q["answerable_from_corpus"]]
    assert len(answerable) == 16
    assert len(controls) == 2
    assert {q["type"] for q in controls} == {"statute_boundary", "out_of_corpus"}


def test_multi_article_needs_all_gold_for_full_credit():
    retrieved = [{"article": "85"}, {"article": "43"}, {"article": "86"}]
    q = {"answerable_from_corpus": True, "gold_articles": ["85", "86"]}
    rec = retrieval_record(q, retrieved)
    assert rec["first_gold_rank"] == 1
    assert rec["all_gold_retrieved"] is True
    # drop one side and full credit disappears
    rec2 = retrieval_record(q, retrieved[:2])
    assert rec2["first_gold_rank"] == 1
    assert rec2["all_gold_retrieved"] is False


def test_gold_ranks_come_from_structured_fields_not_prose():
    # a control question's explanatory text may mention articles; the
    # runner must score from gold_articles, which is empty
    q = {"answerable_from_corpus": False, "gold_articles": []}
    rec = retrieval_record(q, [{"article": "86"}])
    assert rec == {"scored": False}
    assert gold_ranks([], [{"article": "86"}]) == {}


def test_control_questions_excluded_from_retrieval_metrics(questions):
    from eval.run_eval import aggregate
    records = []
    for q in questions:
        records.append({
            "retrieval": retrieval_record(q, []),
            "answer": {"refused": not q["answerable_from_corpus"]},
        })
    agg = aggregate(records, 5)
    assert agg["n_retrieval_scored"] == 16
    assert agg["n_controls"] == 2
