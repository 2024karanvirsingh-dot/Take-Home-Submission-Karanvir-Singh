import os
import pytest
from rag.pipeline import RagPipeline


@pytest.fixture(scope="module")
def pipe():
    return RagPipeline()


def test_no_credentials_required(pipe, monkeypatch=None):
    # the environment carries no API key of any kind and the pipeline
    # must not care
    assert not any(k.endswith("_API_KEY") and os.environ[k]
                   for k in os.environ), \
        "test environment unexpectedly has an API key set"
    out = pipe.answer("What rights does Article 31 provide?")
    assert out["answer"]


def test_article_31_end_to_end(pipe):
    out = pipe.answer("What rights does Article 31 provide?")
    assert not out["refused"]
    assert "31" in out["cited_articles"]
    assert "(Art. 31, UCMJ)" in out["answer"]


def test_control_question_end_to_end(pipe):
    out = pipe.answer(
        "What does Rule for Courts-Martial 707 require for a speedy trial?")
    assert out["refused"] and out["boundary"]


def test_retrieval_shape(pipe):
    out = pipe.answer("What are the three types of courts-martial?")
    assert len(out["retrieved"]) == 5
    for r in out["retrieved"]:
        assert {"citation", "article", "type", "title", "score", "id"} <= set(r)


def test_permissive_pipeline_runs():
    p = RagPipeline(policy="permissive")
    out = p.answer("What is the maximum confinement for being AWOL for more than 30 days?")
    assert not out["refused"]  # permissive never refuses, that is its point
