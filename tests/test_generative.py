import importlib.util
import re
import sys
import pytest

needs_model = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="generative answerer is an optional extra (requirements-gen.txt)")
from rag.generate import (build_prompt, parse_citations, REFUSAL_MARKER,
                          GEN_REFUSAL_TEXT)


def chunk(number, title, text):
    return {"id": f"sec_{number}.txt::0", "number": number, "title": title,
            "text": text, "type": "punitive",
            "citation": f"Art. {number}, UCMJ"}


RETRIEVED = [(chunk("86", "Absence without leave",
                    "Any member who absents himself without authority shall "
                    "be punished as a court-martial may direct."), 9.0),
             (chunk("85", "Desertion",
                    "Any member who deserts in time of war shall suffer "
                    "death."), 5.0)]


def test_prompt_contains_only_retrieved_text():
    # context isolation: every non-instruction line of the prompt comes
    # from the retrieved chunks (or is the question), nothing else
    q = "When is a member AWOL?"
    prompt = build_prompt(q, RETRIEVED)
    context_part = prompt.split("\n\n", 1)[1].rsplit("Question:", 1)[0]
    allowed = " ".join(c["text"] + " " + c["title"] + " " + c["number"]
                       for c, _ in RETRIEVED)
    for word in re.findall(r"[A-Za-z]+", context_part.replace("Article", "")):
        assert word in allowed, f"prompt word {word!r} not from retrieved chunks"
    assert q in prompt
    assert REFUSAL_MARKER in prompt  # the refusal instruction is present


def test_prompt_respects_context_budget():
    long_chunk = chunk("2", "Persons subject", "word " * 2000)
    prompt = build_prompt("Who is subject?", [(long_chunk, 1.0)],
                          max_context_words=100)
    assert len(prompt.split()) < 220  # instructions + capped context


def test_prompt_preserves_retrieval_order():
    prompt = build_prompt("q", RETRIEVED)
    assert prompt.index("[Article 86") < prompt.index("[Article 85")


def test_parse_citations():
    text = ("Desertion in wartime is capital (Art. 85, UCMJ); AWOL is not "
            "(Article 86). Article 85 again.")
    assert parse_citations(text) == ["85", "86"]


def test_missing_transformers_fails_helpfully(monkeypatch):
    # simulate the core-only install: importing transformers raises, the
    # answerer must exit with instructions rather than degrade silently
    monkeypatch.setitem(sys.modules, "transformers", None)
    from rag.generate import GenerativeAnswerer
    with pytest.raises(SystemExit) as e:
        GenerativeAnswerer()
    assert "requirements-gen.txt" in str(e.value)


def test_boundary_check_runs_before_the_model():
    # RCM/MCM questions are refused by system policy without loading any
    # model; construct the object shell without __init__ to prove no model
    # is consulted
    from rag.generate import GenerativeAnswerer
    g = GenerativeAnswerer.__new__(GenerativeAnswerer)
    out = g.construct("What does Rule for Courts-Martial 707 require for a "
                      "speedy trial?", RETRIEVED)
    assert out["refused"] and out["boundary"]


# a live decode; skipped on the core install and anywhere the model is not
# available. Greedy decoding, so the checks are deterministic per platform.


@pytest.fixture(scope="module")
def answerer():
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("generative answerer needs the full install")
    from rag.generate import GenerativeAnswerer
    try:
        return GenerativeAnswerer()
    except OSError:
        pytest.skip("flan-t5-small not downloadable in this environment")


@needs_model
def test_generative_end_to_end_shape(answerer):
    out = answerer.construct("Can a deserter be sentenced to death in time "
                             "of war?", RETRIEVED)
    assert set(out) >= {"text", "refused", "boundary", "cited_articles",
                        "sentences"}
    assert out["sentences"] == []  # generated text carries no verbatim map
    if out["refused"]:
        assert out["text"] == GEN_REFUSAL_TEXT


@needs_model
def test_generative_deterministic(answerer):
    q = "Can a deserter be sentenced to death in time of war?"
    a = answerer.construct(q, RETRIEVED)
    b = answerer.construct(q, RETRIEVED)
    assert a == b
