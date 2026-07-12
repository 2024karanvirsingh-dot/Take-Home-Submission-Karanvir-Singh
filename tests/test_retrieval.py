import pytest
from rag.chunk import build_chunks
from rag.index import BM25, tokenize, _STOP


@pytest.fixture(scope="module")
def index():
    return BM25(build_chunks())


def test_named_article_query_retrieves_it(index):
    results = index.search(
        "What rights does Article 31 give a servicemember suspected of an offense?", 5)
    assert results[0][0]["number"] == "31"


def test_function_words_are_stopped():
    # regression for the recorded q16 failure where 'about' and 'does'
    # decided the top retrieval slot
    for w in ("about", "does", "what", "how"):
        assert w in _STOP
    assert tokenize("What does the UCMJ say about disobeying an order?") == \
        ["ucmj", "disobeying", "order"]


def test_legally_meaningful_words_survive():
    for w in ("subject", "right", "order", "charge", "court", "person"):
        assert w not in _STOP
        assert w in tokenize(f"the {w} is important")


def test_q16_regression_art_92_top(index):
    results = index.search(
        "What does the UCMJ say about disobeying an order or regulation?", 3)
    assert any(c["number"] == "92" for c, _ in results), \
        "Art. 92 should be retrieved for the disobedience question"


def test_matched_terms_exclude_stopwords(index):
    q = "What does the UCMJ say about disobeying an order or regulation?"
    top = index.search(q, 1)[0][0]
    idx = next(i for i, c in enumerate(index.chunks) if c["id"] == top["id"])
    terms = [t for t, _ in index.matched_terms(q, idx)]
    assert "about" not in terms and "does" not in terms


def test_deterministic_ranking(index):
    q = "Who may convene a general court-martial?"
    a = [(c["id"], s) for c, s in index.search(q, 5)]
    b = [(c["id"], s) for c, s in index.search(q, 5)]
    assert a == b
