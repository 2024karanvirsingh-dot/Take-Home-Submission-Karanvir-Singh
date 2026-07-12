import importlib.util
import pytest
from rag.index import rrf_fuse
from rag.chunk import build_chunks

needs_dense = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="dense retriever is an optional extra (requirements-dense.txt)")


def test_rrf_fuse_rewards_agreement():
    # an id ranked mid list by both inputs beats an id ranked first by one
    # and absent from the other: 1/62 + 1/62 > 1/61
    order, scores = rrf_fuse([["a", "b", "c"], ["d", "b", "c"]], k=4)
    assert order[0] == "b"
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["a"] == pytest.approx(1 / 61)


def test_rrf_fuse_deterministic_tie_break():
    # 'a' and 'z' get identical fused scores; ties break by id, not by
    # dict insertion order
    order1, _ = rrf_fuse([["a"], ["z"]], k=2)
    order2, _ = rrf_fuse([["z"], ["a"]], k=2)
    assert order1 == order2 == ["a", "z"]


def test_rrf_fuse_respects_k():
    order, _ = rrf_fuse([list("abcdef")], k=3)
    assert order == ["a", "b", "c"]


# everything below needs the dense encoder; the sparse core install skips it


@pytest.fixture(scope="module")
def dense():
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("dense retriever needs the full install")
    from rag.index import DenseEmbed
    return DenseEmbed(build_chunks())


@needs_dense
def test_dense_embeds_every_chunk(dense):
    assert dense.mat.shape == (len(dense.chunks), 384)


@needs_dense
def test_dense_embeddings_are_normalized(dense):
    import numpy as np
    norms = np.linalg.norm(dense.mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


@needs_dense
def test_dense_fixes_the_vocabulary_gap(dense):
    # q07's central finding: 'who is subject to the UCMJ' misses Art. 2
    # under every sparse configuration and dense retrieves it
    results = dense.search("Who is subject to the UCMJ?", 5)
    assert any(c["number"] == "2" for c, _ in results)


@needs_dense
def test_final_system_configuration():
    # the final selected system: hybrid retrieval, grounded extractive
    # answers, article routing active
    from rag.pipeline import RagPipeline
    pipe = RagPipeline(retriever="hybrid")
    assert pipe.answerer == "extractive" and pipe.policy == "grounded"
    out = pipe.answer("What rights does Article 31 give a servicemember "
                      "suspected of an offense?")
    assert not out["refused"]
    assert "31" in out["cited_articles"]


@needs_dense
def test_hybrid_search_deterministic():
    from rag.index import HybridRRF
    idx = HybridRRF(build_chunks())
    q = "What is the difference between desertion and absence without leave?"
    a = [(c["id"], s) for c, s in idx.search(q, 5)]
    b = [(c["id"], s) for c, s in idx.search(q, 5)]
    assert a == b
