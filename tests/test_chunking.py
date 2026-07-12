import json, os
import pytest
from rag.chunk import build_chunks, load_manifest

CORPUS = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")


@pytest.fixture(scope="module")
def chunks():
    return build_chunks()


def test_corpus_loads_all_provisions():
    manifest = load_manifest()
    assert len(manifest) == 198
    for doc in manifest:
        path = os.path.join(CORPUS, doc["file"])
        assert os.path.exists(path), doc["file"]
        assert doc["type"] in ("punitive", "procedural")


def test_chunks_never_cross_provision_boundaries(chunks):
    # every chunk's text must come from exactly the file its id names
    texts = {}
    for c in chunks:
        fname = c["id"].split("::")[0]
        if fname not in texts:
            with open(os.path.join(CORPUS, fname)) as f:
                texts[fname] = " ".join(f.read().split())
        for para in c["text"].split("\n"):
            assert " ".join(para.split()) in texts[fname], \
                f"chunk {c['id']} contains text not in {fname}"


def test_every_chunk_maps_to_one_article(chunks):
    manifest = {d["file"]: d for d in load_manifest()}
    for c in chunks:
        doc = manifest[c["id"].split("::")[0]]
        assert c["number"] == doc["number"]
        assert c["citation"].startswith(doc["citation"].split(" (part")[0][:12])


def test_overlap_repeats_tail_words():
    multi = {}
    for c in build_chunks(chunk_words=180, overlap_words=40):
        multi.setdefault(c["id"].split("::")[0], []).append(c)
    split_docs = {k: v for k, v in multi.items() if len(v) > 1}
    assert split_docs, "expected at least one multi chunk provision"
    for parts in split_docs.values():
        parts.sort(key=lambda c: int(c["id"].split("::")[1]))
        for a, b in zip(parts, parts[1:]):
            shared = set(a["text"].split()) & set(b["text"].split())
            assert len(shared) >= 5, "consecutive chunks should overlap"


def test_zero_overlap_supported():
    cs = build_chunks(chunk_words=180, overlap_words=0)
    assert len(cs) > 198  # long articles still split
