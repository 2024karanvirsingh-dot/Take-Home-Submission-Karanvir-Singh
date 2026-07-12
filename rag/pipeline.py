"""
End to end pipeline: chunk -> index -> retrieve -> generate.

The RagPipeline object builds the index once and answers many questions. The
retriever is swappable ("bm25" or "tfidf") and chunking params are exposed so
the ablation can rebuild with a different chunk size and compare.
"""
from .chunk import build_chunks
from .index import BM25, TfidfCosine
from .generate import generate, build_prompt


class RagPipeline:
    def __init__(self, retriever="bm25", chunk_words=180, overlap_words=40, k=5,
                 stem=False):
        self.k = k
        self.chunks = build_chunks(chunk_words=chunk_words, overlap_words=overlap_words)
        if retriever == "bm25":
            self.index = BM25(self.chunks, stem=stem)
        elif retriever == "tfidf":
            self.index = TfidfCosine(self.chunks, stem=stem)
        else:
            raise ValueError(retriever)
        self.retriever = retriever

    def retrieve(self, question, k=None):
        return self.index.search(question, k=k or self.k)

    def answer(self, question, k=None):
        retrieved = self.retrieve(question, k=k)
        ans = generate(question, retrieved)
        return {
            "question": question,
            "answer": ans,
            "retrieved": [
                {"citation": c["citation"], "type": c["type"], "title": c["title"],
                 "score": round(float(s), 4), "id": c["id"]}
                for c, s in retrieved
            ],
        }


if __name__ == "__main__":
    import sys
    p = RagPipeline()
    q = " ".join(sys.argv[1:]) or "What is the right to be forgotten?"
    out = p.answer(q)
    print("Q:", out["question"])
    print("\nRetrieved:")
    for r in out["retrieved"]:
        print(f"  {r['score']:.3f}  {r['citation']:22s} [{r['type']}] {r['title'][:50]}")
    print("\nAnswer:\n", out["answer"])
