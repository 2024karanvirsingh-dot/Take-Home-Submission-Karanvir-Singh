"""
End to end pipeline: chunk -> index -> retrieve -> route -> construct answer.

The RagPipeline object builds the index once and answers many questions. The
retriever is swappable ("bm25", "tfidf", "dense", "hybrid") and chunking
params are exposed so the ablations can rebuild with different settings. The
answer layer is swappable too: "extractive" is the local deterministic
builder in rag/answer.py (no model call anywhere in that path, the
default), and "generative" is the optional local flan-t5 layer in
rag/generate.py (a model run on this machine; still no API call and no
credential). The default reproducible system is bm25 + extractive; hybrid
is the best measured retrieval configuration and needs the optional dense
extra. See the README.

Article number routing: practitioners ask about provisions by number
("Article 31"), but a provision's own text never states its own number, so
no retriever, sparse or dense, can find it from the query terms. When the
question names an article that the retriever did not surface, the router
swaps that article's best chunk into the last slot. This runs identically
for every retriever, so the ablations remain a fair comparison.
"""
import re
from .chunk import build_chunks
from .index import BM25, TfidfCosine, DenseEmbed, HybridRRF
from .answer import construct_answer

_ARTICLE_REF = re.compile(r"\barticle\s+(\d+[a-z]?)\b", re.I)


class RagPipeline:
    def __init__(self, retriever="bm25", chunk_words=180, overlap_words=40, k=5,
                 stem=False, policy="grounded", answerer="extractive",
                 gen_model=None):
        self.k = k
        self.policy = policy
        if answerer not in ("extractive", "generative"):
            raise ValueError(answerer)
        self.answerer = answerer
        self.gen = None
        if answerer == "generative":
            from .generate import GenerativeAnswerer, GEN_MODEL_DEFAULT
            self.gen = GenerativeAnswerer(gen_model or GEN_MODEL_DEFAULT)
        self.chunks = build_chunks(chunk_words=chunk_words, overlap_words=overlap_words)
        if retriever == "bm25":
            self.index = BM25(self.chunks, stem=stem)
        elif retriever == "tfidf":
            self.index = TfidfCosine(self.chunks, stem=stem)
        elif retriever == "dense":
            self.index = DenseEmbed(self.chunks, stem=stem)
        elif retriever == "hybrid":
            self.index = HybridRRF(self.chunks, stem=stem)
        else:
            raise ValueError(retriever)
        self.retriever = retriever

    def retrieve(self, question, k=None):
        k = k or self.k
        results = self.index.search(question, k=k)
        return self._route_named_articles(question, results, k)

    def _route_named_articles(self, question, results, k):
        named = [n for n in _ARTICLE_REF.findall(question.lower())]
        if not named:
            return results
        have = {c["number"] for c, _ in results}
        missing = [n for n in named if n not in have]
        by_number = {}
        for c in self.chunks:
            by_number.setdefault(c["number"], []).append(c)
        for n in missing:
            if n not in by_number:
                continue  # the question may name an MCM rule, not a provision
            # best chunk of the named article under the same index, falling
            # back to its first chunk when nothing in it matches the query
            full = {c["id"]: s for c, s in
                    self.index.search(question, k=len(self.chunks))}
            best = max(by_number[n],
                       key=lambda c: (full.get(c["id"], 0.0),
                                      -int(c["id"].rsplit("::", 1)[1])))
            results = results[:-1] + [(best, full.get(best["id"], 0.0))]
        return results

    def answer(self, question, k=None, qtype=None):
        retrieved = self.retrieve(question, k=k)
        if self.answerer == "generative":
            ans = self.gen.construct(question, retrieved, qtype=qtype)
        else:
            ans = construct_answer(question, retrieved, policy=self.policy,
                                   qtype=qtype)
        return {
            "question": question,
            "answer": ans["text"],
            "refused": ans["refused"],
            "boundary": ans["boundary"],
            "cited_articles": ans["cited_articles"],
            "sentences": ans["sentences"],
            "retrieved": [
                {"citation": c["citation"], "article": c["number"],
                 "type": c["type"], "title": c["title"],
                 "score": round(float(s), 4), "id": c["id"]}
                for c, s in retrieved
            ],
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="ask one question against the corpus")
    ap.add_argument("question", nargs="*",
                    default=["What rights does Article 31 provide?"])
    ap.add_argument("--retriever", default="bm25",
                    choices=["bm25", "tfidf", "dense", "hybrid"],
                    help="bm25 is the fast diagnostic default; hybrid is "
                         "the final system (needs the full install)")
    ap.add_argument("--answerer", default="extractive",
                    choices=["extractive", "generative"])
    args = ap.parse_args()
    p = RagPipeline(retriever=args.retriever, answerer=args.answerer)
    out = p.answer(" ".join(args.question))
    print("Q:", out["question"])
    print("\nRetrieved:")
    for r in out["retrieved"]:
        print(f"  {r['score']:.3f}  {r['citation']:34s} {r['title'][:44]}")
    print("\nAnswer:\n" + out["answer"])
