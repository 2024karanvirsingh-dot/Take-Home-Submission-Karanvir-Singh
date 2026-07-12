"""
Run the pipeline over eval/questions.json and write a run file to outputs/.

Each record has the question, the ordered retrieved passages with scores, the
BM25 matched terms that explain the top hit, the generated answer, and a
retrieval_hit flag computed against the gold citation. Scoring against the
rubric is done by hand in the README, this script produces the evidence.

Usage:
    python -m eval.run_eval                      # bm25, default chunking
    python -m eval.run_eval --retriever tfidf    # swap retriever
    python -m eval.run_eval --retriever dense    # needs sentence-transformers
    python -m eval.run_eval --chunk-words 300    # ablation
"""
import os, json, argparse, re
from rag.pipeline import RagPipeline
from rag.index import BM25

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def gold_numbers(gc):
    """Pull UCMJ article numbers (86, 146a, ...) out of a gold citation string."""
    return set(re.findall(r"Art\.?\s*(\d+[a-z]?)", gc))


def retrieval_hit(gold_citation, retrieved):
    arts = gold_numbers(gold_citation)
    if not arts:
        return None  # control questions, no single gold provision
    for rank, r in enumerate(retrieved):
        m = re.search(r"Art\.?\s*(\d+[a-z]?)", r["citation"])
        if m and m.group(1) in arts:
            return rank + 1  # 1-indexed rank of first correct provision
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--chunk-words", type=int, default=180)
    ap.add_argument("--overlap-words", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stem", action="store_true",
                    help="light suffix stripping on both query and documents")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "questions.json")) as f:
        questions = json.load(f)

    pipe = RagPipeline(retriever=args.retriever, chunk_words=args.chunk_words,
                       overlap_words=args.overlap_words, k=args.k, stem=args.stem)

    records = []
    for q in questions:
        out = pipe.answer(q["question"])
        rank = retrieval_hit(q["gold_citation"], out["retrieved"])
        rec = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "gold_citation": q["gold_citation"], "gold_answer": q["gold_answer"],
            "retrieved": out["retrieved"], "answer": out["answer"],
            "gold_rank": rank,
        }
        if args.retriever == "bm25" and out["retrieved"]:
            idx = next(i for i, c in enumerate(pipe.chunks)
                       if c["id"] == out["retrieved"][0]["id"])
            terms = pipe.index.matched_terms(q["question"], idx)
            rec["top_matched_terms"] = [t for t, _ in terms[:6]]
        records.append(rec)

    tag = args.tag or f"{args.retriever}_cw{args.chunk_words}"
    path = os.path.join(OUT, f"run_{tag}.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args), "records": records}, f, indent=2)

    hits = [r["gold_rank"] for r in records if r["gold_rank"] is not None]
    top1 = sum(1 for h in hits if h == 1)
    top3 = sum(1 for h in hits if 1 <= h <= 3)
    miss = sum(1 for h in hits if h == 0)
    print(f"wrote {path}")
    print(f"gold in top1: {top1}/{len(hits)}   top3: {top3}/{len(hits)}   missed: {miss}/{len(hits)}")


if __name__ == "__main__":
    main()
