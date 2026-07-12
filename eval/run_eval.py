"""
Run the pipeline over eval/questions.json and write a run file to outputs/.

Each record has the question, the ordered retrieved passages, the rank at
which each gold article first appears (read from the structured
gold_articles field, never parsed out of prose), the BM25 matched terms that
explain the top hit, and the locally constructed answer with its refusal and
citation metadata. Retrieval metrics are computed here; answer metrics are
computed by eval/score_answers.py from the run file.

Retrieval accounting: the 16 questions with answerable_from_corpus true are
the retrieval scored set. The two controls (q17 statute boundary, q18 out of
corpus) have no gold articles by definition and are excluded from retrieval
metrics; their scoring lives entirely on the answer side (a refusal is the
correct behavior).

Usage:
    python3 -m eval.run_eval                      # bm25, grounded, default chunking
    python3 -m eval.run_eval --retriever tfidf    # tf-idf vector embedding baseline
    python3 -m eval.run_eval --retriever hybrid   # optional: best measured retrieval
    python3 -m eval.run_eval --retriever dense    # optional: dense ablation
    python3 -m eval.run_eval --retriever hybrid --answerer generative
                                                  # optional: generative experiment
    python3 -m eval.run_eval --policy permissive  # swap answer policy
    python3 -m eval.run_eval --chunk-words 300    # chunking ablation

Dense/hybrid need requirements-dense.txt; generative needs
requirements-gen.txt. Both are optional extras with committed outputs.
"""
import os, json, argparse, platform
from rag.pipeline import RagPipeline
from rag.answer import ANSWER_BUILDER_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def gold_ranks(gold_articles, retrieved):
    """1 indexed rank of the first chunk of each gold article, 0 if absent."""
    ranks = {}
    for art in gold_articles:
        ranks[art] = 0
        for i, r in enumerate(retrieved):
            if r["article"] == art:
                ranks[art] = i + 1
                break
    return ranks


def retrieval_record(q, retrieved):
    if not q["answerable_from_corpus"]:
        return {"scored": False}
    ranks = gold_ranks(q["gold_articles"], retrieved)
    first = min((r for r in ranks.values() if r > 0), default=0)
    return {
        "scored": True,
        "gold_ranks": ranks,
        "first_gold_rank": first,
        "all_gold_retrieved": all(r > 0 for r in ranks.values()),
    }


def aggregate(records, k):
    scored = [r for r in records if r["retrieval"]["scored"]]
    firsts = [r["retrieval"]["first_gold_rank"] for r in scored]
    n = len(scored)
    agg = {
        "n_retrieval_scored": n,
        "hit_at_1": sum(1 for f in firsts if f == 1),
        "hit_at_3": sum(1 for f in firsts if 1 <= f <= 3),
        "hit_at_5": sum(1 for f in firsts if 1 <= f <= k),
        "missed": sum(1 for f in firsts if f == 0),
        "mrr": round(sum(1.0 / f for f in firsts if f > 0) / n, 4) if n else 0,
        "all_gold_retrieved": sum(1 for r in scored
                                  if r["retrieval"]["all_gold_retrieved"]),
        "multi_gold_questions": sum(1 for r in scored
                                    if len(r["retrieval"]["gold_ranks"]) > 1),
        "multi_gold_fully_covered": sum(
            1 for r in scored if len(r["retrieval"]["gold_ranks"]) > 1
            and r["retrieval"]["all_gold_retrieved"]),
        "answerable_refused": sum(1 for r in records
                                  if r["retrieval"]["scored"] and r["answer"]["refused"]),
        "controls_refused": sum(1 for r in records
                                if not r["retrieval"]["scored"] and r["answer"]["refused"]),
        "n_controls": sum(1 for r in records if not r["retrieval"]["scored"]),
    }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="bm25",
                    choices=["bm25", "tfidf", "dense", "hybrid"])
    ap.add_argument("--chunk-words", type=int, default=180)
    ap.add_argument("--overlap-words", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stem", action="store_true",
                    help="light suffix stripping on both query and documents")
    ap.add_argument("--policy", default="grounded",
                    choices=["grounded", "permissive"])
    ap.add_argument("--answerer", default="extractive",
                    choices=["extractive", "generative"])
    ap.add_argument("--gen-model", default=None,
                    help="hub id for the generative answerer "
                         "(default google/flan-t5-small)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "questions.json")) as f:
        questions = json.load(f)

    pipe = RagPipeline(retriever=args.retriever, chunk_words=args.chunk_words,
                       overlap_words=args.overlap_words, k=args.k,
                       stem=args.stem, policy=args.policy,
                       answerer=args.answerer, gen_model=args.gen_model)

    records = []
    for q in questions:
        out = pipe.answer(q["question"], qtype=q["type"])
        rec = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "gold_articles": q["gold_articles"],
            "answerable_from_corpus": q["answerable_from_corpus"],
            "retrieved": out["retrieved"],
            "retrieval": retrieval_record(q, out["retrieved"]),
            "answer": {
                "text": out["answer"], "refused": out["refused"],
                "boundary": out["boundary"],
                "cited_articles": out["cited_articles"],
                "sentences": out["sentences"],
            },
        }
        if args.retriever == "bm25" and out["retrieved"]:
            idx = next(i for i, c in enumerate(pipe.chunks)
                       if c["id"] == out["retrieved"][0]["id"])
            terms = pipe.index.matched_terms(q["question"], idx)
            rec["top_matched_terms"] = [t for t, _ in terms[:6]]
        records.append(rec)

    agg = aggregate(records, args.k)
    config = dict(vars(args))
    config.update(answer_builder_version=ANSWER_BUILDER_VERSION,
                  python_version=platform.python_version())

    tag = args.tag or (f"{args.retriever}_cw{args.chunk_words}"
                       + ("_gen" if args.answerer == "generative" else ""))
    path = os.path.join(OUT, f"run_{tag}.json")
    with open(path, "w") as f:
        json.dump({"config": config, "aggregate": agg, "records": records},
                  f, indent=2)

    print(f"wrote {path}")
    print(f"retrieval ({agg['n_retrieval_scored']} scored): "
          f"hit@1 {agg['hit_at_1']}  hit@3 {agg['hit_at_3']}  "
          f"hit@5 {agg['hit_at_5']}  missed {agg['missed']}  MRR {agg['mrr']}")
    print(f"all gold retrieved: {agg['all_gold_retrieved']}/{agg['n_retrieval_scored']}  "
          f"(multi gold fully covered {agg['multi_gold_fully_covered']}/"
          f"{agg['multi_gold_questions']})")
    print(f"answers: {agg['answerable_refused']} refusals on answerable, "
          f"{agg['controls_refused']}/{agg['n_controls']} controls refused")


if __name__ == "__main__":
    main()
