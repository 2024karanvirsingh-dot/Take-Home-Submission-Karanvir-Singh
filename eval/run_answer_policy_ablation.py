"""
Answer policy ablation: grounded versus permissive, retrieval held fixed.

One thing changes between the two arms: the answer policy in rag/answer.py.
Both are the same deterministic extractive code path; the permissive arm
skips the support test, the punitive article guard for punishment amount
questions, and both corpus boundary checks. Nothing here is staged: whatever
the permissive policy produces on weak retrieval is what gets recorded.

Writes outputs/answer_policy_ablation.json with both answers per question
and the automatic scores for each arm.

    python3 -m eval.run_answer_policy_ablation
"""
import os, json, platform
from rag.pipeline import RagPipeline
from rag.answer import construct_answer, ANSWER_BUILDER_VERSION
from eval.score_answers import score_record, aggregate
from rag.chunk import build_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    with open(os.path.join(HERE, "questions.json")) as f:
        questions = json.load(f)

    pipe = RagPipeline()  # bm25 baseline retrieval for both arms
    chunk_text = {c["id"]: c["text"] for c in build_chunks()}

    records, scores = [], {"grounded": [], "permissive": []}
    for q in questions:
        retrieved = pipe.retrieve(q["question"])
        arms = {}
        for policy in ("grounded", "permissive"):
            ans = construct_answer(q["question"], retrieved, policy=policy,
                                   qtype=q["type"])
            arms[policy] = {
                "text": ans["text"], "refused": ans["refused"],
                "boundary": ans["boundary"],
                "cited_articles": ans["cited_articles"],
                "sentences": ans["sentences"],
            }
            rec = {"id": q["id"], "answer": arms[policy]}
            scores[policy].append(score_record(rec, q, chunk_text))
        records.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "answerable_from_corpus": q["answerable_from_corpus"],
            "grounded": arms["grounded"], "permissive": arms["permissive"],
        })

    out = {
        "config": {
            "retriever": "bm25", "chunk_words": 180, "overlap_words": 40,
            "k": 5, "answer_builder_version": ANSWER_BUILDER_VERSION,
            "python_version": platform.python_version(),
            "note": "retrieval identical in both arms; only the answer "
                    "policy differs",
        },
        "aggregate": {p: aggregate(s) for p, s in scores.items()},
        "per_question_scores": {p: s for p, s in scores.items()},
        "records": records,
    }
    path = os.path.join(OUT, "answer_policy_ablation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {path}")
    for p in ("grounded", "permissive"):
        a = out["aggregate"][p]
        print(f"{p:10s}: {a['false_refusals']} refusals on answerable, "
              f"controls refused {a['control_refusal_correct']}/{a['n_controls']}, "
              f"citation hit {a['citation_hit']}/{a['answered']}, "
              f"citation precision {a['mean_citation_precision']}")


if __name__ == "__main__":
    main()
