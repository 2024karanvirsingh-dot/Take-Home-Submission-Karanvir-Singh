"""
Automatic answer scoring. Reads a run file produced by eval/run_eval.py and
scores every answer against the structured fields in eval/questions.json.
No judgment call in this file is manual; qualitative interpretation lives in
the README and is clearly labelled as such there.

Per question metrics:

  refusal_correct    answerable questions must be answered, control
                     questions must be refused (with a boundary explanation)
  citation_hit       at least one cited article is a gold article
  citation_precision fraction of cited articles that are gold or related
  content_coverage   fraction of the question's must_include phrases present
                     in the answer text (phrases are validated against the
                     gold article text by eval/validate_dataset.py, so a
                     covering answer always exists in the corpus)
  support_ok         every supporting sentence is a verbatim substring of a
                     retrieved chunk, re-verified here against the rebuilt
                     chunk set rather than trusted from the answer builder

    python3 -m eval.score_answers outputs/run_bm25_baseline.json
"""
import os, json, re, sys
from rag.chunk import build_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"')
                  .replace("’", "'").replace("‘", "'")).lower()


def score_record(rec, q, chunk_text):
    a = rec["answer"]
    answerable = q["answerable_from_corpus"]
    out = {"id": q["id"], "type": q["type"], "answerable": answerable,
           "refused": a["refused"], "boundary": a["boundary"]}

    if answerable:
        out["refusal_correct"] = not a["refused"]
    else:
        out["refusal_correct"] = a["refused"] and a["boundary"]

    if answerable and not a["refused"]:
        cited = a["cited_articles"]
        gold = set(q["gold_articles"])
        ok_set = gold | set(q.get("related_articles", []))
        out["citation_hit"] = bool(gold & set(cited))
        out["citation_precision"] = (round(sum(1 for c in cited if c in ok_set)
                                           / len(cited), 3) if cited else 0.0)
        text = _norm(a["text"])
        must = q["must_include"]
        out["content_coverage"] = (round(sum(1 for m in must
                                             if _norm(m) in text) / len(must), 3)
                                   if must else None)
    else:
        out["citation_hit"] = None
        out["citation_precision"] = None
        out["content_coverage"] = None

    # faithfulness: independent verification that every answer sentence is
    # a verbatim extract of the chunk it claims to come from
    support = True
    for s in a["sentences"]:
        src = chunk_text.get(s["chunk_id"], "")
        if _norm(s["text"]) not in _norm(src):
            support = False
    out["support_ok"] = support
    return out


def aggregate(scores):
    answerable = [s for s in scores if s["answerable"]]
    controls = [s for s in scores if not s["answerable"]]
    answered = [s for s in answerable if not s["refused"]]
    cov = [s["content_coverage"] for s in answered
           if s["content_coverage"] is not None]
    return {
        "n_answerable": len(answerable),
        "n_controls": len(controls),
        "answered": len(answered),
        "false_refusals": sum(1 for s in answerable if s["refused"]),
        "control_refusal_correct": sum(1 for s in controls if s["refusal_correct"]),
        "citation_hit": sum(1 for s in answered if s["citation_hit"]),
        "mean_citation_precision": round(
            sum(s["citation_precision"] for s in answered) / len(answered), 3)
            if answered else None,
        "mean_content_coverage": round(sum(cov) / len(cov), 3) if cov else None,
        "support_pass": sum(1 for s in scores if s["support_ok"]),
        "support_total": len(scores),
    }


def score_run(run_path):
    with open(run_path) as f:
        run = json.load(f)
    with open(os.path.join(HERE, "questions.json")) as f:
        questions = {q["id"]: q for q in json.load(f)}
    cfg = run["config"]
    chunks = build_chunks(chunk_words=cfg["chunk_words"],
                          overlap_words=cfg["overlap_words"])
    chunk_text = {c["id"]: c["text"] for c in chunks}
    scores = [score_record(r, questions[r["id"]], chunk_text)
              for r in run["records"]]
    return {"run": os.path.basename(run_path), "config": cfg,
            "aggregate": aggregate(scores), "per_question": scores}


def main():
    run_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(OUT, "run_bm25_baseline.json")
    result = score_run(run_path)
    tag = os.path.basename(run_path).replace("run_", "").replace(".json", "")
    out_path = os.path.join(OUT, f"answer_scores_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    a = result["aggregate"]
    print(f"wrote {out_path}")
    print(f"answered {a['answered']}/{a['n_answerable']} answerable "
          f"({a['false_refusals']} refusals), "
          f"controls refused correctly {a['control_refusal_correct']}/{a['n_controls']}")
    print(f"citation hit {a['citation_hit']}/{a['answered']}, "
          f"mean citation precision {a['mean_citation_precision']}, "
          f"mean content coverage {a['mean_content_coverage']}")
    print(f"support (verbatim extraction) {a['support_pass']}/{a['support_total']}")


if __name__ == "__main__":
    main()
