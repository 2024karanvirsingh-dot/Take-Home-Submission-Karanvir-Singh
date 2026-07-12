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
  support_ok         extractive runs: every supporting sentence is a verbatim
                     substring of a retrieved chunk, re-verified here against
                     the rebuilt chunk set rather than trusted from the
                     answer builder. Generative runs: every sentence of the
                     generated answer must align with the retrieved chunks
                     under the fuzzy check below.
  numbers_supported  every bare number in the answer (citations stripped)
                     appears in the retrieved chunk text. Trivial for the
                     extractive layer, load bearing for the generative one:
                     a model can emit a plausible figure its context never
                     contained, and this is the check that catches it.

Generative support check (a clearly labelled heuristic, not entailment):
each answer sentence, with its citation parentheses removed, must have at
least 70 percent of its content tokens present in the retrieved chunks and
every number it states present in the retrieved text. This catches novel
wording and fabricated figures; it cannot catch a fluent recombination of
retrieved words into a claim the statute does not make, and the README says
so. Extractive runs keep the strict verbatim gate.

    python3 -m eval.score_answers outputs/run_bm25_baseline.json
"""
import os, json, re, sys
from rag.chunk import build_chunks
from rag.index import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"')
                  .replace("’", "'").replace("‘", "'")).lower()


_CITATION_PAREN = re.compile(r"\((?:art|article)[^)]*\)", re.I)


def _strip_citations(text):
    return _CITATION_PAREN.sub(" ", text)


def _retrieved_text(rec, chunk_text):
    return " ".join(chunk_text.get(r["id"], "") for r in rec.get("retrieved", []))


def _gen_sentence_support(answer_text, retrieved_text):
    """Fuzzy alignment of generated sentences against the retrieved chunks.
    Returns (n_sentences, unsupported_sentences)."""
    ctx_tokens = set(tokenize(retrieved_text))
    ctx_norm = _norm(retrieved_text)
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s+", answer_text)
             if len(s.split()) >= 3]
    unsupported = []
    for s in sents:
        body = _strip_citations(s)
        toks = tokenize(body)
        recall = (sum(1 for t in toks if t in ctx_tokens) / len(toks)) if toks else 1.0
        nums_ok = all(re.search(r"\b%s\b" % n, ctx_norm)
                      for n in re.findall(r"\b\d+\b", body))
        if recall < 0.7 or not nums_ok:
            unsupported.append(s)
    return len(sents), unsupported


def _numbers_supported(answer_text, retrieved_text):
    """Every bare number stated in the answer (citations stripped) occurs in
    the retrieved chunk text."""
    body = _strip_citations(answer_text)
    ctx = _norm(retrieved_text)
    return all(re.search(r"\b%s\b" % n, ctx)
               for n in re.findall(r"\b\d+\b", body))


def score_record(rec, q, chunk_text, mode="extractive"):
    a = rec["answer"]
    answerable = q["answerable_from_corpus"]
    out = {"id": q["id"], "type": q["type"], "answerable": answerable,
           "refused": a["refused"], "boundary": a["boundary"]}

    if answerable:
        out["refusal_correct"] = not a["refused"]
    elif mode == "generative":
        # the generative layer refuses through the model's refusal marker,
        # which carries no boundary taxonomy; any refusal on a control is
        # counted correct, and whether the explanation names the boundary
        # is discussed qualitatively in the README
        out["refusal_correct"] = a["refused"]
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

    # faithfulness. Extractive: independent verification that every answer
    # sentence is a verbatim extract of the chunk it claims to come from.
    # Generative: fuzzy alignment of the generated text with the retrieved
    # chunks, plus the number check.
    retrieved_text = _retrieved_text(rec, chunk_text)
    if mode == "generative" and not a["refused"]:
        n_sents, unsupported = _gen_sentence_support(a["text"], retrieved_text)
        out["n_sentences"] = n_sents
        out["unsupported_sentences"] = unsupported
        out["support_ok"] = not unsupported
    else:
        support = True
        for s in a["sentences"]:
            src = chunk_text.get(s["chunk_id"], "")
            if _norm(s["text"]) not in _norm(src):
                support = False
        out["support_ok"] = support
    if not a["refused"] and rec.get("retrieved"):
        out["numbers_supported"] = _numbers_supported(a["text"], retrieved_text)
    else:
        out["numbers_supported"] = None
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
        "numbers_supported": sum(1 for s in scores if s.get("numbers_supported")),
        "numbers_checked": sum(1 for s in scores
                               if s.get("numbers_supported") is not None),
        "unsupported_answers": sum(1 for s in scores
                                   if s.get("unsupported_sentences")),
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
    mode = cfg.get("answerer", "extractive")
    scores = [score_record(r, questions[r["id"]], chunk_text, mode=mode)
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
    print(f"support check {a['support_pass']}/{a['support_total']}, "
          f"numbers in context {a['numbers_supported']}/{a['numbers_checked']}")


if __name__ == "__main__":
    main()
