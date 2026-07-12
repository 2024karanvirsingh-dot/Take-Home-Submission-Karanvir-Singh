"""
Build the aggregate summary from every run file in outputs/. Writes
outputs/summary.json and prints the markdown tables used in the README, so
no number in the README is hand copied.

    python3 -m eval.build_summary
"""
import os, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

RUN_ORDER = ["bm25_baseline", "bm25_cw120", "bm25_cw300", "tfidf_baseline",
             "bm25_stem", "dense_baseline", "hybrid_baseline",
             "hybrid_generative"]
# answer scored configurations, in the order the README compares them.
# bm25_baseline + grounded extraction is the default reproducible system;
# hybrid_baseline (hybrid retrieval + grounded extraction) is the best
# measured configuration, produced with the optional dense extra.
ANSWER_ORDER = ["bm25_baseline", "dense_baseline", "hybrid_baseline",
                "hybrid_generative"]
BEST_MEASURED = "hybrid_baseline"


def load_runs():
    runs = {}
    for path in glob.glob(os.path.join(OUT, "run_*.json")):
        tag = os.path.basename(path)[len("run_"):-len(".json")]
        with open(path) as f:
            runs[tag] = json.load(f)
    return runs


def retrieval_table(runs):
    lines = ["| run | hit@1 | hit@3 | hit@5 | MRR | missed | all gold in top k |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for tag in RUN_ORDER:
        if tag not in runs:
            continue
        a = runs[tag]["aggregate"]
        n = a["n_retrieval_scored"]
        lines.append(
            f"| {tag} | {a['hit_at_1']}/{n} | {a['hit_at_3']}/{n} | "
            f"{a['hit_at_5']}/{n} | {a['mrr']:.2f} | {a['missed']}/{n} | "
            f"{a['all_gold_retrieved']}/{n} |")
    return "\n".join(lines)


def per_question_table(runs):
    tags = [t for t in RUN_ORDER if t in runs]
    lines = ["| question | " + " | ".join(tags) + " |",
             "| --- |" + " --- |" * len(tags)]
    base = runs[tags[0]]["records"]
    for i, rec in enumerate(base):
        if not rec["retrieval"]["scored"]:
            continue
        row = [rec["id"]]
        for t in tags:
            r = runs[t]["records"][i]["retrieval"]
            f = r["first_gold_rank"]
            cell = "MISS" if f == 0 else str(f)
            if len(r["gold_ranks"]) > 1:
                cell += "*" if r["all_gold_retrieved"] else "!"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("multi gold questions: * = all gold articles retrieved, "
                 "! = only one side retrieved")
    return "\n".join(lines)


def load_answer_scores():
    scores = {}
    for tag in ANSWER_ORDER:
        path = os.path.join(OUT, f"answer_scores_{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                scores[tag] = json.load(f)
    return scores


def answers_table(scores):
    """One column per answer scored configuration."""
    if not scores:
        return None, ""
    tags = [t for t in ANSWER_ORDER if t in scores]
    header = ["metric"] + [t + (" (best measured)" if t == BEST_MEASURED
                                else "")
                           for t in tags]
    lines = ["| " + " | ".join(header) + " |",
             "| --- |" + " --- |" * len(tags)]

    def row(name, fmt):
        cells = [fmt(scores[t]["aggregate"]) for t in tags]
        lines.append("| " + name + " | " + " | ".join(str(c) for c in cells) + " |")

    row("answered (of 16 answerable)", lambda a: a["answered"])
    row("refusals on answerable questions", lambda a: a["false_refusals"])
    row("controls refused correctly",
        lambda a: f"{a['control_refusal_correct']}/{a['n_controls']}")
    row("answers citing a gold article",
        lambda a: f"{a['citation_hit']}/{a['answered']}")
    row("mean citation precision", lambda a: a["mean_citation_precision"])
    row("mean content coverage", lambda a: a["mean_content_coverage"])
    row("support check passed", lambda a: f"{a['support_pass']}/{a['support_total']}")
    row("answers with every stated number in context",
        lambda a: f"{a.get('numbers_supported', '?')}/{a.get('numbers_checked', '?')}")
    return {t: scores[t]["aggregate"] for t in tags}, "\n".join(lines)


def policy_summary():
    path = os.path.join(OUT, "answer_policy_ablation.json")
    if not os.path.exists(path):
        return None, ""
    with open(path) as f:
        p = json.load(f)
    lines = ["| metric | grounded | permissive |", "| --- | --- | --- |"]
    g, m = p["aggregate"]["grounded"], p["aggregate"]["permissive"]
    rows = [
        ("refusals on answerable", g["false_refusals"], m["false_refusals"]),
        ("controls refused correctly",
         f"{g['control_refusal_correct']}/{g['n_controls']}",
         f"{m['control_refusal_correct']}/{m['n_controls']}"),
        ("answers citing a gold article",
         f"{g['citation_hit']}/{g['answered']}",
         f"{m['citation_hit']}/{m['answered']}"),
        ("mean citation precision",
         g["mean_citation_precision"], m["mean_citation_precision"]),
    ]
    for name, gv, mv in rows:
        lines.append(f"| {name} | {gv} | {mv} |")
    return p["aggregate"], "\n".join(lines)


def main():
    runs = load_runs()
    ans_agg, ans_tbl = answers_table(load_answer_scores())
    pol_agg, pol_tbl = policy_summary()
    summary = {
        "default_system": {
            "run": "bm25_baseline",
            "retriever": "bm25 with article number routing",
            "answerer": "extractive (grounded policy)",
            "note": "the clean clone reproducible baseline",
        },
        "best_measured_configuration": {
            "run": BEST_MEASURED,
            "retriever": "hybrid (bm25 + MiniLM dense, reciprocal rank "
                         "fusion, article number routing)",
            "answerer": "extractive (grounded policy)",
            "note": "needs the optional dense extra; outputs committed",
        },
        "retrieval": {t: runs[t]["aggregate"] for t in RUN_ORDER if t in runs},
        "answers": ans_agg,
        "answer_policy_ablation": pol_agg,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(OUT, "summary.json"))
    print("\n## Retrieval\n")
    print(retrieval_table(runs))
    print("\n## Per question first gold rank\n")
    print(per_question_table(runs))
    if ans_tbl:
        print("\n## Answer metrics by configuration\n")
        print(ans_tbl)
    if pol_tbl:
        print("\n## Answer policy ablation\n")
        print(pol_tbl)


if __name__ == "__main__":
    main()
