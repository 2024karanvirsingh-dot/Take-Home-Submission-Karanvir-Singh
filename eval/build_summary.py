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
             "bm25_stem", "dense_baseline", "hybrid_baseline"]


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


def answer_summary():
    path = os.path.join(OUT, "answer_scores_bm25_baseline.json")
    if not os.path.exists(path):
        return None, ""
    with open(path) as f:
        s = json.load(f)
    a = s["aggregate"]
    lines = ["| metric | value |", "| --- | --- |",
             f"| answered (of {a['n_answerable']} answerable) | {a['answered']} |",
             f"| refusals on answerable questions | {a['false_refusals']} |",
             f"| controls refused correctly | {a['control_refusal_correct']}/{a['n_controls']} |",
             f"| answers citing a gold article | {a['citation_hit']}/{a['answered']} |",
             f"| mean citation precision | {a['mean_citation_precision']} |",
             f"| mean content coverage | {a['mean_content_coverage']} |",
             f"| verbatim support check | {a['support_pass']}/{a['support_total']} |"]
    return a, "\n".join(lines)


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
    ans_agg, ans_tbl = answer_summary()
    pol_agg, pol_tbl = policy_summary()
    summary = {
        "retrieval": {t: runs[t]["aggregate"] for t in RUN_ORDER if t in runs},
        "answers_bm25_baseline_grounded": ans_agg,
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
        print("\n## Answers (bm25 baseline, grounded)\n")
        print(ans_tbl)
    if pol_tbl:
        print("\n## Answer policy ablation\n")
        print(pol_tbl)


if __name__ == "__main__":
    main()
