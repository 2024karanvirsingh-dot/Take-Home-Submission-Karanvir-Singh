"""
Dataset validation. Run before evaluating; exits non zero on any violation.

Checks that the question file is internally consistent and, crucially, that
it is consistent with the corpus: every must_include phrase used for
automatic content scoring has to actually occur in the text of one of that
question's gold articles. That keeps the answer key itself grounded; a
content check that the corpus cannot satisfy would silently score every
system at zero.

    python3 -m eval.validate_dataset
"""
import os, json, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "data", "corpus")

TYPES = {"numeric", "limitations", "definition", "enumeration", "conditional",
         "factual_lookup", "comparative", "comparative_near_duplicate",
         "statute_boundary", "out_of_corpus"}
REQUIRED = ["id", "type", "question", "answerable_from_corpus",
            "gold_articles", "related_articles", "must_include",
            "gold_answer", "notes"]


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"')
                  .replace("’", "'").replace("‘", "'")).lower()


def article_text(number, manifest):
    for doc in manifest:
        if doc["number"] == number:
            with open(os.path.join(CORPUS, doc["file"])) as f:
                return f.read()
    return None


def validate(questions, manifest):
    errors = []
    ids = [q.get("id") for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")
    known = {d["number"] for d in manifest}
    for q in questions:
        qid = q.get("id", "?")
        for field in REQUIRED:
            if field not in q:
                errors.append(f"{qid}: missing field {field}")
        if q.get("type") not in TYPES:
            errors.append(f"{qid}: unknown type {q.get('type')}")
        gold = q.get("gold_articles", [])
        if q.get("answerable_from_corpus"):
            if not gold:
                errors.append(f"{qid}: answerable but no gold_articles")
        else:
            if gold:
                errors.append(f"{qid}: control question must have empty gold_articles")
            if q.get("must_include"):
                errors.append(f"{qid}: control question must have empty must_include")
        for art in gold + q.get("related_articles", []):
            if art not in known:
                errors.append(f"{qid}: article {art} not in corpus manifest")
        # every content requirement must exist in some gold article's text
        for phrase in q.get("must_include", []):
            found = any(_norm(phrase) in _norm(article_text(a, manifest) or "")
                        for a in gold)
            if not found:
                errors.append(f"{qid}: must_include phrase not found in any "
                              f"gold article text: {phrase!r}")
    return errors


def main():
    with open(os.path.join(HERE, "questions.json")) as f:
        questions = json.load(f)
    with open(os.path.join(CORPUS, "manifest.json")) as f:
        manifest = json.load(f)
    errors = validate(questions, manifest)
    n_ans = sum(1 for q in questions if q["answerable_from_corpus"])
    if errors:
        for e in errors:
            print("FAIL:", e)
        sys.exit(1)
    print(f"dataset ok: {len(questions)} questions "
          f"({n_ans} answerable, {len(questions) - n_ans} controls), "
          f"all must_include phrases verified against gold article text")


if __name__ == "__main__":
    main()
