# Scoring rubric

Retrieval and answer quality are scored separately because they fail for
different reasons: a system can get the right answer from the wrong context
(lucky) or the wrong answer from the right context (selection fault).
Keeping them apart is the whole point of the exercise.

Everything in the first two sections is computed automatically by
`eval/run_eval.py` and `eval/score_answers.py` from the structured fields in
`eval/questions.json`. Nothing is parsed out of prose and nothing is scored
by hand. The qualitative reading of the numbers lives in the README and is
labelled as interpretation there.

## 1. Retrieval metrics (automatic)

Computed over the 16 questions with `answerable_from_corpus: true`. The two
controls have no gold articles by definition and are excluded here; their
scoring happens entirely on the answer side.

- **hit@1 / hit@3 / hit@5**: rank of the first gold article chunk.
- **MRR**: mean reciprocal rank of the first gold article.
- **all gold retrieved**: for multi article questions (q03 desertion versus
  AWOL, q14 conduct unbecoming versus the general article), full credit
  requires every gold article in the top k. Retrieving one side of a
  comparison is scored as partial (the per question tables mark it).

## 2. Answer metrics (automatic)

- **refusal correctness**: answerable questions must be answered; the two
  controls must be refused with a boundary explanation. A refusal on an
  answerable question is a false refusal and is counted, not hidden.
- **citation hit**: at least one cited article is a gold article.
- **citation precision**: fraction of cited articles that are gold or
  explicitly related. Tagalong citations cost precision.
- **content coverage**: fraction of the question's `must_include` phrases
  present in the answer. The phrases are statutory wording (validated
  against the gold article text by `eval/validate_dataset.py`), so full
  coverage is always achievable from the corpus. For numeric questions the
  phrase contains the exact figure ("shall be 12", "five years"), so a
  right rule with a missing number scores it as absent: numbers are the
  point.
- **support**: every sentence in every answer must be a verbatim extract of
  a retrieved chunk, re verified independently by `eval/score_answers.py`
  against the rebuilt chunk set. This is a gate, not a gradient. The
  extractive builder passes it by construction; the check exists so that
  any future change to the builder cannot silently start paraphrasing.

## 3. What "good" means per question type (interpretation)

- **factual_lookup / definition**: the specific rule or definition with the
  correct article cited. For definitions, quoting a usage of the term
  instead of its definition is wrong even though the words match.
- **numeric / limitations**: the figure must be exact and present.
- **enumeration / conditional**: measured on recall of the list. Answering
  "who is subject to the UCMJ" with three categories out of thirteen is
  partial coverage, and the `must_include` fields sample representative
  list elements to measure exactly that.
- **comparative / comparative_near_duplicate**: both provisions must be
  retrieved and both sides must appear in the answer. One sided answers are
  visible as a failed "all gold retrieved" plus reduced content coverage.
- **statute_boundary / out_of_corpus**: the only good answer names the
  boundary. The UCMJ frequently provides punishment "as a court-martial may
  direct"; the actual maximum confinement tables live in the Manual for
  Courts-Martial, which is deliberately not in this corpus. A confident
  number for q17, or a fabricated speedy trial rule for q18, is the worst
  outcome the system can produce, which is why these two questions exist.
