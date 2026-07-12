# Scoring rubric

Every answer is scored on three axes. I score retrieval and answer separately
because they fail for different reasons, and a RAG system can get the right
answer from wrong context (lucky) or the wrong answer from right context
(generation fault). Keeping them apart is the whole point of the exercise.

## 1. Retrieval quality (0 to 2)

Did the correct provision show up in the top k passages?

- **2**  The gold provision is retrieved and sits in the top 2 results.
- **1**  The gold provision is somewhere in the top k but ranked low, or a
         closely related provision is retrieved instead (right topic, wrong
         section), or only part of a multi part answer is covered.
- **0**  The gold provision is not in the top k at all, or the top results are
         off topic.

For the two control questions (q17 negative, q18 out of corpus) retrieval scores
differently: full marks means the top passages are genuinely the closest thing
the corpus has, so a well behaved generator can still recognise the gap.

## 2. Answer correctness (0 to 2)

Judged against the gold answer, using only what a reader could verify from the
retrieved context.

- **2**  Correct and complete: states the rule and its key numbers or conditions,
         and cites the right provision.
- **1**  Partially correct: right direction but drops a condition, a number, or
         the citation, or hedges when the context actually supported an answer.
- **0**  Wrong, hallucinated, or cites a provision that does not support the claim.

For q17 and q18 the scoring is inverted: **2** means the system correctly refused
or flagged the gap, **0** means it fabricated a rule.

## 3. Faithfulness (pass or fail)

Is every factual claim in the answer supported by a retrieved passage? This is a
gate, not a gradient. A fluent answer that adds facts not in the context fails
faithfulness even if those facts happen to be true, because in a legal setting an
unsourced claim is a liability. Faithfulness failures are the ones I care about
most in the writeup.

## What "good" means per question type

- **factual_lookup / definition**: the specific rule or definition, with the
  correct Article cited. Paraphrase is fine, invented scope is not.
- **deadline / numeric_threshold / factual_with_number**: the number has to be
  exact (72 hours, one month, 16 years, 20 million euro or 4%). A right rule with
  a wrong number scores 0 on correctness, numbers are the point.
- **enumeration / conditional**: measured on recall of the list. Missing one of
  six lawful bases is a 1, not a 2.
- **comparative / comparative_near_duplicate**: both sides must be retrieved and
  contrasted. Retrieving only one of the two provisions caps correctness at 1.
- **negative_control / out_of_corpus**: the only good answer is a refusal or an
  explicit "not in the context". Any confident fabricated rule is a 0.
