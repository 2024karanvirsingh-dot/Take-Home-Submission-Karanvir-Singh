# RAG over the UCMJ, with an honest evaluation

A small retrieval augmented generation pipeline over the Uniform Code of
Military Justice, plus a writeup that scores its own answers and digs into where
and why retrieval breaks. The pipeline itself is deliberately simple. The point
of this repo is the evaluation, so most of this README is about the failures.

## Why this corpus

The corpus is the UCMJ, the statutory text of 10 U.S.C. chapter 47, all 198
provisions in force, split into its two natural document types:

* **93 punitive articles** (Subchapter X, Articles 77 to 134): the offenses.
  Desertion, AWOL, insubordination, murder, conduct unbecoming, the general
  article.
* **105 procedural articles** (everything else): jurisdiction, apprehension,
  non-judicial punishment, court-martial composition, trial procedure,
  sentencing, appeals.

I picked this corpus for three reasons. First, it is the body of law I know the
context around, which matters when you are writing gold answers and judging
correctness by hand. Second, it has structural traps that make retrieval
evaluation interesting: every provision has two numbers (Article 86 is 10 U.S.C.
886, the famous "Article 15" is section 815), practitioners use colloquial
vocabulary the statute never uses ("panel", "AWOL", even "UCMJ" itself), and the
punitive articles share heavy boilerplate ("Any person subject to this chapter
who...") that erodes term statistics. Third, it has a hard, well defined
knowledge boundary: the statute defines offenses but the actual punishment
tables live in the Manual for Courts-Martial, which is deliberately not in the
corpus. That boundary gives the evaluation a built in hallucination test with
real stakes, because the model knows the MCM numbers from training and has to be
prevented from using them.

Everything is fetched from the Cornell Legal Information Institute by
`data/fetch_corpus.py` and committed under `data/corpus/`, so a clean clone
needs no network.

## Quick start

```bash
git clone <this repo>
cd Take-Home-Submission-Karanvir-Singh
pip install -r requirements.txt      # scikit-learn + numpy is enough for retrieval

# ask one question
python3 -m rag.pipeline "can a deserter be sentenced to death?"

# reproduce every run in this README (no API key needed for retrieval or scoring)
./run_all.sh

# optional: the dense/hybrid ablation (downloads a 90 MB encoder via torch)
pip install sentence-transformers
python3 -m eval.run_eval --retriever dense --tag dense_baseline
python3 -m eval.run_eval --retriever hybrid --tag hybrid_baseline
```

Retrieval, indexing and scoring run with no API key. Generation uses the
Anthropic API when `ANTHROPIC_API_KEY` is set (model overridable with
`RAG_MODEL`); with no key it falls back to returning the top passage verbatim so
the pipeline still runs end to end. See "How the recorded answers were produced"
below for how the graded answers in `outputs/` were generated.

## How the pipeline works

Four stages, each in its own module, nothing hidden behind a framework.

**Ingest and chunk** (`rag/chunk.py`). Each provision is one file. Most are
short enough to be a single chunk. The long ones (Art. 2 persons subject to the
code, Art. 15 non-judicial punishment, Art. 120 sexual offenses) are split on
paragraph boundaries into windows of about 180 words with 40 words of overlap.
Chunking never crosses a provision boundary, so every chunk maps to exactly one
article and every citation is honest. This gives 572 chunks at the default size.

**Index and retrieve** (`rag/index.py`). Two sparse retrievers by default, both
fully inspectable:

* **BM25**, implemented from scratch (about 40 lines) rather than pulled from a
  library, so the term frequency saturation (`k1`) and length normalisation
  (`b`) are visible and so I can print the exact terms that made a chunk win.
  This is the default.
* **TF-IDF cosine** via scikit-learn, used as the alternative in the ablation.

I chose sparse retrieval as the default on purpose. A dense embedding model
turns every retrieval decision into an opaque dot product, while sparse keeps
the clone small and, more importantly for this exercise, keeps every retrieval
decision explainable: the eval runner records which query terms fired in the
winning chunk. The tradeoff is that sparse retrieval has no idea that
"deserter" and "desertion" are the same word, and the failure analysis shows
exactly what that costs. Because the failure analysis ends up predicting
exactly which misses a dense model should fix, there is also an optional dense
retriever (all-MiniLM-L6-v2 embeddings, `--retriever dense`) and a BM25 plus
dense reciprocal rank fusion (`--retriever hybrid`) to put that prediction to
the test; they need `pip install sentence-transformers` and are kept out of
requirements.txt so the default install stays small. Ablation E has the
results, and they cut both ways.

**Generate** (`rag/generate.py`). The prompt is strict: answer only from the
retrieved context, cite the article, treat "as a court-martial may direct" as a
signal that the real number lives in the MCM rather than an invitation to supply
one, and if the answer is not in the context say so instead of guessing. In a
legal setting the most damaging error is a fluent answer that smooths over a
retrieval gap, so "not in the context" is a first class allowed answer.

## The questions and the rubric

18 questions in `eval/questions.json` across eight types: numeric and
limitations questions where an exact figure is the answer, definitions,
enumerations, conditionals, comparative pairs (including the near duplicate
desertion/AWOL pair), and two controls (a statute versus MCM boundary question
and a pure out of corpus question about the Rules for Courts-Martial).

Full rubric in `eval/rubric.md`. Each answer gets three scores, kept separate
because retrieval and generation fail for different reasons:

* **Retrieval (0 to 2)**: was the correct provision in the top k, and near the
  top.
* **Correctness (0 to 2)**: right and complete, with the right article cited.
  For number questions the number must be exact.
* **Faithfulness (pass or fail)**: is every claim supported by a retrieved
  passage. A gate, not a gradient.

## Results at a glance

Baseline: BM25, chunk size 180, top k = 5, strict prompt.

| Metric | Score |
| --- | --- |
| Retrieval (16 scored questions) | 20/32, avg 0.63 |
| Correctness (all 18) | 26/36, avg 0.72 |
| Faithfulness | 18/18 pass |
| Clean wins (retrieval 2, correctness 2) | 9 (q01, q03, q08, q09, q10, q11, q13, q15, q16) |
| Retrieval misses (gold not in top k) | 5 (q04, q05, q06, q07, q12) |

Faithfulness is 18/18 only because the strict prompt converts every retrieval
miss into a visible refusal. The prompt ablation below shows how fast that
collapses when the guardrail is removed. Full per question answers, scores and
notes are in `outputs/graded_answers.json`.

## Three cases where it worked, and why

**q01, punishment for premeditated murder.** Art. 118 takes the top two ranks
and the rank 2 chunk contains the operative clause: death or imprisonment for
life. This works because "premeditated" is a rare term concentrated in exactly
one article. When the query shares a distinctive content word with one short
provision, BM25 is close to unbeatable. It also matters that Art. 118 is one of
the few punitive articles whose punishment is fixed in the statute itself, so
the answer is actually in the corpus to be found.

**q03, desertion versus AWOL.** The near duplicate comparative pair, and the
retriever gets both sides: Art. 85 at ranks 1 and 2, Art. 86 at rank 3. Both
articles have distinctive title vocabulary ("desertion", "absence without
leave") that the query repeats, so both survive into the top k and the generated
answer can draw the real dividing line, intent to remain away permanently. I am
showing this win deliberately because the structurally identical question q14
fails, and the contrast between them is the most instructive pair in the eval.

**q11, the Article 15 question.** "What punishments can a commanding officer
impose under Article 15 without a court-martial?" All five retrieved chunks are
parts of Art. 15, and the enumeration in the answer (admonition, reprimand,
correctional custody, forfeiture, reduction, extra duties, restriction, the
demand for court-martial instead) is fully grounded. The interesting part is why
the colloquial name worked: section 815's own text and its neighbours contain
the token "15" in cross references, so the nickname is actually present in the
index. Compare q07, where the far more common nickname "UCMJ" is nearly absent
from the corpus and actively misleads. Whether a colloquialism retrieves is a
fact about the corpus, not about the colloquialism.

## Where it failed, and why

Five misses, four distinct mechanisms, all visible in the matched term logs the
eval runner records.

**Failure 1: morphology. "Deserter" never matches "desertion" (q04).** The
query "Can a deserter be sentenced to death?" scores zero against Art. 85,
because the statute says "desertion" and "deserts" while the query says
"deserter", and a bag of words matcher with no stemming treats those as three
unrelated tokens. The retriever instead chases "sentenced to death" into the
capital procedure articles (Art. 25, 25a), which are about how a capital
court-martial is composed, not about desertion. The strict prompt then correctly
refuses. The fix and its cost are quantified in Ablation C below.

**Failure 2: boilerplate kills the query's key terms (q05, q07).** "Who is
subject to the UCMJ?" should retrieve Art. 2, which enumerates exactly that. It
never surfaces. The phrase "subject to this chapter" appears in 97 of the 198
provisions, because every punitive article opens with "Any person subject to
this chapter who...". The words that carry the question's meaning are the least
informative words in the entire corpus. Same mechanism for "commanding officer"
(q05): defined once in Art. 1, used in dozens of provisions, so the definition
is invisible behind the usage. These two failures are unchanged at every chunk
size and under both retrievers, which is the fingerprint of a term statistics
problem: no amount of re-chunking fixes a query whose terms have no IDF.

**Failure 3: the corpus does not speak the user's language (q06, q10, and the
"UCMJ" trap).** The statute never calls itself the UCMJ; it says "this chapter".
The token "UCMJ" appears in exactly two provisions, one of which is Art. 146,
the Military Justice Review Panel. So for any question phrased "under the
UCMJ", that phrase is a corpus-rare, high IDF term pointing at the wrong
article, and Art. 146 keeps surfacing at rank 1 (q06, q07, and rank 1 for q10
via the equally colloquial word "panel", which in this statute only ever means
the review body). The vocabulary users add for clarity is exactly the vocabulary
that misleads the retriever. This is the strongest argument in the whole eval
for dense or hybrid retrieval: no term weighting scheme can learn that "UCMJ"
means "this chapter" from term statistics alone. That is a falsifiable claim,
and Ablation E tests it directly.

**Failure 4: a provision whose name is generic is structurally invisible
(q14).** "Conduct unbecoming an officer versus the general article": Art. 133
retrieves at rank 1 on its distinctive title, and Art. 134 never appears,
because its name is made of the two most generic words in the corpus, "general"
and "article". One side of the comparison is grounded, the other missing,
correctness capped at 1. Together with q03 (the same question shape succeeding
when both provisions have distinctive names), this isolates the variable
cleanly: comparative questions live or die on the weaker-named side's
vocabulary.

**Honourable mention, the function word bug (q16).** The right answer was
generated from rank 2, but rank 1 was Art. 103, Spies, matched entirely on the
words "about" and "does". My deliberately small stopword list does not remove
them, and "disobeying" matched nothing because the statute says "disobeys"
(morphology again). The top slot of a retrieval was decided by two function
words. Stopword lists tuned by intuition rather than measurement fail in
exactly this way, and I am keeping the bug in the writeup because the matched
term log that exposed it is the reason the eval runner records matched terms at
all.

## Changing one thing: three ablations

### Ablation A, chunk size

Full runs in `outputs/run_bm25_cw*.json`.

| chunk size | gold in top 1 | gold in top 3 | missed | chunks |
| --- | --- | --- | --- | --- |
| 120 | 8/17 | 11/17 | 6 | 1041 |
| 180 (baseline) | 9/17 | 11/17 | 6 | 572 |
| 300 | 8/17 | 11/17 | 5 | 343 |

Chunk size moves little here, and the reason is worth stating: most UCMJ
provisions are shorter than one chunk at any tested size, so re-chunking only
redistributes the handful of long articles. The one real effect: at 300 words
the Art. 1 definitions article holds together well enough that q06 (accuser)
creeps back in at rank 5. The important negative result is that the q04/q05/q07
misses do not move at any size, confirming their causes are morphology and term
statistics, not chunk boundaries. Within a single long article, though, chunking
decides which paragraph you get: in q02 all five slots are Art. 43 chunks but
the five year rule itself sits at rank 5, behind the tolling paragraphs. The
right document is not the same thing as the right paragraph.

### Ablation B, retriever (TF-IDF cosine instead of BM25)

`outputs/run_tfidf_baseline.json`: 9/17 top 1, 9/17 top 3, 7 missed, slightly
worse than BM25 overall (it additionally drops q13, demotes q10 to rank 4). The
two retrievers agree on every hard failure, which is expected: both are bag of
words scorers and share the same blind spots. Swapping between two sparse
retrievers is a lateral move; the misses need a different representation, not a
different formula over the same terms.

### Ablation C, tokenisation (light stemming), the found-and-fixed one

The q04 failure has an obvious cause (deserter/desertion) and an obvious cheap
fix: a light suffix stripper on both query and document tokens
(`rag/index.py`, `_light_stem`), toggled with `--stem`. Run in
`outputs/run_bm25_stem.json`. The results are the most instructive in the
project precisely because they are mixed:

| question | baseline | with stemming | why |
| --- | --- | --- | --- |
| q04 deserter/death | MISS | **rank 1** | deserter and desertion both stem to desert, exactly as intended |
| q13 insulting the President | 1 | 2 | "officer" stems to "offic" and now collides with "offices" |
| q10 capital panel size | 2 | 4 | new collisions promote Art. 66 and Art. 146 chunks |
| q16 disobeying orders | 2 | MISS | stemmed generic terms boost competing procedural articles past Art. 92 |
| everything else | unchanged | unchanged | |

Aggregate: top 1 stays 9/17, top 3 drops from 11 to 10. So the fix works
perfectly on the failure it was designed for and pays for it with collision
damage elsewhere, netting out to roughly zero on this question set. That is the
honest shape of most retrieval interventions: a targeted gain, a diffuse cost,
and the eval set is what tells you whether the trade is worth it. A proper
stemmer (Porter) would keep most of the q04 gain while avoiding the crude
"officer to offic" collisions, and that, not more chunk tuning, is the next
change I would make. The vocabulary gap failures that no term-level fix can
reach get their own experiment in Ablation E.

### Ablation D, the prompt

One thing changed: the system prompt (strict versus permissive in
`rag/generate.py`), retrieval held fixed. Full text in
`outputs/prompt_ablation.json`.

| question | strict prompt | permissive prompt |
| --- | --- | --- |
| q05 define commanding officer | "not in the context" (faithful) | full Art. 1 definition from memory, cited to an article never retrieved |
| q07 who is subject to the UCMJ | "not in the context" (faithful) | confident Art. 2 enumeration from memory |
| q17 max confinement, AWOL over 30 days | names the statute/MCM boundary | "one year", the real MCM figure, which exists nowhere in the corpus |
| q18 RCM 707 speedy trial | "not in this corpus" | a detailed 120 day rule for a document the system has never seen |

q17 deserves special attention because retrieval made it actively dangerous: the
query's "30 days" matched the day-limit tables in Art. 15, so the context is
full of real, plausible looking numbers (14 days, 60 days) that belong to a
different legal instrument entirely, non-judicial punishment limits rather than
court-martial confinement maxima. The permissive answer skipped those decoys and
supplied the true MCM number from training instead, which is arguably worse: a
correct-in-reality number the system never read, indistinguishable to the user
from a grounded answer. Every permissive answer in the table fails faithfulness.

This is the single most important thing the project demonstrates. The strict
prompt is not a nicety; it is the only reason faithfulness is 18/18, and it
works by converting retrieval failures into visible refusals instead of
invisible fabrications. Better retrieval reduces how often the refusal fires.
The prompt is what makes the residual failures safe.

### Ablation E, dense and hybrid retrieval, testing the failure analysis on its own prediction

The failure analysis above keeps making one falsifiable claim: the q05/q07
boilerplate misses and the "UCMJ" vocabulary trap cannot be fixed by any term
level intervention, only by a representation that knows synonymy. So the last
experiment swaps the retriever for a small pretrained sentence encoder
(all-MiniLM-L6-v2, cosine over normalised embeddings, `--retriever dense`) and
for a reciprocal rank fusion of BM25 and dense (`--retriever hybrid`). Both
need `pip install sentence-transformers` and are deliberately not in
requirements.txt; everything else in the repo runs without them. Runs in
`outputs/run_dense_baseline.json` and `outputs/run_hybrid_baseline.json`.

| 17 scored questions | BM25 | dense | hybrid RRF |
| --- | --- | --- | --- |
| gold at rank 1 | 9 | 12 | 10 |
| gold in top 3 | 11 | 13 | 12 |
| missed | 6 | 4 | 4 |

Every question that moved, and why:

| question | BM25 | dense | hybrid | mechanism |
| --- | --- | --- | --- | --- |
| q04 deserter/death | MISS | **1** | MISS | encoder knows deserter and desertion are the same thing, no stemming needed and none of Ablation C's collateral damage |
| q05 define commanding officer | MISS | **1** | 5 | the boilerplate that zeroed the IDF of "commanding officer" is invisible to the encoder |
| q07 who is subject to the UCMJ | MISS | **2** | 2 | Art. 2 finally surfaces; the encoder maps "subject to the UCMJ" near "persons subject to this chapter" |
| q06 who is an accuser | MISS | MISS | **2** | the pure fusion win, see below |
| q10 capital panel size | 2 | 1 | 1 | "panel" no longer tunnels to the Art. 146 review body |
| q16 disobeying orders | 2 | 1 | 1 | the "about"/"does" function word bug vanishes, no terms to match |
| q08 rights under Article 31 | **1** | MISS | MISS | the new failure dense introduced, see below |
| q12 referral of charges | MISS | MISS | MISS | not a representation problem |
| q17 AWOL confinement max | MISS | MISS | MISS | the answer is in the MCM, not the corpus, by design |

The prediction held. Dense fixes q05 and q07 outright, and it fixes q04 at
rank 1 without stemming and without the officer/offices collisions that
stemming charged for the same repair. Three failure mechanisms (morphology,
boilerplate IDF erosion, vocabulary gap), one representational fix.

But dense buys those wins by breaking the question sparse was best at. On q08,
"What rights does Article 31 give a servicemember suspected of an offense?",
all five dense slots are chunks of Art. 6b, rights of the victim of an
offense. To the encoder, rights of a suspect and rights of a victim are nearly
the same sentence, while the one token that actually pins the question down,
"31", is just another subword. BM25 answered this at rank 1 on exactly that
token. The symmetry is the finding: dense wins when the user's words are not
the statute's words, and loses when the user's words were the statute's words
all along. In a legal corpus, where the exact article number is often the
entire meaning of the question, that is not a small tail case.

The hybrid is instructive in a different way. Its fusion win is q06: Art. 1
(which defines "accuser") was in neither retriever's top 5, but it sat high
enough in both top 50 lists that reciprocal rank fusion lifts it to rank 2.
That is fusion doing exactly what it is for: rewarding agreement between
retrievers that fail differently. But the same consensus bias costs q04 and
q08, where one retriever was simply right (rank 1) and the other blind, so the
gold gets a single 1/(60+1) vote and loses to mediocre chunks present in both
lists. RRF with a plain sum has no way to know that one confident list should
outvote two lukewarm ones. On this question set the honest summary is: dense
is the best single retriever, hybrid trades its peaks for fewer catastrophes,
and which you want depends on whether your users ask by concept or by article
number. A production system would want dense recall with an exact-identifier
boost, which is what BM25 was providing for free.

And the number that matters most does not move: q12 and q17 miss under every
retriever tested, because their failures are not representational. q17's
answer is outside the corpus by construction, and q12's answer (the Art. 32
preliminary hearing requirement) is phrased in procedural language spread
across half a dozen referral provisions, so nothing anchors it. Retrieval
upgrades fix retrieval-shaped failures and nothing else; knowing which
failures are which is what the eval is for.

## How the recorded answers were produced

This environment had no `ANTHROPIC_API_KEY`, so the graded answers in
`outputs/graded_answers.json` were generated with Claude acting as the
generation backend over the exact passages the pipeline retrieved (the same
contexts `rag/generate.py` builds, dumped per question). Retrieval, ranking,
matched term attribution and the run files are all produced by the committed
code and are fully reproducible with `./run_all.sh`. Setting `ANTHROPIC_API_KEY`
makes `eval/run_eval.py` generate answers automatically through the API instead.
I kept generation and retrieval cleanly separated precisely so that the part a
reviewer most needs to trust, the retrieval behaviour and its scoring, does not
depend on any key or any model.

## Honest limitations

* Scoring is mine, applied by hand from the rubric. An LLM judge or a second
  annotator would make it less subjective, and a few correctness 1 versus 2
  calls are genuinely arguable.
* 18 questions is enough to surface failure modes, not enough for stable
  metrics. The per type counts are small.
* The dense results come from one small encoder (MiniLM) on one question set.
  Ablation E's direction is clear but its margins (12 versus 9 at rank 1) are
  well within what three or four question flips can produce, which on an 18
  question set is exactly what happened. A bigger encoder or a legal domain
  one might also handle the Article 31 identifier failure differently.
* The hybrid is the simplest possible fusion (unweighted RRF). Weighted
  fusion, or BM25 as an exact-identifier booster on top of dense recall,
  is the obvious next step and I would expect it to keep q08 without losing
  the dense wins.
* The stopword list was tuned by intuition and q16 shows it (rank 1 decided by
  "about" and "does"). Measured stopword selection or IDF floors would fix it.
* The corpus is the statute only. A production military law assistant would need
  the MCM (punishment tables, Rules for Courts-Martial), and the q17/q18
  controls exist to prove the system knows where its corpus ends.

## Layout

```
data/fetch_corpus.py     fetches the UCMJ from Cornell LII (corpus is committed)
data/corpus/             198 provisions, one file each, + manifest.json
rag/chunk.py             ingestion and paragraph aware chunking
rag/index.py             BM25 (from scratch), TF-IDF cosine, light stemmer,
                         optional dense (MiniLM) and hybrid (RRF) retrievers
rag/generate.py          strict and permissive grounding prompts
rag/pipeline.py          chunk -> index -> retrieve -> generate
eval/questions.json      18 questions, 8 types, gold answers and citations
eval/rubric.md           the scoring rubric
eval/run_eval.py         runs the pipeline, records retrieval rank + matched terms
outputs/graded_answers.json   per question answers with hand scores and notes
outputs/prompt_ablation.json  strict vs permissive prompt on the retrieval gaps
outputs/run_*.json       raw runs for the chunk, retriever and stemming ablations
run_all.sh               reproduce every run
```
