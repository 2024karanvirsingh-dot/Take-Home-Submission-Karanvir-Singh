# RAG over the UCMJ with local grounded answer construction and honest evaluation

A retrieval augmented question answering pipeline over the Uniform Code of
Military Justice, with a fully local, deterministic, citation constrained
answer layer and an evaluation that scores its own outputs and digs into
where and why they break. There is no language model and no API call
anywhere in this repository: retrieval is BM25 / TF-IDF (with optional dense
embeddings), and answers are constructed by a transparent extractive
synthesizer that either quotes the statute with citations or refuses. The
failure analysis matters more than the pipeline, so most of this README is
about the failures.

## Key findings

- Sparse retrieval is highly interpretable and strong when the query shares
  rare vocabulary with one provision, and an explicit article number router
  makes exact identifiers reliable for every retriever.
- Dense retrieval (optional, MiniLM) fixes the vocabulary mismatch failures
  the sparse analysis predicts (morphology, boilerplate, colloquial names)
  but loses the exact identifier signal sparse matching gets for free.
- Hybrid rank fusion trades peaks for fewer catastrophes; it is the only
  configuration that fully covers both comparative questions, and it still
  does not dominate every query.
- A conservative local answer policy converts unsupported retrieval into
  visible refusals instead of unsupported legal claims. The permissive
  policy, same code minus the guardrails, answers every question and cites
  the wrong legal instrument on exactly the questions where that is most
  dangerous.
- Two failures survive every retriever tested (q12, and the q02 paragraph
  selection miss), because they are not representational failures at all.

## Quick start

Tested with Python 3.9.6, scikit-learn 1.6.1, numpy 2.0.2, pytest 8.4.2,
and (for the optional dense ablation) sentence-transformers 5.1.2.

```bash
git clone https://github.com/2024karanvirsingh-dot/Take-Home-Submission-Karanvir-Singh.git
cd Take-Home-Submission-Karanvir-Singh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ask one question (local, deterministic, no credentials)
python3 -m rag.pipeline "What rights does Article 31 provide?"

# reproduce every run, score and table in this README
./run_all.sh

# tests
pip install -r requirements-dev.txt && python3 -m pytest tests/ -q

# optional: dense + hybrid retrievers (pulls in torch)
pip install -r requirements-dense.txt && ./run_all.sh
```

The default pipeline needs no environment variables, no network, and no
model downloads. `run_all.sh` regenerates every output file; the README
tables are printed by `eval/build_summary.py` from those files, so no number
here is hand copied.

## Architecture

```
UCMJ source documents (198 provisions, committed under data/corpus/)
        |
provision aware parsing (data/fetch_corpus.py, run once, output committed)
        |
paragraph aware chunking, never crossing a provision (rag/chunk.py)
        |
BM25 / TF-IDF / MiniLM indexes (rag/index.py)
        |
top k retrieval + article number routing (rag/pipeline.py)
        |
local grounded answer construction with refusal (rag/answer.py)
        |
retrieval, citation, content, support and refusal metrics (eval/)
```

## Why this corpus

The corpus is the UCMJ, the statutory text of 10 U.S.C. chapter 47, all 198
provisions in force, split into its two natural document types:

* **93 punitive articles** (Subchapter X, Articles 77 to 134): the offenses.
  Desertion, AWOL, insubordination, murder, conduct unbecoming, the general
  article.
* **105 procedural articles** (everything else): jurisdiction, apprehension,
  non-judicial punishment, court-martial composition, trial procedure,
  sentencing, appeals.

I picked it for three reasons. First, it is the body of law I know the
context around, which matters when you are writing the answer key. Second,
it has structural traps that make retrieval evaluation interesting: every
provision has two numbers (Article 86 is 10 U.S.C. 886, the famous "Article
15" is section 815), practitioners use colloquial vocabulary the statute
never uses ("panel", "AWOL", even "UCMJ" itself), and the punitive articles
share heavy boilerplate ("Any person subject to this chapter who...") that
erodes term statistics. Third, it has a hard, well defined knowledge
boundary: the statute defines offenses but the actual punishment tables live
in the Manual for Courts-Martial, which is deliberately not in the corpus.
That boundary gives the evaluation a built in fabrication test with real
stakes.

Everything is fetched from the Cornell Legal Information Institute by
`data/fetch_corpus.py` and committed under `data/corpus/`, so a clean clone
needs no network. The fetch script exists for provenance and is never run
during evaluation.

## Ingestion and chunking

Each provision is one file. Most are short enough to be a single chunk. The
long ones (Art. 2 persons subject to the code, Art. 15 non-judicial
punishment, Art. 120 sexual offenses) are split on paragraph boundaries into
windows of about 180 words with 40 words of overlap. Chunking never crosses
a provision boundary, so every chunk maps to exactly one article and every
citation is honest. This gives 572 chunks at the default size.
`tests/test_chunking.py` pins both properties.

## Retrieval methods

The assignment asks for embedding and indexing, and the system implements
and evaluates three representations plus a fusion:

* **BM25** (`rag/index.py`), implemented from scratch (about 40 lines)
  rather than pulled from a library, so term frequency saturation and length
  normalisation are visible and the eval runner can print the exact terms
  that made a chunk win. This is the default, chosen for interpretability
  and minimal setup.
* **TF-IDF cosine** via scikit-learn: the sparse vector embedding baseline.
* **Dense semantic embeddings**, all-MiniLM-L6-v2 + cosine
  (`--retriever dense`): optional because it pulls in torch, installed via
  `requirements-dense.txt`. Exists to test the failure analysis's own
  prediction that the vocabulary gap misses need a representation that
  knows synonymy.
* **Hybrid** (`--retriever hybrid`): reciprocal rank fusion of BM25 and
  dense, the boring standard because it works.

**Article number routing** (`rag/pipeline.py`): practitioners ask about
provisions by number, but a provision's own text never states its own
number, so no retriever, sparse or dense, can find "Article 31" from the
query terms. When the question names an article the retriever did not
surface, the router swaps that article's best chunk into the last result
slot. It runs identically for every retriever, so the ablations stay a fair
comparison. Its effect is visible in the dense results below.

Tokenisation notes: the stopword list stops function words including the
ones questions contribute ("about", "does"), after a recorded failure where
those two words decided a top retrieval slot; legally loaded common words
(subject, right, order, charge, court, person) are deliberately kept. This
fix changed the retrieval baseline relative to earlier commits, and
`tests/test_retrieval.py` pins the regression. An optional light stemmer
(`--stem`) is evaluated in the ablations, where its collision costs are
measured rather than assumed.

## Local answer construction

`rag/answer.py` is a deterministic, citation constrained extractive
synthesizer. There is no model. Given the question and the top k chunks it:

1. splits the chunks into statutory sentences and list items (reflowing the
   source's mid sentence line breaks, dropping amendment history notes);
2. scores each sentence with transparent lexical features: pool idf weighted
   token overlap (with plural folding), title matches that cover most of a
   title, bonuses for exact article number matches, quantities when the
   question asks for one, definitional clauses for definition questions and
   operative legal verbs (shall, may not, prohibited, punished);
3. applies a support test per sentence (the grounded policy): at least two
   topical shared tokens, or a rare anchor token, or the question named the
   sentence's article, or a definitional clause for a definitional question.
   Punishment amount questions additionally require punitive article
   provenance, because procedural provisions like Art. 15 are full of day
   limits that belong to a different legal instrument;
4. selects a small non duplicative set (comparative questions reserve slots
   for both sides), completes enumerations so list items survive extraction
   whole, and composes the answer with one citation per article group;
5. refuses with a standardized message when nothing clears support, states
   the Manual for Courts-Martial boundary when a punishment amount question
   finds only "as a court-martial may direct", and recognises Rules for
   Courts-Martial / MCM questions as out of corpus before retrieval is even
   consulted.

Every sentence in every answer is a verbatim extract of a retrieved chunk,
so faithfulness to the retrieved context holds by construction, and
`eval/score_answers.py` re verifies it independently anyway. The honest
consequence: this layer cannot fabricate out of corpus facts at all, so the
residual answer risk moves to misattribution (quoting the wrong article
fluently), and the metrics below are designed to catch exactly that.

The answer layer performs one vocabulary normalisation (the statute never
calls itself "the UCMJ"; the token maps to "chapter" for ranking only, at a
fixed small weight, and never counts toward support). Retrieval does not get
that mapping; the vocabulary gap it causes is a core finding below.

## Evaluation dataset

18 questions in `eval/questions.json` across eight types (numeric,
limitations, definition, enumeration, conditional, factual lookup, and two
comparative variants), plus two controls. Every question carries structured
fields; nothing is parsed out of prose:

- `answerable_from_corpus`: 16 true, 2 false.
- `gold_articles`: explicit article numbers; comparative questions list
  both sides (["85", "86"]); controls have none by definition.
- `related_articles`: near miss articles that do not count against
  citation precision (for q17 the statute boundary control, Art. 86 defines
  the offense whose punishment the question asks about).
- `must_include`: statutory phrases a complete answer must contain, each
  validated against the gold articles' text by `eval/validate_dataset.py`,
  so the answer key itself is provably grounded.

Accounting used everywhere in this README, the code and the output files:
**16 retrieval scored answerable questions, 1 statute boundary control
(q17), 1 pure out of corpus control (q18)**. Controls are excluded from
retrieval metrics and scored on refusal behavior.

The scoring rubric is in `eval/rubric.md`: automatic retrieval metrics
(hit@1/3/5, MRR, all gold retrieved), automatic answer metrics (refusal
correctness, citation hit and precision, content coverage, verbatim
support), and a clearly separated interpretation section.

## Baseline results

BM25, chunk size 180, k = 5, grounded answer policy. All tables printed by
`eval/build_summary.py`; raw runs in `outputs/`.

| run | hit@1 | hit@3 | hit@5 | MRR | missed | all gold in top k |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_baseline | 9/16 | 11/16 | 11/16 | 0.62 | 5/16 | 10/16 |
| bm25_cw120 | 9/16 | 11/16 | 11/16 | 0.61 | 5/16 | 10/16 |
| bm25_cw300 | 10/16 | 11/16 | 11/16 | 0.66 | 5/16 | 10/16 |
| tfidf_baseline | 9/16 | 9/16 | 10/16 | 0.58 | 6/16 | 9/16 |
| bm25_stem | 9/16 | 10/16 | 12/16 | 0.62 | 4/16 | 11/16 |
| dense_baseline | 12/16 | 13/16 | 14/16 | 0.79 | 2/16 | 13/16 |
| hybrid_baseline | 10/16 | 12/16 | 14/16 | 0.71 | 2/16 | 14/16 |

Answer metrics on the baseline (automatic, `outputs/answer_scores_bm25_baseline.json`):

| metric | value |
| --- | --- |
| answered (of 16 answerable) | 13 |
| refusals on answerable questions | 3 |
| controls refused correctly | 2/2 |
| answers citing a gold article | 11/13 |
| mean citation precision | 0.577 |
| mean content coverage | 0.692 |
| verbatim support check | 18/18 |

The three false refusals are q05, q06 and q07, all questions whose gold
article never reached the top k: the grounded policy is converting retrieval
misses into refusals rather than wrong answers, which is the designed
behavior. The two answered questions that cite no gold article (q04, q12)
are also retrieval misses; there the support test was satisfied by topically
adjacent but wrong provisions, which is the layer's honest residual risk and
gets its own failure entry below.

## Three or more cases where it worked, and why

**q01, punishment for premeditated murder.** Art. 118 takes the top ranks;
"premeditated" is a rare term concentrated in exactly one article, and Art.
118 is one of the few punitive articles whose punishment is fixed in the
statute itself. The constructed answer quotes the murder definition and the
operative clause, and the content check confirms "death or imprisonment for
life" verbatim. When the query shares a distinctive content word with one
short provision, BM25 plus extraction is close to unbeatable.

**q03, desertion versus AWOL.** The near duplicate comparative pair, and
the only sparse win that needs both sides: Art. 85 and Art. 86 are both
retrieved (marked * in the per question table), and the comparative
selection rule reserves answer slots per article, so the answer quotes Art.
86's absence prongs and Art. 85's "intent to remain away therefrom
permanently", which is the legal dividing line. The structurally identical
q14 fails on one side, and the contrast between the two is the most
instructive pair in the eval.

**q09, the three types of courts-martial.** Retrieval is trivial (distinct
title vocabulary), but this is the enumeration completion showcase: the
selected sentence introduces the list and ends mid enumeration, and the
builder pulls items (1) through (3) back in whole. Content coverage 3/3
(general, special, summary). Without list completion this answer names one
kind of court-martial and silently narrows the law.

**q11, the Article 15 question.** The famous colloquial name works here for
two stacked reasons: the query literally names the article, so both the
named article scoring bonus and the router guarantee Art. 15 presence, and
all five retrieved chunks are parts of Art. 15. The answer quotes the
operative "may impose one or more of the following disciplinary punishments"
clause with items. Compare q17 below, where Art. 15's very same day limit
tables become the trap.

**q17 and q18, the controls.** The statute boundary question ("maximum
confinement for AWOL over 30 days") is refused with the Manual for
Courts-Martial explanation, and the Rules for Courts-Martial question is
recognised as out of corpus before retrieval is consulted. Both behaviors
are pinned by tests. These only look easy; the policy ablation below shows
what the same retrieval produces without the guardrails.

## Where it failed, and why

**Failure 1: morphology. "Deserter" never matches "desertion" (q04).** The
query "Can a deserter be sentenced to death?" scores zero against Art. 85,
because the statute says "desertion" and "deserts" while the query says
"deserter", and a bag of words matcher treats those as unrelated tokens.
The retriever chases "sentenced to death" into capital procedure articles
instead, and the grounded answer quotes Art. 25a (capital panel size) with
a citation the scorer flags as wrong: support was satisfied by "sentenced"
and "death", which establish topic but not entailment. Stemming fixes the
retrieval (rank 1 in `bm25_stem`) and dense fixes it without stemming's
side effects; see the ablations.

**Failure 2: boilerplate kills the query's key terms (q05, q07).** "Who is
subject to the UCMJ?" should retrieve Art. 2, which enumerates exactly
that. It never surfaces under any sparse configuration. The phrase "subject
to this chapter" appears in 97 of the 198 provisions, so the words carrying
the question's meaning are the least informative words in the corpus. Same
mechanism for "commanding officer" (q05): defined once in Art. 1, used in
dozens of provisions, the definition invisible behind the usage. These
misses are unchanged at every chunk size and under both sparse retrievers,
the fingerprint of a term statistics problem. The grounded policy refuses
both; the definitional gate in the support test is what stops it from
quoting a usage of "commanding officer" as if it were the definition, which
is precisely what the permissive policy does (it cites Art. 23).

**Failure 3: the corpus does not speak the user's language (the "UCMJ"
trap).** The statute never calls itself the UCMJ; the token appears in two
provisions, one being Art. 146, the Military Justice Review Panel. For any
question phrased "under the UCMJ", that token is corpus rare, high idf and
points at the wrong article. The permissive q07 answer makes the trap
concrete: it confidently quotes Art. 146 panel term limits for "who is
subject to the UCMJ". The vocabulary users add for clarity is exactly the
vocabulary that misleads a term statistics retriever, and no term weighting
scheme can learn that "UCMJ" means "this chapter". That is a falsifiable
claim, and the dense ablation tests it.

**Failure 4: a provision whose name is generic is structurally invisible
(q14).** "Conduct unbecoming an officer versus the general article": Art.
133 retrieves at rank 1 on its distinctive title, and Art. 134 never
appears under any sparse configuration, because its name is made of two of
the most generic words in the corpus. One side of the comparison is
grounded, the other missing, and the answer metrics show it (content 1/2,
"all gold" failed). Together with q03, the same question shape succeeding
when both provisions have distinctive names, this isolates the variable
cleanly. Only the hybrid retriever covers both sides.

**Failure 5: right article, wrong paragraph (q02).** Art. 43 fills the top
ranks, the citation is right, and the content check still fails: the
operative five year clause never says "statute of limitations" in its body
(the tolling and extension clauses do), so within a pool of near tied Art.
43 sentences the operative clause loses the tie on chunk rank and misses
the three slots. Retrieval granularity has a selection layer twin: heading
vocabulary beats operative vocabulary. The same mechanism truncates q11's
enumeration (the correctional custody items live in an unretrieved chunk
part) and q16 picks up an off topic Art. 63 tagalong that legitimately
shares "order" and "regulation", which is what a lexical support test
cannot distinguish and why citation precision is reported.

**Failure 6: not representational at all (q12).** "What has to happen
before charges can be referred?" misses under every retriever including
dense and hybrid, because the answer (the Art. 32 preliminary hearing) is
phrased in procedural language spread across half a dozen referral
provisions and the question shares no distinctive anchor with any of them.
Retrieval upgrades fix retrieval shaped failures and nothing else; knowing
which failures are which is what the eval is for.

## The main change: answer policy ablation (grounded versus permissive)

One thing changes: the answer policy in `rag/answer.py`. Retrieval is held
fixed at the BM25 baseline, both arms run the same deterministic extractive
code, and the permissive arm simply skips the support test, the punitive
article guard and the corpus boundary checks. Nothing is staged; the
permissive failures below are whatever the code produced. Full record in
`outputs/answer_policy_ablation.json`.

| metric | grounded | permissive |
| --- | --- | --- |
| refusals on answerable | 3 | 0 |
| controls refused correctly | 2/2 | 0/2 |
| answers citing a gold article | 11/13 | 11/16 |
| mean citation precision | 0.577 | 0.469 |

What the permissive policy actually does on the questions that matter:

| question | grounded | permissive |
| --- | --- | --- |
| q17 max confinement, AWOL over 30 days | Manual for Courts-Martial boundary refusal | quotes Art. 15's non-judicial punishment day limits, cited to Art. 15 |
| q18 RCM 707 speedy trial | out of corpus refusal | quotes Art. 6b victim rights enforcement text |
| q05 define commanding officer | refusal | quotes a usage of "commanding officer" from Art. 23 |
| q07 who is subject to the UCMJ | refusal | quotes Art. 146 review panel term limits |

q17 deserves the close look. The query's "30 days" matches real day limit
tables in Art. 15, so the retrieved context is full of true, plausible
looking numbers that belong to a different legal instrument entirely (non
judicial punishment limits, not court-martial confinement maxima). The
permissive policy quotes them with a confident citation. Note what this
layer cannot do: being extractive, it cannot produce the MCM's actual one
year figure, because that number exists nowhere in the corpus. A generative
model would know it from training and supply it, which reads better and is
epistemically worse. The extractive permissive failure mode is
misattribution with a real citation attached, which for a legal user may be
the most dangerous shape of all: every quoted sentence is genuine law,
verbatim, and the composition is still wrong.

This is the single most important thing the project demonstrates. The
grounded policy's guardrails (support threshold, punitive article guard,
boundary checks) are the only difference between "safely refuses on its
five retrieval misses" and "answers all eighteen questions, citing the
wrong legal instrument on the two questions engineered to punish it".

## Additional ablations

### Chunk size (120 / 180 / 300)

Chunk size moves little (hit@1 9, 9, 10), and the reason is worth stating:
most UCMJ provisions are shorter than one chunk at any tested size, so
re-chunking only redistributes the handful of long articles. At 300 words
q13 improves to rank 1: Art. 88 is a single chunk at every size, but its
competitors are long articles whose consolidation at 300 words changes
their length normalisation. The important negative result: the
q05/q07/q12 misses do not
move at any size, confirming term statistics rather than chunk boundaries
as the cause. Within a long article, chunking still decides which paragraph
you get; q02 and q11 above are the demonstration.

### Retriever (TF-IDF cosine)

Slightly worse than BM25 across the board (hit@3 drops to 9, q13 becomes a
miss), and the two sparse retrievers agree on every hard failure, which is
expected: both are bag of words scorers with the same blind spots. Swapping
between two sparse retrievers is a lateral move; the misses need a
different representation, not a different formula over the same terms.

### Tokenisation (light stemming), the found and fixed and paid for one

`--stem` folds deserter/desertion and fixes q04 at rank 1, exactly as
designed. The same folding demotes q10 (officer collisions promote
competing articles past Art. 25a, rank 2 to 4) and q16 (rank 1 to 4).
Aggregate hit@1 unchanged at 9, misses down from 5 to 4. A targeted gain, a
diffuse cost, netting out near zero: the honest shape of most retrieval
interventions, and the eval set is what tells you whether the trade is
worth it. The answer layer sidesteps the collision problem with plural only
folding, which fixes nothing morphological but breaks nothing either.

### Dense and hybrid retrieval, testing the failure analysis on its own prediction

Failures 2 and 3 above claim their misses cannot be fixed by any term level
intervention, only by a representation that knows synonymy. The dense run
(all-MiniLM-L6-v2) confirms it: q05 rank 1, q07 rank 2, q04 rank 1 without
stemming and without its collisions, hit@1 12/16, misses down to 2. The
prediction held.

What dense pays: on q08, "What rights does Article 31 give...", the
encoder's five nearest chunks are all Art. 6b, rights of the victim of an
offense. To the encoder, rights of a suspect and rights of a victim are
nearly the same sentence, and the one token that pins the question down,
"31", is just another subword. This is where the article number router
earns its place: it swaps Art. 31's best chunk into the last slot, so the
dense configuration answers correctly from rank 5 instead of missing
outright. The symmetry stands: dense wins when the user's words are not the
statute's words and loses when the user's words were the statute's words
all along, and in a legal corpus the exact identifier is often the entire
meaning of the question.

Hybrid RRF is the only configuration that retrieves both sides of both
comparative questions (all gold 14/16, q14 finally covered) and it recovers
q06 (Art. 1 sat high enough in both top 50 lists that fusion lifts it to
rank 2 when neither retriever alone had it in the top 5). Its consensus
bias also costs it q04, where dense alone was simply right and fusion let
mediocre chunks present in both lists outvote a single confident list.
Dense is the best single retriever on this set; hybrid trades its peaks for
fewer catastrophes; which one you want depends on whether your users ask by
concept or by article number. And q12 misses under all of them, because its
failure was never representational.

## Honest limitations

- The answer builder's support test is lexical. Two common shared tokens
  can admit an off topic sentence (q16's Art. 63 tagalong), and topical
  overlap can pass on the wrong article when retrieval missed the right one
  (q04, q12). Mean citation precision 0.577 is the honest price, reported
  rather than hidden. Entailment, not overlap, is the real fix, and that
  needs a model this repo deliberately does not have.
- Being extractive, answers can be stilted: statutory clauses stitched with
  citations, not prose. The upside is that faithfulness is structural
  (18/18 verbatim, independently re verified), not behavioral.
- Enumeration completion cannot cross into chunks retrieval never returned
  (q11 truncation), and paragraph selection within a provision inherits
  retrieval's near tie ordering (q02).
- `must_include` content coverage samples representative phrases; it is a
  floor on completeness, not a full semantic grade. The dataset validator
  guarantees every phrase is achievable from the gold articles.
- 18 questions surface failure modes; they are not enough for stable
  metrics, and single question flips move the aggregates. The dense margins
  (12 versus 9 at hit@1) are directionally clear but small sample.
- The dense results come from one small encoder on one question set. A
  legal domain encoder might behave differently on the identifier failure.
- The corpus is the statute only. A production military law assistant would
  need the MCM (punishment tables, Rules for Courts-Martial), and the two
  controls exist to prove the system knows where its corpus ends. This is
  an engineering demonstration, not legal advice.

## Reproducibility

- `./run_all.sh` regenerates every run file, the answer policy ablation,
  the answer scores and `outputs/summary.json`, and validates the dataset
  first. The dense and hybrid runs regenerate when sentence-transformers is
  installed and are skipped with a notice otherwise.
- Every output file carries its full configuration (retriever, chunking,
  k, policy, answer builder version, python version). No timestamps, so
  identical runs diff identically.
- Everything is deterministic: same inputs, same outputs, byte for byte.
  There is no API, no key, no network and no sampled randomness anywhere.
- `python3 -m pytest tests/ -q` covers chunk/provision integrity, retrieval
  regressions (the q16 stopword case, Article 31 by name), dataset schema,
  multi article credit, refusal behavior, boundary recognition, exact
  number preservation, verbatim extraction and determinism. 33 tests.
- `make setup`, `make test`, `make eval`, `make all` wrap the same
  commands.

## Repository layout

```
data/fetch_corpus.py     fetches the UCMJ from Cornell LII (provenance only,
                         the corpus is committed)
data/corpus/             198 provisions, one file each, + manifest.json
rag/chunk.py             ingestion and paragraph aware chunking
rag/index.py             BM25 (from scratch), TF-IDF cosine, light stemmer,
                         optional dense (MiniLM) and hybrid (RRF) retrievers
rag/answer.py            local grounded answer construction: sentence
                         splitting, scoring, support test, enumeration
                         completion, refusals and corpus boundaries
rag/pipeline.py          chunk -> index -> retrieve -> route -> answer
eval/questions.json      18 questions with structured gold fields
eval/rubric.md           the scoring rubric (automatic + interpretation)
eval/validate_dataset.py schema and groundedness checks for the dataset
eval/run_eval.py         runs the pipeline, computes retrieval metrics
eval/score_answers.py    automatic answer metrics from a run file
eval/run_answer_policy_ablation.py  grounded vs permissive, retrieval fixed
eval/build_summary.py    builds outputs/summary.json and the README tables
outputs/                 all runs, scores, the policy ablation and summary
tests/                   pytest suite (33 tests)
run_all.sh               reproduce everything
Makefile                 setup / test / eval / all
```
