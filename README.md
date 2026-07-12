# RAG over the GDPR, with an honest evaluation

A small retrieval augmented generation pipeline over the General Data Protection
Regulation, plus a writeup that scores its own answers and digs into where and
why retrieval breaks. The pipeline itself is deliberately simple. The point of
this repo is the evaluation, so most of this README is about the failures.

## Why this corpus

The corpus is the GDPR, split into its two natural document types:

* **99 Articles**, the binding operative text of the Regulation.
* **173 Recitals**, the non binding preamble that explains and interprets the
  Articles.

I picked this pairing on purpose. Articles and Recitals cover the same topics in
different registers: Recitals are written in fuller, more natural language while
Articles are terse and enumerated. That overlap is where a lot of interesting
retrieval behaviour lives, because a recital often matches a plain English
question better than the article that actually answers it. A legal user wants the
binding article. The retriever does not know that by default. Several of the
failures below come straight out of this tension.

Everything is fetched from gdpr-info.eu by `data/fetch_corpus.py` and committed
under `data/corpus/` so a clean clone needs no network.

## Quick start

```bash
git clone <this repo>
cd Take-Home-Submission-Karanvir-Singh
pip install -r requirements.txt      # scikit-learn + numpy is enough for retrieval

# ask one question
python -m rag.pipeline "within how long must a data breach be reported?"

# reproduce every run in this README (no API key needed for retrieval or scoring)
./run_all.sh
```

Retrieval, indexing and scoring run with no API key. Generation uses the
Anthropic API when `ANTHROPIC_API_KEY` is set (model overridable with
`RAG_MODEL`); with no key it falls back to returning the top passage verbatim so
the pipeline still runs end to end. See "How the recorded answers were produced"
below for how the graded answers in `outputs/` were generated.

## How the pipeline works

Four stages, each in its own module, nothing hidden behind a framework.

**Ingest and chunk** (`rag/chunk.py`). Each provision is one file. Most are short
enough to be a single chunk. The handful of long ones (Art. 4 definitions, Art. 6
lawfulness, Art. 9, Art. 83 fines) are split on paragraph boundaries into windows
of about 180 words with 40 words of overlap. Chunking never crosses a provision
boundary, so every chunk maps to exactly one Article or Recital and every
citation is honest. This gives 579 chunks at the default size.

**Index and retrieve** (`rag/index.py`). Two sparse retrievers, both fully
inspectable:

* **BM25**, implemented from scratch (about 40 lines) rather than pulled from a
  library, so the term frequency saturation (`k1`) and length normalisation (`b`)
  are visible and so I can print the exact terms that made a chunk win. This is
  the default.
* **TF-IDF cosine** via scikit-learn, used as the alternative in the ablation.

I chose sparse retrieval on purpose. A dense embedding model would mean a several
hundred MB download and would turn every retrieval decision into an opaque dot
product. Sparse keeps the clone small and, more importantly for this exercise,
keeps every retrieval decision explainable: I can tell you which query term
pulled which chunk. The tradeoff is that sparse retrieval has no idea that
"erasure" and "deletion" are the same thing, and you will see that cost in the
failures.

**Generate** (`rag/generate.py`). The prompt is strict: answer only from the
retrieved context, cite the provision, prefer binding Articles over Recitals and
say when you are leaning on a Recital, and if the answer is not in the context
say so instead of guessing. In a legal setting the most damaging error is a
fluent answer that smooths over a retrieval gap, so "not in the context" is a
first class allowed answer.

## The questions and the rubric

18 questions in `eval/questions.json`, spread across eight types so different
failure modes get exercised: single provision lookups, definitions, exact number
and deadline questions, enumerations, conditional tests, near duplicate
comparisons, and two controls (one where the premise is a myth not in the text,
one that asks about a different law entirely).

Full rubric in `eval/rubric.md`. In short, each answer gets three scores, kept
separate on purpose because retrieval and generation fail for different reasons:

* **Retrieval (0 to 2)**: was the correct provision in the top k, and near the top.
* **Correctness (0 to 2)**: is the answer right and complete, with the right
  citation. For number questions the number has to be exact.
* **Faithfulness (pass or fail)**: is every claim actually supported by a
  retrieved passage. This is a gate, not a gradient.

## Results at a glance

Baseline: BM25, chunk size 180, top k = 5, strict prompt.

| Metric | Score |
| --- | --- |
| Retrieval (gold provision in top k) | 19/32 across the 16 scored questions, avg 0.59 |
| Correctness | 25/36, avg 0.69 |
| Faithfulness | 18/18 pass |
| Clean wins (retrieval 2 and correctness 2) | 8 (q02, q03, q04, q12, q13, q14, q15, q16) |
| Retrieval misses (gold not in top k) | 5 (q05, q06, q07, q08, q09) |

Faithfulness is 18/18 only because the strict prompt turns every retrieval miss
into a refusal rather than a fabrication. That number is doing a lot of work and
the prompt ablation below shows how fast it collapses when the guardrail is
removed. Full per question answers, scores and notes are in
`outputs/graded_answers.json`.

## Three cases where it worked, and why

**q02, breach notification deadline.** "Within what time must a controller notify
the supervisory authority of a personal data breach?" Top result is Art. 33 part
1, which contains the exact clause "not later than 72 hours after having become
aware of it." This works because the query terms "notify", "breach" and
"supervisory authority" are all mid to high IDF (they do not appear in most
provisions) and they all co-occur in one short chunk, so BM25 concentrates the
score there. The answer states 72 hours and the reasons for delay rule, cited to
Art. 33(1). Retrieval 2, correctness 2.

**q13, right to object.** "When can a data subject object to processing?" Art. 21
parts 1 to 3 take the top three ranks. "Object" is a strong, relatively rare
signal in this corpus and it is concentrated in Art. 21, so the retriever locks
onto the right provision and even orders the sub parts sensibly. The answer keeps
both halves that matter: the situational objection under 21(1) and the absolute
direct marketing objection under 21(2). Retrieval 2, correctness 2.

**q15, when to appoint a DPO.** "When must an organization appoint a Data
Protection Officer?" The rank 1 chunk is Art. 37 part 1, which holds all three
trigger conditions in one place. This is the ideal case for chunking: the whole
answer is one self contained enumerated list that fits inside a single chunk, so
there is no partial retrieval problem. The three conditions come back complete.
Retrieval 2, correctness 2.

The pattern in the wins: the answer is short, self contained, and keyed on a
distinctive term that lives in one provision. Sparse retrieval is very good at
exactly this.

## Three (plus) cases where it failed, and why

This is the part that matters. The failures cluster into three mechanisms.

**Failure 1: common term definitions die on low IDF (q05, q06, q07).**
"How does the GDPR define personal data?" The gold provision, Art. 4(1), is never
retrieved. Top results are about professional secrecy of authority staff (Art.
54), security of processing (Art. 32) and Board opinions (Art. 64). The reason is
mechanical and, once you see it, obvious: "personal data" is the single most
frequent phrase in the entire corpus, so its inverse document frequency is close
to zero. The word that should point straight at the definition carries almost no
retrieval signal, because it points at everything. The query "define personal
data" has no distinctive term to grab onto. The same thing sinks "processing"
(q06) and "controller versus processor" (q07): these are the most ubiquitous
terms in the Regulation, so the provisions that define them are invisible to a
bag of words scorer. Dense embeddings or a field weighted index that boosts the
Article 4 "Definitions" title would fix this; plain BM25 cannot.

**Failure 2: recitals out-rank the binding articles (q08, q09, and partially q01,
q03).** "How long does a controller have to respond to an access request?" The
binding deadline is Art. 12(3). It is not retrieved. Instead the top hit is
Recital 59, which says the controller should respond "at the latest within one
month" in flowing prose. The recital wins because it phrases the idea the way the
question does, while Art. 12 buries the deadline among procedural clauses. Same
story for the lawful bases (q09): Recital 39 paraphrases lawfulness in rich
language and out-ranks the terse six item list in Art. 6(1), which never surfaces.
This is the article versus recital tension I built the corpus to expose. It is a
real world problem: the system keeps handing back interpretive, non binding text
when the user needs the operative rule. In q08 the recital at least carries the
right number (one month), so the answer is partly rescued, but it drops the two
month extension that only Art. 12(3) contains, and it is sourced to non binding
text.

**Failure 3: right article, wrong paragraph (q01, q10).** This is the subtlest
one and the one I would most want a reviewer to see. "What is the maximum fine?"
BM25 does retrieve Art. 83, so a naive "is the gold article in the top k?" check
scores it a hit. But it retrieves parts 1 to 3 of Art. 83, the chunks about how a
fine is calculated, and not part 4/5, where the actual ceiling of 20 million euro
or 4% of worldwide turnover lives. The answer bearing paragraph is absent. A
faithful model therefore cannot state the number even though the number is sitting
in the corpus, and it correctly says so. A model answering from memory would
confidently emit "20 million or 4%" with an Art. 83 citation it never actually
read. The lesson: document level retrieval metrics lie here. What matters is
whether the specific answer bearing chunk was retrieved, not whether some chunk of
the right provision was. Q10 (special category conditions) fails the same way: it
retrieves the tail of Art. 9 rather than the enumerated 9(2) exceptions.

**Bonus, the near duplicate trap (q11).** "Difference between erasure and
restriction of processing?" The query explicitly names both rights. BM25 commits
hard to Art. 18 (restriction) and its neighbours and never surfaces Art. 17
(erasure), so only one side of a two sided question is grounded. Comparative
questions are structurally hard for a top k retriever that has no notion that the
query is asking for a contrast.

## Changing one thing: two ablations

### Ablation A, chunk size (retrieval strategy)

I reran the whole eval at chunk sizes 120, 180 (baseline) and 300 words. Run
files are in `outputs/run_bm25_cw*.json`.

| chunk size | gold in top 1 | gold in top 3 | missed | chunks |
| --- | --- | --- | --- | --- |
| 120 | 10/16 | 11/16 | 5 | 1053 |
| 180 (baseline) | 9/16 | 11/16 | 5 | 579 |
| 300 | 8/16 | 11/16 | 3 | 362 |

There is a clean precision versus recall tradeoff. Bigger chunks **recover buried
provisions**: q08 (access deadline) goes from a miss to rank 5, and q09 (lawful
bases) from a miss to rank 4, because a 300 word window is large enough to pull
the operative article text in alongside whatever matched. Total misses drop from
5 to 3. But bigger chunks **hurt precision at the top**: q11 slips from rank 1 to
rank 2 and q16 from rank 1 to rank 2, because a larger chunk mixes the answer with
neighbouring text and dilutes the exact match. Smaller chunks (120) do the
opposite, nudging top 1 up to 10 but recovering nothing that was missed.

Before and after, concretely, q08 "how long to respond to an access request":

* chunk 180: Art. 12(3) not in top 5, answer leans on Recital 59, no extension.
* chunk 300: Art. 12 now reachable in the top k, the one month plus two month
  extension can be answered from the binding article.

The most useful result from this ablation is a negative one. The three definition
failures (q05, q06, q07) do not move at any chunk size, and do not move under the
TF-IDF retriever either. That invariance is the proof that their cause is term
frequency, not chunk boundaries. Changing chunk size cannot fix a problem whose
root is that the query term has no IDF signal. The fix for those is a different
lever: field weighting on the provision title, a dense retriever, or query
expansion. I would not have known that for sure without running the sweep and
watching those three refuse to budge.

(The TF-IDF cosine run, `outputs/run_tfidf_baseline.json`, lands very close to
BM25: 9/16 top 1, 12/16 top 3. It recovers q08 to rank 3 but demotes q11 to rank
3. Same family of behaviour, which is expected since both are bag of words.)

### Ablation B, the prompt

Here the one thing changed is the system prompt, retrieval held fixed, on the
three questions that had a retrieval miss. Full text in
`outputs/prompt_ablation.json`.

| question | strict prompt | permissive prompt |
| --- | --- | --- |
| q05 define personal data | "not in the context" (faithful) | full Art. 4(1) definition from memory (unsupported) |
| q09 lawful bases | "not in the context" (faithful) | all six bases enumerated from memory (unsupported) |
| q18 CCPA definition | "corpus is GDPR only" (faithful) | a confident CCPA definition, for a law not in the corpus |

Every permissive answer is fluent and, for q05 and q09, factually correct in the
real world. All three fail faithfulness, because not one token of them came from
the retrieved context. The model reconstructed them from training and stapled on
a citation it never read. q18 is the dangerous one: the corpus contains no CCPA at
all, yet the permissive prompt produces an authoritative CCPA definition with no
signal to the user that it is ungrounded.

This is the single most important thing I learned building it. The strict prompt
is not a nicety. It is the only reason the baseline scores 18/18 on faithfulness,
and it works by converting retrieval failures into visible refusals instead of
invisible fabrications. Better retrieval reduces how often the refusal fires;
the prompt is what makes the residual failures safe.

## How the recorded answers were produced

This environment had no `ANTHROPIC_API_KEY`, so the graded answers in
`outputs/graded_answers.json` were generated with Claude acting as the generation
backend over the exact passages the pipeline retrieved (the same contexts
`rag/generate.py` builds, dumped per question). Retrieval, ranking, matched term
attribution and scoring are all produced by the committed code and are fully
reproducible with `./run_all.sh`. Setting `ANTHROPIC_API_KEY` makes
`eval/run_eval.py` generate the answers automatically through the API instead. I
kept generation and retrieval cleanly separated precisely so that the part a
reviewer most needs to trust, the retrieval behaviour and its scoring, does not
depend on any key or any model.

## Honest limitations

* Scoring is mine, applied by hand from the rubric. An LLM judge or a second
  annotator would make it less subjective, and a few of the correctness 1 versus 2
  calls are genuinely arguable.
* 18 questions is enough to surface failure modes, not enough for stable metrics.
  The per type counts are small.
* Sparse only. The definition failures would very likely be fixed by a dense
  retriever, and adding one (with a hybrid score) is the first thing I would do
  next. I chose not to here so the clone stays small and every retrieval decision
  stays explainable, which served the evaluation goal better.
* No reranker and no query expansion, both of which would help the near duplicate
  and article versus recital cases.

## Layout

```
data/fetch_corpus.py     fetches GDPR articles + recitals (corpus is committed)
data/corpus/             272 provisions, one file each, + manifest.json
rag/chunk.py             ingestion and paragraph aware chunking
rag/index.py             BM25 (from scratch) and TF-IDF cosine
rag/generate.py          strict and permissive grounding prompts
rag/pipeline.py          chunk -> index -> retrieve -> generate
eval/questions.json      18 questions, 8 types, gold answers and citations
eval/rubric.md           the scoring rubric
eval/run_eval.py         runs the pipeline, records retrieval rank + matched terms
outputs/graded_answers.json   per question answers with hand scores and notes
outputs/prompt_ablation.json  strict vs permissive prompt on the retrieval misses
outputs/run_*.json       raw runs for the chunk size and retriever ablations
run_all.sh               reproduce every run
```
