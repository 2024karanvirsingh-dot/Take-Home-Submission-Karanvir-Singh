"""
Local grounded answer construction.

There is no language model here. The answer layer is a deterministic,
citation constrained extractive synthesizer: it splits the retrieved chunks
into statutory sentences, scores each sentence against the question with
transparent lexical features, selects a small set of non duplicative
supporting sentences, and either composes them into a cited answer or
refuses. Every sentence in an answer is copied verbatim from a retrieved
chunk, so faithfulness to the retrieved context holds by construction and
the interesting failures move to selection and citation, which is where the
evaluation looks for them.

Two policies, compared in the answer policy ablation:

  grounded    : the default. Requires each supporting sentence to clear a
                support test before it may appear in an answer, refuses when
                nothing clears it, and recognises two corpus boundaries (the
                Manual for Courts-Martial punishment tables and the Rules
                for Courts-Martial, neither of which is in this corpus).
  permissive  : always answers with the best scoring sentences, no support
                test, no boundary checks. Exists to show what the guardrails
                are worth; its failures are produced by the same code path,
                not fabricated by hand.

Everything is pure functions over the inputs. Same question, same retrieved
chunks, same policy: same answer, byte for byte.
"""
import re
from collections import defaultdict
from .index import tokenize

ANSWER_BUILDER_VERSION = "local-v1"

# Standardised refusal and boundary texts. score_answers.py detects these by
# the structured flags on the result, not by string matching, but keeping the
# wording fixed makes the output files diffable.
REFUSAL_INSUFFICIENT = (
    "The retrieved provisions do not answer this question. Rather than "
    "construct an answer from weakly matching passages, the system declines."
)
BOUNDARY_MCM = (
    "The statute does not contain this figure. The UCMJ typically provides "
    "punishment 'as a court-martial may direct'; the specific maximum "
    "punishments are set in the Manual for Courts-Martial punishment tables, "
    "which are not part of this corpus."
)
BOUNDARY_OUT_OF_CORPUS = (
    "This question asks about material from the Manual for Courts-Martial "
    "(which includes the Rules for Courts-Martial), not the UCMJ statute. "
    "This corpus contains only the statutory text of 10 U.S.C. chapter 47, "
    "so it cannot answer this."
)

_CMD = "as a court-martial may direct"
_NUM_WORDS = set("one two three four five six seven eight nine ten eleven "
                 "twelve thirty sixty ninety".split())
_OPERATIVE = ("shall", "may not", "must", "prohibited", "punished",
              "is guilty", "no person may")
_DEF_MARKERS = re.compile(r"\bmeans\b|\bincludes\b|\bthe term\b|"
                          r"\bthe following\b|\bare the following\b", re.I)
# Source and amendment history notes fetched along with the statutory text.
# They are not law and never belong in an answer. Every provision ends with a
# parenthetical history block that starts with a date or 'Added'/'Pub. L.',
# so we cut from that marker to the end of the chunk before splitting. The
# per sentence check is a second line of defence: the statute body spells
# out cross references ('section 825 of this title'), so a section sign or a
# Statutes at Large cite is a reliable history fingerprint.
_HISTORY_START = re.compile(
    r"\(\s?(Aug|Sept|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|June|July)\.? \d"
    r"|\(\s?(Added|Amended|Pub\. L\.)")
_HISTORY = re.compile(r"^\(?(Aug\.|Added|Amended|Pub\. L\.)|§|Stat\.")

# The one vocabulary normalisation the answer layer performs: the statute
# never calls itself "the UCMJ", it says "this chapter". Retrieval
# deliberately does NOT get this mapping (the vocabulary gap it causes is a
# core finding of the eval); the answer layer gets it only for ranking
# sentences that retrieval has already surfaced, and the token it introduces
# is excluded from the support test below because "this chapter" is
# boilerplate in 97 of 198 provisions.
_QUERY_SYNONYMS = {"ucmj": "chapter"}
_BOILERPLATE_TOKENS = {"chapter"}


def _clean(text):
    return re.sub(r"\s+", " ", text.replace("“", '"').replace("”", '"')
                  .replace("’", "'").replace("‘", "'")).strip()


def _fold(tok):
    """Plural folding for the answer layer only: 'limitations' matches
    'limitation', 'offenses' matches 'offense'. This is not stemming (no
    suffix families, no officer/offices collisions); retrieval keeps its
    own optional stemmer so the tokenisation ablation stays clean."""
    return tok[:-1] if tok.endswith("s") and len(tok) > 3 else tok


def _fold_set(tokens):
    return {_fold(t) for t in tokens}


def split_sentences(text):
    """Split a chunk into statutory sentences and list items.

    The fetched corpus files contain hard line breaks in the middle of
    sentences (the source markup puts linked entities on their own lines),
    so newlines cannot be trusted as boundaries. The text is reflowed into
    one stream first, then split at sentence ends and at the '; (2)' style
    joints between enumerated items, so each list item can be scored on its
    own. Lowercase continuations stay attached to their clause. Amendment
    history notes are dropped: they are not law and never belong in an
    answer.
    """
    flowed = _clean(text.replace("\n", " "))
    m = _HISTORY_START.search(flowed)
    if m:
        flowed = flowed[: m.start()]
    parts = re.split(r"(?<=[.;])\s+(?=\(?[A-Z0-9(])", flowed)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # glue fragments that are too short to stand alone (citation
        # numbers, subsection markers) back onto the previous piece
        if buf and len(p.split()) < 5:
            buf += " " + p
            continue
        if buf:
            out.append(buf)
        buf = p
    if buf:
        out.append(buf)
    return [s for s in out if len(s.split()) >= 4 and not _HISTORY.search(s)]


def _query_info(question, qtype=None):
    """Lightweight question analysis. During evaluation the dataset's type
    field is passed in; interactively we fall back to surface heuristics."""
    ql = question.lower()
    toks = [_fold(_QUERY_SYNONYMS.get(t, t)) for t in tokenize(question)]
    named = re.findall(r"article\s+(\d+[a-z]?)\b", ql)
    definitional = bool(re.search(
        r"\b(define|definition|meaning|who counts as|who is considered|"
        r"what is an?|who is subject)\b", ql))
    enumeration = bool(re.search(r"\b(what are|list|types of|categories)\b", ql))
    if qtype in ("definition",):
        definitional = True
    if qtype in ("enumeration",):
        enumeration = True
    punishment_amount = bool(
        re.search(r"\b(maximum|minimum|how long|how much)\b", ql)
        and re.search(r"\b(confinement|punishment|imprisonment|sentence|penalty)\b", ql))
    wants_number = bool(re.search(r"\b(how many|how long|maximum|minimum|number)\b", ql)
                        or re.search(r"\d", ql)
                        or qtype in ("numeric", "limitations"))
    comparative = bool((qtype or "").startswith("comparative")
                       or "difference between" in ql)
    return {
        "comparative": comparative,
        "tokens": set(toks),
        "core_tokens": {t for t in toks
                        if t not in _BOILERPLATE_TOKENS and not _is_numeric(t)},
        "named_articles": set(named),
        "definitional": definitional,
        "enumeration": enumeration,
        "punishment_amount": punishment_amount,
        "wants_number": wants_number,
    }


def _is_numeric(tok):
    return bool(re.fullmatch(r"\d+[a-z]?", tok)) or tok in _NUM_WORDS


def _has_number(sentence):
    """Does the sentence state a quantity? List markers like '(1)' and
    subsection cross references are structure, not quantities, so they are
    stripped before looking."""
    stripped = re.sub(r"\([a-zA-Z0-9]{1,3}\)", " ", sentence)
    stripped = re.sub(r"\bsections? \d+[a-z]?\b", " ", stripped)
    low = stripped.lower()
    return bool(re.search(r"\b\d+\b", stripped)) or any(
        re.search(r"\b%s\b" % w, low) for w in _NUM_WORDS)


def detect_corpus_boundary(question):
    """True when the question is about the MCM or the Rules for
    Courts-Martial, which are outside this corpus by construction."""
    ql = question.lower()
    return bool(re.search(
        r"rule[s]? for courts?-martial|\brcm\b|manual for courts?-martial|"
        r"\bmcm\b|punishment table", ql))


def _candidates(retrieved):
    """Flatten retrieved chunks into scored sentence candidates, keeping
    chunk rank and in chunk position for deterministic tie breaking."""
    cands = []
    for rank, (chunk, _score) in enumerate(retrieved):
        for pos, sent in enumerate(split_sentences(chunk["text"])):
            cands.append({
                "text": sent,
                "tokens": _fold_set(tokenize(sent)),
                "article": chunk["number"],
                "chunk_type": chunk["type"],
                "title_tokens": _fold_set(tokenize(chunk["title"])),
                "chunk_rank": rank,
                "pos": pos,
                "chunk_id": chunk["id"],
                "chunk_part": int(chunk["id"].rsplit("::", 1)[1]),
            })
    return cands


def _score(cand, q, n_containing, n_cands, policy):
    """Transparent additive relevance score. Every feature is a thing you
    could check by eye against the sentence."""
    shared = q["tokens"] & (cand["tokens"] | cand["title_tokens"])
    # idf over the candidate pool, so a query term found in one sentence
    # outweighs a query term found in fifty. Known boilerplate ('chapter',
    # introduced by the ucmj synonym) gets a fixed small weight instead:
    # it can be locally rare in a pool while being corpus wide noise, and
    # pool idf would reward exactly the wrong sentences for it.
    import math
    s = sum(0.3 if t in _BOILERPLATE_TOKENS
            else math.log(1 + n_cands / max(1, n_containing[t]))
            for t in shared)
    low = cand["text"].lower()
    if q["named_articles"] and cand["article"] in q["named_articles"]:
        s += 3.0
    if q["wants_number"] and _has_number(cand["text"]):
        s += 2.0
    if any(op in low for op in _OPERATIVE):
        s += 0.5
    if (q["definitional"] or q["enumeration"]) and _DEF_MARKERS.search(cand["text"]):
        s += 2.0
    if policy == "grounded" and not _core_shared(cand, q):
        # weak match: shares numbers or boilerplate only. Permissive skips
        # this penalty, which is most of what the ablation measures.
        s *= 0.25
    return s


def _core_shared(cand, q):
    """Distinct shared tokens that actually establish topicality: not bare
    numbers (a matching '30' says nothing about what the sentence is about)
    and not corpus wide boilerplate ('chapter'). Title tokens count only
    when at least two of them match; a single shared title word ('military'
    in 'Military judge') is how off topic chunks sneak weak sentences past
    the support test."""
    text_shared = {t for t in q["core_tokens"] & cand["tokens"]
                   if not _is_numeric(t)}
    title_shared = {t for t in q["core_tokens"] & cand["title_tokens"]
                    if not _is_numeric(t)}
    # a title match counts when it covers most of the title: two words of
    # 'Statute of limitations' is a match, and so is all of 'Desertion',
    # but the lone word 'military' out of 'Military judge of a general or
    # special court-martial' is not
    if len(title_shared) >= min(2, len(cand["title_tokens"])) and title_shared:
        return text_shared | title_shared
    return text_shared


def is_context_sufficient(cand, q, n_containing=None):
    """The grounded support test, per candidate sentence. A sentence may
    appear in an answer only if one of these holds:
      1. it shares at least two distinct topical tokens with the question,
      2. the question names an article and the sentence is from it,
      3. the question is definitional and the sentence is a definitional
         clause (means / includes / the following) sharing at least one
         topical token, or
      4. it shares a single rare anchor: a topical token found in at most
         two candidate sentences, or a matching title ('Desertion' for a
         desertion question). One rare word is real evidence; one common
         word is noise.
    Two type specific tightenings. Punishment amount questions additionally
    require the sentence to come from a punitive article: procedural
    provisions like Art. 15 are full of day limits that belong to a
    different legal instrument, and quoting them for an offence question is
    exactly the error the eval punishes. Definition questions require an
    actual definitional clause (means / includes / the following): the words
    being defined ('commanding officer') are by nature the most ubiquitous
    words in the corpus, so plain token overlap would happily quote a usage
    of the term from any article instead of its definition."""
    # a sentence from an article the question names by number is responsive
    # even with zero lexical overlap: the statute never numbers itself in
    # its own text, so overlap physically cannot exist for these queries
    if q["named_articles"] and cand["article"] in q["named_articles"]:
        return True
    core = _core_shared(cand, q)
    if not core:
        return False
    if q["punishment_amount"] and cand["chunk_type"] != "punitive":
        return False
    if q["definitional"] or q["enumeration"]:
        return bool(_DEF_MARKERS.search(cand["text"]))
    if len(core) >= 2:
        return True
    title_shared = {t for t in q["core_tokens"] & cand["title_tokens"]
                    if not _is_numeric(t)}
    if title_shared and len(title_shared) >= min(2, len(cand["title_tokens"])):
        return True
    if n_containing and any(n_containing.get(t, 0) <= 2 for t in core):
        return True
    return False


def _dedupe(selected):
    kept = []
    for c in selected:
        dup = False
        for k in kept:
            inter = len(c["tokens"] & k["tokens"])
            union = len(c["tokens"] | k["tokens"]) or 1
            if inter / union > 0.7 or c["text"] in k["text"] or k["text"] in c["text"]:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


_LIST_ITEM = re.compile(r"^\((\d+[a-zA-Z]?|[A-Z]|i{1,3}|iv|v)\)\s")


def _expand_lists(selected, all_cands):
    """Complete enumerations. A selected sentence that ends mid list (';',
    em dash, colon) or that introduces one ('the following') pulls the
    subsequent list items back in, in order, following the article across
    chunk part boundaries when a long list was split by the chunker.
    Statutory meaning lives in complete lists; extracting prong (1) of
    Art. 92 without prongs (2) and (3) would silently narrow the law.
    Expansion stops at the first sentence that is not a list item, so it
    never wanders into the next subsection."""
    by_article = defaultdict(list)
    for x in all_cands:
        by_article[x["article"]].append(x)
    for arts in by_article.values():
        arts.sort(key=lambda x: (x["chunk_part"], x["pos"]))
    out = list(selected)
    have = {(c["chunk_id"], c["pos"]) for c in out}
    texts = {c["text"] for c in out}
    for c in selected:
        tail = c["text"].rstrip()
        if not (tail.endswith((";", ":")) or tail.endswith("—")
                or "the following" in tail.lower()):
            continue
        arts = by_article[c["article"]]
        start = next(i for i, x in enumerate(arts)
                     if (x["chunk_id"], x["pos"]) == (c["chunk_id"], c["pos"]))
        added = 0
        for it in arts[start + 1:]:
            if not _LIST_ITEM.match(it["text"]) or added >= 14:
                break
            # chunk overlap can repeat an item across parts; skip repeats
            if it["text"] not in texts and (it["chunk_id"], it["pos"]) not in have:
                out.append(it)
                have.add((it["chunk_id"], it["pos"]))
                texts.add(it["text"])
                added += 1
    return out


def select_supporting_sentences(question, retrieved, policy="grounded", qtype=None):
    """Rank candidate sentences and pick the supporting set. Returns
    (selected, query_info, all_candidates)."""
    q = _query_info(question, qtype)
    cands = _candidates(retrieved)
    n_containing = defaultdict(int)
    for c in cands:
        for t in q["tokens"] & c["tokens"]:
            n_containing[t] += 1
    for c in cands:
        c["score"] = round(_score(c, q, n_containing, len(cands), policy), 4)
    # deterministic order: score, then retrieval rank, then position in chunk
    ranked = sorted(cands, key=lambda c: (-c["score"], c["chunk_rank"], c["pos"]))
    if policy == "grounded":
        pool = [c for c in ranked if is_context_sufficient(c, q, n_containing)]
    else:
        # permissive always attempts an answer from whatever was retrieved,
        # zero overlap included; that is the point of the ablation
        pool = list(ranked)
    # relative pruning: sentences far below the best match are tagalongs,
    # and every extra sentence is an extra citation the answer must defend
    if pool:
        cutoff = 0.5 * pool[0]["score"]
        pool = [c for c in pool if c["score"] >= cutoff]
    limit = 4 if (q["enumeration"] or q["comparative"]) else 3
    deduped = _dedupe(pool[: limit * 3])
    if q["comparative"]:
        # comparative questions live or die on covering both sides, so no
        # single article may take more than two of the answer slots
        selected, per_art = [], defaultdict(int)
        for c in deduped:
            if per_art[c["article"]] < 2 and len(selected) < limit:
                selected.append(c)
                per_art[c["article"]] += 1
    else:
        selected = deduped[:limit]
    if selected:
        selected = _expand_lists(selected, cands)
    return selected, q, cands


def _compose(selected):
    """Order the chosen sentences by retrieval rank and position, group by
    article, and attach one citation per article group."""
    ordered = sorted(selected, key=lambda c: (c["chunk_rank"], c["pos"]))
    parts, cur_art, buf = [], None, []
    for c in ordered:
        if c["article"] != cur_art and buf:
            parts.append("%s (Art. %s, UCMJ)" % (" ".join(buf), cur_art))
            buf = []
        cur_art = c["article"]
        buf.append(c["text"])
    if buf:
        parts.append("%s (Art. %s, UCMJ)" % (" ".join(buf), cur_art))
    return "\n\n".join(parts)


def construct_answer(question, retrieved, policy="grounded", qtype=None):
    """Build a grounded, cited answer from the retrieved chunks, or refuse.

    Returns a dict:
      text            the answer (or refusal) text
      refused         True when no answer was constructed
      boundary        True when the refusal names a corpus boundary
      cited_articles  articles cited in the answer, in order
      sentences       the supporting sentences with article and score
      policy          which policy produced this
    """
    result = {"policy": policy, "refused": False, "boundary": False,
              "cited_articles": [], "sentences": []}

    if policy == "grounded" and detect_corpus_boundary(question):
        result.update(text=BOUNDARY_OUT_OF_CORPUS, refused=True, boundary=True)
        return result

    selected, q, _ = select_supporting_sentences(question, retrieved, policy, qtype)

    if not selected:
        if policy == "grounded" and q["punishment_amount"]:
            result.update(text=BOUNDARY_MCM, refused=True, boundary=True)
        else:
            result.update(text=REFUSAL_INSUFFICIENT, refused=True)
        return result

    text = _compose(selected)
    if policy == "grounded" and q["punishment_amount"] \
            and not any(_has_number(c["text"]) for c in selected) \
            and any(_CMD in c["text"].lower() for c in selected):
        text += "\n\n" + BOUNDARY_MCM
        result["boundary"] = True

    seen, cited = set(), []
    for c in sorted(selected, key=lambda c: (c["chunk_rank"], c["pos"])):
        if c["article"] not in seen:
            cited.append(c["article"])
            seen.add(c["article"])
    result.update(
        text=text, cited_articles=cited,
        sentences=[{"text": c["text"], "article": c["article"],
                    "chunk_id": c["chunk_id"], "score": c["score"]}
                   for c in selected])
    return result
