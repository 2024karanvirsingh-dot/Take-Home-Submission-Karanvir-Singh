"""
Indexing and retrieval scoring.

The default retrievers are sparse and fully inspectable. That is a deliberate
choice for this project: with sparse retrieval you can print the exact terms
that made a chunk win, which is what the evaluation is about. The dense and
hybrid retrievers are optional (they need sentence-transformers, which pulls
in torch) and exist to test whether the vocabulary gap failures the sparse
analysis predicts are dense-fixable actually are.

  BM25    : the standard bag of words ranking function. Implemented here from
            scratch (about 40 lines) rather than pulled from a library so the
            term frequency saturation and length normalisation are visible.
            This is the default.
  TF-IDF  : scikit-learn TfidfVectorizer + cosine. Used as the alternative
            sparse retriever in the ablation.
  Dense   : all-MiniLM-L6-v2 sentence embeddings + cosine. Optional.
  Hybrid  : reciprocal rank fusion of BM25 and dense. Optional.

Tokenisation is shared between the sparse pair so they are compared on equal
footing.
"""
import re, math
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")
# light stopword list. kept small on purpose: legal text is terse and dropping
# too much ("right", "data", "subject") would erase the signal we retrieve on.
_STOP = set("the a an and or of to in for on by with as is are be this that "
            "shall may such which its it his her their our your at from into "
            "under upon any all each other where when who whom".split())


def _light_stem(t):
    """Crude suffix stripper, not Porter. The goal is only that morphological
    variants of the same legal root land on the same token on both the query
    and document side: deserter / desertion / deserts -> desert,
    disobeying / disobeys -> disobey, charges -> charge. It will mangle some
    words, but it mangles them identically in queries and documents, which is
    all a bag of words matcher needs."""
    for suf in ("ations", "ation", "ions", "ion", "ings", "ing", "ers", "er",
                "ies", "es", "s", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def tokenize(text, stem=False):
    toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]
    if stem:
        toks = [_light_stem(t) for t in toks]
    return toks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75, stem=False):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.stem = stem
        self.docs = [tokenize(c["text"] + " " + c["title"], stem=stem) for c in chunks]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in self.docs]
        df = defaultdict(int)
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        # standard bm25 idf with the +0.5 smoothing
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query):
        q = tokenize(query, stem=self.stem)
        scores = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores[i] = s
        return scores

    def search(self, query, k=5):
        scores = self.score(query)
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in order]

    def matched_terms(self, query, chunk_idx):
        """Which query terms actually fired in this chunk, ranked by contribution."""
        q = set(tokenize(query, stem=self.stem))
        tf = self.tf[chunk_idx]
        hits = [(t, self.idf.get(t, 0.0)) for t in q if t in tf]
        return sorted(hits, key=lambda x: x[1], reverse=True)


class DenseEmbed:
    """Dense retrieval with a small pretrained sentence encoder
    (all-MiniLM-L6-v2, about 90 MB). This is the one retriever here that is
    not inspectable: there are no terms to print, only a cosine over learned
    vectors. It exists to test the central claim of the failure analysis,
    that the vocabulary gap misses (UCMJ, panel, general article) need a
    representation that knows synonymy, which no term statistic can supply.

    Kept out of requirements.txt on purpose; it pulls in torch. Install with
    `pip install sentence-transformers` to run the dense ablation."""

    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks, stem=False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit(
                "dense retrieval needs sentence-transformers. "
                "pip install sentence-transformers (about 90 MB model on first run)")
        # stem is accepted for interface parity but ignored: the encoder has
        # its own subword tokenizer and stemming would only hurt it.
        self.chunks = chunks
        self.model = SentenceTransformer(self.MODEL)
        corpus = [c["title"] + ". " + c["text"] for c in chunks]
        self.mat = self.model.encode(corpus, normalize_embeddings=True,
                                     show_progress_bar=False)

    def search(self, query, k=5):
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.mat @ qv
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order]


class HybridRRF:
    """Reciprocal rank fusion of BM25 and dense. No score normalisation
    gymnastics, just 1/(60+rank) summed over both lists, which is the boring
    standard because it works. Tests whether fusing the two recovers the
    dense wins without giving back the sparse ones."""

    def __init__(self, chunks, stem=False, pool=50):
        self.chunks = chunks
        self.pool = pool
        self.bm25 = BM25(chunks, stem=stem)
        self.dense = DenseEmbed(chunks)

    def search(self, query, k=5):
        fused = defaultdict(float)
        for ranked in (self.bm25.search(query, k=self.pool),
                       self.dense.search(query, k=self.pool)):
            for rank, (chunk, _) in enumerate(ranked):
                fused[chunk["id"]] += 1.0 / (60 + rank + 1)
        by_id = {c["id"]: c for c in self.chunks}
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(by_id[cid], fused[cid]) for cid in order]


class TfidfCosine:
    def __init__(self, chunks, stem=False):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        corpus = [c["text"] + " " + c["title"] for c in chunks]
        tok = (lambda t: tokenize(t, stem=True)) if stem else tokenize
        self.vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None)
        self.mat = self.vec.fit_transform(corpus)

    def search(self, query, k=5):
        from sklearn.metrics.pairwise import cosine_similarity
        qv = self.vec.transform([query])
        sims = cosine_similarity(qv, self.mat)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order]
