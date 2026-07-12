"""
Indexing and retrieval scoring.

Two retrievers, both sparse and both fully inspectable. That is a deliberate
choice for this project: with sparse retrieval you can print the exact terms
that made a chunk win, which is what the evaluation is about. A dense embedding
model would need a multi hundred MB download and would turn every retrieval
decision into an opaque dot product. Sparse keeps the clone small and the
failures explainable.

  BM25    : the standard bag of words ranking function. Implemented here from
            scratch (about 40 lines) rather than pulled from a library so the
            term frequency saturation and length normalisation are visible.
  TF-IDF  : scikit-learn TfidfVectorizer + cosine. Used as the alternative
            retriever in the ablation.

Tokenisation is shared so the two are compared on equal footing.
"""
import re, math
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")
# light stopword list. kept small on purpose: legal text is terse and dropping
# too much ("right", "data", "subject") would erase the signal we retrieve on.
_STOP = set("the a an and or of to in for on by with as is are be this that "
            "shall may such which its it his her their our your at from into "
            "under upon any all each other where when who whom".split())


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c["text"] + " " + c["title"]) for c in chunks]
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
        q = tokenize(query)
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
        q = set(tokenize(query))
        tf = self.tf[chunk_idx]
        hits = [(t, self.idf.get(t, 0.0)) for t in q if t in tf]
        return sorted(hits, key=lambda x: x[1], reverse=True)


class TfidfCosine:
    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        corpus = [c["text"] + " " + c["title"] for c in chunks]
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.mat = self.vec.fit_transform(corpus)

    def search(self, query, k=5):
        from sklearn.metrics.pairwise import cosine_similarity
        qv = self.vec.transform([query])
        sims = cosine_similarity(qv, self.mat)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order]
