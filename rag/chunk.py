"""
Ingestion and chunking.

Each file in data/corpus is one legal provision (an Article or a Recital).
Most provisions are short enough to be a single chunk. A few (Art. 4 with the
definitions, Art. 6 lawfulness, Art. 9, Art. 83 fines) are long, so we split
them on paragraph boundaries into windows of roughly `chunk_words` words with
`overlap_words` of overlap so a definition that straddles a split still shows
up whole in one of the windows.

Chunking respects the provision boundary: we never merge two Articles into one
chunk. That keeps citations honest, every chunk maps to exactly one provision.
"""
import os, json, re

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "data", "corpus")


def load_manifest():
    with open(os.path.join(CORPUS, "manifest.json")) as f:
        return json.load(f)


def _split_paragraphs(text):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def _windows(paras, chunk_words, overlap_words):
    """Greedily pack paragraphs into word windows with overlap."""
    if not paras:
        return []
    chunks, cur, cur_n = [], [], 0
    for p in paras:
        n = len(p.split())
        if cur and cur_n + n > chunk_words:
            chunks.append("\n".join(cur))
            # start next window with a tail of the previous one for overlap
            if overlap_words > 0:
                tail, tn = [], 0
                for q in reversed(cur):
                    tail.insert(0, q)
                    tn += len(q.split())
                    if tn >= overlap_words:
                        break
                cur, cur_n = list(tail), tn
            else:
                cur, cur_n = [], 0
        cur.append(p)
        cur_n += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def build_chunks(chunk_words=180, overlap_words=40):
    """Return a list of chunk dicts with text and provenance metadata."""
    manifest = load_manifest()
    chunks = []
    for doc in manifest:
        with open(os.path.join(CORPUS, doc["file"])) as f:
            text = f.read().strip()
        paras = _split_paragraphs(text)
        pieces = _windows(paras, chunk_words, overlap_words)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{doc['file']}::{i}",
                "text": piece,
                "type": doc["type"],
                "number": doc["number"],
                "title": doc["title"],
                "citation": doc["citation"] + (f" (part {i+1})" if len(pieces) > 1 else ""),
                "source": doc["source"],
                "n_words": len(piece.split()),
            })
    return chunks


if __name__ == "__main__":
    import statistics
    for cw in (120, 180, 300):
        cs = build_chunks(chunk_words=cw)
        wc = [c["n_words"] for c in cs]
        print(f"chunk_words={cw:4d}  ->  {len(cs):4d} chunks  "
              f"(median {statistics.median(wc):.0f} words, max {max(wc)})")
