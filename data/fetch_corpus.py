"""
Fetch the GDPR corpus (Articles + Recitals) from gdpr-info.eu into data/corpus/.

We use two document *types* on purpose:
  - Articles  : the binding, operative text of the Regulation (99 articles)
  - Recitals  : non-binding, interpretive preamble text (173 recitals)

They cover the same topics in different registers. That overlap is what makes
the retrieval evaluation interesting: recitals are written in richer natural
language and often out-retrieve the terser binding article for the same query.

Output is committed to the repo so a clean clone needs no network to run.
Re-run this script only if you want to refresh the raw text.
"""
import os, re, time, sys, json
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus")
os.makedirs(OUT, exist_ok=True)
HEADERS = {"User-Agent": "rag-legal-eval/1.0 (educational take-home; contact via github)"}


def clean(text: str) -> str:
    # collapse the ragged whitespace gdpr-info emits, keep paragraph breaks
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def fetch(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    s = BeautifulSoup(r.text, "html.parser")
    h1 = s.find("h1")
    body = s.select_one("div.entry-content")
    if not (h1 and body):
        return None
    # drop the "Suitable Recitals" / navigation cruft appended to article pages
    for sel in body.select("div, ul.gdpr-vertical-menu, .su-note"):
        cls = " ".join(sel.get("class", []))
        if "recital" in cls.lower() or "menu" in cls.lower():
            sel.decompose()
    return h1.get_text(" ", strip=True), clean(body.get_text("\n", strip=True))


def main():
    manifest = []

    # --- Articles 1..99 ---
    for n in range(1, 100):
        url = f"https://gdpr-info.eu/art-{n}-gdpr/"
        res = fetch(url)
        if not res:
            print(f"  skip article {n}", file=sys.stderr)
            continue
        title, text = res
        title = re.sub(r"^Art\.?\s*\d+\s*GDPR", "", title).strip()
        fn = f"article_{n:03d}.txt"
        with open(os.path.join(OUT, fn), "w") as f:
            f.write(text)
        manifest.append({"file": fn, "type": "article", "number": n,
                         "title": title, "citation": f"Art. {n} GDPR", "source": url})
        print(f"article {n}: {title[:60]}")
        time.sleep(0.15)

    # --- Recitals 1..173 ---
    for n in range(1, 174):
        url = f"https://gdpr-info.eu/recitals/no-{n}/"
        res = fetch(url)
        if not res:
            print(f"  skip recital {n}", file=sys.stderr)
            continue
        title, text = res
        title = re.sub(r"^Recital\s*\d+\*?", "", title).strip().lstrip("*").strip()
        fn = f"recital_{n:03d}.txt"
        with open(os.path.join(OUT, fn), "w") as f:
            f.write(text)
        manifest.append({"file": fn, "type": "recital", "number": n,
                         "title": title, "citation": f"Recital {n} GDPR", "source": url})
        print(f"recital {n}: {title[:60]}")
        time.sleep(0.15)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} documents to {OUT}")


if __name__ == "__main__":
    main()
