"""
Fetch the UCMJ corpus (10 U.S.C. chapter 47, sections 801 to 946a) from the
Cornell Legal Information Institute into data/corpus/.

The corpus splits into two natural document types:

  punitive    : Subchapter X, the punitive articles (Art. 77 to 134). These
                define the offenses: desertion, AWOL, insubordination, murder,
                conduct unbecoming, the general article, and so on.
  procedural  : everything else. Jurisdiction, apprehension, non-judicial
                punishment, court-martial composition, trial procedure,
                sentencing, appellate review.

The two registers overlap heavily in vocabulary (accused, court-martial,
convening authority appear everywhere) which is exactly what makes retrieval
over this corpus interesting. There is also a numbering trap built into the
material: every provision has both a U.S. Code section number and a UCMJ
article number offset by 800 (Article 86 is section 886, the famous
"Article 15" is section 815). Questions use the colloquial article numbers.

Output is committed to the repo so a clean clone needs no network. Re-run this
script only to refresh the raw text.
"""
import os, re, time, sys, json
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus")
os.makedirs(OUT, exist_ok=True)
BASE = "https://www.law.cornell.edu"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (ucmj-rag-eval; educational)"}

SUBCHAPTERS = ["I", "II", "III", "IV", "V", "VI", "VII",
               "VIII", "IX", "X", "XI", "XII"]


def clean(text):
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def sections_for(sub):
    s = get(f"{BASE}/uscode/text/10/subtitle-A/part-II/chapter-47/subchapter-{sub}")
    secs = []
    for a in s.select("a[href]"):
        m = re.fullmatch(r"/uscode/text/10/(\d{3}[a-z]?)", a.get("href", ""))
        if m and m.group(1) not in secs:
            secs.append(m.group(1))
    return secs


def fetch_section(sec):
    url = f"{BASE}/uscode/text/10/{sec}"
    s = get(url)
    h1 = s.find("h1")
    body = s.select_one("div.section") or s.select_one("#tab_default_1")
    if not (h1 and body):
        return None
    heading = h1.get_text(" ", strip=True)
    # UCMJ article number is always the section number minus 800, with any
    # letter suffix carried over: section 886 is Article 86, 946a is 146a.
    # Deriving it arithmetically is more reliable than parsing the heading,
    # which varies between "Art. 86." and "Article 1." across pages.
    num = re.match(r"(\d+)([a-z]?)", sec)
    article = f"{int(num.group(1)) - 800}{num.group(2)}"
    # strip everything up to and including the article label to get the title
    title = re.sub(r"^10 U.S. Code § \S+\s*-?\s*(Art(icle)?\.?\s*\d+[a-z]?\.\s*)?",
                   "", heading).strip()
    return {"url": url, "article": article, "title": title,
            "text": clean(body.get_text("\n", strip=True))}


def main():
    manifest = []
    for sub in SUBCHAPTERS:
        secs = sections_for(sub)
        kind = "punitive" if sub == "X" else "procedural"
        for sec in secs:
            try:
                d = fetch_section(sec)
            except Exception as e:
                print(f"  skip {sec}: {e}", file=sys.stderr)
                continue
            if not d or not d["text"]:
                print(f"  skip {sec}: empty", file=sys.stderr)
                continue
            fn = f"sec_{sec}.txt"
            with open(os.path.join(OUT, fn), "w") as f:
                f.write(d["text"])
            art = d["article"] or sec
            manifest.append({
                "file": fn, "type": kind, "number": art, "section": sec,
                "title": d["title"],
                "citation": f"Art. {art}, UCMJ (10 U.S.C. § {sec})",
                "source": d["url"], "subchapter": sub,
            })
            print(f"{sub:4s} sec {sec:5s} Art. {art or '?':5s} {d['title'][:55]}")
            time.sleep(0.25)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} provisions to {OUT}")


if __name__ == "__main__":
    main()
