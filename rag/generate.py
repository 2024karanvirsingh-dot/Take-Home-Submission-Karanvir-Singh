"""
Generative answer layer (optional, local, no API).

This is the actual generation path of the RAG pipeline: a small local
instruction tuned model (google/flan-t5-small, about 300 MB, downloaded
once from the Hugging Face hub and cached; swap in a larger model with
--gen-model) writes the answer conditioned on the retrieved chunks and
nothing else. flan-t5-small is deliberately the smallest instruction model
that produces a meaningful comparison: it runs on CPU in seconds per
question from a clean clone, and its failure modes under this harness are
themselves a finding, not an embarrassment to hide. The prompt contains only the retrieved
statute text; the model is instructed to cite articles inline and to output
a fixed refusal marker when the context does not contain the answer.

Contract with the rest of the system:

  - construct() returns the same dict shape as rag.answer.construct_answer,
    so eval/run_eval.py and eval/score_answers.py treat both answerers
    uniformly. Generated answers carry an empty `sentences` list; their
    support is checked by the fuzzy alignment in eval/score_answers.py
    instead of the verbatim gate the extractive layer earns by construction.
  - The question level corpus boundary check (Rules for Courts-Martial /
    MCM questions) runs before the model, exactly as it does for the
    grounded extractive policy: that check is system policy, not model
    behavior, and holding it fixed keeps the answerer comparison clean.
    Everything past that point is the model's own behavior, evaluated as is.
  - Decoding is greedy (no sampling, beam size 1) with a fixed token budget,
    so the same question, chunks and model version give the same answer on
    the same platform. Floating point differences across hardware can in
    principle flip a near tied token; the committed outputs record the
    platform they were produced on.

This path is an optional experiment, completely isolated from the default
clean clone workflow: the model dependencies (transformers, torch) live in
requirements-gen.txt, nothing in the baseline or the core evaluation
imports this module, and using it without the extra installed fails with
instructions rather than silently changing the evaluated system. The
committed generative run in outputs/ documents what it does.
"""
import re
from .answer import detect_corpus_boundary, BOUNDARY_OUT_OF_CORPUS

GEN_MODEL_DEFAULT = "google/flan-t5-small"
REFUSAL_MARKER = "NOT ANSWERABLE FROM CONTEXT"
GEN_REFUSAL_TEXT = (
    "The retrieved provisions do not contain the answer to this question. "
    "(Generative answerer refusal: the model returned the refusal marker.)"
)

_CITED = re.compile(r"\bart(?:icle)?\.?\s*(\d+[a-z]?)\b", re.I)


def build_prompt(question, retrieved, max_context_words=900):
    """Build the generation prompt from the retrieved chunks and nothing
    else. Pure function, importable and testable without torch. The chunk
    budget keeps the prompt inside what a small T5 handles well; chunks are
    included in retrieval order and truncated, never reordered, so the
    prompt is a faithful window onto what retrieval produced."""
    blocks, used = [], 0
    for chunk, _score in retrieved:
        words = chunk["text"].split()
        room = max_context_words - used
        if room <= 20:
            break
        take = words[:room]
        used += len(take)
        blocks.append("[Article %s - %s]\n%s"
                      % (chunk["number"], chunk["title"], " ".join(take)))
    context = "\n\n".join(blocks)
    return (
        "You are answering a question about the Uniform Code of Military "
        "Justice using ONLY the statute articles quoted below. Do not use "
        "any other knowledge. Cite the article number for every claim, in "
        "the form (Art. 86, UCMJ). If the quoted articles do not contain "
        "the answer, reply with exactly: " + REFUSAL_MARKER + "\n\n"
        + context + "\n\nQuestion: " + question + "\nAnswer:")


def parse_citations(text):
    """Article numbers cited in generated text, deduplicated in order."""
    seen, out = set(), []
    for m in _CITED.findall(text):
        n = m.lower()
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


class GenerativeAnswerer:
    def __init__(self, model_name=GEN_MODEL_DEFAULT):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise SystemExit(
                "the generative answerer needs transformers and torch. "
                "Install the optional generative extra with "
                "`pip install -r requirements-gen.txt` (first use downloads "
                "%s, about 300 MB, cached by the Hugging Face hub). The "
                "default install and the extractive answerer do not need "
                "this."
                % model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def construct(self, question, retrieved, qtype=None):
        """Same return shape as rag.answer.construct_answer."""
        result = {"policy": "generative", "refused": False, "boundary": False,
                  "cited_articles": [], "sentences": []}
        if detect_corpus_boundary(question):
            result.update(text=BOUNDARY_OUT_OF_CORPUS, refused=True,
                          boundary=True)
            return result
        import torch
        prompt = build_prompt(question, retrieved)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=1536)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=192,
                                      do_sample=False, num_beams=1)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if not text or REFUSAL_MARKER in text.upper():
            result.update(text=GEN_REFUSAL_TEXT, refused=True)
            return result
        result.update(text=text, cited_articles=parse_citations(text))
        return result
