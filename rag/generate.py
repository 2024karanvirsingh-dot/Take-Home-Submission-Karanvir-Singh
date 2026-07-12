"""
Generation.

The prompt is deliberately strict: answer only from the retrieved context, cite
the provision by its label, and if the context does not contain the answer say
so instead of guessing. Most RAG hallucinations in a legal setting come from the
model smoothing over a gap in retrieval, so the prompt makes "not in the context"
a first class allowed answer.

Two backends:
  anthropic  : used when ANTHROPIC_API_KEY is set. Model is overridable with
               RAG_MODEL (default claude-3-5-sonnet, a broadly available model
               so a reviewer's key works without edits).
  extractive : zero dependency fallback so the pipeline runs on a clean clone
               with no key. It returns the single best chunk verbatim. This is
               not a real generated answer, it just proves retrieval end to end.

The recorded evaluation run in outputs/ was generated over the exact contexts
this file produces. See the README for how that run was done.
"""
import os, textwrap

# Strict prompt: the guardrailed default. "Not in the context" is an allowed,
# even encouraged, answer. This is what makes a retrieval miss fail safe.
STRICT = (
    "You are a careful legal research assistant answering questions about the "
    "Uniform Code of Military Justice. Answer using only the provided context "
    "passages, which are the statutory text of 10 U.S.C. chapter 47. Cite the "
    "provision you rely on by its UCMJ article, for example (Art. 86, UCMJ). "
    "Statutory punishment language such as 'as a court-martial may direct' means "
    "the specific maximum punishment is set in the Manual for Courts-Martial, "
    "not this statute; say that rather than inventing a number. If the context "
    "does not contain the answer, say 'The provided context does not answer "
    "this' and stop. Do not use outside knowledge."
)

# Permissive prompt: same task, guardrail removed. Used only for the prompt
# ablation, to show that when the context is missing the model falls back to its
# training memory and produces confident but unsupported answers.
PERMISSIVE = (
    "You are a helpful legal assistant answering questions about the Uniform "
    "Code of Military Justice. Use the provided context passages to help you "
    "and answer the question as fully and helpfully as you can."
)

SYSTEM = STRICT  # backwards compatible default


def _system():
    return PERMISSIVE if os.environ.get("RAG_PROMPT") == "permissive" else STRICT


def build_prompt(question, retrieved):
    blocks = []
    for c, score in retrieved:
        blocks.append(f"[{c['citation']} | {c['type']} | {c['title']}]\n{c['text']}")
    context = "\n\n".join(blocks)
    user = f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return _system(), user


def generate(question, retrieved):
    system, user = build_prompt(question, retrieved)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic(system, user)
        except Exception as e:
            return f"[anthropic call failed: {e}]\n\n" + _extractive(retrieved)
    return _extractive(retrieved)


def _anthropic(system, user):
    import anthropic
    model = os.environ.get("RAG_MODEL", "claude-3-5-sonnet-20241022")
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=600, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


def _extractive(retrieved):
    if not retrieved:
        return "The provided context does not answer this."
    top, _ = retrieved[0]
    body = textwrap.shorten(top["text"].replace("\n", " "), width=500, placeholder=" ...")
    return f"(extractive fallback, no LLM key set) Best passage {top['citation']}:\n{body}"
