#!/usr/bin/env bash
# Reproduce every run and every number used in the README. Everything is
# local and deterministic; no credentials, no network.
set -e
echo "[1/7] dataset validation"
python3 -m eval.validate_dataset
echo "[2/7] baseline (bm25, chunk_words=180, grounded answers)"
python3 -m eval.run_eval --tag bm25_baseline
echo "[3/7] chunk size ablation"
python3 -m eval.run_eval --chunk-words 120 --tag bm25_cw120
python3 -m eval.run_eval --chunk-words 300 --tag bm25_cw300
echo "[4/7] retriever ablation (tfidf cosine) and tokenisation ablation (stemming)"
python3 -m eval.run_eval --retriever tfidf --tag tfidf_baseline
python3 -m eval.run_eval --stem --tag bm25_stem
echo "[5/7] dense + hybrid ablation (skipped unless sentence-transformers is installed)"
if python3 -c "import sentence_transformers" 2>/dev/null; then
    python3 -m eval.run_eval --retriever dense --tag dense_baseline
    python3 -m eval.run_eval --retriever hybrid --tag hybrid_baseline
else
    echo "    sentence-transformers not installed, skipping (pip install -r requirements-dense.txt to run)"
fi
echo "[6/7] answer policy ablation and answer scoring"
python3 -m eval.run_answer_policy_ablation
python3 -m eval.score_answers outputs/run_bm25_baseline.json
echo "[7/7] aggregate summary (the README tables come from this)"
python3 -m eval.build_summary
