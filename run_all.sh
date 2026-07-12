#!/usr/bin/env bash
# Reproduce every run used in the README. Retrieval and scoring need no API key.
set -e
echo "[1/6] baseline (bm25, chunk_words=180)"
python3 -m eval.run_eval --tag bm25_baseline
echo "[2/6] chunk size ablation"
python3 -m eval.run_eval --chunk-words 120 --tag bm25_cw120
python3 -m eval.run_eval --chunk-words 300 --tag bm25_cw300
echo "[3/6] retriever ablation (tfidf cosine)"
python3 -m eval.run_eval --retriever tfidf --tag tfidf_baseline
echo "[4/6] tokenisation ablation (light stemming)"
python3 -m eval.run_eval --stem --tag bm25_stem
echo "[5/6] dense + hybrid ablation (skipped unless sentence-transformers is installed)"
if python3 -c "import sentence_transformers" 2>/dev/null; then
    python3 -m eval.run_eval --retriever dense --tag dense_baseline
    python3 -m eval.run_eval --retriever hybrid --tag hybrid_baseline
else
    echo "    sentence-transformers not installed, skipping (pip install sentence-transformers to run)"
fi
echo "[6/6] done. see outputs/ for run files."
echo "graded answers and prompt ablation are in outputs/graded_answers.json and outputs/prompt_ablation.json"
