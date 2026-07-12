#!/usr/bin/env bash
# Reproduce every run used in the README. Retrieval and scoring need no API key.
set -e
echo "[1/4] baseline (bm25, chunk_words=180)"
python3 -m eval.run_eval --tag bm25_baseline
echo "[2/4] chunk size ablation"
python3 -m eval.run_eval --chunk-words 120 --tag bm25_cw120
python3 -m eval.run_eval --chunk-words 300 --tag bm25_cw300
echo "[3/4] retriever ablation (tfidf cosine)"
python3 -m eval.run_eval --retriever tfidf --tag tfidf_baseline
echo "[4/4] done. see outputs/ for run files."
echo "graded answers and prompt ablation are in outputs/graded_answers.json and outputs/prompt_ablation.json"
