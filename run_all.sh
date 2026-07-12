#!/usr/bin/env bash
# Reproduce every run, score, report and table used in the README.
# Everything is local; no credentials and no API. The default install
# (requirements.txt) runs every baseline stage. The dense/hybrid stage
# needs requirements-dense.txt and the generative stage needs
# requirements-gen.txt (each downloads its model from the Hugging Face hub
# on first use); without those optional extras the stages are skipped with
# a notice and the committed outputs for them remain in outputs/.
set -e
echo "[1/9] dataset validation"
python3 -m eval.validate_dataset
echo "[2/9] diagnostic baseline (bm25, chunk_words=180, grounded answers)"
python3 -m eval.run_eval --tag bm25_baseline
echo "[3/9] chunk size ablation"
python3 -m eval.run_eval --chunk-words 120 --tag bm25_cw120
python3 -m eval.run_eval --chunk-words 300 --tag bm25_cw300
echo "[4/9] retriever ablation (tfidf cosine) and tokenisation ablation (stemming)"
python3 -m eval.run_eval --retriever tfidf --tag tfidf_baseline
python3 -m eval.run_eval --stem --tag bm25_stem
echo "[5/9] optional dense + hybrid ablation"
if python3 -c "import sentence_transformers" 2>/dev/null; then
    python3 -m eval.run_eval --retriever dense --tag dense_baseline
    python3 -m eval.run_eval --retriever hybrid --tag hybrid_baseline
else
    echo "    sentence-transformers not installed, skipping (pip install -r requirements-dense.txt)"
fi
echo "[6/9] optional generative answerer experiment (flan-t5-small, local)"
if python3 -c "import transformers, sentence_transformers" 2>/dev/null; then
    python3 -m eval.run_eval --retriever hybrid --answerer generative --tag hybrid_generative
else
    echo "    transformers/sentence-transformers not installed, skipping (pip install -r requirements-dense.txt -r requirements-gen.txt)"
fi
echo "[7/9] answer policy ablation and answer scoring for every configuration"
python3 -m eval.run_answer_policy_ablation
for tag in bm25_baseline dense_baseline hybrid_baseline hybrid_generative; do
    if [ -f "outputs/run_${tag}.json" ]; then
        python3 -m eval.score_answers "outputs/run_${tag}.json"
    fi
done
echo "[8/9] aggregate summary (the README tables come from this)"
python3 -m eval.build_summary
echo "[9/9] per question report and before/after examples"
if [ -f outputs/run_hybrid_baseline.json ]; then
    python3 -m eval.build_report hybrid_baseline
else
    python3 -m eval.build_report bm25_baseline
fi
python3 -m eval.build_before_after > /dev/null && echo "wrote outputs/before_after.md"
