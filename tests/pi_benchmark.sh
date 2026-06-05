#!/bin/bash
# Benchmark gemma2:2b on repair vs pool vs hybrid, 20 cycles each.
# Run from the repo root: bash tests/pi_benchmark.sh
# Results written to results/pi_repair.json, pi_pool.json, pi_hybrid.json

set -e
cd "$(dirname "$0")/.."

echo ""
echo "==============================="
echo " Pi Benchmark — gemma2:2b"
echo " 20 cycles x 3 strategies"
echo "==============================="
echo ""

for strategy in repair pool hybrid; do
    echo "--- Strategy: $strategy ---"
    python3 tests/haiku_benchmark_v2.py \
        --model gemma2:2b \
        --cycles 20 \
        --seed 5 \
        --strategy "$strategy" \
        --audit \
        --json "results/pi_${strategy}.json" \
        --once
    echo ""
done

echo "==============================="
echo " Done. Comparing repair vs pool:"
echo "==============================="
python3 tests/compare_runs.py results/pi_repair.json results/pi_pool.json

echo ""
echo "==============================="
echo " Repair vs hybrid:"
echo "==============================="
python3 tests/compare_runs.py results/pi_repair.json results/pi_hybrid.json
