#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/rtdetrv2/rtdetrv2_r50vd_6x_poly_lowmem.yml"
RESUME="output/rtdetrv2_r50vd_6x_poly/best.pth"
LOG_DIR="output/sweep_fix_query"

mkdir -p "$LOG_DIR"

echo "===== fix_query sweep (infer_adapt=False) ====="
echo "Config : $CONFIG"
echo "Resume : $RESUME"
echo "Log dir: $LOG_DIR"
echo ""

for fix_query in $(seq 50 25 300); do
    log_file="$LOG_DIR/fix_query_${fix_query}.log"
    echo "---------- fix_query=${fix_query} ----------"
    python tools/train.py \
        -c "$CONFIG" \
        -r "$RESUME" \
        --test-only \
        --fix_query "$fix_query" \
        -u "RTDETRTransformerv2.infer_adapt=False" \
        2>&1 | tee "$log_file"
    echo "  -> log saved: $log_file"
    echo ""
done

echo "===== Sweep complete ====="
echo ""
echo "----- Summary (AP from each run) -----"
for fix_query in $(seq 50 25 300); do
    log_file="$LOG_DIR/fix_query_${fix_query}.log"
    ap=$(grep -oP 'Average Precision.*?=\s*\K[0-9.]+' "$log_file" | head -1 || echo "N/A")
    echo "fix_query=${fix_query}: AP=${ap}"
done
