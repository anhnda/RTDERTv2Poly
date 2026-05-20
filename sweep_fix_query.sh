#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/rtdetrv2/rtdetrv2_r50vd_6x_poly_lowmem.yml"
RESUME="output/rtdetrv2_r50vd_6x_poly/best.pth"
LOG_DIR="output/sweep_fix_query"
LOG_FILE="$LOG_DIR/fix_query.log"

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

echo "===== fix_query sweep (infer_adapt=False) =====" | tee -a "$LOG_FILE"
echo "Config : $CONFIG"                                 | tee -a "$LOG_FILE"
echo "Resume : $RESUME"                                 | tee -a "$LOG_FILE"
echo "Log    : $LOG_FILE"                               | tee -a "$LOG_FILE"
echo ""                                                 | tee -a "$LOG_FILE"

for fix_query in $(seq 50 25 300); do
    echo "---------- fix_query=${fix_query} ----------" | tee -a "$LOG_FILE"
    python tools/train.py \
        -c "$CONFIG" \
        -r "$RESUME" \
        --test-only \
        --fix_query "$fix_query" \
        -u "RTDETRTransformerv2.infer_adapt=False" \
        2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "===== Sweep complete =====" | tee -a "$LOG_FILE"
echo ""                           | tee -a "$LOG_FILE"
echo "----- Summary (AP from each run) -----" | tee -a "$LOG_FILE"
for fix_query in $(seq 50 25 300); do
    ap=$(grep -A2 "fix_query=${fix_query} ---" "$LOG_FILE" | grep -oP 'Average Precision.*?=\s*\K[0-9.]+' | head -1 || true)
    echo "fix_query=${fix_query}: AP=${ap:-N/A}" | tee -a "$LOG_FILE"
done
