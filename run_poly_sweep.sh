#!/bin/bash

CONFIG="configs/rtdetrv2/include/rtdetrv2_r50vd.yml"
LOG="poly_result.log"

# Set RESUME to your checkpoint path, e.g.: RESUME="-r output/rtdetrv2_r50vd_6x_poly/best.pth"
RESUME="-r output/rtdetrv2_r50vd_6x_poly/best.pth"

RUN_CMD="python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_poly_lowmem.yml --test-only ${RESUME}"

set_yaml() {
    local key=$1
    local val=$2
    sed -i "s/^\([[:space:]]*${key}:[[:space:]]*\).*/\1${val}/" "$CONFIG"
}

> "$LOG"

# --- baseline: infer_adapt False ---
echo "=== infer_adapt: False ===" | tee -a "$LOG"
set_yaml "infer_adapt" "False"
${RUN_CMD} 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- sweep: infer_adapt True, offset 10..150 ---
set_yaml "infer_adapt" "True"

for offset in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150; do
    echo "=== infer_adapt: True  offset: ${offset} ===" | tee -a "$LOG"
    set_yaml "offset" "${offset}"
    ${RUN_CMD} 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

# --- restore defaults ---
set_yaml "infer_adapt" "True"
set_yaml "offset" "100"

echo "=== Sweep complete. Restored infer_adapt: True, offset: 100 ===" | tee -a "$LOG"
echo "Results saved to ${LOG}"
