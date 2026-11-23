#!/bin/bash
# Try every command in a few minutes on CPU, without any dataset: a toy corpus of synthetic
# drone clips vs noise and steady tones, a short training run, the benchmark protocol, a
# labelled "long recording" for the field protocol, and detection with a timeline plot.
# The toy task is easy (validation F1 reaches 1.0 within a few epochs, and the first best epoch
# is kept, so scores sit close to the threshold); it shows the workflow, not the paper's numbers.
set -euo pipefail
OUT=${1:-toy}
python examples/toy_data.py --out "$OUT"
dronedet-train --data "$OUT/data" --out "$OUT/run" --epochs 15 --synthetic 20
dronedet-eval bench --data "$OUT/data" --checkpoints "$OUT/run/best_detection.pth" --out "$OUT/bench.json"
dronedet-eval field --audio "$OUT/long.wav" --labels "$OUT/labels.csv" \
    --checkpoints "$OUT/run/best_detection.pth" --cutoffs none 4000 2000 --out "$OUT/field.json"
dronedet-detect "$OUT/long.wav" --checkpoint "$OUT/run/best_detection.pth" --lowpass 4000 \
    --csv "$OUT/scores.csv" --plot "$OUT/timeline.png"
echo "done: see $OUT/"
