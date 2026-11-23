# dronedet: compact acoustic drone detection

A 0.47 M-parameter CNN that detects multirotor drones from 3-s audio windows. This repository
contains the model, the training recipe and the two evaluation protocols of the paper:

> K. Raphael and G. Bognár, "Integrated Drone Detection, Localization and Path Tracking using Deep Learning" (under review).

The code is a clean re-implementation of the paper's detection stage. Its features and model
reproduce the paper's checkpoints exactly (see [Verification](#verification)).

## What the paper found, in short

- **Benchmark** (held-out recordings, source-grouped split): F1 0.961 ± 0.005.
  - On par with a CRNN (0.965) and fine-tuned PANNs (0.952), and below fine-tuned AST (0.978).
  - 180× fewer parameters than AST and 27× faster on a CPU: 34 ms per window on one thread.
- **Field** (outdoor recordings of 1–5 drones, labelled per window from GPS logs, with drone-free periods):
  - Benchmark accuracy does not predict field performance. CRNN, PANNs and fine-tuned AST rank the field windows at or below chance.
  - Low-passing the input at 2 kHz (cutoff chosen on calibration flights) plus 15-s smoothing lifts this detector to AUROC 0.946 on a reference microphone and 0.911 on a low-cost MEMS microphone. That is the best of the learned models.
  - The training measures (synthetic data, SpecAugment, balanced sampling) change nothing on the benchmark but matter in the field (AUROC 0.88 vs 0.60 without synthetic data).
- **A simple baseline to keep in mind:** a training-free band-energy detector reaches 0.97 at that quiet site, but only 0.70–0.75 on the benchmark's loud non-drone sounds. Always compare a field result with it (`dronedet-eval field` does).

## Install

```bash
git clone <this repository> && cd acoustic-drone-detection
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[test]" for the tests
```

Python ≥ 3.9 and PyTorch ≥ 2.0. A GPU is optional; everything runs on a CPU.

## Detect drones in a recording

```bash
dronedet-detect recording.wav --checkpoint best_detection.pth \
    --lowpass 2000 --smooth-s 15 --csv scores.csv --plot timeline.png
```

The output is one row per 3-s window (step 1.5 s): `start_s, end_s, score, smoothed, drone`.
Any length and sample rate work; long files are read in chunks. `--channel` picks one channel of a multichannel file.

Two settings depend on your site. Set them from your own labelled recordings, not from the defaults:
- **`--lowpass`:** the input low-pass cutoff. 2 kHz was best at the paper's site; choose yours with `dronedet-eval field`.
- **`--threshold`:** by default the checkpoint's validation threshold, which does **not** transfer to new sites. `dronedet-eval field` reports the threshold that meets a false-alarm budget on your drone-free audio.

## Train

```
data/
  train/drone/*.wav   train/non_drone/*.wav
  val/drone/*.wav     val/non_drone/*.wav
  test/drone/*.wav    test/non_drone/*.wav     (optional)
```

```bash
dronedet-train --data data/ --out runs/seed42 --seed 42 --synthetic 139
```

- **Split by source recording, not by clip.** All clips cut from one recording, and all mixtures made from it, go into the same partition. A clip-level shuffle leaks recordings into the test set and inflates the score.
- **The first 3 s of each file are used.** Cut long recordings into 3-s clips first.
- **Recipe (as in the paper):**
  - focal loss (γ 2, α 0.6, label smoothing 0.02);
  - class-balanced sampler and SpecAugment-style masking;
  - AdamW (3e-4, weight decay 1e-3), warm-up then cosine decay, gradient clipping at 1.0;
  - 30 epochs, batch 32;
  - the epoch with the best validation F1 is kept, with its validation threshold.
- **Features** are cached on disk (`<out>/feature_cache`).
- **`--synthetic N`** adds N physics-informed synthetic drone clips: a blade-pass harmonic series, propagation delay and attenuation, and a noise floor. `dronedet-synth --backgrounds <folder>` writes such clips mixed into your own background recordings, which is more realistic.
- **Ablations:** `--no-specaug` and `--no-sampler`.

The paper trained on the DroneAudioDataset (positives and noise), selected UrbanSound8K classes and three public drone videos. Check each dataset's licence before use: the DroneAudioDataset licence excludes security and surveillance applications.

## Evaluate

**Benchmark:** a held-out test folder, at each checkpoint's validation threshold, mean ± std over checkpoints:

```bash
dronedet-eval bench --data data/ --checkpoints runs/seed*/best_detection.pth
```

**Field:** a long recording with labelled 3-s windows, split into calibration and test windows:

```bash
dronedet-eval field --audio site.wav --labels windows.csv \
    --checkpoints runs/seed*/best_detection.pth --out field_results.json
```

`windows.csv` has the columns `start_s,label,split`:
- `start_s`: seconds from the start of the file;
- `label`: 1 = drone airborne, 0 = drone-free; any other value is skipped;
- `split`: `cal` or `test`.

The protocol keeps the test windows out of every decision:
1. The input low-pass cutoff is chosen by the mean calibration AUROC over checkpoints.
2. For false-alarm budgets of 120, 60, 30 and 12 per hour, the threshold is the lowest one that meets the budget on calibration.
3. Test windows are used only to report AUROC/AUPRC (raw and smoothed) and recall / false alarms per hour at those thresholds.
4. A band-energy reference (the band with the best calibration AUROC) is reported alongside.

How to label well (what the paper did):
- **Drone:** at least one drone more than 1 m above ground, from the flight logs.
- **Drone-free:** no drone powered within ±30 s.
- **Leave out** take-off and landing.
- **Separate calibration and test** flights.
- **Collect enough drone-free audio:** with 11 minutes, one false alarm already counts as 5.4 per hour.

## Use from Python

```python
from dronedet import load_audio, load_checkpoint, predict_proba, window_features

model, threshold, fe = load_checkpoint("best_detection.pth")
y = load_audio("clip.wav", fe.sr)
p = predict_proba(model, window_features(y, cutoff_hz=2000, fe=fe)[None])[0]
```

## Pretrained weights

Not included yet. The paper's checkpoints were trained on data whose licences restrict use, and their release is being clarified. Until then, train your own with the recipe above.

## What is not here

- **The field recordings.** They belong to the HUN-REN Centre for Energy Research and are not public. Ask the data owners for access.
- **The paper's lab-measured indoor noise floors** used by its synthetic generator. They are partner data. The generator here uses generic noise floors or your own background recordings instead.
- **The localization and tracking stages**, and the baseline models (CRNN, PANNs, AST). This release covers the detector.

## Verification

Checked against the paper's original implementation and its Komondor results (2026-11-23):

- **Benchmark.** `dronedet-eval bench` with the paper's five detector checkpoints, on its
  source-grouped test partition (2,163 clips), reproduces Table II exactly. Every seed gives the
  same true positives, false positives, threshold and F1 as the original runs (F1 0.961 ± 0.005).
- **Field.** `dronedet-eval field` on the paper's array-W calibration and test windows reproduces:
  - the calibration AUROC of every cutoff and the selection of 2 kHz;
  - the unsmoothed test AUROC (0.393 ± 0.189 unfiltered, 0.919 ± 0.024 at 2 kHz);
  - the band-energy reference (0.968);
  - the 60 FA/h operating point.

  (The field recordings are private, so this check can't be rerun from this repository.)
- **Tests.** `tests/` covers features, model, metrics, the synthetic generator, and an end-to-end
  train → bench → field → detect run on generated audio:

```bash
pip install -e ".[test]" && pytest -q        # about a minute on a CPU
```

## Citation

```bibtex
@misc{raphael2026dronedet,
  author = {Raphael, Kipchirchir and Bogn{\'a}r, Gerg{\H{o}}},
  title  = {Integrated Drone Detection, Localization and Path Tracking using Deep Learning},
  year   = {2026},
  note   = {Under review}
}
```

## Licence

MIT (see `LICENSE`). Datasets used to train models keep their own licences.
