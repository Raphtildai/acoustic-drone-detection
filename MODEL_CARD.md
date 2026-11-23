# Model card: dronedet v0.1.0 detector weights

These are the five detector checkpoints evaluated in the paper, one per training seed (42–46). They
are released as assets of the GitHub release `v0.1.0`, not in the git history.

| File | Seed | Kept epoch | Validation threshold | Validation F1 | Test F1 (Table II) |
|---|---|---|---|---|---|
| `dronedet-v0.1.0-seed42.pth` | 42 | 18 | 0.780 | 0.974 | 0.958 |
| `dronedet-v0.1.0-seed43.pth` | 43 | 25 | 0.725 | 0.976 | 0.960 |
| `dronedet-v0.1.0-seed44.pth` | 44 | 18 | 0.665 | 0.975 | 0.971 |
| `dronedet-v0.1.0-seed45.pth` | 45 | 12 | 0.870 | 0.973 | 0.956 |
| `dronedet-v0.1.0-seed46.pth` | 46 | 26 | 0.865 | 0.977 | 0.961 |

SHA-256 sums are in `SHA256SUMS` next to the files. Each file is about 1.9 MB.

## Model

- **Architecture:** `DetectionCNN`, a 0.47 M-parameter CNN that outputs the probability that a drone is present.
- **Input:** one 3-s window at 22.05 kHz, turned into a 3 × 64 × 259 stack of log-mel, PCEN and delta channels (`dronedet.window_features`).
- **Training recipe:** the one in the README; 30 epochs at most, and the epoch with the best validation F1 is kept.
- **Speed:** 34 ms per window on one CPU thread.

## Intended use

- Research on acoustic drone detection.
- Reproducing the paper's results.
- A starting point for training on your own data.

The model ranks windows by how drone-like they sound. It does not localize or count drones.

## Out of scope

The weights' licence forbids these uses, because the DroneAudioDataset licence does:
- military, defence or intelligence applications;
- security, weapons or surveillance applications.

Commercial use is also not allowed (CC BY-NC 4.0).

## Training data

The paper's source-grouped corpus (Section III-A):
- **Drone clips (2,860):**
  - DroneAudioDataset recordings (Bebop and Membo, including the authors' noise-mixed versions);
  - three public consumer-drone recordings (two video soundtracks and one Pixabay flyover), plus four clips from a fourth public video;
  - 139 physics-informed synthetic clips, whose noise floors were measured by the partner laboratory;
  - background mixtures of the above.
- **Non-drone clips (7,366):**
  - the ESC-50 and Speech Commands noise slices of the DroneAudioDataset;
  - seven UrbanSound8K classes;
  - silence.

None of the training data is included here.

## Performance

| Setting | Result |
|---|---|
| Benchmark (2,163 held-out clips, source-grouped split) | F1 0.961 ± 0.005, AUROC 0.991 |
| Field: outdoor, 1–5 drones, GPS-labelled, reference microphone; input low-passed at 2 kHz and smoothed over 15 s | AUROC 0.946 |
| Field: unfiltered input | AUROC 0.36, below chance |

Re-running `dronedet-eval bench` with these files reproduces the benchmark row exactly.

## Limitations

- **Site dependence.** Without the low-pass filter the model fails at the paper's field site. Choose the cutoff and the threshold on your own calibration recordings (`dronedet-eval field`). The stored validation threshold does not transfer.
- **Narrow drone diversity.** The benchmark drones are two small indoor-recorded types, and the field evaluation covers one site on one day.
- **Simple baseline.** At a quiet site, a training-free band-energy detector can do as well as the model; at the paper's site it did better (AUROC 0.97). Always report it next to the model.
- **Known label issue.** The training set includes 13 clips (39 s, about 0.5 % of the drone clips) cut from a private outdoor recording at times when no drone was flying. Their effect is expected to be negligible. They are not part of any test set.

## Licence

The weights are licensed under CC BY-NC 4.0, with the use restriction above; see `WEIGHTS_LICENSE.md`. The code is MIT.

## Citation

See the README.
