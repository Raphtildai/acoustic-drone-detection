"""Evaluate detectors with the paper's two protocols.

bench   held-out test folder (<data>/test/{drone,non_drone}): F1/precision/recall at each
        checkpoint's validation threshold, AUROC, AUPRC; mean +- std over checkpoints.

    dronedet-eval bench --data data/ --checkpoints runs/seed*/best_detection.pth

field   a long recording with labelled 3-s windows, split into calibration and test windows
        (e.g. by flight). Every field-specific choice is made on calibration windows only:
          1. the input low-pass cutoff (max mean calibration AUROC over checkpoints),
          2. thresholds for false-alarm budgets (lowest threshold with calibration FA/h <= B);
        test windows are used only for reporting: AUROC/AUPRC raw and after causal smoothing,
        and recall / false alarms per hour at each budget. A training-free reference (log power
        in the frequency band with the best calibration AUROC) is reported alongside.

    labels CSV columns: start_s (seconds from the start of the audio file), label (1 = drone,
    0 = drone-free; other values are ignored), split (cal / test)

    dronedet-eval field --audio site.wav --labels windows.csv --checkpoints runs/seed*/best_detection.pth \
        --cutoffs none 4000 3000 2000 1500 1000 --out field_results.json

How to label: a window is "drone" if a drone is airborne (e.g. from GPS logs), "drone-free"
only if no drone is powered within a margin (the paper: +-30 s), and ambiguous windows
(take-off, landing) are left out. Keep the calibration and test windows in separate flights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import DEFAULT, window_features
from .metrics import binary_metrics, budget_threshold, operating_point, smooth
from .model import load_checkpoint, predict_proba
from .train import list_split

BUDGETS = [120, 60, 30, 12]
EDGES = [20, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]


def ms(v):
    v = np.asarray(v, float)
    return [float(np.mean(v)), float(np.std(v))]


def cmd_bench(args):
    files, labels = list_split(Path(args.data), "test")
    if not files:
        raise SystemExit(f"no audio in {args.data}/test/{{drone,non_drone}}")
    from .features import load_audio
    y = np.array(labels)
    rows, feats = [], {}
    for ck in args.checkpoints:
        model, thr, fe = load_checkpoint(ck)
        if fe not in feats:                              # features depend only on the front end
            feats[fe] = np.stack([window_features(load_audio(f, fe.sr), args.lowpass, fe) for f in files])
        p = predict_proba(model, feats[fe])
        rows.append({"checkpoint": ck, **binary_metrics(y, p, thr)})
        print(f"{ck}: F1 {rows[-1]['f1']:.4f}  AUROC {rows[-1]['auroc']:.4f}  AUPRC {rows[-1]['auprc']:.4f}")
    res = {"n_test": len(y), "n_drone": int(y.sum()), "lowpass": args.lowpass,
           **{k: ms([r[k] for r in rows]) for k in ("f1", "precision", "recall", "auroc", "auprc")}, "per_checkpoint": rows}
    print(f"mean +- std over {len(rows)}: F1 {res['f1'][0]:.3f} +- {res['f1'][1]:.3f}, AUPRC {res['auprc'][0]:.3f}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


def read_windows(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("label", "") in ("0", "1") and r.get("split") in ("cal", "test")]
    t = np.array([float(r["start_s"]) for r in rows]); o = np.argsort(t)
    return t[o], np.array([int(r["label"]) for r in rows])[o], np.array([r["split"] for r in rows])[o]


def load_windows(audio, t, fe, channel=None):
    """Raw 3-s windows at the label times (whole file loaded once)."""
    y, _ = librosa.load(str(audio), sr=fe.sr, mono=channel is None)
    if channel is not None and y.ndim > 1:
        y = y[channel]
    n = fe.n_samples
    keep = [i for i, s in enumerate(t) if int(round(s * fe.sr)) + n <= len(y)]
    W = np.stack([y[int(round(t[i] * fe.sr)): int(round(t[i] * fe.sr)) + n] for i in keep]).astype(np.float32)
    return np.array(keep), W


def band_logpower(W, sr, bands):
    f = np.fft.rfftfreq(W.shape[1], 1 / sr)
    X = np.abs(np.fft.rfft(W, axis=1)) ** 2
    return np.log10(np.stack([X[:, (f >= lo) & (f <= hi)].mean(1) for lo, hi in bands], 1) + 1e-20)


def field_report(t, y, split, P, k, step, label):
    te, ca = split == "test", split == "cal"
    rows = []
    for p in P:
        s = smooth(t, p, k, step)
        r = {"cal_auroc": float(roc_auc_score(y[ca], s[ca])), "test_auroc": float(roc_auc_score(y[te], s[te])),
             "test_auprc": float(average_precision_score(y[te], s[te])), "budgets": {}}
        for b in BUDGETS:
            r["budgets"][str(b)] = operating_point(y[te], s[te], budget_threshold(y[ca], s[ca], b, step), step)
        rows.append(r)
    out = {"cal_auroc": ms([r["cal_auroc"] for r in rows]), "test_auroc": ms([r["test_auroc"] for r in rows]),
           "test_auprc": ms([r["test_auprc"] for r in rows]),
           "budgets": {str(b): {"recall": ms([r["budgets"][str(b)]["recall"] for r in rows]),
                                "false_alarms_per_hour": ms([r["budgets"][str(b)]["false_alarms_per_hour"] for r in rows])}
                       for b in BUDGETS}, "per_checkpoint": rows}
    b60 = out["budgets"]["60"]
    print(f"  {label:28s} test AUROC {out['test_auroc'][0]:.3f} +- {out['test_auroc'][1]:.3f}   "
          f"at 60 FA/h (calibrated): recall {b60['recall'][0]:.2f}, test FA/h {b60['false_alarms_per_hour'][0]:.0f}")
    return out


def cmd_field(args):
    t, y, split = read_windows(args.labels)
    models = [load_checkpoint(c) for c in args.checkpoints]
    fe = models[0][2]
    keep, W = load_windows(args.audio, t, fe, args.channel)
    t, y, split = t[keep], y[keep], split[keep]
    step = fe.window_s
    for s in ("cal", "test"):
        m = split == s
        if len(set(y[m])) < 2:
            raise SystemExit(f"the {s} windows need both drone (1) and drone-free (0) labels")
    print(f"{len(t)} windows: cal {np.sum(split == 'cal')} ({np.sum((split == 'cal') & (y == 0)) * step / 60:.0f} min drone-free), "
          f"test {np.sum(split == 'test')} ({100 * y[split == 'test'].mean():.0f} % drone)")
    cuts = [None if c == "none" else float(c) for c in args.cutoffs]
    P = {}
    for c in cuts:
        X = np.stack([window_features(w, c, fe) for w in W])
        P[c] = [predict_proba(m, X) for m, _, _ in models]
    ca = split == "cal"
    cal_auc = {c: float(np.mean([roc_auc_score(y[ca], p[ca]) for p in P[c]])) for c in cuts}
    best = max(cal_auc, key=cal_auc.get)
    name = lambda c: "none" if c is None else f"{c:g}"
    print("mean calibration AUROC by cutoff:", {name(c): round(v, 3) for c, v in cal_auc.items()}, "-> selected", name(best))
    k = max(1, int(round(args.smooth_s / step)))
    res = {"n_windows": int(len(t)), "calibration_auroc_by_cutoff": {name(c): v for c, v in cal_auc.items()},
           "selected_cutoff": name(best), "smoothing_windows": k, "results": {}}
    print("test results (cutoff chosen on calibration):")
    for c in sorted({None, best}, key=lambda c: (c is not None, c or 0)):
        for kk in sorted({1, k}):
            lab = f"detector, cutoff {name(c)}, k={kk}"
            res["results"][lab] = field_report(t, y, split, P[c], kk, step, lab)
    bands = [(lo, hi) for lo in EDGES for hi in EDGES if hi > lo and hi <= fe.sr / 2]
    E = band_logpower(W, fe.sr, bands)
    j = int(np.argmax([roc_auc_score(y[ca], E[ca, i]) for i in range(len(bands))]))
    res["energy_reference_band_hz"] = list(bands[j])
    for kk in sorted({1, k}):
        lab = f"band energy {bands[j][0]}-{bands[j][1]} Hz, k={kk}"
        res["results"][lab] = field_report(t, y, split, [E[:, j]], kk, step, lab)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="held-out test folder")
    b.add_argument("--data", required=True)
    b.add_argument("--checkpoints", nargs="+", required=True)
    b.add_argument("--lowpass", type=float, default=None)
    b.add_argument("--out", default="bench_results.json")
    f = sub.add_parser("field", help="labelled long recording, calibration/test split")
    f.add_argument("--audio", required=True)
    f.add_argument("--labels", required=True)
    f.add_argument("--checkpoints", nargs="+", required=True)
    f.add_argument("--channel", type=int, default=None)
    f.add_argument("--cutoffs", nargs="+", default=["none", "6000", "4000", "3000", "2000", "1500", "1000", "500"])
    f.add_argument("--smooth-s", type=float, default=15.0)
    f.add_argument("--out", default="field_results.json")
    args = a.parse_args(argv)
    {"bench": cmd_bench, "field": cmd_field}[args.cmd](args)


if __name__ == "__main__":
    main()
