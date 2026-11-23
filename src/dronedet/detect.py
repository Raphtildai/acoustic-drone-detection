"""Score a recording of any length: one drone probability per 3-s window, optional input low-pass,
causal smoothing, a detection decision, a CSV and an optional timeline plot.

    dronedet-detect recording.wav --checkpoint best_detection.pth --lowpass 2000 \
        --smooth-s 15 --csv scores.csv --plot timeline.png

About the options (Section V-F of the paper):
  --lowpass   low-pass the input before the features (8th-order Butterworth, per window). On the
              paper's outdoor recordings a 2-kHz cutoff, chosen on calibration flights, turned a
              detector that ranked field windows below chance into one with AUROC 0.86-0.95.
              The best cutoff depends on the site and microphone: choose it on your own labelled
              calibration recordings with `dronedet-eval field`.
  --smooth-s  causal moving average of the window scores over this many seconds (0 = off).
  --threshold the validation threshold stored in the checkpoint is used by default. It does not
              transfer to new sites: set one from your own drone-free audio (`dronedet-eval field`
              reports the threshold for a false-alarm budget).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .features import window_features
from .metrics import smooth
from .model import load_checkpoint, predict_proba


def iter_windows(path, sr, win_s, hop_s, channel=None, chunk_s=600.0):
    """Yield (start_s, window) over the whole file, reading `chunk_s` seconds at a time."""
    info = sf.info(str(path)) if Path(path).suffix.lower() in {".wav", ".flac", ".ogg"} else None
    dur = info.duration if info else librosa.get_duration(path=str(path))
    n_win = int(round(win_s * sr))
    start = 0.0
    while start + win_s <= dur + 1e-6:
        span = min(chunk_s, dur - start)
        y, _ = librosa.load(str(path), sr=sr, mono=channel is None, offset=start, duration=span)
        if channel is not None and y.ndim > 1:
            y = y[channel]
        i, last = 0, None
        while i + n_win <= len(y):
            yield start + i / sr, y[i:i + n_win].astype(np.float32)
            last = i
            i += int(round(hop_s * sr))
        if last is None:
            break
        start += (last / sr) + hop_s


def score_file(path, model, fe, lowpass=None, hop_s=1.5, channel=None, batch=64, device="cpu"):
    t, P, buf, tb = [], [], [], []
    for s, w in iter_windows(path, fe.sr, fe.window_s, hop_s, channel):
        buf.append(window_features(w, lowpass, fe)); tb.append(s)
        if len(buf) == batch:
            P.append(predict_proba(model, np.stack(buf), device=device)); t += tb; buf, tb = [], []
    if buf:
        P.append(predict_proba(model, np.stack(buf), device=device)); t += tb
    return np.array(t), (np.concatenate(P) if P else np.zeros(0))


def plot_timeline(t, p, s, thr, out, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2.6))
    m = t / 60
    ax.plot(m, p, color="#9a9993", lw=0.6, label="window score")
    if s is not p:
        ax.plot(m, s, color="#2a78d6", lw=1.3, label="smoothed")
    ax.axhline(thr, color="#eb6834", lw=0.8, ls="--", label=f"threshold {thr:.2f}")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("minutes"); ax.set_ylabel("drone score")
    ax.set_title(title, fontsize=9); ax.legend(loc="upper right", fontsize=7, frameon=False)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("audio")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--lowpass", type=float, default=None, help="input low-pass cutoff in Hz (e.g. 2000)")
    a.add_argument("--hop-s", type=float, default=1.5, help="window step in seconds (paper: 1.5, i.e. 50%% overlap)")
    a.add_argument("--smooth-s", type=float, default=15.0)
    a.add_argument("--threshold", type=float, default=None)
    a.add_argument("--channel", type=int, default=None, help="channel of a multichannel file (default: mean)")
    a.add_argument("--csv", default=None)
    a.add_argument("--plot", default=None)
    a.add_argument("--device", default="cpu")
    args = a.parse_args(argv)
    model, val_thr, fe = load_checkpoint(args.checkpoint, args.device)
    thr = args.threshold if args.threshold is not None else val_thr
    t, p = score_file(args.audio, model, fe, args.lowpass, args.hop_s, args.channel, device=args.device)
    k = max(1, int(round(args.smooth_s / args.hop_s))) if args.smooth_s else 1
    s = smooth(t, p, k, args.hop_s) if k > 1 else p
    det = s >= thr
    print(f"{len(t)} windows, {det.sum()} above threshold {thr:.3f} "
          f"({100 * det.mean() if len(det) else 0:.1f} %), lowpass {args.lowpass or 'none'}, smoothing {k} windows")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_s", "end_s", "score", "smoothed", "drone"])
            for row in zip(t, t + fe.window_s, p, s, det):
                w.writerow([f"{row[0]:.3f}", f"{row[1]:.3f}", f"{row[2]:.5f}", f"{row[3]:.5f}", int(row[4])])
        print(f"wrote {args.csv}")
    if args.plot:
        plot_timeline(t, p, s, thr, args.plot, Path(args.audio).name)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
