"""Train the detector with the paper's recipe.

Data layout (WAV/FLAC/OGG/MP3, any sample rate; the first 3 s of each file are used, shorter
files are zero-padded):

    <data>/train/drone/...      <data>/train/non_drone/...
    <data>/val/drone/...        <data>/val/non_drone/...
    <data>/test/drone/...       <data>/test/non_drone/...      (optional)

Split by SOURCE RECORDING, not by clip: all clips cut from one recording, and all mixtures made
from it, must be in the same partition. A clip-level shuffle leaks recordings between train and
test and inflates the test score (the paper's thesis version lost several points to this).

Recipe (Section III-B): focal loss (gamma 2, alpha 0.6, label smoothing 0.02), class-balanced
sampler, SpecAugment-style masking, AdamW (3e-4, weight decay 1e-3), linear warm-up for
epochs/10 then cosine decay to 5 %, gradient clipping at 1.0, 30 epochs, batch 32. The
checkpoint with the best validation F1 is kept, with its F1-optimal validation threshold.
--synthetic N adds N physics-informed synthetic drone clips to the training set (paper: 139;
they matter in the field, not on the benchmark).

    dronedet-train --data data/ --out runs/seed42 --seed 42 --synthetic 139
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .features import DEFAULT, load_audio, window_features
from .metrics import binary_metrics, best_f1_threshold
from .model import DetectionCNN, FocalLoss, predict_proba, save_checkpoint

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
CLASSES = {"non_drone": 0, "drone": 1}


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def list_split(root: Path, split: str):
    files, labels = [], []
    for name, lab in CLASSES.items():
        d = root / split / name
        fs = sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT) if d.exists() else []
        files += fs; labels += [lab] * len(fs)
    return files, labels


def cached_features(files, cache: Path, cutoff_hz=None):
    """Feature tensors, cached on disk (one .npy per clip, keyed by path, size and mtime)."""
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        st = f.stat()
        key = hashlib.sha1(f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}|{cutoff_hz}".encode()).hexdigest()
        p = cache / f"{key}.npy"
        if not p.exists():
            np.save(p, window_features(load_audio(f), cutoff_hz))
        out.append(p)
    return out


def spec_augment(x: np.ndarray) -> np.ndarray:
    """The paper's masking: random gain, one time mask, one frequency mask, light noise."""
    _, F, T = x.shape
    if random.random() < 0.5:
        x *= random.uniform(0.85, 1.15)
    if random.random() < 0.35:
        tw = random.randint(2, max(2, T // 10)); ts = random.randint(0, max(0, T - tw))
        x[:, :, ts:ts + tw] *= random.uniform(0.0, 0.2)
    if random.random() < 0.25:
        fw = random.randint(2, max(2, F // 8)); fs = random.randint(0, max(0, F - fw))
        x[:, fs:fs + fw, :] *= random.uniform(0.0, 0.2)
    if random.random() < 0.30:
        x += np.random.randn(*x.shape).astype(np.float32) * 0.03
    return x


class FeatureDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths, self.labels, self.augment = list(paths), list(labels), augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = np.load(self.paths[i]).astype(np.float32)
        if self.augment:
            x = spec_augment(x)
        return torch.from_numpy(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)), torch.tensor(self.labels[i])


def warmup_cosine(epochs: int, min_factor: float = 0.05):
    warm, total = max(1, epochs // 10), max(epochs + 1, max(1, epochs // 10) + 1)

    def f(ep):
        if ep < warm:
            return (ep + 1) / warm
        prog = (ep - warm) / max(total - warm, 1)
        return min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * prog))
    return f


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--weight-decay", type=float, default=1e-3)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--synthetic", type=int, default=0, help="number of synthetic drone clips added to train")
    a.add_argument("--no-specaug", action="store_true")
    a.add_argument("--no-sampler", action="store_true")
    a.add_argument("--cache", default=None, help="feature cache folder (default <out>/feature_cache)")
    a.add_argument("--workers", type=int, default=0)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = a.parse_args(argv)
    set_seed(args.seed)
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else out / "feature_cache"

    splits = {s: list_split(root, s) for s in ("train", "val", "test")}
    for s in ("train", "val"):
        if not splits[s][0] or len(set(splits[s][1])) < 2:
            raise SystemExit(f"{root}/{s} needs both drone/ and non_drone/ audio files")
    if args.synthetic:
        from .synth import synthesise_drone
        import soundfile as sf
        sdir = out / "synthetic"; sdir.mkdir(exist_ok=True)
        rng = np.random.default_rng(args.seed + 1)
        for i in range(args.synthetic):
            p = sdir / f"synth_{i:05d}.wav"
            if not p.exists():
                sf.write(p, synthesise_drone(rng), DEFAULT.sr)
            splits["train"][0].append(p); splits["train"][1].append(1)
    for s, (f, l) in splits.items():
        if f:
            print(f"{s:5s}: {sum(l)} drone, {len(l) - sum(l)} non-drone")

    print("computing features (cached) ...")
    feats = {s: cached_features(f, cache) for s, (f, _) in splits.items()}
    tr = FeatureDataset(feats["train"], splits["train"][1], augment=not args.no_specaug)
    va = FeatureDataset(feats["val"], splits["val"][1])
    kw = dict(batch_size=args.batch_size, num_workers=args.workers)
    if args.no_sampler:
        tr_l = DataLoader(tr, shuffle=True, **kw)
    else:
        lab = np.array(tr.labels); cnt = np.bincount(lab, minlength=2).astype(float); cnt[cnt == 0] = 1
        w = (1.0 / cnt)[lab]
        tr_l = DataLoader(tr, sampler=WeightedRandomSampler(w, len(w), replacement=True), **kw)
    va_l = DataLoader(va, shuffle=False, **kw)

    dev = args.device
    model = DetectionCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(args.epochs))
    crit = FocalLoss()
    best, log = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = loss_sum = 0
        for X, y in tr_l:
            X, y = X.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(y); tot += len(y)
        sched.step()
        model.eval()
        yv = np.array(va.labels)
        pv = predict_proba(model, np.stack([np.load(p) for p in va.paths]), device=dev)
        thr = best_f1_threshold(yv, pv)
        m = binary_metrics(yv, pv, thr)
        log.append({"epoch": ep, "train_loss": loss_sum / max(tot, 1), "val_f1": m["f1"], "val_threshold": thr,
                    "val_auprc": m["auprc"]})
        flag = ""
        if m["f1"] > best:
            best = m["f1"]; flag = "  * best"
            save_checkpoint(out / "best_detection.pth", model, thr, epoch=ep, val_metrics=m, seed=args.seed)
        print(f"epoch {ep:3d}  loss {log[-1]['train_loss']:.4f}  val F1 {m['f1']:.4f}  thr {thr:.3f}{flag}")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))

    res = {"best_val_f1": best}
    if splits["test"][0]:
        from .model import load_checkpoint
        model, thr, _ = load_checkpoint(out / "best_detection.pth", dev)
        yt = np.array(splits["test"][1])
        pt = predict_proba(model, np.stack([np.load(p) for p in feats["test"]]), device=dev)
        res["test_at_val_threshold"] = binary_metrics(yt, pt, thr)
        t = res["test_at_val_threshold"]
        print(f"test (threshold {thr:.3f}): F1 {t['f1']:.4f}  precision {t['precision']:.4f}  "
              f"recall {t['recall']:.4f}  AUROC {t['auroc']:.4f}  AUPRC {t['auprc']:.4f}")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"saved {out / 'best_detection.pth'}")


if __name__ == "__main__":
    main()
