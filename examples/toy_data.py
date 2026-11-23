"""Write a toy dataset (folder layout of dronedet-train) and a labelled long recording."""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from dronedet import DEFAULT
from dronedet.synth import synthesise_drone

SR, N = DEFAULT.sr, DEFAULT.n_samples


def non_drone(rng):
    t = np.arange(N) / SR
    kind = rng.integers(3)
    if kind == 0:
        return (0.1 * rng.standard_normal(N)).astype(np.float32)
    if kind == 1:
        return (0.3 * np.sin(2 * np.pi * rng.uniform(800, 3000) * t)).astype(np.float32)
    return (0.05 * np.cumsum(rng.standard_normal(N)) / np.sqrt(N)).astype(np.float32)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--out", default="toy")
    a.add_argument("--n", type=int, default=60, help="clips per class and split")
    args = a.parse_args()
    rng, out = np.random.default_rng(0), Path(args.out)
    for split in ("train", "val", "test"):
        for cls in ("drone", "non_drone"):
            d = out / "data" / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(args.n):
                sf.write(d / f"{i:04d}.wav", synthesise_drone(rng) if cls == "drone" else non_drone(rng), SR)
    blocks, rows, t0 = [], [], 0.0
    for b in range(12):                         # 30-s blocks, alternating drone / no drone
        drone = b % 2 == 1
        for _ in range(10):
            blocks.append(synthesise_drone(rng) if drone else non_drone(rng))
            rows.append({"start_s": t0, "label": int(drone), "split": "cal" if b < 6 else "test"})
            t0 += DEFAULT.window_s
    sf.write(out / "long.wav", np.concatenate(blocks), SR)
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_s", "label", "split"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}/data, {out}/long.wav, {out}/labels.csv")


if __name__ == "__main__":
    main()
