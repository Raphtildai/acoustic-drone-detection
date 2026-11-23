"""Train on a tiny synthetic task, then run every command (CPU, about a minute)."""
import csv
import json

import numpy as np
import soundfile as sf

from dronedet import DEFAULT
from dronedet import detect, evaluate, train
from dronedet.synth import synthesise_drone

SR = DEFAULT.sr


def non_drone(rng):
    n = DEFAULT.n_samples
    kind = rng.integers(3)
    if kind == 0:                                     # broadband noise
        return (0.1 * rng.standard_normal(n)).astype(np.float32)
    t = np.arange(n) / SR                             # a single steady tone (not a harmonic comb)
    if kind == 1:
        return (0.3 * np.sin(2 * np.pi * rng.uniform(800, 3000) * t)).astype(np.float32)
    return (0.05 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)).astype(np.float32)


def make_data(root, rng, n=24):
    for split in ("train", "val", "test"):
        for cls in ("drone", "non_drone"):
            d = root / split / cls; d.mkdir(parents=True)
            for i in range(n):
                y = synthesise_drone(rng) if cls == "drone" else non_drone(rng)
                sf.write(d / f"{i}.wav", y, SR)


def test_train_eval_detect(tmp_path):
    rng = np.random.default_rng(0)
    make_data(tmp_path / "data", rng)
    train.main(["--data", str(tmp_path / "data"), "--out", str(tmp_path / "run"), "--epochs", "3", "--seed", "1"])
    ck = tmp_path / "run" / "best_detection.pth"
    res = json.loads((tmp_path / "run" / "results.json").read_text())
    assert res["test_at_val_threshold"]["auroc"] > 0.9          # easy synthetic task

    evaluate.main(["bench", "--data", str(tmp_path / "data"), "--checkpoints", str(ck),
                   "--out", str(tmp_path / "bench.json")])
    assert json.loads((tmp_path / "bench.json").read_text())["auroc"][0] > 0.9

    # a long recording: alternating 30-s drone / no-drone blocks, labelled per 3-s window
    blocks, rows, t0 = [], [], 0.0
    for b in range(8):
        drone = b % 2 == 1
        for w in range(10):
            blocks.append(synthesise_drone(rng) if drone else non_drone(rng))
            rows.append({"start_s": t0, "label": int(drone), "split": "cal" if b < 4 else "test"})
            t0 += DEFAULT.window_s
    sf.write(tmp_path / "long.wav", np.concatenate(blocks), SR)
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_s", "label", "split"]); w.writeheader(); w.writerows(rows)
    evaluate.main(["field", "--audio", str(tmp_path / "long.wav"), "--labels", str(tmp_path / "labels.csv"),
                   "--checkpoints", str(ck), "--cutoffs", "none", "4000", "--out", str(tmp_path / "field.json")])
    field = json.loads((tmp_path / "field.json").read_text())
    assert field["selected_cutoff"] in ("none", "4000") and field["results"]

    detect.main([str(tmp_path / "long.wav"), "--checkpoint", str(ck), "--lowpass", "4000",
                 "--csv", str(tmp_path / "s.csv"), "--plot", str(tmp_path / "s.png")])
    lines = (tmp_path / "s.csv").read_text().splitlines()
    assert lines[0] == "start_s,end_s,score,smoothed,drone" and len(lines) > 100
    assert (tmp_path / "s.png").exists()
