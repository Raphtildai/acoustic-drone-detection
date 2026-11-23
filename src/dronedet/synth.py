"""Physics-informed synthetic drone clips (Section III-B of the paper).

A blade-pass harmonic series (fundamental drawn around a measured blade-pass frequency, k^-1.3
roll-off, random phases, 0.5-Hz frequency modulation) plus a noise floor is propagated to each
microphone with a fractional delay d/c and d^-0.6 attenuation; the detector uses the mean over
the microphones.

Differences from the paper's generator, stated so results can be compared fairly:
  * the paper added indoor noise floors measured by the partner laboratory; those measurements
    are not redistributed. Here the noise floor is Brownian (wind-like, outdoor) or pink-ish
    broadband, or a real background recording you pass in (`background=`), which is better.
  * the microphone geometry defaults to the three UaVirBASE microphones used in the paper
    (1.72 m radius); any (n, 2) array of positions in metres works.

    dronedet-synth --out data/synthetic/drone --n 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf

from .features import DEFAULT

# blade-pass frequency profiles: (f_min, f_center, f_max, n_harmonics) in Hz.
# Centres for the three DJI types are laboratory measurements reported in the paper.
BPF_PROFILES = {
    "mavic_pro":    (160, 209, 340, 4),
    "mavic_2_pro":  (160, 195, 340, 4),
    "mavic_mini":   (260, 360, 620, 4),
    "generic_quad": (80, 110, 350, 4),
}
TYPE_WEIGHTS = {"mavic_pro": 0.30, "mavic_2_pro": 0.30, "mavic_mini": 0.25, "generic_quad": 0.15}
SPEED_OF_SOUND = 343.0
R = 1.72
UAVIRBASE_MICS = np.array([[0.0, R], [R, 0.0], [-R, 0.0]], dtype=np.float32)


def sample_fundamental(drone_type: str, rng: np.random.Generator) -> float:
    lo, c, hi, _ = BPF_PROFILES[drone_type]
    return float(np.clip(rng.normal(c, (hi - lo) / 6.0), lo, hi))


def fractional_delay(x: np.ndarray, delay: float) -> np.ndarray:
    """Delay by a non-integer number of samples (8-tap windowed sinc + integer shift)."""
    n, d = len(x), max(0.0, float(delay))
    i, f = int(np.floor(d)), d - np.floor(d)
    if f > 1e-6:
        taps = np.arange(-3, 5)
        h = np.sinc(taps - f) * np.hanning(len(taps))
        x = np.convolve(x.astype(np.float64), h / (h.sum() + 1e-12), mode="full")[:n]
    out = np.zeros(n, np.float32)
    if i < n:
        out[i:] = x[:n - i]
    return out


def noise_floor(n: int, sr: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    wn = rng.standard_normal(n)
    if kind == "outdoor":                     # Brownian (f^-2), high-passed at 5 Hz
        x = scipy.signal.sosfilt(scipy.signal.butter(2, 5.0, btype="high", fs=sr, output="sos"), np.cumsum(wn))
    else:                                     # pink-ish broadband
        b, a = scipy.signal.butter(1, 0.05)
        x = scipy.signal.lfilter(b, a, wn)
    return (x / (np.std(x) + 1e-8)).astype(np.float64)


def synthesise_drone(rng: np.random.Generator, drone_type: Optional[str] = None, distance_m: Optional[float] = None,
                     noise_level: Optional[float] = None, noise_kind: Optional[str] = None,
                     mics: np.ndarray = UAVIRBASE_MICS, background: Optional[np.ndarray] = None,
                     snr_db: Optional[float] = None, sr: int = DEFAULT.sr, duration: float = DEFAULT.window_s) -> np.ndarray:
    """One mono synthetic drone clip (float32, peak <= 1). With `background` (a real recording
    of at least `duration` s at `sr`), the clip is mixed into it at `snr_db` instead of the
    synthetic noise floor."""
    types = list(TYPE_WEIGHTS)
    drone_type = drone_type or str(rng.choice(types, p=[TYPE_WEIGHTS[t] for t in types]))
    n = int(sr * duration)
    t = np.arange(n) / sr
    f0 = sample_fundamental(drone_type, rng)
    src = np.zeros(n)
    jit = 1.0 + 0.003 * np.sin(2 * np.pi * 0.5 * t)
    for k in range(1, BPF_PROFILES[drone_type][3] + 1):
        amp = k ** -1.3 * (0.9 + 0.2 * rng.random())
        src += amp * np.sin(2 * np.pi * f0 * k * jit * t + rng.uniform(0, 2 * np.pi))
    if background is None:
        # as in the paper's generator: the floor's std is 0.5 * level^2 (level in [0.05, 0.10]),
        # i.e. a nearly clean harmonic signal; use `background=` for realistic mixtures
        lvl = max(noise_level if noise_level is not None else rng.uniform(0.02, 0.10), 0.05)
        src += lvl * lvl * 0.5 * noise_floor(n, sr, noise_kind or str(rng.choice(["outdoor", "broadband"])), rng)
    r = distance_m if distance_m is not None else rng.uniform(0.3, 30.0)
    th = rng.uniform(0, 2 * np.pi)
    pos = mics.mean(0) + r * np.array([np.cos(th), np.sin(th)])
    chans = []
    for m in mics:
        d = max(float(np.linalg.norm(pos - m)), 0.01)
        chans.append(fractional_delay(src / (d ** 0.6 + 0.1), d / SPEED_OF_SOUND * sr))
    y = np.mean(chans, axis=0)
    if background is not None:
        bg = np.asarray(background[:n], np.float64)
        snr = snr_db if snr_db is not None else rng.uniform(-5, 15)
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * np.sqrt(np.mean(bg ** 2)) * 10 ** (snr / 20) + bg
    return (y / max(1.0, float(np.max(np.abs(y))) / 0.99)).astype(np.float32)


def main(argv=None):
    a = argparse.ArgumentParser(description="Write synthetic drone clips as WAV files.")
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--seed", type=int, default=43)
    a.add_argument("--backgrounds", default=None,
                   help="optional folder of real background recordings to mix into (recommended)")
    args = a.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bgs = []
    if args.backgrounds:
        from .features import load_audio
        bgs = [b for b in (load_audio(p) for p in sorted(Path(args.backgrounds).rglob("*.wav")))
               if len(b) >= DEFAULT.n_samples]
        if not bgs:
            raise SystemExit(f"no background WAV of >= {DEFAULT.window_s} s in {args.backgrounds}")
    for i in range(args.n):
        bg = None
        if bgs:
            b = bgs[rng.integers(len(bgs))]
            s = rng.integers(0, len(b) - DEFAULT.n_samples + 1)
            bg = b[s:s + DEFAULT.n_samples]
        sf.write(out / f"synth_{i:05d}.wav", synthesise_drone(rng, background=bg), DEFAULT.sr)
    print(f"wrote {args.n} clips to {out}")


if __name__ == "__main__":
    main()
