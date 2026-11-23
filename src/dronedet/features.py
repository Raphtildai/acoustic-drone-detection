"""Audio front end: loading, optional input low-pass, and the 3-channel feature tensor.

The tensor is exactly the one used in the paper: log-mel, PCEN and the temporal delta of the
log-mel spectrogram, each standardized to zero mean and unit variance over the window, giving
3 x 64 x 259 for a 3-s window at 22.05 kHz (STFT 1024 / hop 256, 64 mel bands, 20-8000 Hz).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import librosa
import numpy as np
import scipy.signal


@dataclass(frozen=True)
class FrontEnd:
    sr: int = 22050
    window_s: float = 3.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8000.0
    # PCEN (librosa.pcen)
    pcen_gain: float = 0.8
    pcen_bias: float = 10.0
    pcen_power: float = 0.25
    pcen_time_constant: float = 0.4

    @property
    def n_samples(self) -> int:
        return int(self.sr * self.window_s)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = FrontEnd()


def load_audio(path, sr: int = DEFAULT.sr, channel: Optional[int] = None) -> np.ndarray:
    """Load a file as float32 at `sr`. `channel` selects one channel of a multichannel file
    (default: the mean of all channels)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y, _ = librosa.load(str(path), sr=sr, mono=channel is None)
    if channel is not None and y.ndim > 1:
        y = y[channel]
    return np.asarray(y, dtype=np.float32)


def pad_or_truncate(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


@lru_cache(maxsize=16)
def _butter_sos(cutoff_hz: float, sr: int, order: int = 8):
    return scipy.signal.butter(order, cutoff_hz, btype="low", fs=sr, output="sos")


def lowpass(y: np.ndarray, cutoff_hz: Optional[float], sr: int = DEFAULT.sr) -> np.ndarray:
    """8th-order Butterworth low-pass (causal, as in the paper). `cutoff_hz=None` returns y."""
    if not cutoff_hz:
        return np.asarray(y, dtype=np.float32)
    return scipy.signal.sosfilt(_butter_sos(float(cutoff_hz), sr), y).astype(np.float32)


def standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)


def delta(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.diff(x, axis=1, prepend=x[:, :1]).astype(np.float32)


def mel_power(y: np.ndarray, fe: FrontEnd = DEFAULT) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y, sr=fe.sr, n_fft=fe.n_fft, hop_length=fe.hop_length,
        n_mels=fe.n_mels, fmin=fe.fmin, fmax=fe.fmax).astype(np.float32)


def feature_stack(y: np.ndarray, fe: FrontEnd = DEFAULT) -> np.ndarray:
    """[log-mel, PCEN, delta(log-mel)], each standardized -> float32 (3, n_mels, frames)."""
    M = mel_power(np.asarray(y, dtype=np.float32), fe)
    logmel = librosa.power_to_db(M, ref=np.max).astype(np.float32)
    P = librosa.pcen(M, sr=fe.sr, hop_length=fe.hop_length, gain=fe.pcen_gain, bias=fe.pcen_bias,
                     power=fe.pcen_power, time_constant=fe.pcen_time_constant, eps=1e-6).astype(np.float32)
    m = standardize(logmel)
    return np.stack([m, standardize(P), standardize(delta(m))], axis=0).astype(np.float32)


def window_features(y: np.ndarray, cutoff_hz: Optional[float] = None, fe: FrontEnd = DEFAULT) -> np.ndarray:
    """One window of raw audio -> feature tensor, with the optional input low-pass."""
    return feature_stack(lowpass(pad_or_truncate(y, fe.n_samples), cutoff_hz, fe.sr), fe)
