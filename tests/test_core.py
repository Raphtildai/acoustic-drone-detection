import numpy as np
import torch

from dronedet import DEFAULT, DetectionCNN, feature_stack, load_checkpoint, lowpass, save_checkpoint, window_features
from dronedet.metrics import best_f1_threshold, budget_threshold, smooth
from dronedet.synth import synthesise_drone


def test_feature_shape_and_standardization():
    x = feature_stack(np.random.default_rng(0).standard_normal(DEFAULT.n_samples).astype(np.float32))
    assert x.shape == (3, 64, 259) and x.dtype == np.float32
    assert np.allclose(x.reshape(3, -1).mean(1), 0, atol=1e-4)
    assert np.allclose(x.reshape(3, -1).std(1), 1, atol=1e-3)


def test_short_input_is_padded():
    assert window_features(np.zeros(1000, np.float32)).shape == (3, 64, 259)


def test_lowpass_removes_high_band():
    sr = DEFAULT.sr
    t = np.arange(sr) / sr
    y = (np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    Y = np.abs(np.fft.rfft(lowpass(y, 2000, sr)))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    assert Y[np.argmin(abs(f - 6000))] < 1e-3 * Y[np.argmin(abs(f - 500))]
    assert np.allclose(lowpass(y, None), y)


def test_model_size_and_checkpoint_roundtrip(tmp_path):
    m = DetectionCNN().eval()
    n = sum(p.numel() for p in m.parameters())
    assert 460_000 < n < 480_000                       # the paper's 0.47 M
    x = torch.randn(2, 3, 64, 259)
    save_checkpoint(tmp_path / "m.pth", m, 0.7)
    m2, thr, fe = load_checkpoint(tmp_path / "m.pth")
    assert thr == 0.7 and fe == DEFAULT
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x))


def test_synthetic_clip_has_blade_pass_harmonics():
    rng = np.random.default_rng(1)
    y = synthesise_drone(rng, drone_type="mavic_mini", distance_m=5.0)
    assert y.shape == (DEFAULT.n_samples,) and np.max(np.abs(y)) <= 1.0
    f = np.fft.rfftfreq(len(y), 1 / DEFAULT.sr)
    peak = f[np.argmax(np.abs(np.fft.rfft(y)))]
    assert 260 <= peak <= 620                           # fundamental inside the Mavic Mini band


def test_smoothing_is_causal_and_restarts_at_gaps():
    t = np.array([0, 3, 6, 9, 30, 33.0]); p = np.array([0, 0, 1, 1, 1, 0.0])
    s = smooth(t, p, 2, 3.0)
    assert np.allclose(s, [0, 0, 0.5, 1, 1, 0.5])       # restarts after the gap at t = 30


def test_thresholds():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1]); p = np.array([.1, .2, .3, .9, .6, .7, .8, .95])
    assert 0.3 < best_f1_threshold(y, p) <= 0.6
    thr = budget_threshold(y, p, budget_per_h=0, window_s=3.0)     # no false alarm allowed
    assert thr > 0.9
