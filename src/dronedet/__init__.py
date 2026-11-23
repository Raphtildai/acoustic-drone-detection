"""Compact acoustic drone detection (0.47 M-parameter CNN) with the training recipe and the
benchmark/field evaluation protocols of the accompanying paper."""

from .features import DEFAULT, FrontEnd, feature_stack, load_audio, lowpass, window_features
from .model import DetectionCNN, load_checkpoint, predict_proba, save_checkpoint

__version__ = "0.1.0"
__all__ = ["DEFAULT", "FrontEnd", "feature_stack", "load_audio", "lowpass", "window_features",
           "DetectionCNN", "load_checkpoint", "predict_proba", "save_checkpoint"]
