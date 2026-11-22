"""Jarvis wake word: a small classifier trained on this user's voice plus synthetic voices.

It sits on openWakeWord's frozen audio features (the same ones its stock models use): every 80 ms one 96-number embedding is
produced, and the last 16 of them (about 1.3 s) go through a tiny network that outputs "is the word 'Jarvis' just finished?".
It hears only sound features, never words, so nothing is transcribed while Tommy is idle.
"""
import importlib.machinery, types, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "tommy" / "jarvis_wake.npz"
FRAMES = 16


def _features():
    for name, attrs in {"sklearn": [], "sklearn.linear_model": ["LogisticRegression"], "sklearn.pipeline": ["make_pipeline"],
                        "sklearn.preprocessing": ["FunctionTransformer", "StandardScaler"]}.items():
        mod = types.ModuleType(name); mod.__path__ = []
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)      # torch calls find_spec() on every loaded module
        for attr in attrs:
            setattr(mod, attr, object)
        sys.modules.setdefault(name, mod)
    from openwakeword.utils import AudioFeatures
    return AudioFeatures(inference_framework="onnx")


def forward(w, x):
    """x: (n, 16, 96) float32 embeddings -> (n,) scores in 0..1."""
    x = ((x.reshape(len(x), -1) - w["mean"]) / w["std"]).astype(np.float32)
    h = np.maximum(x @ w["w1"] + w["b1"], 0)
    return 1 / (1 + np.exp(-(h @ w["w2"] + w["b2"]).reshape(-1)))


class JarvisWake:
    def __init__(self, weights=WEIGHTS):
        self.w = dict(np.load(weights))
        self.threshold = float(self.w["threshold"]) if "threshold" in self.w else 0.5
        self.feats = _features()
        self.prev = 0.0

    def predict(self, frame):
        """frame: 1280 int16 samples (80 ms). Returns the score for the last ~1.3 s of audio."""
        self.feats(frame)
        if len(self.feats.feature_buffer) < FRAMES:
            return 0.0
        score = float(forward(self.w, self.feats.get_features(FRAMES))[0])
        fired, self.prev = min(score, self.prev), score          # two high scores in a row: one stray spike never wakes it
        return fired

    def reset(self):
        self.feats = _features()
        self.prev = 0.0
