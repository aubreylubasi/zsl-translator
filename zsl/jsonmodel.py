"""Use the phone app's model file (app/www/zsl_model.json) from Python.

train.py writes that file next to the desktop model. Loading it here means
live.py can run straight away on a computer that hasn't trained anything yet.
The maths is the same as scikit-learn's predict_proba and app/www/js/zsl.js.
"""
import json

import numpy as np


class JsonModel:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("activation", "relu") != "relu":
            raise ValueError("only relu models are supported")
        self.labels = list(d["labels"])
        self.moving = list(d.get("moving", []))
        self.feature_dim = int(d["feature_dim"])
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.scale = np.array(d["scale"], dtype=np.float64)
        self.layers = [(np.array(L["W"], dtype=np.float64).reshape(L["in"], L["out"]),
                        np.array(L["b"], dtype=np.float64)) for L in d["layers"]]

    def predict_proba(self, X):
        a = (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mean) / self.scale
        for k, (W, b) in enumerate(self.layers):
            a = a @ W + b
            if k < len(self.layers) - 1:
                a = np.maximum(a, 0.0)
        a = np.exp(a - a.max(axis=1, keepdims=True))
        return a / a.sum(axis=1, keepdims=True)
