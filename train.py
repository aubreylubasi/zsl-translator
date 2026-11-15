"""Train the sign recogniser on everything in data/.

    python train.py
    python train.py --test webcam_P02     # honest test: hold out a recording of a
                                          # person the model has never seen

Saves models/zsl_model.joblib, which live.py uses.
"""
import argparse
import glob
import os
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zsl.features import FEATURE_DIM, mirror

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(data_dir):
    X, y, src, pos = [], [], [], []
    backends = set()
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(path, allow_pickle=False)
        if d["X"].shape[1] != FEATURE_DIM or len(d["X"]) == 0:
            print(f"skipping {os.path.basename(path)} (empty or old format)")
            continue
        name = str(d["src"])
        backends.add(str(d["backend"]) if "backend" in d.files else "unknown")
        X.append(d["X"]); y.append(d["y"]); pos.append(d["pos"])
        src.append(np.full(len(d["X"]), name))
        print(f"  {name}: {len(d['X'])} examples, {len(set(d['y']))} signs")
    if not X:
        sys.exit("No data yet. Run video_to_dataset.py or collect.py first.")
    if len(backends) > 1:
        print(f"Note: data was made with different MediaPipe versions ({', '.join(sorted(backends))}). "
              "That's fine, but tracking can differ slightly between them.")
    load.backends = backends
    return np.concatenate(X), np.concatenate(y), np.concatenate(src), np.concatenate(pos)


def augment(X, y, use_mirror=True, copies=2, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    if use_mirror:
        Xs.append(mirror(X)); ys.append(y)
    base = np.concatenate(Xs)
    yb = np.concatenate(ys)
    out_X, out_y = [base], [yb]
    for _ in range(copies):  # small wobbles, like a slightly different hand
        noise = rng.normal(0, 0.04, base.shape).astype(np.float32) * (base != 0)
        out_X.append(base + noise); out_y.append(yb)
    return np.concatenate(out_X), np.concatenate(out_y)


def make_model():
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300, random_state=0))


def export_json(model, labels, path):
    """Save the model as plain numbers for the phone app (app/www)."""
    import json
    scaler, mlp = model.steps[0][1], model.steps[1][1]
    r = lambda a: [float(f"{v:.6g}") for v in np.asarray(a).ravel()]
    data = {
        "labels": [str(s) for s in labels],
        "feature_dim": FEATURE_DIM,
        "mean": r(scaler.mean_), "scale": r(scaler.scale_),
        "activation": mlp.activation,
        "layers": [{"in": int(W.shape[0]), "out": int(W.shape[1]), "W": r(W), "b": r(b)}
                   for W, b in zip(mlp.coefs_, mlp.intercepts_)],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def report(model, Xt, yt, title):
    pred = model.predict(Xt)
    acc = (pred == yt).mean()
    print(f"\n{title}: {acc * 100:.1f}% of {len(yt)} test frames correct")
    labels = sorted(set(yt))
    per = {lab: (pred[yt == lab] == lab).mean() for lab in labels}
    worst = sorted(per.items(), key=lambda kv: kv[1])[:5]
    print("  hardest signs: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in worst))
    mix = Counter((t, p) for t, p in zip(yt, pred) if t != p).most_common(5)
    if mix:
        print("  most mixed up: " + ", ".join(f"{t}->{p} ({n})" for (t, p), n in mix))
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "zsl_model.joblib"))
    ap.add_argument("--test", help="hold out data files whose name starts with this (e.g. a different signer)")
    ap.add_argument("--no-mirror", action="store_true", help="don't add left-handed copies")
    args = ap.parse_args()

    print("Loading data:")
    X, y, src, pos = load(args.data)
    print(f"Total: {len(X)} examples, {len(set(y))} signs, {len(set(src))} source(s)")

    if args.test:
        test = np.char.startswith(src.astype(str), args.test)
        if not test.any() or test.all():
            sys.exit(f"--test {args.test!r} must match some, but not all, data files")
        title = f"Test on unseen recording(s) '{args.test}'"
    else:
        test = pos >= 0.65  # the later part of every sign
        title = "Test on held-back frames (same signer(s) - real-world accuracy will be lower)"

    Xa, ya = augment(X[~test], y[~test], not args.no_mirror)
    model = make_model().fit(Xa, ya)
    report(model, X[test], y[test], title)

    print("\nTraining final model on all data...")
    Xa, ya = augment(X, y, not args.no_mirror)
    model = make_model().fit(Xa, ya)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": model, "labels": list(model.classes_), "feature_dim": FEATURE_DIM,
                 "sources": sorted(set(src)), "backend": sorted(load.backends)[0]}, args.out)
    print(f"Saved -> {os.path.relpath(args.out, ROOT)}   Now run: python live.py")
    web = os.path.join(ROOT, "app", "www", "zsl_model.json")
    if os.path.isdir(os.path.join(ROOT, "app", "www")):
        export_json(model, model.classes_, web)
        print(f"Phone app model -> {os.path.relpath(web, ROOT)}")


if __name__ == "__main__":
    main()
