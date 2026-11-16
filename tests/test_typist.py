"""Quick self-test of the typing rules. No camera or MediaPipe needed.

    python tests/test_typist.py

Checks that a still hand gets typed, a moving hand doesn't, "not a sign" is
never typed, and (if Node.js is installed) that the phone app's zsl.js gives
exactly the same answers as Python.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zsl.typist import MotionMeter, Typist  # noqa: E402

LABELS = ["A", "B", "_"]
FPS = 30.0


def fake_hand(cx, cy, size=0.08):
    """21 points shaped roughly like an open hand, wrist at (cx, cy)."""
    rng = np.random.default_rng(1)
    pts = np.zeros((21, 3))
    pts[1:, 0] = rng.uniform(-0.6, 0.6, 20) * size
    pts[1:, 1] = -rng.uniform(0.3, 1.6, 20) * size
    pts[9] = (0.0, -size, 0.0)  # middle knuckle: palm length = size
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def stream(seconds, speed_palms, top="A", jitter=0.002, seed=0):
    """Frames of (t, hands, proba). speed_palms: how fast the hand slides sideways."""
    rng = np.random.default_rng(seed)
    proba = np.full(len(LABELS), 0.05)
    proba[LABELS.index(top)] = 0.9
    out = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        hand = fake_hand(0.3 + speed_palms * 0.08 * t / (16 / 9), 0.6)
        hand[:, :2] += rng.normal(0, jitter, (21, 2))
        out.append((t, [hand], proba / proba.sum()))
    return out


def run_python(frames, **kw):
    ty, me, last, speeds = Typist(LABELS, **kw), MotionMeter(), None, []
    for t, hands, proba in frames:
        sp = me.update(hands, 16 / 9, t)
        speeds.append(sp)
        ty.update(proba, 0.0 if last is None else t - last, t, sp)
        last = t
    return ty.text, speeds


def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}, expected {want!r}")
    return ok


def main():
    cases = {
        "still hand is typed": (stream(1.5, 0.0), {}, "A"),
        "moving hand is not typed": (stream(1.5, 3.0), {}, ""),
        "moving hand, rule off (--still 0)": (stream(1.5, 3.0), {"still": 0}, "A"),
        "'not a sign' is never typed": (stream(1.5, 0.0, top="_"), {}, ""),
    }
    print("Python:")
    good = all(check(name, run_python(fr, **kw)[0], want) for name, (fr, kw, want) in cases.items())

    node = shutil.which("node")
    if not node:
        print("Node.js not installed, so the phone app check was skipped.")
    else:
        print("Phone app (zsl.js) vs Python:")
        js = os.path.join(ROOT, "app", "www", "js", "zsl.js")
        for name, (fr, kw, _) in cases.items():
            text, speeds = run_python(fr, **kw)
            payload = {"labels": LABELS, "opts": kw, "frames": [
                {"t": t, "hands": [[{"x": p[0], "y": p[1], "z": p[2]} for p in h] for h in hands],
                 "proba": list(map(float, proba))} for t, hands, proba in fr]}
            script = """
const Z = require(process.argv[1]);
const d = JSON.parse(require("fs").readFileSync(0, "utf8"));
const ty = new Z.Typist(d.labels, d.opts), me = new Z.MotionMeter(0.2);
let last = null; const speeds = [];
for (const f of d.frames) {
  const sp = me.update(f.hands, 16 / 9, f.t); speeds.push(sp);
  ty.update(f.proba, last === null ? 0 : f.t - last, sp); last = f.t;
}
console.log(JSON.stringify({text: ty.text, speeds}));"""
            res = subprocess.run([node, "-e", script, js], input=json.dumps(payload),
                                 capture_output=True, text=True, check=True)
            out = json.loads(res.stdout)
            same_speed = all((a is None) == (b is None) and (a is None or abs(a - b) < 1e-9)
                             for a, b in zip(speeds, out["speeds"]))
            good &= check(name, (out["text"], same_speed), (text, True))
    print("\nAll good." if good else "\nSomething is different - see FAIL above.")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
