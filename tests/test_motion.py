"""Self-test of the motion model. No camera, MediaPipe or files needed.

    python tests/test_motion.py

Makes two fake word signs with the same handshape and the same end position
but opposite movements (LEFT and RIGHT), then checks that:
  * a single-frame model mixes them up, and the motion model doesn't
  * streamed live, the motion model types them correctly
  * (if Node.js is installed) the phone app's zsl.js Window, model and typing
    give the same answers as Python
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train import export_json, make_model, make_windows  # noqa: E402
from zsl.features import FEATURE_DIM, Window  # noqa: E402
from zsl.jsonmodel import JsonModel  # noqa: E402
from zsl.typist import Typist  # noqa: E402

rng = np.random.default_rng(0)
SHAPE = rng.normal(0, 1, 63)


def frame(x, shape):
    f = np.zeros(FEATURE_DIM, np.float32)
    f[:63] = shape + rng.normal(0, 0.05, 63)
    f[63], f[64], f[65], f[66] = x, -0.2, 0.32, 1
    return f


def take(lab, n=45):
    out = []
    for i in range(n):
        u = min(i / (n - 1) / 0.7, 1)  # move for 70% of the take, then hold
        x = {"LEFT": 0.8 - 0.8 * u, "RIGHT": -0.8 + 0.8 * u, "A": 0.0}[lab]
        out.append(frame(x, -SHAPE if lab == "A" else SHAPE))
    return out


X, y, src, seg, t, pos = [], [], [], [], [], []
for k in range(36):
    lab = ["LEFT", "RIGHT", "A"][k % 3]
    for i, f in enumerate(take(lab)):
        X.append(f); y.append(lab); src.append("fake"); seg.append(k); t.append(i / 30); pos.append(i / 44)
X, y, src, seg, t, pos = map(np.array, (X, y, src, seg, t, pos))
test = (seg >= 30) & (pos >= 0.65)

ok = True


def check(name, good, detail):
    global ok
    ok &= bool(good)
    print(f"  {'ok  ' if good else 'FAIL'} {name}: {detail}")


single = make_model().fit(X[~test], y[~test])
acc1 = (single.predict(X[test]) == y[test]).mean()
W = make_windows(X, src, seg, t)
motion = make_model().fit(W[~test], y[~test])
acc2 = (motion.predict(W[test]) == y[test]).mean()
check("single-frame model can't tell LEFT from RIGHT", acc1 < 0.9, f"{acc1 * 100:.0f}% correct")
check("motion model can", acc2 > 0.95, f"{acc2 * 100:.0f}% correct")

tmp = os.path.join(ROOT, "tests", "_motion_model.json")
export_json(motion, motion.classes_, tmp, ["LEFT", "RIGHT"], {"frames": 6, "seconds": 1.2})
m = JsonModel(tmp)
stream, tt = [], 0.0
for lab in ["LEFT", "RIGHT", "A", "RIGHT"]:
    for f in take(lab):
        stream.append((tt, f.tolist())); tt += 1 / 30
    for _ in range(36):
        stream.append((tt, None)); tt += 1 / 30
win, ty, last, probs = Window(6, 1.2), Typist(m.labels, free_move=m.moving), None, []
for tt, f in stream:
    x = win.update(tt, None if f is None else np.array(f, np.float32))
    p = None if x is None else m.predict_proba(x.reshape(1, -1))[0]
    probs.append(None if p is None else p.tolist())
    ty.update(p, 0 if last is None else tt - last, tt, speed=None if f is None else 0.0)
    last = tt
check("streamed live", ty.text == "LEFT RIGHT A RIGHT ", repr(ty.text))

node = shutil.which("node")
if node:
    js = """
const Z=require(process.argv[1]);const m=new Z.Model(require(process.argv[2]));
const s=JSON.parse(require('fs').readFileSync(0,'utf8'));const w=new Z.Window(m.window.frames,m.window.seconds);
const ty=new Z.Typist(m.labels,{freeMove:m.moving});let last=null;const probs=[];
for(const [t,f] of s){const x=w.update(t,f?Float32Array.from(f):null);const p=x?m.predict(x):null;
probs.push(p?Array.from(p):null);ty.update(p,last==null?0:t-last,f?0.0:null);last=t;}
console.log(JSON.stringify({text:ty.text,probs}));"""
    r = subprocess.run([node, "-e", js, os.path.join(ROOT, "app", "www", "js", "zsl.js"), tmp],
                       input=json.dumps(stream), capture_output=True, text=True)
    if r.returncode:
        check("phone app", False, r.stderr.strip().splitlines()[-1])
    else:
        out = json.loads(r.stdout)
        diff = max(abs(a - b) for pa, pb in zip(probs, out["probs"]) if pa for a, b in zip(pa, pb))
        check("phone app types the same", out["text"] == ty.text, repr(out["text"]))
        check("phone app maths matches Python", diff < 1e-6, f"largest difference {diff:.1e}")
else:
    print("Node.js not installed, so the phone app check was skipped.")
os.remove(tmp)
print("\nAll good." if ok else "\nSomething is wrong - see FAIL above.")
sys.exit(0 if ok else 1)
