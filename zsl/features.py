"""Turn tracked landmarks into a fixed-length list of numbers the AI can learn from.

Two hand slots: the "main" hand (the higher one, which is the hand doing the
signing) and the "other" hand. Ordering by height instead of MediaPipe's
Left/Right labels keeps the numbers identical across MediaPipe versions and
platforms (Python, browser, Android), which label hands differently.

For each slot we keep:
  * hand shape: 21 points relative to the wrist, scaled by palm size
    (so it doesn't matter how far the person stands from the camera)
  * where the hand is relative to the shoulders (signs near the face vs chest)
  * hand size relative to shoulder width
  * a 1/0 flag for "this hand is visible"

The phone app (app/www/js/zsl.js) computes exactly the same numbers.
"""
import re

import numpy as np

N_SLOTS = 2
HAND_DIM = 21 * 3 + 2 + 1 + 1
FEATURE_DIM = HAND_DIM * N_SLOTS


def frame_features(lm):
    """Landmarks -> 1-D float32 vector, or None if no hand is visible."""
    if not lm.hands:
        return None
    a = lm.aspect
    if lm.pose is not None:
        l_sh = lm.pose[11, :2] * (a, 1)
        r_sh = lm.pose[12, :2] * (a, 1)
        centre = (l_sh + r_sh) / 2
        width = max(float(np.linalg.norm(l_sh - r_sh)), 1e-3)
    else:  # no body found: assume a person roughly centred in the frame
        centre = np.array([a / 2, 0.6])
        width = 0.35
    hands = sorted(lm.hands.values(), key=lambda h: float(h[0, 1]))[:N_SLOTS]  # highest wrist first
    parts = []
    for slot in range(N_SLOTS):
        if slot >= len(hands):
            parts.append(np.zeros(HAND_DIM, dtype=np.float32))
            continue
        pts = hands[slot].astype(np.float32) * (a, 1, a)  # undo the image's stretched x axis
        rel = pts - pts[0]
        palm = max(float(np.linalg.norm(rel[9, :2])), 1e-4)  # wrist -> middle knuckle
        shape = (rel / palm).ravel()
        pos = (pts[0, :2] - centre) / width
        parts.append(np.concatenate([shape, pos, [palm / width], [1.0]]).astype(np.float32))
    return np.concatenate(parts)


def mirror(X):
    """Flip left/right, so the model also works for left-handed signers."""
    X = np.asarray(X, dtype=np.float32).reshape(-1, N_SLOTS, HAND_DIM).copy()
    X[:, :, 0:63:3] *= -1   # x of every hand point
    X[:, :, 63] *= -1       # x of hand position
    return X.reshape(-1, FEATURE_DIM)


def parse_labels(text):
    """'A-Z' -> ['A'..'Z'];  'HELLO,THANK YOU' -> ['HELLO','THANK YOU'];  path to a .txt -> one per line."""
    import os
    if os.path.isfile(text):
        with open(text, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    m = re.fullmatch(r"\s*([A-Za-z0-9])\s*-\s*([A-Za-z0-9])\s*", text)
    if m:
        lo, hi = m.group(1), m.group(2)
        return [chr(c) for c in range(ord(lo), ord(hi) + 1)]
    return [t.strip() for t in text.split(",") if t.strip()]
