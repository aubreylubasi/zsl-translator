"""Turning per-frame guesses into typed text.

Used by live.py. The phone app (app/www/js/zsl.js) does exactly the same.

A sign is typed when the same guess has lasted `hold` seconds (0.5) AND the
hand has been still for the last `still_time` seconds of that (0.15).

Two rules stop junk letters while someone signs words:
  * the hand must settle. MotionMeter measures how fast the signing hand
    moves, in palm-lengths per second. A held letter settles; word signs and
    the moves between letters mostly don't.
  * the model may say "_" (not a sign). That is never typed.
"""
from collections import deque

import numpy as np

from .features import NOT_A_SIGN

MOVING_SIGNS = ("J", "Z")  # letters that are signed with a movement get a looser limit


class MotionMeter:
    """How fast the signing hand (the highest one) is moving, in palm-lengths per second.

    It compares the hand now with where it was `window` seconds ago, so the
    tracker's small frame-to-frame jitter doesn't count as movement.
    """

    def __init__(self, window=0.2):
        self.window = window
        self.hist = deque()  # (t, (21, 2) points)
        self.speed = None

    def reset(self):
        self.hist.clear()
        self.speed = None

    def update(self, hands, aspect, t):
        """hands: list of (21, 3) arrays in 0..1 image coordinates. Returns speed, or None if unknown."""
        if not hands:
            self.reset()
            return None
        hand = min(hands, key=lambda h: float(h[0][1]))  # highest wrist = signing hand
        pts = np.asarray(hand, dtype=np.float64)[:, :2] * (aspect, 1.0)
        palm = max(float(np.hypot(*(pts[9] - pts[0]))), 1e-4)
        self.hist.append((t, pts))
        while len(self.hist) > 2 and t - self.hist[1][0] >= self.window:
            self.hist.popleft()
        t0, p0 = self.hist[0]
        if t - t0 < self.window / 2:
            self.speed = None
        else:
            moved = float(np.mean(np.hypot(pts[:, 0] - p0[:, 0], pts[:, 1] - p0[:, 1])))
            self.speed = moved / palm / (t - t0)
        return self.speed


class Typist:
    """Turns a stream of per-frame guesses into typed text."""

    def __init__(self, labels, min_conf=0.6, hold=0.5, gap=1.0, smooth=8, still=1.0, still_time=0.15,
                 moving_signs=MOVING_SIGNS, moving_factor=3.0):
        self.labels, self.min_conf, self.hold, self.gap = list(labels), min_conf, hold, gap
        self.still_limit, self.still_time = still, still_time
        self.moving_signs, self.moving_factor = set(moving_signs), moving_factor
        self.buf = deque(maxlen=smooth)
        self.text, self.cand, self.cand_t, self.last, self.no_hand = "", None, 0.0, None, 0.0
        self.still_t = 0.0
        self.guess, self.conf, self.still = None, 0.0, False
        self.log = []

    def update(self, proba, dt, t=0.0, speed=None):
        """proba = model probabilities for this frame, or None if no hands.
        speed = MotionMeter speed (palm-lengths per second), or None if unknown."""
        if proba is None:
            self.buf.clear()
            self.cand, self.guess, self.still = None, None, False
            self.no_hand += dt
            if self.no_hand >= self.gap:
                self.last = None
                if self.text and not self.text.endswith(" "):
                    self.text += " "
                    return "word"
            return None
        self.no_hand = 0.0
        self.buf.append(np.asarray(proba, dtype=np.float64))
        avg = np.mean(self.buf, axis=0)
        top = int(np.argmax(avg))
        lab = self.labels[top]
        self.guess, self.conf = lab, float(avg[top])
        limit = self.still_limit * (self.moving_factor if lab in self.moving_signs else 1.0)
        self.still = self.still_limit <= 0 or (speed is not None and speed <= limit)
        if lab == NOT_A_SIGN or self.conf < self.min_conf:
            self.cand = None
            return None
        if top != self.cand:
            self.cand, self.cand_t, self.still_t = top, 0.0, 0.0
            if top != self.last:
                self.last = None  # a different sign appeared, so a repeat is allowed later
        else:
            self.cand_t += dt
            self.still_t = self.still_t + dt if self.still else 0.0
        settled = self.still_limit <= 0 or self.still_t >= self.still_time
        if self.cand_t >= self.hold and settled and top != self.last:
            self.text += lab if len(lab) == 1 else lab + " "
            self.last = top
            self.log.append((t, lab))
            return "sign"
        return None
