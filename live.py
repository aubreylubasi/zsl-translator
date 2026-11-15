"""Live Zambian Sign Language recognition from the webcam (or a video file).

    python live.py                        # webcam
    python live.py --video clip.mp4       # try it on a video
    python live.py --speak                # say each word out loud when you pause

Hold a sign still for about half a second and it gets typed. Drop your hands
for a second to start a new word. To repeat a letter (the LL in HELLO),
drop your hand briefly between them.

Keys:  Q quit   C clear   BACKSPACE delete   S speak the text
"""
import argparse
import os
import shutil
import subprocess
import time
from collections import deque

import cv2
import joblib
import numpy as np

from zsl.features import FEATURE_DIM, frame_features
from zsl.tracker import Tracker, draw

ROOT = os.path.dirname(os.path.abspath(__file__))


def speak(text):
    text = text.strip()
    if not text:
        return
    if shutil.which("say"):  # built into macOS
        subprocess.Popen(["say", text])
    else:
        print(f"[speak] {text}")


def put(img, text, org, scale=0.8, colour=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


class Typist:
    """Turns a stream of per-frame guesses into typed text."""

    def __init__(self, labels, min_conf=0.6, hold=0.5, gap=1.0, smooth=8):
        self.labels, self.min_conf, self.hold, self.gap = labels, min_conf, hold, gap
        self.buf = deque(maxlen=smooth)
        self.text, self.cand, self.cand_t, self.last, self.no_hand = "", None, 0.0, None, 0.0
        self.guess, self.conf = None, 0.0
        self.log = []

    def update(self, proba, dt, t):
        """proba = model probabilities for this frame, or None if no hands."""
        if proba is None:
            self.buf.clear()
            self.cand, self.guess = None, None
            self.no_hand += dt
            if self.no_hand >= self.gap:
                self.last = None
                if self.text and not self.text.endswith(" "):
                    self.text += " "
                    return "word"
            return None
        self.no_hand = 0.0
        self.buf.append(proba)
        avg = np.mean(self.buf, axis=0)
        top = int(np.argmax(avg))
        self.guess, self.conf = self.labels[top], float(avg[top])
        if self.conf < self.min_conf:
            self.cand = None
            return None
        if top != self.cand:
            self.cand, self.cand_t = top, 0.0
            if top != self.last:
                self.last = None  # a different sign appeared, so a repeat is allowed later
        else:
            self.cand_t += dt
        if self.cand_t >= self.hold and top != self.last:
            lab = self.labels[top]
            self.text += lab if len(lab) == 1 else lab + " "
            self.last = top
            self.log.append((t, lab))
            return "sign"
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "zsl_model.joblib"))
    ap.add_argument("--min-conf", type=float, default=0.6, help="how sure the AI must be (0-1)")
    ap.add_argument("--hold", type=float, default=0.5, help="seconds a sign must be held")
    ap.add_argument("--speak", action="store_true", help="speak each word when you pause")
    ap.add_argument("--no-window", action="store_true", help="just print the text (for testing)")
    ap.add_argument("--backend", default="auto", choices=["auto", "tasks", "legacy"])
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit("No model yet - run train.py first.")
    bundle = joblib.load(args.model)
    if bundle["feature_dim"] != FEATURE_DIM:
        raise SystemExit("Model is from an older version - run train.py again.")
    model, labels = bundle["model"], bundle["labels"]

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Can't open the camera. On a Mac: System Settings > Privacy & Security > "
                         "Camera, and allow Terminal (or VS Code).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = Tracker(backend=args.backend)
    if bundle.get("backend") not in (None, "unknown", tracker.backend):
        print("WARNING: this model was trained with a different MediaPipe version. "
              "Run video_to_dataset.py and train.py again on this computer.")
    typist = Typist(labels, args.min_conf, args.hold)
    t0, last_t, n = time.time(), None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps if args.video else time.time() - t0
        n += 1
        if args.video and frame.shape[1] > 960:
            frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
        dt = 0.0 if last_t is None else t - last_t
        last_t = t
        lm = tracker.process(frame, t * 1000)
        f = frame_features(lm)
        proba = None if f is None else model.predict_proba(f.reshape(1, -1))[0]
        event = typist.update(proba, dt, t)
        if event == "word" and args.speak:
            speak(typist.text.split()[-1])

        if args.no_window:
            continue
        view = draw(frame.copy(), lm)
        if not args.video:
            view = cv2.flip(view, 1)  # mirror the webcam so it feels natural
        h, w = view.shape[:2]
        if typist.guess:
            colour = (0, 255, 0) if typist.conf >= args.min_conf else (0, 165, 255)
            put(view, f"{typist.guess}  {typist.conf * 100:.0f}%", (20, 60), 1.6, colour, 3)
        cv2.rectangle(view, (0, h - 70), (w, h), (0, 0, 0), -1)
        put(view, typist.text[-40:] + "_", (20, h - 25), 1.2, (255, 255, 255), 2)
        put(view, "Q quit  C clear  BKSP delete  S speak", (w - 430, 30), 0.6)
        cv2.imshow("ZSL Translator", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            typist.text = ""
        if key in (8, 127):
            typist.text = typist.text[:-1]
        if key == ord("s"):
            speak(typist.text)

    cap.release()
    tracker.close()
    if not args.no_window:
        cv2.destroyAllWindows()
    print("\nText:", typist.text.strip())
    if args.no_window:
        print("Typed at:", ", ".join(f"{lab}@{t:.1f}s" for t, lab in typist.log))


if __name__ == "__main__":
    main()
