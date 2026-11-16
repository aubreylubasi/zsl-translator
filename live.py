"""Live Zambian Sign Language recognition from the webcam (or a video file).

    python live.py                        # webcam
    python live.py --video clip.mp4       # try it on a video
    python live.py --speak                # say each word out loud when you pause

Hold a sign STILL for about half a second and it gets typed. While your hand
moves (word signs, moving between letters) nothing is typed. Drop your hands
for a second to start a new word. To repeat a letter (the LL in HELLO),
drop your hand briefly between them.

    python live.py --still 0              # turn the still-hand rule off (old behaviour)

Keys:  Q quit   C clear   BACKSPACE delete   S speak the text
"""
import argparse
import os
import shutil
import subprocess
import time

import cv2
import joblib

from zsl.features import FEATURE_DIM, NOT_A_SIGN, frame_features
from zsl.jsonmodel import JsonModel
from zsl.tracker import Tracker, draw
from zsl.typist import MotionMeter, Typist

ROOT = os.path.dirname(os.path.abspath(__file__))
DESKTOP_MODEL = os.path.join(ROOT, "models", "zsl_model.joblib")
PHONE_MODEL = os.path.join(ROOT, "app", "www", "zsl_model.json")


def load_model(path):
    """Returns (model, labels, backend). Uses the phone app's model if nothing is trained here yet."""
    if path is None:
        path = DESKTOP_MODEL if os.path.exists(DESKTOP_MODEL) else PHONE_MODEL
        if path == PHONE_MODEL:
            print("No model trained on this computer yet, so using the phone app's demo model "
                  "(app/www/zsl_model.json).")
    if not os.path.exists(path):
        raise SystemExit("No model yet - run train.py first.")
    if path.endswith(".json"):
        model = JsonModel(path)
        if model.feature_dim != FEATURE_DIM:
            raise SystemExit("Model is from an older version - run train.py again.")
        return model, model.labels, None
    bundle = joblib.load(path)
    if bundle["feature_dim"] != FEATURE_DIM:
        raise SystemExit("Model is from an older version - run train.py again.")
    return bundle["model"], bundle["labels"], bundle.get("backend")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    ap.add_argument("--model", help="model file (.joblib or the phone's .json); default: "
                                     "models/zsl_model.joblib, else app/www/zsl_model.json")
    ap.add_argument("--min-conf", type=float, default=0.6, help="how sure the AI must be (0-1)")
    ap.add_argument("--hold", type=float, default=0.5, help="seconds a sign must be held")
    ap.add_argument("--still", type=float, default=1.0,
                    help="how still the hand must be, in palm-lengths per second (0 = off)")
    ap.add_argument("--still-time", type=float, default=0.15,
                    help="seconds the hand must be still at the end of the hold")
    ap.add_argument("--speak", action="store_true", help="speak each word when you pause")
    ap.add_argument("--no-window", action="store_true", help="just print the text (for testing)")
    ap.add_argument("--backend", default="auto", choices=["auto", "tasks", "legacy"])
    args = ap.parse_args()

    model, labels, trained_with = load_model(args.model)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Can't open the camera. On a Mac: System Settings > Privacy & Security > "
                         "Camera, and allow Terminal (or VS Code).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = Tracker(backend=args.backend)
    if trained_with not in (None, "unknown", tracker.backend):
        print("Note: this model was trained with a different MediaPipe version. If results are "
              "poor, run video_to_dataset.py and train.py again on this computer.")
    typist = Typist(labels, args.min_conf, args.hold, still=args.still, still_time=args.still_time)
    meter = MotionMeter()
    t0, last_t, n = time.time(), None, 0
    seen = {"hands": 0, "not_sign": 0, "still": 0}

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
        speed = meter.update(list(lm.hands.values()), lm.aspect, t)
        event = typist.update(proba, dt, t, speed)
        if proba is not None:
            seen["hands"] += 1
            seen["not_sign"] += typist.guess == NOT_A_SIGN
            seen["still"] += typist.still
        if event == "word" and args.speak:
            speak(typist.text.split()[-1])

        if args.no_window:
            continue
        view = draw(frame.copy(), lm)
        if not args.video:
            view = cv2.flip(view, 1)  # mirror the webcam so it feels natural
        h, w = view.shape[:2]
        if typist.guess == NOT_A_SIGN:
            put(view, "not a sign", (20, 60), 1.2, (170, 170, 170), 3)
        elif typist.guess:
            colour = (0, 255, 0) if typist.conf >= args.min_conf else (0, 165, 255)
            put(view, f"{typist.guess}  {typist.conf * 100:.0f}%", (20, 60), 1.6, colour, 3)
        if typist.guess and args.still > 0:
            put(view, "still" if typist.still else "moving - hold still",
                (20, 100), 0.7, (0, 255, 0) if typist.still else (0, 165, 255))
        cv2.rectangle(view, (0, h - 70), (w, h), (0, 0, 0), -1)
        put(view, typist.text[-40:] + "_", (20, h - 25), 1.2, (255, 255, 255), 2)
        put(view, "Q quit  C clear  BKSP delete  S speak", (w - 430, 30), 0.6)
        cv2.imshow("ZSL Translator", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            typist.text, typist.last = "", None
        if key in (8, 127):
            typist.text, typist.last = typist.text[:-1], None
        if key == ord("s"):
            speak(typist.text)

    cap.release()
    tracker.close()
    if not args.no_window:
        cv2.destroyAllWindows()
    print("\nText:", typist.text.strip())
    if args.no_window:
        print("Typed at:", ", ".join(f"{lab}@{t:.1f}s" for t, lab in typist.log))
        h = max(seen["hands"], 1)
        still = f"{100 * seen['still'] / h:.0f}%" if args.still > 0 else "rule off"
        print(f"Frames with hands: {seen['hands']} | hand still: {still} | "
              f"read as not a sign: {100 * seen['not_sign'] / h:.0f}%")


if __name__ == "__main__":
    main()
