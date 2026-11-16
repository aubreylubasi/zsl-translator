"""Record signs from the webcam. Saves ONLY hand/body numbers - never video.

    python collect.py --labels A-Z --signer P01
    python collect.py --labels _ --signer P01 --samples 10    # "not a sign" takes

"_" means NOT A SIGN: during those takes, sign words, move between letters,
talk with your hands, scratch your face, rest your hands. This teaches the
model what to ignore. Each "_" take lasts twice as long.

Only record people who have agreed to take part. Use a code like P01, P02
instead of real names.

Keys:  SPACE record the sign shown   N next sign   B previous sign   Q save and quit
"""
import argparse
import os
import time

import cv2
import numpy as np

from zsl.features import FEATURE_DIM, NOT_A_SIGN, frame_features, parse_labels
from zsl.tracker import Tracker, draw

ROOT = os.path.dirname(os.path.abspath(__file__))


def put(img, text, org, scale=0.8, colour=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="A-Z, 'HELLO,THANK YOU' or a .txt file")
    ap.add_argument("--signer", default="P01", help="a code for the person signing (not their real name)")
    ap.add_argument("--samples", type=int, default=5, help="takes per sign")
    ap.add_argument("--seconds", type=float, default=1.5, help="length of each take")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", default="auto", choices=["auto", "tasks", "legacy"])
    args = ap.parse_args()

    labels = parse_labels(args.labels)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Can't open the camera. On a Mac: System Settings > Privacy & Security > "
                         "Camera, and allow Terminal (or VS Code).")
    tracker = Tracker(backend=args.backend)
    X, y, seg, pos = [], [], [], []
    done = {lab: 0 for lab in labels}
    idx, take = 0, 0
    state, t_state, take_frames = "idle", 0.0, []
    t0 = time.time()
    print(__doc__)

    while idx < len(labels):
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        lm = tracker.process(frame, (now - t0) * 1000)
        f = frame_features(lm)
        lab = labels[idx]
        seconds = args.seconds * (2 if lab == NOT_A_SIGN else 1)

        if state == "countdown" and now - t_state >= 1.0:
            state, t_state, take_frames = "recording", now, []
        if state == "recording":
            if f is not None:
                take_frames.append((now - t_state, f))
            if now - t_state >= seconds:
                state = "idle"
                if len(take_frames) >= 5:
                    for t, feat in take_frames:
                        X.append(feat); y.append(lab); seg.append(take); pos.append(t / seconds)
                    take += 1
                    done[lab] += 1
                    if done[lab] >= args.samples:
                        idx += 1
                else:
                    print("No hands seen in that take - try again with better light.")

        view = cv2.flip(draw(frame.copy(), lm), 1)  # mirror view, like a selfie
        h, w = view.shape[:2]
        if idx < len(labels):
            cur = labels[idx]
            shown = "NOT A SIGN (move, sign words, rest)" if cur == NOT_A_SIGN else cur
            put(view, f"Sign: {shown}   take {done[cur] + 1}/{args.samples}", (20, 45), 1.1, (0, 255, 255), 3)
        put(view, f"Signer {args.signer} | {sum(done.values())} takes saved", (20, 85), 0.7)
        put(view, "SPACE record   N next   B back   Q save+quit", (20, h - 20), 0.7)
        if f is None:
            put(view, "no hands seen", (w - 230, 45), 0.8, (0, 0, 255))
        if state == "countdown":
            put(view, "get ready...", (w // 2 - 110, h // 2), 1.4, (0, 255, 255), 3)
        elif state == "recording":
            cv2.circle(view, (w - 40, 90), 15, (0, 0, 255), -1)
            put(view, "RECORDING", (w - 210, 98), 0.8, (0, 0, 255))
        cv2.imshow("ZSL - collect", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if state == "idle":
            if key == ord(" "):
                state, t_state = "countdown", now
            elif key == ord("n"):
                idx += 1
            elif key == ord("b"):
                idx = max(0, idx - 1)

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    if not X:
        print("Nothing recorded.")
        return
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    name = f"webcam_{args.signer}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = os.path.join(ROOT, "data", f"{name}.npz")
    np.savez_compressed(out, X=np.array(X, dtype=np.float32).reshape(-1, FEATURE_DIM), y=np.array(y),
                        seg=np.array(seg), pos=np.array(pos, dtype=np.float32), src=np.array(name),
                        backend=np.array(tracker.backend))
    print(f"Saved {take} takes ({len(X)} frames) -> data/{name}.npz   Now run: python train.py")


if __name__ == "__main__":
    main()
