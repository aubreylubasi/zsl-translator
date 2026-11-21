"""Learn signs from a video: cut it into signs, keep only hand/body numbers.

AUTO - videos that show the sign's name on screen (like the Signs of Love ZSL
series, where the letter appears in a yellow circle):

    python video_to_dataset.py Alphabet-ZSL.mp4 --labels A-Z --preview

MANUAL - any other video: write a CSV of  label,start,end  (seconds or m:ss)

    python video_to_dataset.py greetings.mp4 --segments greetings.csv

ONE SIGN PER VIDEO - every video shows just one sign:

    python video_to_dataset.py videos/hello_*.mp4 --label HELLO

NOT A SIGN - teach it what to ignore. Use videos of word signs, natural
signing (a news interpreter), people talking with their hands, hands resting:

    python video_to_dataset.py videos/znbc_news.mp4 --none

In a CSV, "_" also means "not a sign" (for example the intro of a video).

Output: data/<name>.npz (numbers only, no pictures) and data/<name>_segments.csv
(where each sign was found - fix it by hand and re-run with --segments if needed).
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

from zsl.features import FEATURE_DIM, NOT_A_SIGN, frame_features, parse_labels
from zsl.tracker import Tracker, draw

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_time(t):
    t = t.strip()
    if ":" in t:
        m, s = t.split(":", 1)
        return int(m) * 60 + float(s)
    return float(t)


def read_segments_csv(path, fps):
    segs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in ("label", "#") or row[0].startswith("#"):
                continue
            label, start, end = row[0].strip(), parse_time(row[1]), parse_time(row[2])
            segs.append((label, int(round(start * fps)), int(round(end * fps))))
    return segs


def runs_of(mask, min_len):
    """Contiguous True runs as (start, end) frame pairs, end inclusive."""
    runs, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    return runs


def auto_segments(thumbs, card_on, labels, fps, min_len_s=0.5):
    """Cut the video wherever the on-screen label changes."""
    n = len(labels)
    min_len = max(2, int(min_len_s * fps))
    diff = np.zeros(len(thumbs))
    diff[1:] = np.abs(np.diff(thumbs, axis=0)).mean(axis=(1, 2))
    runs = runs_of(card_on, min_len)
    if not runs:
        sys.exit("Could not find the label card. Check --label-box / --card-color, "
                 "or use --segments with a CSV.")
    noise = max(1.0, 3 * float(np.median(diff[card_on])))
    cands = sorted(((diff[i], i) for s, e in runs for i in range(s + 1, e + 1)
                    if diff[i] > noise), reverse=True)
    need = n - len(runs)
    cuts = []
    for strength, i in cands:
        if len(cuts) >= need:
            break
        if all(abs(i - c) >= min_len for c in cuts) and all(i - s >= min_len and e - i >= min_len for s, e in runs if s <= i <= e):
            cuts.append(i)
    if need < 0 or len(cuts) < need:
        found = len(runs) + max(len(cuts), 0)
        return None, f"Expected {n} signs but found {found} on-screen label changes."
    segs = []
    for s, e in runs:
        inner = sorted(c for c in cuts if s < c <= e)
        bounds = [s] + inner + [e + 1]
        segs += [(bounds[k], bounds[k + 1] - 1) for k in range(len(bounds) - 1)]
    return [(lab, s, e) for lab, (s, e) in zip(labels, segs)], None


def save_preview(video, segs, lms, path, width=320):
    cap = cv2.VideoCapture(video)
    tiles = []
    for lab, s, e in segs:
        mid = (s + e) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, fr = cap.read()
        if not ok:
            continue
        if lms[mid] is not None:
            draw(fr, lms[mid])
        h = int(fr.shape[0] * width / fr.shape[1])
        fr = cv2.resize(fr, (width, h))
        cv2.rectangle(fr, (0, 0), (width, 34), (0, 0, 0), -1)
        cv2.putText(fr, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(fr)
    cap.release()
    cols = 6
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    cv2.imwrite(path, grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", metavar="video", help="one or more video files")
    ap.add_argument("--labels", help="signs in the order they appear: A-Z, 'HELLO,THANK YOU' or a .txt file")
    ap.add_argument("--segments", help="CSV with label,start,end instead of auto-detection")
    ap.add_argument("--label", help="the whole video is this one sign")
    ap.add_argument("--none", action="store_true",
                    help="the whole video is NOT a sign (word signs, chatting, resting hands)")
    ap.add_argument("--label-box", default="0.58,0.10,0.84,0.56",
                    help="where the on-screen label sits, x0,y0,x1,y1 as fractions (default fits Signs of Love videos)")
    ap.add_argument("--card-color", default="yellow", choices=["yellow", "any"],
                    help="colour of the label card; 'any' if the video contains only the signs")
    ap.add_argument("--trim", type=float, default=0.2,
                    help="ignore this fraction at the start and end of each sign (hand moving in/out)")
    ap.add_argument("--width", type=int, default=960, help="shrink frames to this width before tracking")
    ap.add_argument("--name", help="output name (default: video file name; only for one video)")
    ap.add_argument("--preview", action="store_true", help="save a picture of each detected sign to check the labels")
    ap.add_argument("--backend", default="auto", choices=["auto", "tasks", "legacy"])
    args = ap.parse_args()
    if args.none:
        args.label, args.trim = NOT_A_SIGN, 0.0
    if sum(bool(v) for v in (args.labels, args.segments, args.label)) != 1:
        ap.error("give one of --labels (auto mode), --segments (CSV), --label or --none")
    if args.name and len(args.videos) > 1:
        ap.error("--name only works with one video")
    for video in args.videos:
        process(video, args)


def process(video, args):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Can't open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    x0, y0, x1, y1 = (float(v) for v in args.label_box.split(","))
    name = args.name or os.path.splitext(os.path.basename(video))[0]
    find_card = bool(args.labels)

    print(f"Reading {os.path.basename(video)}: {total} frames at {fps:.0f} fps "
          f"(tracking with MediaPipe)...")
    tracker = Tracker(backend=args.backend)
    feats, lms, thumbs, card = [], [], [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if find_card:  # watch the on-screen label
            roi = fr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            thumbs.append(cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (32, 32),
                                     interpolation=cv2.INTER_AREA).astype(np.float32))
            if args.card_color == "yellow":
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                yellow = ((hsv[..., 0] >= 18) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 150))
                card.append(yellow.mean() > 0.25)
            else:
                card.append(True)
        small = cv2.resize(fr, (args.width, int(h * args.width / w))) if w > args.width else fr
        lm = tracker.process(small, i * 1000.0 / fps)
        lms.append(lm if lm.hands else None)
        feats.append(frame_features(lm))
        i += 1
        if total and i % max(1, total // 10) == 0:
            print(f"  {100 * i // total}%")
    cap.release()
    tracker.close()
    backend = tracker.backend

    if args.label:
        segs = [(args.label, 0, len(feats) - 1)]
    elif args.segments:
        segs = read_segments_csv(args.segments, fps)
    else:
        labels = parse_labels(args.labels)
        segs, err = auto_segments(np.array(thumbs), np.array(card), labels, fps)
        if err:
            sys.exit(err + " Try adjusting --label-box, or write the times in a CSV and use --segments.")

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    seg_csv = os.path.join(ROOT, "data", f"{name}_segments.csv")
    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "start", "end"])
        for lab, s, e in segs:
            wr.writerow([lab, f"{s / fps:.2f}", f"{(e + 1) / fps:.2f}"])

    X, y, seg_id, pos, ts = [], [], [], [], []
    print("\nSign   time          frames used")
    for k, (lab, s, e) in enumerate(segs):
        length = e - s + 1
        a, b = s + int(args.trim * length), e - int(args.trim * length)
        used = 0
        for j in range(a, min(b, len(feats) - 1) + 1):
            if feats[j] is not None:
                X.append(feats[j]); y.append(lab); seg_id.append(k); pos.append((j - s) / length)
                ts.append(j / fps)
                used += 1
        flag = "" if used >= 10 else "   <- few hand frames, check this one"
        shown = "(not a sign)" if lab == NOT_A_SIGN else lab
        print(f"{shown:6} {s / fps:5.1f}-{(e + 1) / fps:5.1f}s   {used:4d}{flag}")

    out = os.path.join(ROOT, "data", f"{name}.npz")
    np.savez_compressed(out, X=np.array(X, dtype=np.float32).reshape(-1, FEATURE_DIM),
                        y=np.array(y), seg=np.array(seg_id), pos=np.array(pos, dtype=np.float32),
                        t=np.array(ts, dtype=np.float32),
                        src=np.array(name), backend=np.array(backend))
    print(f"\nSaved {len(X)} examples of {len(segs)} signs -> {os.path.relpath(out, ROOT)}")
    print(f"Sign times -> {os.path.relpath(seg_csv, ROOT)}")
    if args.preview:
        prev = os.path.join(ROOT, "data", f"{name}_preview.jpg")
        save_preview(video, segs, lms, prev)
        print(f"Check picture -> {os.path.relpath(prev, ROOT)}  (is every label on the right sign?)")


if __name__ == "__main__":
    main()
