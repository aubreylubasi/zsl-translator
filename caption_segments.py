"""Read the sign names a lesson video shows on screen, and write when each sign is.

For teaching videos that caption every sign, like the "Zambian Sign Language
Communication" lessons (a dark bar near the bottom with "Hello" and how to sign
it). Step 1 reads the captions, step 2 is the usual video_to_dataset.py:

    python caption_segments.py videos/zsl_lesson1_greetings.mp4 --speed 0.75
    python video_to_dataset.py videos/zsl_lesson1_greetings.mp4 \\
        --segments data/zsl_lesson1_greetings_captions.csv --preview

Output:
  data/<name>_captions.csv  label,start,end,speed  (open it and check it)
  data/<name>_captions.jpg  the caption it read for each sign, to check by eye

--speed   how fast the signing is compared with real life. These lessons show
          each sign in slow motion first (about 0.75 = three-quarter speed).
          The motion model uses it so slow-motion signs still look real-speed.
--skip    signs to leave out, e.g. --skip So,And (the video itself says these
          are English words, not natural ZSL).
--box     where the caption bar is, as fractions of the picture: x0,y0,x1,y1.

Rows starting with # in the CSV are ignored by video_to_dataset.py: skipped
signs, signs lost to a damaged video, and captions it could not read ("?").
Fix a "?" row by typing the sign name and removing the #.

Damaged downloads: a half-finished download can have missing or frozen parts.
OpenCV then stops early or gets the times wrong, so this script makes a
repaired copy (videos/<name>_fixed.mp4, needs ffmpeg) and prints the exact
video_to_dataset.py command to use with it. Signs in frozen parts are left out.

Needs Tesseract for reading text:  brew install tesseract   (and  brew install ffmpeg)
"""
import argparse
import csv
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BOX = "0,0.79,0.78,0.94"
SAMPLE_FPS = 4.0          # caption checks per second
FROZEN_DIFF = 0.15        # mean grey-level change below this = the picture did not move
MIN_SAMPLES = 2           # a label must be read at least this many times to count
BRIDGE_S = 1.5            # same label again within this gap = still the same sign
MIN_CLEAN_S = 1.0         # shorter than this after cutting out frozen parts = left out


# ---------- reading text ----------

def clean_label(text):
    """'(alternative option) "ZSL"' -> 'ZSL';  'Hello' -> 'HELLO';  junk -> None."""
    text = re.sub(r"\([^)]*\)?", " ", text)
    text = re.sub(r"[^A-Za-z0-9' -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -'")
    if not text or len(text) > 30 or not re.search(r"[A-Za-z0-9]", text):
        return None
    words = text.split()
    if len(words) > 1 and any(len(w) == 1 and not w.isdigit() for w in words):
        return None  # "Mdd y": a word broken by bad reading
    return text.upper()


def title_mask(gray_bar):
    """White caption title (first line of the bar) as a mask, without the quote marks."""
    h, w = gray_bar.shape
    strip = gray_bar[int(0.11 * h):int(0.51 * h), int(0.06 * w):]
    m = (strip > 190).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = [st[i] for i in range(1, n) if st[i, 4] > 15]
    keep = np.zeros(m.shape, bool)
    if not comps:
        return keep
    top = max(c[3] for c in comps)
    base = float(np.median([c[1] + c[3] for c in comps if c[3] >= 0.5 * top]))
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if area <= 15:
            continue
        on_line = y + ch >= base - 0.2 * top                     # letters sit on the line
        dot = ch <= 0.22 * top and 0.5 * ch <= cw <= 0.45 * top    # the dot on an i
        if on_line or dot:                                         # quote marks float above it
            keep |= lab == i
    return keep


def ocr(mask, tmpdir):
    img = np.where(mask, 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    path = os.path.join(tmpdir, "line.png")
    cv2.imwrite(path, img)
    out = subprocess.run(["tesseract", path, "-", "--psm", "7"], capture_output=True, text=True)
    return out.stdout.strip()


# ---------- grouping ----------

def similar(a, b):
    """Same caption read slightly differently ('LANGUAGE' / 'LANQUAGE')."""
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def group_samples(samples, step, min_samples=MIN_SAMPLES, bridge=BRIDGE_S):
    """samples: time-ordered (t, label or None, frozen).  Returns runs of one caption:
    dicts with label (the most common reading), times and frozen (one flag per reading)."""
    runs = []
    for t, label, frozen in samples:
        if label is None:
            continue
        last = runs[-1] if runs else None
        if last and t - last["times"][-1] <= bridge and any(similar(label, v) for v in last["reads"]):
            last["times"].append(t); last["frozen"].append(frozen); last["reads"].append(label)
        else:
            runs.append({"times": [t], "frozen": [frozen], "reads": [label]})
    merged = []  # a one-off misreading in the middle can split a sign in two
    for r in runs:
        if len(r["times"]) < min_samples:
            continue
        last = merged[-1] if merged else None
        if (last and r["times"][0] - last["times"][-1] <= bridge
                and similar(max(set(r["reads"]), key=r["reads"].count),
                            max(set(last["reads"]), key=last["reads"].count))):
            for k in ("times", "frozen", "reads"):
                last[k] += r[k]
        else:
            merged.append(r)
    for r in merged:
        r["label"] = max(set(r["reads"]), key=r["reads"].count)
        r["start"], r["end"] = r["times"][0], r["times"][-1] + step
        r["clean"] = clean_part(r["times"], r["frozen"], step)
    return merged


def clean_part(times, frozen, step):
    """Longest stretch without frozen readings, as (start, end), or None."""
    best, cur = None, None
    for t, f in zip(times, frozen):
        if f:
            cur = None
            continue
        cur = (cur[0], t + step) if cur and t - cur[1] < step * 1.5 else (t, t + step)
        if best is None or cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return best


def part_lost(run, frozen, near=0.5):
    """True when part of the sign is frozen, or a frozen stretch touches its start or end
    (the caption may have started inside a damaged part). Half a sign teaches it wrong."""
    s, e = run["clean"]
    if (s - run["start"]) + (run["end"] - e) > near:
        return True
    return any(abs(fs_end - s) <= near or abs(fs_start - e) <= near for fs_start, fs_end in frozen)


def stretches(times, flags, step, min_len):
    """(start, end) of runs where flags is True, at least min_len seconds long."""
    out, start = [], None
    for t, f in zip(list(times) + [None], list(flags) + [False]):
        if f and start is None:
            start = t
        elif not f and start is not None:
            end = (t if t is not None else times[-1] + step)
            if end - start >= min_len:
                out.append((start, end))
            start = None
    return out


# ---------- the video ----------

def check_timing(video):
    """OpenCV reads frames one after another. If frames are missing, or it stops early,
    frame counting gives wrong times. Returns (ok, list of gaps in seconds)."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    gaps, last, n = [], None, 0
    while cap.grab():
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if last is not None and ms - last > 3 * 1000.0 / fps:
            gaps.append((last / 1000.0, ms / 1000.0))
        last, n = ms, n + 1
    cap.release()
    stopped_early = total and n < 0.98 * total
    return not gaps and not stopped_early, gaps, stopped_early, n, total


def repair(video):
    fixed = os.path.splitext(video)[0] + "_fixed.mp4"
    if not shutil.which("ffmpeg"):
        sys.exit("This video is damaged and ffmpeg is needed to repair it: brew install ffmpeg")
    print(f"Making a repaired copy with even timing -> {os.path.relpath(fixed)} ...")
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=r_frame_rate", "-of", "default=nw=1:nk=1", video],
                          capture_output=True, text=True).stdout.strip() or "30"
    subprocess.run(["ffmpeg", "-y", "-nostdin", "-v", "error", "-err_detect", "ignore_err",
                    "-i", video, "-vf", f"fps={rate}", "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "18", fixed],
                   stderr=subprocess.DEVNULL, check=True)
    return fixed


def bar_showing(grey, box):
    """The caption bar darkens everything under its top edge: compare a strip just above
    the edge with a strip just below it, column by column."""
    h, w = grey.shape
    x0, y0, x1 = int(box[0] * w), int(box[1] * h), int(box[2] * w)
    d = int(0.05 * h)
    if y0 - d < 0:
        return True
    above = grey[y0 - d:y0 - 2, x0:x1].astype(np.float32).mean(axis=0)
    below = grey[y0 + 3:y0 + d, x0:x1].astype(np.float32).mean(axis=0)
    return float(((above - below) > 20).mean()) > 0.6


def scan(video, box, tmpdir):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    every = max(1, int(round(fps / SAMPLE_FPS)))
    x0, y0, x1, y1 = box
    samples, prev, i = [], None, 0
    print(f"Reading captions in {os.path.basename(video)} ({total / fps:.0f} s)...")
    while cap.grab():
        if i % every == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            h, w = fr.shape[:2]
            small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 90),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            frozen = prev is not None and float(np.abs(small - prev).mean()) < FROZEN_DIFF
            prev = small
            grey = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            bar_on = bar_showing(grey, box)
            bar = grey[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            mask = title_mask(bar) if bar_on else None
            has_text = mask is not None and 0.001 < mask.mean() < 0.12
            label = clean_label(ocr(mask, tmpdir)) if has_text else None
            samples.append((i / fps, label, frozen, bar_on))
            if total and (i // every) % int(30 * SAMPLE_FPS) == 0 and i:
                print(f"  {100 * i // total}%")
        i += 1
    cap.release()
    return samples, every / fps


def save_check_picture(video, rows, box, path, width=480):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x0, y0, x1, y1 = box
    tiles = []
    for tag, label, s, e in rows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((s + e) / 2 * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        bar = fr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        bar = cv2.resize(bar, (width, int(bar.shape[0] * width / bar.shape[1])))
        head = np.zeros((30, width, 3), np.uint8)
        colour = (0, 255, 255) if tag == "" else (80, 80, 255)
        text = f"{label}  {s:.1f}-{e:.1f}s" + (f"  ({tag})" if tag else "")
        cv2.putText(head, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, bar]))
    cap.release()
    if not tiles:
        return False
    cols = 3
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    cv2.imwrite(path, grid)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="signing speed vs real life, e.g. 0.75 for the slow-motion part (default 1)")
    ap.add_argument("--skip", default="", help="signs to leave out, e.g. So,And")
    ap.add_argument("--box", default=DEFAULT_BOX, help=f"caption bar x0,y0,x1,y1 (default {DEFAULT_BOX})")
    ap.add_argument("--name", help="output name (default: video file name)")
    args = ap.parse_args()

    if not shutil.which("tesseract"):
        sys.exit("Tesseract is needed to read the captions: brew install tesseract")
    if not os.path.isfile(args.video):
        sys.exit(f"Can't find {args.video}")
    if not 0.1 <= args.speed <= 2:
        sys.exit("--speed should be between 0.1 and 2 (1 = real speed)")
    box = tuple(float(v) for v in args.box.split(","))
    name = args.name or os.path.splitext(os.path.basename(args.video))[0]
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    video = args.video
    ok, gaps, early, n, total = check_timing(video)
    if not ok:
        print("\nThis video is DAMAGED (probably a download that didn't finish properly):")
        for a, b in gaps:
            print(f"  no pictures from {a:.1f} s to {b:.1f} s")
        if early:
            print(f"  only {n} of {total} frames can be read")
        print("Signs in those parts are lost. Download it again if you can.\n")
        video = repair(video)

    with tempfile.TemporaryDirectory() as tmp:
        samples, step = scan(video, box, tmp)
    runs = group_samples([(t, lab, fr) for t, lab, fr, _ in samples], step)

    covered = np.zeros(len(samples), bool)
    times = np.array([s[0] for s in samples])
    for r in runs:
        covered |= (times >= r["start"]) & (times < r["end"])
    unread = stretches(times, [s[3] and not c for s, c in zip(samples, covered)], step, 2.0)
    frozen = stretches(times, [s[2] for s in samples], step, 2.0)

    rows = []  # (tag, label, start, end)
    for r in runs:
        clean = r["clean"]
        if r["label"] in skip:
            rows.append(("skipped", r["label"], r["start"], r["end"]))
        elif clean is None or clean[1] - clean[0] < MIN_CLEAN_S:
            rows.append(("damaged", r["label"], r["start"], r["end"]))
        elif part_lost(r, frozen):
            rows.append(("part lost", r["label"], *clean))
        else:
            rows.append(("", r["label"], *clean))
    rows += [("unreadable", "?", a, b) for a, b in unread]
    rows.sort(key=lambda r: r[2])

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out = os.path.join(ROOT, "data", f"{name}_captions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Made by caption_segments.py from {os.path.basename(args.video)}. "
                "Rows starting with # are left out.\n")
        wr = csv.writer(f)
        wr.writerow(["label", "start", "end", "speed"])
        for tag, label, s, e in rows:
            row = [label, f"{s:.2f}", f"{e:.2f}", f"{args.speed:g}"]
            if tag:
                note = {"skipped": "# skipped", "damaged": "# damaged (frozen picture)",
                        "part lost": "# part of the sign is in a damaged/frozen part",
                        "unreadable": "# caption not readable, type the sign name"}[tag]
                f.write(f"{note}: {','.join(row)}\n")
            else:
                wr.writerow(row)

    good = [r for r in rows if not r[0]]
    print(f"\n{len(good)} signs found:")
    for tag, label, s, e in rows:
        mark = f"   <- {tag}" if tag else ""
        print(f"  {label:14} {s:6.1f}-{e:6.1f}s{mark}")
    if frozen:
        print("Frozen or still picture (title cards, or damage): "
              + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in frozen))
    pic = os.path.join(ROOT, "data", f"{name}_captions.jpg")
    if save_check_picture(video, rows, box, pic):
        print(f"\nCheck picture -> {os.path.relpath(pic, ROOT)}")
    print(f"Sign times    -> {os.path.relpath(out, ROOT)}")
    extra = f" --name {name}" if video != args.video else ""
    print(f"\nNext:\n  python video_to_dataset.py {video} --segments {os.path.relpath(out, ROOT)}"
          f"{extra} --preview")


if __name__ == "__main__":
    main()
