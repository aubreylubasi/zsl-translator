"""Self-test of reading lesson captions. No video, MediaPipe or Tesseract needed.

    python tests/test_captions.py

Checks that:
  * caption text is cleaned into sign names ('(alternative option) "ZSL"' -> ZSL)
  * readings are grouped into signs, one-off misreadings are ignored, and a
    slightly different reading of the same caption (LANQUAGE) doesn't split it
  * signs in a frozen (damaged) part, or touching one, are left out
  * video_to_dataset.py reads the speed column and ignores # rows
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from caption_segments import clean_label, group_samples, part_lost  # noqa: E402
from video_to_dataset import read_segments_csv  # noqa: E402

STEP = 0.25


def readings(spec):
    """spec: list of (label or None, count, frozen) -> time-ordered samples."""
    out, t = [], 0.0
    for label, n, frozen in spec:
        for _ in range(n):
            out.append((t, label, frozen))
            t += STEP
    return out


def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return ok


def main():
    good = True
    print("Cleaning caption text")
    for raw, want in [('"Hello"', "HELLO"), ("Hi", "HI"), ("1", "1"),
                      ('(alternative option) "ZSL"', "ZSL"), ("Mdd y", None), ("", None),
                      ("; » | Fg", "FG"), ("Thank you", "THANK YOU")]:
        good &= check(f"{raw!r} -> {want}", clean_label(raw) == want)

    print("Grouping readings into signs")
    runs = group_samples(readings([
        (None, 4, False), ("HELLO", 10, False), ("FG", 1, False), (None, 3, False),
        ("GOOD", 8, False), (None, 2, False),
        ("LANGUAGE", 3, False), ("LANQUAGE", 2, False), ("LANGUAGE", 4, False),
    ]), STEP)
    good &= check("three signs found", [r["label"] for r in runs] == ["HELLO", "GOOD", "LANGUAGE"])
    good &= check("HELLO lasts 2.5 s", abs((runs[0]["end"] - runs[0]["start"]) - 2.5) < 1e-6)
    good &= check("LANGUAGE kept in one piece", len(runs[2]["times"]) == 9)

    print("Damaged parts")
    runs = group_samples(readings([("ZSL", 6, False), ("ZSL", 20, True)]), STEP)
    good &= check("frozen end -> part lost", part_lost(runs[0], []))
    runs = group_samples(readings([(None, 20, True), ("VERY", 6, False)]), STEP)
    good &= check("starts right after a frozen stretch -> part lost",
                  part_lost(runs[0], [(0.0, 5.0)]))
    runs = group_samples(readings([(None, 20, False), ("DEAF", 12, False)]), STEP)
    good &= check("clean sign kept", not part_lost(runs[0], [(0.0, 2.0)]))

    print("Reading the CSV")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# made by caption_segments.py\nlabel,start,end,speed\n"
                    "HELLO,10,12,0.75\n# skipped: SO,13,14,0.75\nGOOD,0:15,0:16.5\n")
        segs, speeds = read_segments_csv(path, 30)
    good &= check("2 signs, # rows ignored", [s[0] for s in segs] == ["HELLO", "GOOD"])
    good &= check("speed 0.75, missing speed = 1", speeds == [0.75, 1.0])
    good &= check("times in frames", segs[1][1:] == (450, 495))

    print("\nAll good." if good else "\nSomething failed.")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
