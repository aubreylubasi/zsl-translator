# ZSL Translator: Zambian Sign Language to text

A camera-based translator for **Zambian Sign Language (ZSL)**. It watches
someone sign, recognises the signs and types them out, and it can speak them
aloud. Very little technology supports Zambian Sign Language, so this project
starts with the alphabet and is built to grow into words and sentences.

**Privacy by design:** the app never saves video. It keeps only the positions
of the hands and shoulders as numbers, with no faces and no pictures.

## How it works

1. **Tracking:** Google's MediaPipe finds 21 points on each hand plus the
   shoulders in every camera frame.
2. **Features:** each hand's shape is measured relative to its wrist and palm
   size, and its position relative to the shoulders. The result doesn't change
   with how far you stand from the camera.
3. **Learning from video:** `video_to_dataset.py` cuts a teaching video into
   signs automatically. It reads the on-screen label (for example, the letter in
   a yellow circle), so you don't have to label anything by hand.
4. **Recognition:** a small neural network (scikit-learn) classifies each frame.
   `live.py` smooths the guesses and types a sign once it's held for half a
   second.
5. **Left-handed signers:** training adds mirrored copies, so it works for
   either hand.

## Results so far

| Test | Result |
|---|---|
| Learned the full ZSL alphabet from one 70-second video | 26/26 letters found and labelled automatically |
| Played back on that video | Typed **A to Z** in order, with no mistakes |
| Flipped and lowered to 480p (like a left-handed signer on a worse camera) | All 26 letters in order, plus one stray letter in the intro |
| Held-back frames, same signer | 77% of single frames correct before smoothing |
| Phone app in Chrome, camera shown L, O, V, E | Typed **LOVE** |
| Phone app maths compared with the Python version (92 frames) | Same answer on every frame |

All of these use one signer. The next milestone is testing on people the model
has never seen (see step 5 below).

## Phone app (Android, works offline)

Download **ZSL-Translator.apk** from the
[latest test release](../../releases/tag/latest-apk). Open it on your phone and
allow installs from that source. Point the back camera at the signer, or press
**Flip** to sign to the front camera yourself. If an older test build is
installed, uninstall it first.

- `app/www/`: the app itself (HTML and JavaScript). It uses MediaPipe's hand and
  body tracking and the same model as the desktop version (`zsl_model.json`).
- `app/android/`: a small Android wrapper that serves the app from inside the
  APK and blocks all internet access, so nothing leaves the phone.
- `.github/workflows/android.yml`: GitHub builds a new APK on every push.

To update the phone's model, run `python train.py` (it rewrites
`app/www/zsl_model.json`), then commit and push.

## Roadmap

- [x] Alphabet from video, live webcam recognition, speech output
- [ ] Record Deaf signers with consent (with ZNAD, schools for the Deaf, Signs of Love)
- [ ] Words with movement (sequence model over frames)
- [x] Offline Android app (MediaPipe runs on the phone)
- [ ] Other direction: speech to sign

---

## Setup (Mac)

MediaPipe needs a supported Python version. Check yours with `python3 --version`.

- **Apple Silicon Mac (M1 or newer):** Python 3.10 or newer works.
- **Intel Mac:** use **Python 3.10, 3.11 or 3.12**. On 3.13 or newer, install
  3.12 first with `brew install python@3.12`, then use `python3.12` in place of
  `python3` below.

Open **Terminal** in this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each time you open a new Terminal later, run `source .venv/bin/activate` first.

## 1. Learn from a video

Put the video in `videos/`, then run:

```bash
python video_to_dataset.py videos/Alphabet-ZSL.mp4 --labels A-Z --preview
```

- It finds each letter by watching the yellow circle where the letter appears.
- `--preview` saves `data/<name>_preview.jpg`. Check that every label sits on
  the right sign.

For a video without on-screen labels, write the times in a CSV:

```
label,start,end
HELLO,0:03.5,0:05.0
THANK YOU,0:06.2,0:08.0
```

Then run `python video_to_dataset.py videos/greetings.mp4 --segments greetings.csv`.

## 2. Train

```bash
python train.py
```

This takes under a minute. It prints a test score and the signs it mixes up.

## 3. Try it live

```bash
python live.py            # add --speak to hear each word
```

- Hold a sign still for about half a second and it gets typed.
- Drop your hands for 1 second to start a new word.
- To repeat a letter (the LL in HELLO), drop your hand briefly in between.
- Keys: **Q** quit, **C** clear, **Backspace** delete, **S** speak.

## 4. Add real people (the important part)

```bash
python collect.py --labels A-Z --signer P01 --samples 5
python train.py
```

- **Space** records the sign shown: a 1-second countdown, then 1.5 seconds of signing.
- Use codes like P01 and P02, not names, and only record people who have agreed.

## 5. Honest test

Record someone new (for example P09), then run:

```bash
python train.py --test webcam_P09
```

This scores the model on a person it has never seen.

---

## Files

| File | What it does |
|---|---|
| `video_to_dataset.py` | Cuts a video into signs and keeps only hand numbers |
| `collect.py` | Records signs from the webcam (numbers only) |
| `train.py` | Trains the recogniser on everything in `data/` |
| `live.py` | Live camera to text and speech |
| `zsl/tracker.py` | MediaPipe hand and body tracking (works with old and new MediaPipe) |
| `zsl/features.py` | Turns hand points into numbers the model learns from |
| `app/` | The offline Android app (see above) |

## Limits of this version

- It reads **handshapes** frame by frame. Moving letters such as J and Z can be
  confused with similar still letters.
- One signer so far, so accuracy on new people will be lower until more signers
  are added.

## Credits

The ZSL alphabet video used for the first experiments is by
[Signs of Love](https://www.signsoflove.org). The video and the hand data taken
from it are not in this repository. The demo model (`app/www/zsl_model.json`)
was trained on it for non-commercial testing only.

Inspired by the Sign Champions' BSL translator (Longitude Explorer Prize 2021),
rebuilt for Zambia.

## Troubleshooting

- **"Can't open the camera":** go to System Settings > Privacy & Security >
  Camera and turn on Terminal (or VS Code). Then restart Terminal.
- **"No matching distribution found for mediapipe":** your Python version is
  too new for your Mac. See Setup above.
- **Certificate error when downloading models:** run `pip install --upgrade certifi`.
- **Warning about a different MediaPipe version:** run steps 1 and 2 again on
  this computer.
