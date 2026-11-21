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
   `live.py` smooths the guesses and types a sign once it's shown for half a
   second **and the hand has settled**. While the hand is moving (word signs,
   moving between letters) nothing is typed.
5. **Not a sign:** the model can also learn a "not a sign" answer (label `_`)
   from videos of word signs and everyday hand movement, so it stops forcing
   every handshape into a letter.
6. **Left-handed signers:** training adds mirrored copies, so it works for
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
| ZNBC news interpreter, 30 s filmed off a TV (15 Nov) | Typed only "H A": the model only knows still letters, so word signs confused it |

All of these use one signer. The next milestone is testing on people the model
has never seen (see step 5 below).

### Junk letters on word signs (16 Nov)

Stand-in footage from a public ASL video dictionary: 57 word-sign clips
(94 s), 26 single-letter clips, and lower-quality 25 fps copies of the letters.
The same MediaPipe tracking as the app was used. This is ASL, not ZSL, so treat
the numbers as a direction, not a score.

| | Junk letters on word signs | Letters it already read right, still typed |
|---|---|---|
| Before (hold 0.5 s) | 42 in 57 clips | 7 of 7 |
| Still-hand rule, current model | 27 in 57 clips (35 → 20 when played back to back) | 6 of 7 (also 6 of 7 at 25 fps) |
| `live.py` on 34 s of unseen word signs | 20 → 13 | |
| Demo model that also learned "not a sign" from 29 *other* word clips | 20 → 1 on the same 34 s; 19 → 1 over 28 unseen clips | Same as without "not a sign" (23 of 26 letter clips with the still-hand rule, 26 without) |

The "not a sign" demo still read fingerspelling by the word-clip signers as
letters, so it learned handshapes rather than just "this video looks
different". The real model needs ZSL "not a sign" data (step 4b below).

## Phone app (Android, works offline)

Download **ZSL-Translator.apk** from the
[latest test release](../../releases/tag/latest-apk). Open it on your phone and
allow installs from that source. Point the back camera at the signer, or press
**Flip** to sign to the front camera yourself. Every build is signed with the
same key (`app/android/app/zsl-test.jks`), so a new APK installs straight over
the last one. Builds from before 21 Nov 2026 used a different key: uninstall
those once first. The test key is public, so for a real release save a private
key as the secrets `ZSL_KEYSTORE_B64`, `ZSL_KEYSTORE_PASSWORD`, `ZSL_KEY_ALIAS`
and `ZSL_KEY_PASSWORD`; the workflow then uses it instead.

If your head and shoulders aren't in view, or the hand is right up against the
camera, the app says "Step back" and types nothing. The model measures the hand
against the shoulders, so without them its guesses can't be trusted.

- `app/www/`: the app itself (HTML and JavaScript). It uses MediaPipe's hand and
  body tracking and the same model as the desktop version (`zsl_model.json`).
- `app/android/`: a small Android wrapper that serves the app from inside the
  APK and blocks all internet access, so nothing leaves the phone.
- `.github/workflows/android.yml`: GitHub builds a new APK on every push.

The **Letters / Words** button switches between two models: the alphabet
(`app/www/zsl_model.json`) and word signs from the lesson videos
(`app/www/zsl_words.json`, 17 greetings from Lesson 1 so far). The typed text
is kept when you switch.

To update the phone's models, run `python train.py` (it rewrites
`app/www/zsl_model.json`), then commit and push. To rebuild only the Words
model, put the word-sign data in its own folder and run
`python train.py --data data/words --phone-out app/www/zsl_words.json`.
Training on letters and words together in one model is the long-term plan;
until the alphabet data is back on the Mac, keep them separate so the phone
doesn't lose the alphabet.

## Roadmap

- [x] Alphabet from video, live webcam recognition, speech output
- [ ] Record Deaf signers with consent (with ZNAD, schools for the Deaf, Signs of Love)
- [x] Motion model over the last 1.2 s, and 54 word signs plus numbers from the ZNAD poster ready to record (`labels/`)
- [ ] Record the word signs and train them
- [x] Offline Android app (MediaPipe runs on the phone)
- [x] Don't type while the hand is moving (still-hand rule)
- [ ] "Not a sign" examples from ZSL signers and ZNBC clips (the code is ready)
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

An optional 4th column, `speed`, says how fast the signing is compared with
real life (`0.75` for slow motion, blank or `1` for normal). The motion model
uses it so slow-motion signs are learnt at real speed.

### Lesson videos with captions

Some teaching videos write each sign's name in a caption bar, like the "Zambian
Sign Language Communication" lessons. `caption_segments.py` reads the captions
and writes the CSV for you (it needs `brew install tesseract ffmpeg`):

```bash
python caption_segments.py videos/zsl_lesson1_greetings.mp4 --speed 0.75 --skip So,And
python video_to_dataset.py videos/zsl_lesson1_greetings.mp4 \
    --segments data/zsl_lesson1_greetings_captions.csv --preview
```

- Check `data/<name>_captions.jpg` (the caption read for each sign) and the
  preview. Rows starting with `#` in the CSV are left out: skipped signs,
  captions it couldn't read (`?`: type the name and remove the `#`), and signs
  in a damaged part of the video.
- **Damaged downloads:** a download that didn't finish can have missing or
  frozen parts, which makes OpenCV stop early or get the times wrong. The script
  spots this, makes a repaired copy (`videos/<name>_fixed.mp4`) and prints the
  exact `video_to_dataset.py` command to use. Signs in the damaged parts are
  lost, so download the video again if you can.
- These lessons show each sign in slow motion first (about 0.75 of real speed,
  measured from hand speeds), then at normal speed without captions.

## 2. Train

```bash
python train.py
```

This takes under a minute. It prints a test score and the signs it mixes up.

## 3. Try it live

```bash
python live.py            # add --speak to hear each word
```

If you haven't trained anything on this computer yet, it uses the phone app's
model (`app/www/zsl_model.json`), so you can try it straight away.

- Hold a sign **still** for about half a second and it gets typed. The screen
  says "moving - hold still" while your hand is moving.
- `--still 0` turns the still-hand rule off. `--still 1.5` is looser and
  `--still 0.75` stricter (palm-lengths per second). `--still-time` sets how
  long the hand must settle (default 0.15 s).
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

## 4a. Add numbers and word signs

`labels/` has ready-made lists from the ZNAD poster "Easy to Use Zambian Sign
Language": numbers 0 to 10 and 54 word signs (health, questions, family, food,
work). Record a list, then train:

```bash
python collect.py --labels labels/health.txt --signer P01 --seconds 2.5
python train.py
```

Signs marked `*` in a list are signed with a movement, so the app types them
without waiting for the hand to stop. When the data has such signs,
`train.py` builds a **motion model** that reads the last 1.2 s of signing
instead of one frame, so it learns the movement too (`--motion off` turns it
off). The phone app and `live.py` pick it up automatically. See
`labels/README.md` for tips.

## 4b. Teach it what is NOT a sign

This stops it typing junk while someone signs words. Give it examples of hands
doing things that aren't letters: word signs, natural signing, moving between
letters, talking with your hands, scratching your face, resting.

```bash
python video_to_dataset.py videos/znbc_news.mp4 --none     # a whole video of word signs
python collect.py --labels _ --signer P01 --samples 10     # record your own
python train.py
```

- In a times CSV, a `_` row also means "not a sign", for example the intro of
  a video: `_,0:00,0:07`.
- `train.py` now also prints **junk let through** (not-a-sign frames read as a
  letter) and **signs wrongly ignored**. You want both low.
- If a video has one sign only, use `--label HELLO` instead of a CSV.

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
| `caption_segments.py` | Reads the sign names a lesson video shows on screen and writes the times CSV |
| `collect.py` | Records signs from the webcam (numbers only) |
| `labels/` | Sign lists from the ZNAD poster: numbers and word signs |
| `train.py` | Trains the recogniser on everything in `data/` |
| `live.py` | Live camera to text and speech |
| `zsl/tracker.py` | MediaPipe hand and body tracking (works with old and new MediaPipe) |
| `zsl/features.py` | Turns hand points into numbers the model learns from |
| `zsl/typist.py` | Still-hand meter and the rules for when a sign gets typed |
| `zsl/jsonmodel.py` | Lets `live.py` use the phone app's model file |
| `tests/test_typist.py` | Quick check of the typing rules, and that the phone app matches Python |
| `tests/test_captions.py` | Quick check of caption reading, damaged-video handling and the speed column |
| `app/` | The offline Android app (see above) |

## Limits of this version

- The alphabet-only model reads **handshapes** frame by frame. Moving letters
  such as J and Z can be confused with similar still letters. They get a looser
  still-hand limit. Training with word signs switches to the motion model,
  which sees movement (tested on made-up data so far, not real signers).
- Letters flashed very quickly (still for less than about 0.15 s) are skipped.
  That is the price of not typing junk; use `--still 0` for the old behaviour.
- One signer so far, so accuracy on new people will be lower until more signers
  are added.

## Credits

The ZSL alphabet video used for the first experiments is by
[Signs of Love](https://www.signsoflove.org). The video and the hand data taken
from it are not in this repository. The demo model (`app/www/zsl_model.json`)
was trained on it for non-commercial testing only.

The greetings signs come from the video "Zambian Sign Language Communication,
Lesson 1: Greetings". Only the sign times (`data/zsl_lesson1_greetings_captions.csv`)
are in this repository, not the video or its hand data. Ask its makers before
releasing a model trained on it.

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
