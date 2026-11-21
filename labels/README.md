# Sign lists

Ready-made lists for `collect.py`, taken from the ZNAD / Peace Corps poster
"Easy to Use Zambian Sign Language" (2009). The poster is the guide for *how*
to sign each word; it is not included here (it is marked "not for sale").

| File | Signs |
|---|---|
| `numbers.txt` | 0 to 10 |
| `health.txt` | 19 health signs (sick, pain, medicine, malaria…) |
| `questions.txt` | why, what, when, who, where |
| `family.txt` | family, mother, father, brother, sister |
| `food.txt` | 11 food signs (water, nshima, beans…) |
| `work.txt` | 14 work and everyday signs (teacher, school, good, bad, Zambia…) |
| `greetings.txt` | 19 greetings and everyday signs (hello, morning, night, deaf, sign, language…) from the ZSL lesson 1 video, not the poster |

A `*` after a sign means it is signed with a movement. The app types those
without waiting for the hand to stop. Remove the `*` from a sign you hold
still. Lines starting with `#` are notes.

## Recording

```bash
python collect.py --labels labels/numbers.txt --signer P01
python collect.py --labels labels/health.txt  --signer P01 --seconds 2.5
python collect.py --labels _ --signer P01 --samples 10   # "not a sign": always record some
python train.py
```

- Give word signs a longer take (`--seconds 2.5`) so the whole movement fits.
- Do every sign the same way each time, start to finish, inside the take.
- Keep your head and both shoulders in view.
- Record several people (P01, P02…), with their consent, then test honestly on
  one the model never saw: `python train.py --test webcam_P02`.
- Train only what you've recorded: `train.py` learns every label in `data/`,
  so you can add one list at a time.

## Motion model

Once your data has word signs marked `*`, `train.py` switches to the **motion
model** by itself. Instead of one frame, each guess looks at the last 1.2 s of
signing as 6 snapshots, so it learns the movement, not just the handshape.
Two signs with the same handshape and place but a different movement can then
be told apart. It also handles the letters, so one model does everything.

- `python train.py --motion off` forces the old one-frame model;
  `--motion on` forces the motion model.
- Recordings made before 21 Nov 2026 have no frame times, so 30 fps is assumed.
  Record word signs again with the new `collect.py` for the best results.
- Moving word signs are typed after 0.3 s of the same guess (letters: 0.5 s).
- Check it: `python tests/test_motion.py`.
