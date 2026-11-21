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

## Limits

The model reads one frame at a time, so it learns the *handshape and position*
of a word sign, not the movement itself. Signs with the same handshape in the
same place but a different movement will be confused. The fix is a model that
reads about 1 to 2 seconds of movement (next step in the main README).
