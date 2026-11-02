# Bonisa

**Bonisa** ("show" in isiZulu) turns a phone video of a physics experiment into a
verified scientific measurement. Film a pendulum, and Bonisa measures *g* with an
honest uncertainty, or refuses to give a number and tells you how to fix your setup.

Built for the Meta Global AI Developer Hackathon (Oct–Nov 2026). All code in this
repository was written during the hackathon.

## How it works

1. **SAM 3.1** (Meta Model API) tracks the pendulum bob through every frame from a
   plain-language description such as "orange ball", and finds an A4 sheet on the
   wall to convert pixels to metres.
2. **Physics fit**: Bonisa fits a damped oscillation to the motion to get the period,
   fits the arc to locate the pivot and measure the string length from the video,
   corrects for swing angle, and computes g = 4π²L/T² with propagated uncertainty.
3. **Verification**: eight physics checks (tracking coverage, time resolution,
   number of swings, model fit, swing angle, camera tilt via the A4 sheet's aspect
   ratio, video-vs-tape length, precision). Any failure means no number is reported.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then paste your Meta Model API key into .env
python scripts/measure.py path\to\pendulum.mp4 --object "orange ball" --length 0.80
```

Results land in `runs/<video name>/`: `result.json`, `plot.png` and the cached SAM tracks.

## Tests (no API key needed)

```bash
pip install -r requirements-dev.txt
pytest
python tests/make_video.py data/fake_pendulum.mp4
BONISA_FAKE_SAM=1 python scripts/measure.py data/fake_pendulum.mp4 --length 0.70
```
